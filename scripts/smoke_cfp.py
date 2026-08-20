"""Forward/backward smoke test + throughput report for the CFP policy.

Run this first on Colab. It deliberately touches nothing outside
pcbworld/agents/cfp/, so unlike everything under pcbworld/env/ it does not
need pcbworld_pns_bridge built -- if this fails, the problem is the model,
not the Colab build.

Which of the two reported numbers binds depends on how the trainer is built,
and both readings are right in their own regime:

  * Synchronous VectorEnv (the gymnasium/SB3 default): all workers step, then
    one batched forward. **ms/batch** must be much less than T_pns, the cost
    of ONE net-route -- the workers routed concurrently, so dividing by batch
    size flatters the model by exactly the factor they already gave you.
  * Pipelined / async: workers step independently and the GPU serves whatever
    is ready. The GPU is a shared server, so only **throughput** matters, and
    ms/board is the honest number.

docs/AI_ARCHITECTURE.md picks pipelined, which makes the requirement
`T_pns > (ms/board) * num_workers` rather than `T_pns >> ms/batch`. Both are
printed below; read the one matching the trainer you are building.

T_pns has not been measured yet -- every conclusion drawn from these numbers
is provisional until it is. Measure it alongside the waypoint-fidelity test,
since both need the bridge built.

    python scripts/smoke_cfp.py --device cuda --batch-size 32 --amp --profile

Note that --amp is a batch->=32 tool: at batch 8 the autocast cast overhead on
the attention path costs more than the canvas encoder saves.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

# The repo isn't pip-installed anywhere (see notebooks/00_setup.ipynb --
# Colab runs it straight from the checkout), so put the root on sys.path
# rather than requiring the caller to set PYTHONPATH or cd first.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pcbworld.agents.cfp import CFPConfig, CFPPolicy, make_dummy_observation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-nets", type=int, default=24)
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dim", type=int, default=CFPConfig().dim)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="run act() under fp16 autocast; on a T4 the conv-heavy canvas "
        "encoder is the bottleneck and tensor cores are worth ~2-3x there",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="report the canvas-encoder / everything-else time split",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    policy = CFPPolicy(CFPConfig(dim=args.dim)).to(device)
    print(policy.describe())
    print(f"device={device}")

    obs = make_dummy_observation(
        batch_size=args.batch_size,
        num_nets=args.num_nets,
        canvas_size=args.canvas_size,
        device=device,
    )
    obs.validate()

    action = policy.act(obs)
    kind, slot = action.split(obs.num_nets)
    print(
        f"\nact(): action_index={tuple(action.action_index.shape)} "
        f"field={tuple(action.field.shape)} "
        f"log_prob={action.log_prob.mean().item():+.3f} "
        f"cat_entropy={action.score.cat_entropy.mean().item():.3f} "
        f"field_entropy={action.score.field_entropy.mean().item():.1f} "
        f"value={action.value.mean().item():+.3f}"
    )
    print(f"        kinds sampled={sorted(set(kind.tolist()))} slots={sorted(set(slot.tolist()))}")
    print(f"        planner field range=[{action.planner_field().min():+.2f}, "
          f"{action.planner_field().max():+.2f}]")

    score = policy.evaluate_actions(obs, action.action_index, action.field)
    drift = (score.log_prob - action.log_prob).abs().max().item()
    print(f"\nevaluate_actions() reproduces act() log_prob to {drift:.2e} (should be ~0)")
    assert drift < 1e-3, "act()/evaluate_actions() disagree -- PPO ratios would be wrong"

    # Two entropy coefficients, not one -- see CFPScore.
    loss = (
        -score.log_prob.mean()
        + score.value.pow(2).mean()
        - 0.01 * score.cat_entropy.mean()
        - 1e-4 * score.field_entropy.mean()
    )
    loss.backward()
    ungrad = [n for n, p in policy.named_parameters() if p.grad is None]
    assert not ungrad, f"parameters received no gradient: {ungrad}"
    print(f"backward(): all {sum(1 for _ in policy.parameters())} parameter tensors got gradients")

    # Throughput. Timed under no_grad since rollout collection is the part
    # that has to keep up with the env workers.
    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    def bench(fn, iters: int) -> float:
        with torch.no_grad():
            for _ in range(3):  # warmup, and lets cudnn pick its algorithms
                fn()
            sync()
            start = time.perf_counter()
            for _ in range(iters):
                fn()
            sync()
        return (time.perf_counter() - start) / iters

    if args.amp and device.type == "cpu":
        # CPU has no fp16 conv kernels; torch emulates them and a benchmark
        # that takes seconds on a T4 runs for minutes here. The correctness
        # of the autocast path is covered by tests instead.
        raise SystemExit(
            "--amp is GPU-only in practice: fp16 convolution is emulated on CPU and "
            "the benchmark will appear to hang. Drop --amp, or use --device cuda."
        )

    autocast = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp)

    def one_act() -> None:
        with autocast:
            policy.act(obs)

    per_batch_ms = 1e3 * bench(one_act, args.iters)
    per_board_ms = per_batch_ms / args.batch_size
    print(
        f"\nact() latency: {per_batch_ms:.2f} ms/batch of {args.batch_size}"
        f"{' [fp16 autocast]' if args.amp else ''}"
    )
    print(
        f"  synchronous trainer : needs T_pns >> {per_batch_ms:.1f} ms "
        f"(one net-route vs one whole batched forward)"
    )
    print(
        f"  pipelined trainer   : {1e3 / per_board_ms:.0f} net-decisions/s "
        f"= {per_board_ms:.3f} ms/board; needs T_pns > {per_board_ms:.2f} ms x num_workers"
    )

    if args.profile:

        def one_canvas() -> None:
            with autocast:
                policy.net.canvas_encoder(obs.canvas)

        canvas_ms = 1e3 * bench(one_canvas, args.iters)
        print(
            f"\nprofile: canvas_encoder {canvas_ms:.2f} ms "
            f"({100 * canvas_ms / per_batch_ms:.0f}% of act()), "
            f"towers + heads {per_batch_ms - canvas_ms:.2f} ms"
        )
        print(f"         blocks per canvas stage = {policy.config.stage_blocks(4)}")

    print("\nOK")


if __name__ == "__main__":
    main()
