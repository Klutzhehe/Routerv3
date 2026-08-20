"""Forward/backward smoke test + throughput report for the CFP policy.

Run this first on Colab. It deliberately touches nothing outside
pcbworld/agents/cfp/, so unlike everything under pcbworld/env/ it does not
need pcbworld_pns_bridge built -- if this fails, the problem is the model,
not the Colab build.

The number to actually look at is the reported per-forward time versus the
router's per-net cost. The CFP design bets that GPU time stays far below
env time (docs/AI_ARCHITECTURE.md); if a forward pass is not comfortably
under a millisecond per board on the batch sizes you plan to train with,
that bet is wrong and the config needs shrinking before any training run.

    python scripts/smoke_cfp.py --device cuda --batch-size 32
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
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(3):  # warmup
            policy.act(obs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            policy.act(obs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    per_batch_ms = 1e3 * elapsed / args.iters
    per_board_ms = per_batch_ms / args.batch_size
    print(
        f"\nact() throughput: {per_batch_ms:.2f} ms/batch of {args.batch_size} "
        f"= {per_board_ms:.3f} ms/board ({1e3 / per_board_ms:.0f} boards/s)"
    )
    print("\nOK")


if __name__ == "__main__":
    main()
