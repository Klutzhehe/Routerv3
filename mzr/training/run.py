"""Training entry point.

    python -m mzr.training.run --stage 0 --device cuda --updates 500

One process, batched at the tensor level: `--batch` boards advance together and
one PPO update consumes `--rollout` macro-steps of them. There is no worker
pool -- the thing that capped every previous thread in this repo at 2 CPUs.

Eval reports **both** argmax and sampled completion on the same held-out
boards, every time. `neuroroute/` had a real mode/mean gap -- sampled beat
argmax by 11-19 points even untrained -- and a policy that only works sampled
is leaning on exploration noise to reach the goal. The gate is argmax,
sustained across 3 consecutive evals (the stage-0 "met the gate once at u275
then regressed" scar).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mzr.env.route_env import EnvConfig, RouteEnv
from mzr.models.policy import PriorPolicy
from mzr.training.curriculum import EVAL_SEEDS, STAGES
from mzr.training.ppo import PPOConfig, RolloutBuffer, ppo_update
from mzr.world.engine import STATUS_DONE, WorldConfig


def make_env(stage, batch: int, device: str, seed: int) -> RouteEnv:
    return RouteEnv(
        EnvConfig(
            spec=stage.board_spec(),
            world=WorldConfig(
                batch_size=batch,
                max_nets=stage.generator.num_nets + 6,
                max_macro_steps=stage.max_macro_steps,
                max_steps_per_frontier=stage.max_macro_steps,
                ripup=stage.ripup,
                device=device,
            ),
            generator=stage.generator,
            reward=stage.reward,
            max_episode_steps=stage.max_macro_steps,
            seed=seed,
        )
    )


def expert_completion(stage, device: str, n_boards: int) -> float:
    """Sequential + PathFinder expert on the held-out seeds. Computed once --
    the boards are fixed, so the baseline does not move."""
    from mzr.world.expert import ExpertConfig, route_world_board

    seeds = EVAL_SEEDS[:n_boards]
    env = make_env(stage, batch=len(seeds), device=device, seed=0)
    env.reset(seeds=seeds)
    comp = []
    for b in range(len(seeds)):
        r = route_world_board(env.world, b, ExpertConfig(iterations=6), negotiate=True)
        comp.append(r.completion)
    return sum(comp) / len(comp)


@torch.no_grad()
def evaluate(policy, stage, device: str, n_boards: int) -> dict:
    """Argmax and sampled completion on the fixed held-out seeds."""
    policy.eval()
    seeds = EVAL_SEEDS[:n_boards]
    out = {}
    for arm, det in (("argmax", True), ("sampled", False)):
        env = make_env(stage, batch=len(seeds), device=device, seed=0)
        obs = env.reset(seeds=seeds)
        while True:
            act = policy.act(obs, deterministic=det)
            step = env.step(act["action"])
            obs = step.obs
            if step.done:
                break
        c = env.completion()
        out[f"{arm}_completion"] = float(c.mean())
        out[f"{arm}_perfect"] = float((c >= 0.999).float().mean())
    policy.train()
    return out


def collect(env: RouteEnv, policy, buf: RolloutBuffer, n_steps: int, obs):
    """Run `n_steps` macro-steps, filling `buf`. Returns the trailing obs."""
    for _ in range(n_steps):
        act = policy.act(obs, deterministic=False)
        step = env.step(act["action"])
        # board reward for the critic = sum of per-frontier reward + board term
        board_r = step.reward.sum(dim=1) + step.board_reward
        buf.add(
            obs=obs,
            action=act["action"],
            logp=act["logp"],
            mask=act["mask"],
            board_reward=board_r,
            value=act["value"],
            done=step.done * torch.ones(board_r.shape[0], device=board_r.device),
        )
        obs = step.obs
        if step.done:
            obs = env.reset()
    return obs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--field-width", type=int, default=48)
    p.add_argument("--token-width", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-boards", type=int, default=64)
    p.add_argument("--checkpoint-dir", default="mzr_ckpt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    stage = STAGES[args.stage]
    dev = args.device
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / f"stage{args.stage}.jsonl"

    env = make_env(stage, args.batch, dev, seed=1_000 + args.seed)
    policy = PriorPolicy(
        num_layers=stage.layers,
        field_width=args.field_width,
        token_width=args.token_width,
    ).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    ppo_cfg = PPOConfig(lr=args.lr, bc_coef=stage.bc_coef0)

    start = 0
    best = -1.0
    latest = ckpt_dir / f"stage{args.stage}_latest.pt"
    if args.resume and latest.exists():
        blob = torch.load(latest, map_location=dev)
        # strict=False: new heads (h/g/f at stage 2) will not exist in an
        # earlier checkpoint, and the loader must not choke on that.
        missing, unexpected = policy.load_state_dict(blob["policy"], strict=False)
        if not missing and not unexpected:
            try:
                opt.load_state_dict(blob["opt"])
            except ValueError:
                print("optimizer state shape mismatch -- starting Adam cold")
        else:
            print(f"partial load ({len(missing)} missing, {len(unexpected)} unexpected) -- Adam cold")
        start = blob.get("update", 0)
        best = blob.get("best", -1.0)
        print(f"resumed from update {start}, best {best:.3f}")

    expert_baseline = None
    if stage.gate[0] == "vs_expert":
        expert_baseline = expert_completion(stage, dev, args.eval_boards)
        print(f"expert (sequential + PathFinder) baseline: {expert_baseline:.3f} "
              f"-- gate is argmax >= {expert_baseline + stage.gate[1]:.3f}")

    obs = env.reset()
    hits = 0  # consecutive evals clearing the gate
    print(f"stage {args.stage}: {stage.name}")
    print(f"gate {stage.gate} | kill: {stage.kill}")

    for u in range(start, args.updates):
        t0 = time.time()
        buf = RolloutBuffer()
        obs = collect(env, policy, buf, args.rollout, obs)
        with torch.no_grad():
            last_v = policy.act(obs, deterministic=False)["value"]
        m = ppo_update(policy, opt, buf, last_v, ppo_cfg)
        dt = time.time() - t0

        line = {"update": u, "sec": round(dt, 2), **{k: round(v, 4) for k, v in m.items()}}

        if (u + 1) % args.eval_every == 0 or u == args.updates - 1:
            ev = evaluate(policy, stage, dev, args.eval_boards)
            line.update({k: round(v, 4) for k, v in ev.items()})

            kind, thr = stage.gate
            score = ev["argmax_completion"]
            if kind == "absolute":
                passed = score >= thr
            elif kind == "vs_expert" and expert_baseline is not None:
                passed = score >= expert_baseline + thr
                line["expert_baseline"] = round(expert_baseline, 4)
            else:
                passed = False  # vs_prior is decided by the stage-3 search eval, not here
            hits = hits + 1 if passed else 0
            line["gate_hits"] = hits

            if score > best:
                best = score
                torch.save(
                    {"policy": policy.state_dict(), "opt": opt.state_dict(),
                     "update": u, "best": best, "stage": args.stage},
                    ckpt_dir / f"stage{args.stage}_best.pt",
                )
            torch.save(
                {"policy": policy.state_dict(), "opt": opt.state_dict(),
                 "update": u, "best": best, "stage": args.stage},
                latest,
            )
            print(
                f"u{u:4d} | argmax {ev['argmax_completion']:.3f} "
                f"sampled {ev['sampled_completion']:.3f} "
                f"(perfect {ev['argmax_perfect']:.2f}) | best {best:.3f} | "
                f"hits {hits} | kl {m['approx_kl']:.3f} clip {m['clip_frac']:.2f} "
                f"ent {m['entropy']:.2f} | {dt:.1f}s"
            )
            if hits >= 3 and kind == "absolute":
                print(f"GATE CLEARED: argmax {score:.3f} >= {thr} for 3 consecutive evals")

        with open(log_path, "a") as f:
            f.write(json.dumps(line) + "\n")

    print(f"done. best argmax completion {best:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
