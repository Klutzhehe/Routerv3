"""Training entry point.

    python -m neuroroute.training.run --stage 1 --device cuda --updates 200

Reports **completion rate against the non-learned baselines on held-out board
seeds** every eval, not just reward. That framing is not cosmetic: this repo
has a measured case of a policy scoring worse reward while completing more nets
(`docs/RL_PLAN.md`), so a reward curve on its own can move in the wrong
direction and look like progress.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from neuroroute.env.baselines import detour_action, greedy_safe_action, layer_hop_action
from neuroroute.env.observation import FIELD_CHANNELS, head_feature_dim, net_feature_dim
from neuroroute.env.rewards import terminal_reward
from neuroroute.env.route_env import EnvConfig, NeuroRouteEnv
from neuroroute.models.forecaster import forecast_gate
from neuroroute.models.policy import NeuroRoutePolicy
from neuroroute.training.curriculum import (
    default_curriculum,
    demand_baseline,
    episode_targets,
    stage_env_config,
)
from neuroroute.training.ppo import PPOConfig, RolloutBuffer, compute_gae, ppo_update
from neuroroute.world.engine import WorldConfig


def build(args) -> tuple[NeuroRouteEnv, NeuroRoutePolicy, list]:
    stages = default_curriculum(layers_max=args.layers, size=args.size)
    stage = stages[args.stage]

    world = WorldConfig(
        batch_size=args.batch,
        max_heads=args.heads,
        max_nets=max(64, stage.generator.num_nets + 8),
        device=args.device,
        geodesic_refresh=args.geodesic_refresh,
    )
    env = NeuroRouteEnv(stage_env_config(stage, world, EnvConfig(seed=args.seed)))

    L = stage.board.num_layers
    rules = stage.board.rules
    policy = NeuroRoutePolicy(
        field_channels=FIELD_CHANNELS,
        head_features=head_feature_dim(L),
        net_features=net_feature_dim(),
        num_layers=L,
        num_via_classes=rules.num_via_classes,
        num_width_classes=rules.num_width_classes,
        width=args.width,
    ).to(args.device)
    return env, policy, stages


@torch.no_grad()
def evaluate(env: NeuroRouteEnv, policy: NeuroRoutePolicy, seeds: list[int], deterministic: bool = True) -> dict:
    """Roll out the policy and every baseline on the *same* held-out boards."""
    results: dict[str, float] = {}

    def run(policy_fn) -> tuple[float, float, dict]:
        obs = env.reset(seeds)
        rejected = acted = 0
        for _ in range(env.cfg.max_episode_steps):
            act = policy_fn(obs)
            obs, rew, done, info = env.step(act)
            rejected += int(info["rejected"].sum())
            acted += int(info["active"].sum())
            if bool(done.all()):
                break
        _term, metrics = terminal_reward(env.world, env.cfg.reward)
        return (
            float(env.world.completion().mean()),
            rejected / max(1, acted),
            {k: float(v.mean()) for k, v in metrics.items()},
        )

    comp, rej, metrics = run(lambda o: policy.act(o, deterministic=deterministic).actions)
    results["policy/completion"] = comp
    results["policy/rejected_action_rate"] = rej
    results.update({f"policy/{k}": v for k, v in metrics.items()})

    for name, fn in (("greedy", greedy_safe_action), ("detour", detour_action), ("layer_hop", layer_hop_action)):
        c, r, _ = run(fn)
        results[f"{name}/completion"] = c
        results[f"{name}/rejected_action_rate"] = r
    return results


def train(args) -> None:
    torch.manual_seed(args.seed)
    env, policy, stages = build(args)
    stage = stages[args.stage]
    dev = torch.device(args.device)

    ppo = PPOConfig(
        rollout_steps=args.rollout,
        lr=args.lr,
        store_device=args.store_device,
        entropy_coef=args.entropy,
    )
    opt = torch.optim.Adam(policy.parameters(), lr=ppo.lr, eps=1e-5)
    buf = RolloutBuffer(ppo, dev)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (ckpt_dir / "latest.pt").exists():
        state = torch.load(ckpt_dir / "latest.pt", map_location=dev)
        policy.load_state_dict(state["policy"])
        opt.load_state_dict(state["optimiser"])
        print(f"resumed from update {state.get('update', 0)}")

    n_params = sum(p.numel() for p in policy.parameters())
    print("=" * 72)
    print(f"NeuroRoute -- stage {args.stage}: {stage.name}")
    print(f"  {stage.introduces}")
    print(f"  board {stage.board.height_cells}x{stage.board.width_cells} x {stage.board.num_layers} layers, "
          f"{stage.generator.num_nets} nets, pitch {stage.board.pitch_mm:.2f}mm")
    print(f"  B={args.batch} K={args.heads}  ->  {args.batch * args.heads} decisions/step")
    print(f"  policy {n_params/1e6:.2f}M params on {args.device}")
    print("=" * 72, flush=True)

    obs = env.reset()
    history: list[dict] = []
    t_start = time.perf_counter()

    for update in range(args.updates):
        frac = update / max(1, args.updates - 1)
        entropy_coef = ppo.entropy_coef + frac * (ppo.entropy_final - ppo.entropy_coef)

        buf.clear()
        ep_metrics: dict[str, float] = {}
        for _ in range(ppo.rollout_steps):
            out = policy.act(obs)
            next_obs, reward, done, info = env.step(out.actions)
            buf.add(obs, out.actions, out.log_prob, out.value, reward,
                    done.unsqueeze(-1).expand_as(reward), obs.head_mask)
            if "terminal_reward" in info:
                # Board-level terminal reward is shared by that board's heads:
                # completion, pair gap, length matching are all properties of
                # the finished board, not of any one head's trajectory.
                buf.reward[-1] = buf.reward[-1] + (
                    info["terminal_reward"].unsqueeze(-1).expand_as(reward) / max(1, args.heads)
                ).to(buf.reward[-1].device)
                ep_metrics = {k.split("/", 1)[1]: float(v.mean()) for k, v in info.items() if k.startswith("final/")}
            obs = next_obs
            if bool(done.all()):
                targets = episode_targets(env.world, args_latent_shape(env, policy))
                base = demand_baseline(env._demand, args_latent_shape(env, policy))
                obs = env.reset()

        with torch.no_grad():
            last_value = policy.act(obs).value

        rewards = torch.stack(buf.reward).to(dev)
        values = torch.stack(buf.value).to(dev)
        dones = torch.stack(buf.done).to(dev)
        adv, ret = compute_gae(rewards, values, dones, last_value, ppo.gamma, ppo.gae_lambda)

        targets = episode_targets(env.world, args_latent_shape(env, policy))
        stats = ppo_update(policy, opt, buf, adv, ret, ppo, entropy_coef, targets)

        row = {
            "update": update,
            "reward": float(rewards.sum(0).mean()),
            "completion": float(env.world.completion().mean()),
            "entropy_coef": entropy_coef,
            **stats,
            **ep_metrics,
        }
        history.append(row)

        if update % args.log_every == 0:
            el = time.perf_counter() - t_start
            sps = (update + 1) * ppo.rollout_steps * args.batch * args.heads / max(el, 1e-6)
            print(
                f"[{update:5d}] reward {row['reward']:8.2f}  completion {row['completion']:6.1%}  "
                f"pi {stats['policy_loss']:+.3f}  v {stats['value_loss']:.3f}  "
                f"H {stats['entropy']:.2f}  fcast {stats['forecast']:.3f}  "
                f"{sps:,.0f} dec/s",
                flush=True,
            )

        if update > 0 and update % args.eval_every == 0:
            seeds = list(range(args.eval_seed_base, args.eval_seed_base + args.batch))
            ev = evaluate(env, policy, seeds)
            with torch.no_grad():
                out = policy.act(obs)
                gate = forecast_gate(out.forecast, targets, demand_baseline(env._demand, args_latent_shape(env, policy)))
            print("  " + "-" * 68)
            print(f"  eval on held-out seeds {seeds[0]}..{seeds[-1]}")
            print(f"    policy    {ev['policy/completion']:6.1%}   rejected {ev['policy/rejected_action_rate']:5.2%}")
            print(f"    greedy    {ev['greedy/completion']:6.1%}")
            print(f"    detour    {ev['detour/completion']:6.1%}")
            print(f"    layer_hop {ev['layer_hop/completion']:6.1%}")
            print(f"    vias {ev.get('policy/vias', 0):.1f}  detour-ratio {ev.get('policy/detour', 0):.3f}  "
                  f"pair-gap-err {ev.get('policy/pair_gap_error', 0):.3f}  "
                  f"len-in-tol {ev.get('policy/length_within_tol', 0):.1%}")
            print(f"    FORECAST GATE: mae {gate['forecast_mae']:.4f} vs baseline {gate['baseline_mae']:.4f}"
                  f"   corr {gate['forecast_corr']:+.3f} vs {gate['baseline_corr']:+.3f}"
                  f"   {'BEATS baseline' if gate['beats_baseline'] else 'does NOT beat baseline'}")
            print("  " + "-" * 68, flush=True)
            history[-1].update(ev)
            history[-1].update(gate)

            if ev["policy/completion"] >= stage.gate:
                print(f"  *** stage gate {stage.gate:.0%} met -- ready for stage {args.stage + 1} ***", flush=True)

            obs = env.reset()

        if update > 0 and update % args.checkpoint_every == 0:
            # Colab sessions die. Checkpointing every N updates is not
            # optional (docs/RL_PLAN.md lists it as non-negotiable).
            torch.save(
                {"policy": policy.state_dict(), "optimiser": opt.state_dict(),
                 "update": update, "stage": args.stage, "args": vars(args)},
                ckpt_dir / "latest.pt",
            )
            (ckpt_dir / "history.json").write_text(json.dumps(history, indent=1))

    torch.save(
        {"policy": policy.state_dict(), "optimiser": opt.state_dict(),
         "update": args.updates, "stage": args.stage, "args": vars(args)},
        ckpt_dir / "latest.pt",
    )
    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=1))
    print(f"done. checkpoints in {ckpt_dir}")


def args_latent_shape(env: NeuroRouteEnv, policy: NeuroRoutePolicy) -> tuple[int, int, int]:
    """The encoder's output grid, derived rather than hardcoded -- the stem's
    stride is the single source of truth for it."""
    L, H, W = env.world.shape
    return (L, max(1, H // 4), max(1, W // 4))


def main() -> None:
    p = argparse.ArgumentParser(description="Train NeuroRoute")
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--geodesic-refresh", type=int, default=0)
    p.add_argument("--store-device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed-base", type=int, default=900000,
                   help="held-out seeds, deliberately far from the training range")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--checkpoint-dir", default="checkpoints/neuroroute")
    p.add_argument("--resume", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
