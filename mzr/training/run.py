"""Training entry point.

    python -m mzr.training.run --stage 0 --device cuda --updates 500

One process, batched at the tensor level: `--batch` boards advance together and
one PPO update consumes `--rollout` macro-steps of them. There is no worker
pool -- the thing that capped every previous thread in this repo at 2 CPUs.

**Pure RL.** `--bc-coef` defaults to the stage's `bc_coef0`, which is 0 for
every implemented stage. Raise it only after a measured plateau, to blend in
expert behaviour cloning.

Boards are generated fresh from seeds -- no solvability pre-filter. The gate is
still absolute 1.00, on the understanding that if the policy stalls a few points
short, the handful of failing eval seeds get **reviewed by hand** (run
`python -m mzr.world.pool --stage S --seeds ...` -- it reports whether the
expert can route each) rather than auto-filtered out of the distribution.

Eval reports **both** argmax and sampled completion, and names the seeds that
did not reach 100% under argmax, so that review is a copy-paste. `neuroroute/`
had a real mode/mean gap -- sampled beat argmax by 11-19 points even untrained
-- and a policy that only works sampled is leaning on exploration noise. The
gate is argmax, sustained 3 consecutive evals (the stage-0 "met the gate once
at u275 then regressed" scar).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mzr.env.route_env import EnvConfig, RouteEnv
from mzr.models.policy import PriorPolicy
from mzr.eval.quality import ProfileAccumulator, quality_verdict, route_quality
from mzr.training.bc import ExpertActions
from mzr.training.curriculum import EVAL_SEEDS, STAGES
from mzr.training.ppo import PPOConfig, RolloutBuffer, ppo_update
from mzr.world.engine import WorldConfig


def make_env(stage, batch: int, device: str, seed: int,
             leg_budget: float = 0.0, copper_seeded: bool | None = None,
             geodesic_refresh: int | None = None) -> RouteEnv:
    """Build the env a stage describes.

    `copper_seeded` / `geodesic_refresh` default to the STAGE's values; passing
    them explicitly is the CLI override. They used to default to
    ``False`` / ``16`` here regardless of the stage, which meant the curriculum
    could not actually choose its own growth mode.
    """
    ds = stage.geodesic_downsample
    H, W = stage.height, stage.width
    # Relaxation propagates one cell per iteration, so the cap has to scale
    # with the grid it runs on. At ds=4 a 48x48 board was a 12x12 grid and 96
    # was plenty; at ds=1 it is 48x48 and a route that winds around obstacles
    # needs more than that, and an under-relaxed field is worse than a coarse
    # one. The loop breaks as soon as it converges, so this is a cap, not a cost.
    iters = max(96, 3 * (H + W) // max(1, ds))
    return RouteEnv(
        EnvConfig(
            spec=stage.board_spec(),
            world=WorldConfig(
                batch_size=batch,
                max_nets=stage.generator.num_nets + 6,
                # A k-pin net needs k-1 legs; a diff pair needs 2. This
                # multiplies F and therefore fr_geo, the dominant memory term.
                max_legs=max(2, stage.generator.max_pins_per_net - 1),
                leg_budget_frac=leg_budget,
                copper_seeded=stage.copper_seeded if copper_seeded is None else copper_seeded,
                geodesic_refresh=(stage.geodesic_refresh if geodesic_refresh is None
                                  else geodesic_refresh),
                geodesic_downsample=ds,
                geodesic_iterations=iters,
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


_EVAL_ENV: RouteEnv | None = None


@torch.no_grad()
def evaluate(policy, stage, device: str, eval_seeds: list[int], with_sampled: bool = True,
             *, copper_seeded: bool | None = None, geodesic_refresh: int | None = None) -> dict:
    """Held-out completion on the fixed seeds.

    Reuses one persistent env across calls -- rebuilding it (and regenerating
    every board, and recomputing every geodesic field) each eval was ~half the
    eval cost. `with_sampled` gates the second full episode: the gate is argmax,
    so the sampled arm is a diagnostic and does not need to run every time.

    Returns `argmax_fail_seeds` -- the seeds not at 100% under argmax. On an
    absolute-1.0 gate those are the whole story; a mean hides them.
    """
    global _EVAL_ENV
    policy.eval()
    if _EVAL_ENV is None or _EVAL_ENV.cfg.world.batch_size != len(eval_seeds):
        _EVAL_ENV = make_env(stage, batch=len(eval_seeds), device=device, seed=0,
                             copper_seeded=copper_seeded,
                             geodesic_refresh=geodesic_refresh)
    env = _EVAL_ENV

    out: dict = {}
    arms = [("argmax", True)] + ([("sampled", False)] if with_sampled else [])
    prof = ProfileAccumulator()
    for arm, det in arms:
        obs = env.reset(seeds=eval_seeds)
        while True:
            action = policy.act(obs, deterministic=det)["action"]
            # Sampled per step while frontiers are still live -- reading this
            # after the loop averages over an empty set. See ProfileAccumulator.
            if arm == "argmax":
                prof.update(policy, obs, action)
            step = env.step(action)
            obs = step.obs
            if step.done:
                break
        c = env.completion()
        out[f"{arm}_completion"] = float(c.mean())
        out[f"{arm}_perfect"] = float((c >= 0.999).float().mean())
        if arm == "argmax":
            out["argmax_fail_seeds"] = [
                int(sd) for sd, ok in zip(eval_seeds, (c >= 0.999).tolist()) if not ok
            ][:12]
            # Route quality and what the policy chose, on the SAME rollout --
            # completion says a net connected, these say whether it was routed
            # well and whether the policy is steering or just following the
            # geodesic field. See mzr/eval/quality.py for why both are gated.
            out.update(route_quality(env.world))
            out.update(prof.result())
    if not with_sampled:
        out["sampled_completion"] = out["argmax_completion"]
        out["sampled_perfect"] = out["argmax_perfect"]
    policy.train()
    return out


def collect(env: RouteEnv, policy, buf: RolloutBuffer, n_steps: int, obs,
            expert: ExpertActions | None = None):
    """Run `n_steps` macro-steps, filling `buf`. Returns the trailing obs.

    `expert` supplies the behaviour-cloning demonstration. Without it
    `bc_action` stays None and ppo.py's BC term short-circuits to zero -- which
    is what `--bc-coef` silently did before this argument existed.
    """
    for _ in range(n_steps):
        act = policy.act(obs, deterministic=False)
        bc_action = expert.action(obs) if expert is not None else None
        step = env.step(act["action"])
        board_r = step.reward.sum(dim=1) + step.board_reward
        # Per-frontier reward, with the board-level term (failure penalty,
        # terminal completion) shared across live frontiers -- those really are
        # joint outcomes, but the dense shaping above them is not.
        live = act["mask"].float()
        n_live = live.sum(dim=1, keepdim=True).clamp_min(1.0)
        frontier_r = step.reward + (step.board_reward.unsqueeze(1) / n_live) * live
        buf.add(
            obs=obs,
            action=act["action"],
            logp=act["logp"],
            mask=act["mask"],
            board_reward=board_r,
            value=act["value"],
            frontier_reward=frontier_r,
            frontier_value=act["value_f"],
            done=step.done * torch.ones(board_r.shape[0], device=board_r.device),
            bc_action=bc_action,
        )
        obs = step.obs
        if step.done:
            obs = env.reset()
    return obs


def _save(policy, opt, path, u, best, stage):
    torch.save(
        {"policy": policy.state_dict(), "opt": opt.state_dict(),
         "update": u, "best": best, "stage": stage},
        path,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=2, help="PPO epochs per update (was 4)")
    p.add_argument("--minibatches", type=int, default=2)
    p.add_argument("--entropy-coef", type=float, default=0.004)
    p.add_argument("--progress-coef", type=float, default=None,
                   help="override RewardConfig.progress for this run")
    p.add_argument("--wirelength", type=float, default=None,
                   help="terminal weight on excess routed length over the straight "
                        "line, charged per completed net across BOTH frontiers")
    p.add_argument("--corner", type=float, default=None,
                   help="per 45-degree octant of bend beyond the first")
    p.add_argument("--max-macro-steps", type=int, default=None,
                   help="override the stage's episode length. Single-ended "
                        "(copper-seeded) growth needs roughly double, since one "
                        "frontier covers the whole route instead of two halves")
    p.add_argument("--per-frontier-adv", action="store_true",
                   help="per-frontier advantage instead of one board advantage "
                        "broadcast to every frontier (GPAE-style). The MAPPO "
                        "shared advantage is the one blocker that gets WORSE "
                        "as net count grows -- SNR falls as 1/N")
    p.add_argument("--copper-seeded", action=argparse.BooleanOptionalAction, default=None,
                   help="one field per NET (distance to its trunk) instead of one "
                        "per frontier targeting a static pad; implies trunk+spokes. "
                        "Unset = whatever the STAGE asks for (stages 0-3 ask for it). "
                        "See mzr/DESIGN_COPPER_SEEDED.md")
    p.add_argument("--geodesic-refresh", type=int, default=None,
                   help="macro-steps between field refreshes when copper-seeded "
                        "(unset = the stage's value)")
    p.add_argument("--leg-budget", type=float, default=0.0,
                   help="fraction of the leg geodesic ONE frontier may route "
                        "before retiring (0.6 = half plus slack; 0 disables). "
                        "Makes a double-traverse impossible, not just unrewarded")
    p.add_argument("--tip-progress", type=float, default=None,
                   help="dense reward for closing on the partner frontier (the "
                        "other end of the same leg); the only term that charges "
                        "back a redundant traverse")
    p.add_argument("--leg-progress", type=float, default=None,
                   help="shape on the leg's closing gap instead of per-frontier "
                        "distance, so a leg is paid once for ground covered")
    p.add_argument("--bc-decay", type=int, default=0,
                   help="linearly decay --bc-coef to 0 over this many updates "
                        "(0 = constant). Keyed on UPDATE COUNT, not completion: "
                        "DESIGN.md prescribes decaying 'as completion rises', but "
                        "a run where BC is holding completion flat never triggers "
                        "that -- measured, bc_coef 0.5 sat at completion 0.72 for "
                        "250 updates with entropy RISING, so a completion-keyed "
                        "anneal would have held it at 0.5 forever.")
    p.add_argument("--bc-negotiate", action="store_true",
                   help="run PathFinder negotiation when planning demonstrations. "
                        "Off by default: negotiation is what makes the expert a "
                        "strong BASELINE, but for a demonstration the extra "
                        "iterations mostly cost wall-clock in the collect loop.")
    p.add_argument("--bc-coef", type=float, default=None,
                   help="override the stage's BC weight (default: 0 -- pure RL)")
    p.add_argument("--field-width", type=int, default=40)
    p.add_argument("--token-width", type=int, default=96)
    p.add_argument("--encoder-levels", type=int, default=2,
                   help="0 = lite encoder (no U-Net pyramid); right size for stages 0-1")
    p.add_argument("--token-depth", type=int, default=2,
                   help="0 = per-frontier MLP only, no cross-frontier attention")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-boards", type=int, default=32)
    p.add_argument("--sampled-every", type=int, default=3,
                   help="run the (diagnostic) sampled eval arm every Nth eval")
    p.add_argument("--checkpoint-dir", default="mzr_ckpt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    stage = STAGES[args.stage]
    if args.max_macro_steps is not None:
        stage.max_macro_steps = args.max_macro_steps
    dev = args.device
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / f"stage{args.stage}.jsonl"
    eval_seeds = EVAL_SEEDS[: args.eval_boards]

    env = make_env(stage, args.batch, dev, seed=1_000 + args.seed,
                   leg_budget=args.leg_budget, copper_seeded=args.copper_seeded,
                   geodesic_refresh=args.geodesic_refresh)
    policy = PriorPolicy(
        num_layers=stage.layers,
        field_width=args.field_width,
        token_width=args.token_width,
        encoder_levels=args.encoder_levels,
        token_depth=args.token_depth,
    ).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    if args.progress_coef is not None:
        stage.reward.progress = args.progress_coef
    if args.tip_progress is not None:
        stage.reward.tip_progress = args.tip_progress
    if args.leg_progress is not None:
        stage.reward.leg_progress = args.leg_progress
    if args.wirelength is not None:
        stage.reward.wirelength = args.wirelength
    if args.corner is not None:
        stage.reward.corner = args.corner
    bc = stage.bc_coef0 if args.bc_coef is None else args.bc_coef
    ppo_cfg = PPOConfig(
        lr=args.lr, bc_coef=bc, epochs=args.epochs,
        minibatches=args.minibatches, entropy_coef=args.entropy_coef,
        per_frontier_adv=args.per_frontier_adv,
    )

    start, best = 0, -1.0
    latest = ckpt_dir / f"stage{args.stage}_latest.pt"
    if args.resume and latest.exists():
        blob = torch.load(latest, map_location=dev)
        missing, unexpected = policy.load_state_dict(blob["policy"], strict=False)
        if not missing and not unexpected:
            try:
                opt.load_state_dict(blob["opt"])
            except ValueError:
                print("optimizer state shape mismatch -- Adam cold")
        else:
            print(f"partial load ({len(missing)} missing, {len(unexpected)} unexpected) -- Adam cold")
        # A torch optimizer's state_dict carries `lr` inside param_groups, so
        # load_state_dict above silently overwrites the --lr just passed on the
        # command line. That makes the documented remedy for a kl blow-up
        # ("kl spiking > 0.2 -> try --lr 1.5e-4", ANTIGRAVITY_PROMPT.md) a
        # no-op on exactly the runs that need it -- and the run then reports
        # that a lower LR did not help, which is a wrong conclusion rather
        # than a null one. Re-assert the requested LR after the load.
        for g in opt.param_groups:
            g["lr"] = args.lr
        start, best = blob.get("update", 0), blob.get("best", -1.0)
        print(f"resumed from update {start}, best {best:.3f}, lr {args.lr:g}")

    kind, thr = stage.gate
    print(f"stage {args.stage}: {stage.name}")
    print(f"gate: argmax {kind} >= {thr}, sustained 3 evals | bc_coef {bc} | kill: {stage.kill}")

    obs = env.reset()
    expert = None
    if bc > 0.0:
        expert = ExpertActions(env, negotiate=args.bc_negotiate)
        expert.plan()
        sched = (f"decaying to 0 over {args.bc_decay} updates"
                 if args.bc_decay > 0 else "CONSTANT (no decay)")
        print(f"BC on: expert demonstrations, coef {bc} {sched}, "
              f"negotiate={args.bc_negotiate}, cloning direction+step only")
    hits = 0
    for u in range(start, args.updates):
        t0 = time.time()
        buf = RolloutBuffer()
        if expert is not None and env.t == 0:
            expert.plan()
        obs = collect(env, policy, buf, args.rollout, obs, expert=expert)
        t_collect = time.time() - t0
        with torch.no_grad():
            _last = policy.act(obs, deterministic=False)
        last_v, last_vf = _last["value"], _last["value_f"]
        t1 = time.time()
        # Anneal BC. Strong early to teach the expert's step/direction habits
        # (right-angle 27% by update 24, against 75% without), then out of the
        # way so PPO can actually converge -- at a constant 0.5 the two
        # objectives fought and neither won.
        if bc > 0.0 and args.bc_decay > 0:
            ppo_cfg.bc_coef = bc * max(0.0, 1.0 - u / float(args.bc_decay))
        m = ppo_update(policy, opt, buf, last_v, ppo_cfg, last_value_f=last_vf)
        t_ppo = time.time() - t1
        dt = time.time() - t0

        line = {"update": u, "sec": round(dt, 2), "collect_s": round(t_collect, 2),
                "ppo_s": round(t_ppo, 2), "bc_coef": round(ppo_cfg.bc_coef, 4),
                **{k: round(v, 4) for k, v in m.items()}}
        # A heartbeat every update -- so "is it stuck or just slow" is answerable
        # without waiting for the next eval.
        if u < start + 3 or u % max(1, args.eval_every // 5) == 0:
            print(f"  u{u:4d} {dt:.1f}s (collect {t_collect:.1f} / ppo {t_ppo:.1f}) "
                  f"kl {m['approx_kl']:.3f} clip {m['clip_frac']:.2f} vloss {m['value_loss']:.1f}",
                  flush=True)

        if (u + 1) % args.eval_every == 0 or u == args.updates - 1:
            n_eval = (u + 1) // args.eval_every
            ev = evaluate(policy, stage, dev, eval_seeds,
                          with_sampled=(n_eval % args.sampled_every == 0),
                          copper_seeded=args.copper_seeded,
                          geodesic_refresh=args.geodesic_refresh)
            line.update(
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()}
            )

            score = ev["argmax_completion"]
            # The gate is completion AND quality. A run that completes every
            # net by wandering, double-routing, or blindly following the
            # geodesic field has not solved the stage -- it has saturated the
            # one metric that cannot see any of those.
            q_ok, q_why = quality_verdict(
                ev, ev,
                max_copper=stage.max_copper,
                max_right_angle=stage.max_right_angle,
                min_dir_entropy=stage.min_dir_entropy,
                max_d0_frac=stage.max_d0_frac,
            )
            hits = hits + 1 if (score >= thr and q_ok) else 0
            line["gate_hits"] = hits
            line["quality_ok"] = q_ok
            line["quality_why"] = q_why

            if score > best:
                best = score
                _save(policy, opt, ckpt_dir / f"stage{args.stage}_best.pt", u, best, args.stage)
            _save(policy, opt, latest, u, best, args.stage)

            print(
                f"       copper {ev['copper_median']:.3f}x med / {ev['copper_mean']:.3f}x mean"
                f" | right-angle {ev['right_angle_frac']:.0%}"
                f" | doubled {ev['doubled']}"
                f" | dir d0 {ev['dir_d0_frac']:.0%} ent {ev['ent_direction']:.2f}"
                + ("" if q_ok else f"  <-- QUALITY FAIL: {q_why}")
            )
            fails = ev["argmax_fail_seeds"]
            print(
                f"u{u:4d} | argmax {ev['argmax_completion']:.3f} "
                f"sampled {ev['sampled_completion']:.3f} "
                f"(perfect {ev['argmax_perfect']:.2f}) | best {best:.3f} | hits {hits} | "
                f"kl {m['approx_kl']:.3f} clip {m['clip_frac']:.2f} ent {m['entropy']:.2f} | {dt:.1f}s"
            )
            if fails:
                print(f"       argmax < 100% on seeds: {fails}"
                      f"{'  (review with: python -m mzr.world.pool --stage ' + args.stage + ' --seeds ' + ' '.join(map(str, fails)) + ')' if len(fails) <= 6 else ''}")
            if hits >= 3:
                print(f"GATE CLEARED: argmax {score:.3f} >= {thr} for 3 consecutive evals")

        with open(log_path, "a") as f:
            f.write(json.dumps(line) + "\n")

    print(f"done. best argmax completion {best:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
