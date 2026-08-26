"""Training entry point, instrumented for remote (Colab) operation.

    python -m neuroroute.training.run --stage 1 --device cuda --updates 500

This is written to be run by someone who cannot debug it interactively. Whoever
is driving Colab reports output back rather than diagnosing (`AGENTS.md`), and
the session may die at any point, so the run has to explain itself as it goes:

* every update appends to `train_log.jsonl`, fsync-ed, so a killed VM still
  leaves a complete record;
* health checks run after every optimiser step and print loudly -- a silent NaN
  otherwise burns the rest of the run producing a checkpoint of garbage;
* evals render **failed boards** to PNG, because a contact sheet shows the
  failure mode and a reward curve never will;
* an optional real-KiCad DRC pass during eval tracks the sim-to-real gap while
  training, not just before it;
* any exception writes a self-contained `crash_report.txt` with environment,
  config, recent metrics and tensor state.

**Completion rate is the headline metric, not reward.** `docs/RL_PLAN.md`
measured a policy scoring worse reward while routing more nets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from dataclasses import replace

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
from neuroroute.training.telemetry import Telemetry, check_model_health, tensor_debug
from neuroroute.world.engine import WorldConfig


def latent_shape(env: NeuroRouteEnv) -> tuple[int, int, int]:
    """The encoder's output grid. Derived from the stem's stride, not hardcoded."""
    L, H, W = env.world.shape
    return (L, max(1, H // 4), max(1, W // 4))


def build(args):
    stages = default_curriculum(layers_max=args.layers, size=args.size)
    if not 0 <= args.stage < len(stages):
        raise SystemExit(f"--stage must be 0..{len(stages) - 1}")
    stage = stages[args.stage]

    world = WorldConfig(
        batch_size=args.batch,
        max_heads=args.heads,
        max_nets=max(64, stage.generator.num_nets + 8),
        device=args.device,
        geodesic_refresh=args.geodesic_refresh,
    )
    env = NeuroRouteEnv(stage_env_config(stage, world, EnvConfig(seed=args.seed)))

    # A SEPARATE, wider environment for eval. Reusing the training env pins the
    # eval set to `--batch` boards, and at stage 0 that made each board worth
    # 6.25% of the reported number: 75.0% was 12/16 and 87.5% was 14/16, so
    # real progress and one board of noise were indistinguishable. Eval is
    # infrequent, so the extra memory is cheap.
    eval_world = replace(world, batch_size=args.eval_boards)
    eval_env = NeuroRouteEnv(
        stage_env_config(stage, eval_world, EnvConfig(seed=args.seed + 10_000))
    )

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
    return env, eval_env, policy, stage


@torch.no_grad()
def evaluate(env: NeuroRouteEnv, policy: NeuroRoutePolicy, seeds: list[int]) -> dict:
    """Policy and every baseline on the **same held-out boards**.

    Same seeds for all of them: a completion number is meaningless without the
    boards being identical, and board-to-board variance here is large.

    The policy is scored **twice, on those same boards**: once with
    ``deterministic=True`` (argmax) and once **sampled**, which is the
    distribution training actually rolls out under. Reporting only the argmax
    number made a real effect unattributable: stage 0 logged ~100% completion
    on every training update while every held-out eval read 73-90%, and there
    was no way to tell whether the eval boards were harder or the mode was
    simply worse than the mean.

    Measured locally on an **untrained** policy, 16 held-out boards, so that a
    trained gap can be read against something rather than against zero:

        stage              argmax   sampled    argmax vias   sampled vias
        0 (1 net,   2L)    75.00%    93.75%           0.00           2.88
        1 (20 nets, 2L)    26.56%    37.19%           0.00          42.88

    **+10.6 to +18.8 points of the gap is present before any learning**, and
    stage 0's argmax number is exactly the 75.0% plateau run A sat at for three
    consecutive evals. The mechanism is visible in the same measurement: at
    init the argmax policy takes direction 0 -- straight down the geodesic
    gradient -- on **100%** of steps, never proposes a via, and cycles (stage
    1: 23,455 head-steps over 5,146 distinct cells, a 78.1% revisit rate,
    against 48.0% sampled). Nothing in the observation tells a head it has been
    in this cell before: the dead-zone channel keys off *rejection*, and argmax
    is rejected 0.05% of the time. Sampling is what places the vias and what
    breaks the cycles.
    """
    results: dict[str, float] = {}

    def run(fn) -> tuple[float, float, dict]:
        obs = env.reset(seeds)
        rejected = acted = 0
        for _ in range(env.cfg.max_episode_steps):
            obs, rew, done, info = env.step(fn(obs))
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

    # `policy/*` stays the deterministic arm: it is what the stage gate, the
    # checkpoint's `best_completion` and every logged history row already mean,
    # and silently changing which number those refer to would make this run
    # incomparable with the two stage-0 runs and the stage-1 run already on
    # record.
    comp, rej, metrics = run(lambda o: policy.act(o, deterministic=True).actions)
    results["policy/completion"] = comp
    results["policy/rejected_action_rate"] = rej
    results.update({f"policy/{k}": v for k, v in metrics.items()})

    comp_s, rej_s, metrics_s = run(lambda o: policy.act(o).actions)
    results["policy_sampled/completion"] = comp_s
    results["policy_sampled/rejected_action_rate"] = rej_s
    results.update({f"policy_sampled/{k}": v for k, v in metrics_s.items()})
    results["policy/sample_minus_argmax"] = comp_s - comp

    for name, fn in (
        ("greedy", greedy_safe_action),
        ("detour", detour_action),
        ("layer_hop", layer_hop_action),
    ):
        c, r, _ = run(fn)
        results[f"{name}/completion"] = c
        results[f"{name}/rejected_action_rate"] = r
    return results


def drc_check(env: NeuroRouteEnv, out_dir: Path, tag: str, n_boards: int = 2) -> dict:
    """Export boards and run KiCad's real DRC on them, mid-training.

    The lattice claims to be DRC-clean by construction. That was verified before
    training, but a policy is an adversary: it will find whatever the geometry
    model permits. Checking during training is how a regression gets caught
    while it is still cheap to explain.

    Returns `{}` if `kicad-cli` is unavailable -- never fatal.
    """
    if shutil.which("kicad-cli") is None:
        return {}
    from neuroroute.eval.kicad_export import export_board
    from neuroroute.scripts.validate_kicad import classify, run_drc, summarise

    out_dir.mkdir(parents=True, exist_ok=True)
    legality = advisory = routed = 0
    for b in range(min(n_boards, env.world.occ.shape[0])):
        pcb = out_dir / f"{tag}_board{b}.kicad_pcb"
        export_board(env.world, b, pcb)
        routed += int(((env.world.net_status[b] == 2) & env.world.net_valid[b]).sum())
        ok, report = run_drc(pcb, out_dir / f"{tag}_board{b}.drc.json")
        if not ok:
            return {"drc/exporter_broken": 1.0}
        groups = classify(summarise(report))
        legality += sum(groups["legality"].values()) + sum(groups["unknown"].values())
        advisory += sum(groups["advisory"].values())
    return {
        "drc/legality_violations": float(legality),
        "drc/advisory_warnings": float(advisory),
        "drc/routed_nets": float(routed),
        "drc/per_1000_nets": 1000.0 * legality / max(1, routed),
    }


def _tune_backend(device: str) -> None:
    """Cheap, safe CUDA backend settings.

    `cudnn.benchmark` lets cuDNN pick the fastest algorithm per convolution
    shape. It costs a few slow iterations while it probes, then pays for the
    rest of the run -- and this workload has completely fixed shapes, which is
    exactly the case it is designed for.

    TF32 is a no-op on Turing (a T4 has no TF32 units) but a real speedup on
    Ampere and later, so it is set unconditionally rather than gated on a
    capability check that would just add a failure mode.
    """
    if not device.startswith("cuda"):
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train(args) -> int:
    torch.manual_seed(args.seed)
    _tune_backend(args.device)
    out = Path(args.checkpoint_dir)
    tel = Telemetry(out, run_name=f"stage{args.stage}")

    env, eval_env, policy, stage = build(args)
    dev = torch.device(args.device)
    n_params = sum(p.numel() for p in policy.parameters())

    tel.banner(
        f"NeuroRoute training -- stage {args.stage}: {stage.name}",
        {
            "introduces": stage.introduces,
            "board": f"{stage.board.height_cells}x{stage.board.width_cells} "
                     f"x {stage.board.num_layers} layers",
            "pitch_mm": stage.board.pitch_mm,
            "nets": stage.generator.num_nets,
            "batch(B)": args.batch,
            "heads(K)": args.heads,
            "decisions/step": args.batch * args.heads,
            "rollout": args.rollout,
            "ppo_chunk": f"{args.ppo_chunk}  ({args.ppo_chunk * args.batch} boards/pass)",
            "amp": args.amp,
            "eval_boards": args.eval_boards,
            "entropy": f"{args.entropy} -> {args.entropy_final}",
            "updates": args.updates,
            "params_M": round(n_params / 1e6, 3),
            "lr": args.lr,
            "device": args.device,
            "stage_gate": stage.gate,
            "checkpoint_dir": str(out),
        },
    )

    ppo = PPOConfig(
        rollout_steps=args.rollout, lr=args.lr,
        store_device=args.store_device, entropy_coef=args.entropy,
        entropy_final=args.entropy_final, chunk=args.ppo_chunk, amp=args.amp,
    )
    opt = torch.optim.Adam(policy.parameters(), lr=ppo.lr, eps=1e-5)
    # fp16 GradScaler, only meaningful with --amp on CUDA.
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and dev.type == "cuda"))
    buf = RolloutBuffer(ppo, dev)

    start_update = 0
    if args.resume and (out / "latest.pt").exists():
        state = torch.load(out / "latest.pt", map_location=dev, weights_only=False)
        policy.load_state_dict(state["policy"])
        opt.load_state_dict(state["optimiser"])
        start_update = int(state.get("update", 0))
        tel.history = state.get("history", [])
        tel.print(f"resumed from update {start_update}")

    obs = env.reset()
    best_completion = -1.0

    try:
        for update in range(start_update, args.updates):
            frac = update / max(1, args.updates - 1)
            entropy_coef = ppo.entropy_coef + frac * (ppo.entropy_final - ppo.entropy_coef)

            buf.clear()
            ep_metrics: dict[str, float] = {}
            rejected = acted = 0

            with tel.section("rollout"):
                for _ in range(ppo.rollout_steps):
                    pol_out = policy.act(obs)
                    next_obs, reward, done, info = env.step(pol_out.actions)
                    buf.add(obs, pol_out.actions, pol_out.log_prob, pol_out.value, reward,
                            done.unsqueeze(-1).expand_as(reward), obs.head_mask)
                    rejected += int(info["rejected"].sum())
                    acted += int(info["active"].sum())
                    if "terminal_reward" in info:
                        # Board-level terminal reward is shared by that board's
                        # heads: completion, pair gap and length matching are
                        # properties of the finished board, not of one head.
                        buf.reward[-1] = buf.reward[-1] + (
                            info["terminal_reward"].unsqueeze(-1).expand_as(reward)
                            / max(1, args.heads)
                        ).to(buf.reward[-1].device)
                        ep_metrics = {
                            k.split("/", 1)[1]: float(v.mean())
                            for k, v in info.items() if k.startswith("final/")
                        }
                    obs = next_obs
                    if bool(done.all()):
                        obs = env.reset()

            with torch.no_grad():
                last_value = policy.act(obs).value

            rewards = torch.stack(buf.reward).to(dev)
            values = torch.stack(buf.value).to(dev)
            dones = torch.stack(buf.done).to(dev)
            adv, ret = compute_gae(rewards, values, dones, last_value, ppo.gamma, ppo.gae_lambda)

            with tel.section("update"):
                targets = episode_targets(env.world, latent_shape(env))
                stats = ppo_update(policy, opt, buf, adv, ret, ppo, entropy_coef,
                                   targets, scaler=scaler)

            health = check_model_health(policy)
            for w in health.warnings:
                tel.print(f"    [WARN] {w}")
            for e in health.errors:
                tel.print(f"    [FATAL] {e}")

            row = {
                "update": update,
                "reward": float(rewards.sum(0).mean()),
                "completion": float(env.world.completion().mean()),
                "rejected_action_rate": rejected / max(1, acted),
                "entropy_coef": entropy_coef,
                "gpu_memory": tel.gpu_memory(),
                **stats,
                **ep_metrics,
            }
            metric_health = tel.log(row)

            if health.fatal or metric_health.fatal:
                tel.print("\n[FATAL] model or metrics are non-finite -- stopping before "
                          "this corruption is written to a checkpoint.")
                tel.crash(
                    RuntimeError("non-finite model or metrics"),
                    extra={
                        "update": update,
                        "errors": health.errors + metric_health.errors,
                        "tensors": tensor_debug(rewards=rewards, values=values,
                                                advantages=adv, returns=ret),
                    },
                )
                return 2

            if update % args.log_every == 0:
                done_n = ppo.rollout_steps * args.batch * args.heads * (update - start_update + 1)
                sps = done_n / max(tel.timings.get("rollout", 1e-6), 1e-6)
                tel.print(
                    f"[{update:5d}/{args.updates}] "
                    f"completion {row['completion']:6.1%}  reward {row['reward']:8.2f}  "
                    f"rej {row['rejected_action_rate']:5.1%}  "
                    f"pi {stats['policy_loss']:+.3f}  v {stats['value_loss']:7.3f}  "
                    f"H {stats['entropy']:.3f}  fcast {stats['forecast']:.3f}  "
                    f"{sps:,.0f} dec/s  {tel.gpu_memory()}  [{tel.timing_summary()}]"
                )

            # -- eval -----------------------------------------------------
            if update > 0 and update % args.eval_every == 0:
                with tel.section("eval"):
                    seeds = list(range(args.eval_seed_base,
                                       args.eval_seed_base + args.eval_boards))
                    ev = evaluate(eval_env, policy, seeds)
                    with torch.no_grad():
                        gate = forecast_gate(
                            policy.act(obs).forecast,
                            episode_targets(env.world, latent_shape(env)),
                            demand_baseline(env._demand, latent_shape(env)),
                        )

                    tel.print("  " + "-" * 74)
                    tel.print(f"  EVAL @ update {update}  ({args.eval_boards} held-out "
                              f"boards, seeds {seeds[0]}..{seeds[-1]}, never trained on)")
                    tel.print(f"    policy     {ev['policy/completion']:6.1%}   "
                              f"rejected {ev['policy/rejected_action_rate']:6.2%}   "
                              f"vias {ev.get('policy/vias', 0):5.1f}   (argmax)")
                    tel.print(f"    policy     {ev['policy_sampled/completion']:6.1%}   "
                              f"rejected {ev['policy_sampled/rejected_action_rate']:6.2%}   "
                              f"vias {ev.get('policy_sampled/vias', 0):5.1f}   (sampled -- "
                              f"the distribution training rolls out under)")
                    tel.print(f"    sampled - argmax {ev['policy/sample_minus_argmax']:+6.1%}"
                              "   (untrained stage-1 reference: +10.6%)")
                    for nm in ("greedy", "detour", "layer_hop"):
                        beat = "  <-- policy ahead" if ev["policy/completion"] > ev[f"{nm}/completion"] else ""
                        tel.print(f"    {nm:<10} {ev[f'{nm}/completion']:6.1%}{beat}")
                    tel.print(f"    detour {ev.get('policy/detour', 0):.3f}   "
                              f"pair-gap-err {ev.get('policy/pair_gap_error', 0):.3f}   "
                              f"split {ev.get('policy/split_fraction', 0):.2f}   "
                              f"len-in-tol {ev.get('policy/length_within_tol', 0):.1%}")
                    tel.print(
                        f"    FORECAST GATE  mae {gate['forecast_mae']:.4f} vs baseline "
                        f"{gate['baseline_mae']:.4f}   corr {gate['forecast_corr']:+.3f} vs "
                        f"{gate['baseline_corr']:+.3f}   "
                        f"{'BEATS baseline' if gate['beats_baseline'] else 'does NOT beat baseline'}"
                    )

                    drc = drc_check(eval_env, out / "drc", f"u{update}") if args.drc_every and \
                        update % args.drc_every == 0 else {}
                    if drc:
                        if drc.get("drc/exporter_broken"):
                            tel.print("    DRC: exporter produced a file KiCad cannot read")
                        else:
                            tel.print(f"    DRC (real KiCad): "
                                      f"{int(drc['drc/legality_violations'])} legality violations "
                                      f"over {int(drc['drc/routed_nets'])} routed nets "
                                      f"= {drc['drc/per_1000_nets']:.1f}/1000")

                    if ev["policy/completion"] >= stage.gate:
                        tel.print(f"    *** STAGE GATE {stage.gate:.0%} MET "
                                  f"-- ready for stage {args.stage + 1} ***")
                    tel.print("  " + "-" * 74)

                    tel.history[-1].update(ev)
                    tel.history[-1].update(gate)
                    tel.history[-1].update(drc)

                    # Visual debugging. Renders the WORST boards, not the batch
                    # order -- the interesting ones are the failures.
                    if args.render_every and update % args.render_every == 0:
                        try:
                            from neuroroute.eval.render import (
                                contact_sheet, learning_curves, render_board,
                            )
                            art = out / "renders"
                            worst = int(eval_env.world.completion().argmin())
                            render_board(eval_env.world, worst, art / f"u{update}_worst.png",
                                         title=f"update {update}, worst board")
                            contact_sheet(eval_env.world, art / f"u{update}_sheet.png")
                            learning_curves(tel.history, out / "curves.png")
                            tel.print(f"    renders -> {art}   curves -> {out/'curves.png'}")
                        except Exception as exc:  # rendering must never kill a run
                            tel.print(f"    [WARN] render failed: {exc!r}")

                    best_completion = max(best_completion, ev["policy/completion"])
                obs = env.reset()

            # -- checkpoint ------------------------------------------------
            if update > 0 and update % args.checkpoint_every == 0:
                save(out, policy, opt, update, args, tel, best_completion)
                tel.print(f"    checkpoint @ update {update} -> {out/'latest.pt'}")

        # Inside the `try`, so this runs BEFORE `finally: tel.close()`.
        save(out, policy, opt, args.updates, args, tel, best_completion)
        tel.print(f"\ndone. best held-out completion {best_completion:.1%}")
        tel.print(f"artifacts in {out}:")
        for f in sorted(out.glob("*")):
            tel.print(f"    {f.name}")
        return 0

    except KeyboardInterrupt:
        tel.print("\ninterrupted -- saving before exit")
        save(out, policy, opt, update, args, tel, best_completion)
        return 130
    except Exception as exc:
        tel.crash(exc, extra={"stage": args.stage, "args": vars(args)})
        return 1
    finally:
        tel.close()


def save(out: Path, policy, opt, update: int, args, tel: Telemetry, best: float) -> None:
    """Atomic checkpoint write.

    Written to a temp file and renamed: a Colab VM reclaimed mid-`torch.save`
    otherwise leaves a truncated `latest.pt` and takes the whole run's progress
    with it.
    """
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "latest.pt.tmp"
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimiser": opt.state_dict(),
            "update": update,
            "stage": args.stage,
            "args": vars(args),
            "history": tel.history,
            "best_completion": best,
        },
        tmp,
    )
    tmp.replace(out / "latest.pt")
    (out / "history.json").write_text(json.dumps(tel.history, indent=1, default=float),
                                      encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Train NeuroRoute")
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=16, help="boards in flight (B)")
    p.add_argument("--heads", type=int, default=8, help="simultaneous routing heads per board (K)")
    p.add_argument("--width", type=int, default=64, help="encoder base channel width")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--size", type=int, default=128, help="lattice cells per side")
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--entropy-final", type=float, default=0.003,
                   help="entropy coefficient floor. 0.001 let the policy go "
                        "near-deterministic (H 2.1 -> 0.29) less than halfway "
                        "through a run, locking in a via-less local optimum")
    p.add_argument("--ppo-chunk", type=int, default=8,
                   help="rollout timesteps folded into one forward/backward. "
                        "The GPU utilisation knob; lower it first on OOM")
    p.add_argument("--amp", action="store_true",
                   help="fp16 autocast on CUDA. UNVERIFIED -- no GPU was "
                        "available to test it; watch for [FATAL] non-finite")
    p.add_argument("--eval-boards", type=int, default=64,
                   help="held-out boards per eval. 16 gave 6.25%% resolution, "
                        "too coarse to tell progress from noise")
    p.add_argument("--geodesic-refresh", type=int, default=0)
    p.add_argument("--store-device", default="cpu",
                   help="where rollout observations live; 'cpu' saves VRAM")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed-base", type=int, default=900000,
                   help="held-out seeds, deliberately far from the training range")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--render-every", type=int, default=50, help="0 disables")
    p.add_argument("--drc-every", type=int, default=100,
                   help="run real KiCad DRC during eval; 0 disables")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--checkpoint-dir", default="checkpoints/neuroroute")
    p.add_argument("--resume", action="store_true")
    sys.exit(train(p.parse_args()))


if __name__ == "__main__":
    main()
