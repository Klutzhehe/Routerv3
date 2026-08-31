"""Why does argmax swing 0.08-0.73 while sampled sits stable at ~0.86?

    python -m mzr.scripts.diagnose_stage0 --stage 0 \
        --ckpt /content/drive/MyDrive/mzr_ckpt/stage0/stage0_latest.pt --device cuda

The stage-0 log has a shape that a plain "it needs more updates" story does not
explain. Completion on stage 0 is *binary per board* (one net), so
`argmax_completion == argmax_perfect`, and the trace reads:

    0.578  0.078  0.516  0.469  0.469  0.078  0.109  0.094  0.562  0.484  0.734

Sampled, over the same evals, never leaves 0.78-0.89. A deterministic policy
that is *worse* than its own stochastic version, and that collapses globally
(the failing-seed list becomes a contiguous run from the first eval seed), is
not an underfitting signature. It is one of:

* a deterministic trap -- argmax picks a move that changes nothing, so the next
  observation is identical, so it picks it again, forever, until the frontier
  starves. Sampling escapes; argmax cannot.
* a head that is irrelevant on this stage but still *acted on*, whose argmax
  flipped to a constant bad value and took every board down at once.
* a critic that cannot see the quantity the return depends on, making every
  advantage noise and the policy a random walk that happens to pass through
  good argmaxes.

This script measures all three, plus the non-learned floor, and attributes each
argmax failure to either the board or the policy. Nothing here trains; it is
read-only on the checkpoint.
"""

from __future__ import annotations

import argparse

import torch

from mzr.models.policy import PriorPolicy
from mzr.training.curriculum import EVAL_SEEDS, STAGES
from mzr.training.ppo import PPOConfig, compute_gae
from mzr.training.run import make_env
from mzr.world import baselines
from mzr.world.engine import STATUS_DONE, STATUS_FAILED

HEADS = ("direction", "step", "layer", "via", "width", "couple")


# -- helpers ---------------------------------------------------------------


def infer_model_kwargs(sd: dict) -> dict:
    """Recover the constructor arguments from a checkpoint's state_dict.

    Loading with the wrong widths and `strict=False` produces a *randomly
    initialised* policy that silently scores like an untrained one -- which
    would look exactly like the training failure we are trying to explain. So
    the shapes are read off the tensors rather than taken from flags.
    """
    field_width = sd["field.stem.0.weight"].shape[0]
    token_width = sd["head.weight"].shape[1]
    levels = 2 if any(k.startswith("field.down1.") for k in sd) else 0
    depth = 0
    while any(k.startswith(f"tokens.blocks.layers.{depth}.") for k in sd):
        depth += 1
    # head.bias is sum(_head_sizes) = 8 + 3 + (1+L) + 4 + 4 + 2
    num_layers = sd["head.bias"].shape[0] - (8 + 3 + 1 + 4 + 4 + 2)
    return dict(
        num_layers=num_layers,
        field_width=field_width,
        token_width=token_width,
        encoder_levels=levels,
        token_depth=depth,
    )


def load_policy(ckpt: str | None, stage, device: str):
    if ckpt is None:
        kw = dict(num_layers=stage.layers, field_width=40, token_width=96,
                  encoder_levels=0, token_depth=0)
        print(f"  no --ckpt: using an UNTRAINED policy {kw}")
        return PriorPolicy(**kw).to(device).eval(), -1, -1.0
    blob = torch.load(ckpt, map_location=device)
    sd = blob["policy"]
    kw = infer_model_kwargs(sd)
    print(f"  checkpoint {ckpt}")
    print(f"  update {blob.get('update')} best {blob.get('best')} | inferred {kw}")
    pol = PriorPolicy(**kw).to(device)
    pol.load_state_dict(sd, strict=True)
    return pol.eval(), blob.get("update", -1), blob.get("best", -1.0)


def _act_dict(t6) -> dict:
    return dict(zip(HEADS, t6))


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if float(d) > 1e-9 else float("nan")


# -- 1. the non-learned floor ---------------------------------------------


@torch.no_grad()
def baseline_floor(stage, device, seeds) -> dict:
    """greedy / detour / layer_hop on the same eval seeds.

    If the trained argmax does not clear `greedy`, nothing downstream about
    learning rates or update counts matters -- the policy is below the floor it
    was initialised at, and that is a reward or credit-assignment fact.
    """
    out = {}
    for name, fn in baselines.BASELINES.items():
        env = make_env(stage, batch=len(seeds), device=device, seed=0)
        env.reset(seeds=seeds)
        while True:
            if env.step(_act_dict(fn(env.world))).done:
                break
        c = env.completion()
        out[name] = (float(c.mean()), float((c >= 0.999).float().mean()))
    return out


# -- 2. behaviour trace: argmax vs sampled --------------------------------


@torch.no_grad()
def trace(policy, stage, device, seeds, deterministic: bool) -> dict:
    env = make_env(stage, batch=len(seeds), device=device, seed=0)
    obs = env.reset(seeds=seeds)

    prev_pos = obs.frontier_pos.clone()
    frozen = torch.zeros(obs.frontier_mask.shape, dtype=torch.long, device=device)
    worst_frozen = torch.zeros_like(frozen)
    head_hist = {h: torch.zeros(policy.sizes[h], dtype=torch.long) for h in HEADS}
    n_rej = n_con = n_mov = n_via = n_live = 0
    n_joint_bad = n_chosen = 0
    steps = 0

    while True:
        act = policy.act(obs, deterministic=deterministic)["action"]
        m = obs.frontier_mask

        for h in HEADS:
            head_hist[h] += torch.bincount(
                act[h][m].flatten().cpu(), minlength=policy.sizes[h]
            )

        # Was the *chosen* (direction, step) pair jointly safe? The suppression
        # mask in policy._suppress applies the two marginals independently --
        # a direction safe at some step length, a step length safe in some
        # direction -- which cannot rule out a pair that is unsafe together.
        d, s = act["direction"], act["step"]
        chose_safe = obs.safety.gather(
            2, d.view(*d.shape, 1, 1).expand(*d.shape, 1, obs.safety.shape[-1])
        ).squeeze(2).gather(2, s.unsqueeze(-1)).squeeze(-1)
        n_joint_bad += int((~chose_safe & m).sum())
        n_chosen += int(m.sum())

        step = env.step(act)

        same = (step.obs.frontier_pos == prev_pos).all(-1) & m
        frozen = torch.where(same, frozen + 1, torch.zeros_like(frozen))
        worst_frozen = torch.maximum(worst_frozen, frozen)
        prev_pos = step.obs.frontier_pos.clone()

        n_rej += int(step.info["rejected"].sum())
        n_con += int(step.info["contended"].sum())
        n_mov += int(step.info["moved"].sum())
        n_via += int(step.info["vias"].sum())
        n_live += int(m.sum())
        steps += 1
        obs = step.obs
        if step.done:
            break

    w = env.world
    c = env.completion()
    valid = w.net_valid
    return {
        "completion": float(c.mean()),
        "perfect": float((c >= 0.999).float().mean()),
        "steps": steps,
        "rej_rate": n_rej / max(1, n_live),
        "contended_rate": n_con / max(1, n_live),
        "move_rate": n_mov / max(1, n_live),
        "via_rate": n_via / max(1, n_live),
        "joint_unsafe_rate": n_joint_bad / max(1, n_chosen),
        "worst_frozen": int(worst_frozen.max()),
        "frontiers_frozen_10plus": int((worst_frozen >= 10).sum()),
        "nets_failed": int(((w.net_status == STATUS_FAILED) & valid).sum()),
        "nets_done": int(((w.net_status == STATUS_DONE) & valid).sum()),
        "head_hist": {h: head_hist[h].tolist() for h in HEADS},
        "fail_seeds": [int(sd) for sd, ok in zip(seeds, (c >= 0.999).tolist()) if not ok],
    }


# -- 3. critic quality ------------------------------------------------------


@torch.no_grad()
def critic_quality(policy, stage, device, batch, rollout) -> dict:
    """Explained variance of V against the returns PPO actually fits.

    The value head reads only `g`, the *globally mean-pooled* field embedding
    (models/encoder.py: `g = global_proj(z.mean(dim=(2,3,4)))`). It has no
    frontier position and no distance-to-target input. This repo has four
    recorded failures at decoding distance out of a pooled embedding, so the
    question is whether V(s) tracks the return at all.

    ev <= 0 means the critic is no better than predicting the mean, i.e. every
    advantage handed to the policy gradient is noise.
    """
    env = make_env(stage, batch=batch, device=device, seed=7)
    obs = env.reset()
    cfg = PPOConfig()
    rew, val, dn, dist = [], [], [], []
    for _ in range(rollout):
        a = policy.act(obs, deterministic=False)
        step = env.step(a["action"])
        rew.append((step.reward.sum(dim=1) + step.board_reward).detach())
        val.append(a["value"].detach())
        dn.append(step.done * torch.ones(batch, device=device))
        live = obs.frontier_mask.float()
        d = torch.nan_to_num(env.world.fr_prev, posinf=0.0, nan=0.0)
        dist.append((d * live).sum(1) / live.sum(1).clamp_min(1.0))
        obs = step.obs
        if step.done:
            obs = env.reset()
    last_v = policy.act(obs, deterministic=False)["value"]
    rewards, values = torch.stack(rew), torch.stack(val)
    _, returns = compute_gae(rewards, values, last_v, torch.stack(dn).float(), cfg)

    resid = (returns - values).var()
    ev = float(1.0 - resid / returns.var().clamp_min(1e-8))
    return {
        "explained_variance": ev,
        "value_std": float(values.std()),
        "return_std": float(returns.std()),
        "value_mean": float(values.mean()),
        "return_mean": float(returns.mean()),
        "corr_V_vs_remaining_distance": _corr(values, torch.stack(dist)),
    }


# -- 4. board or policy? ----------------------------------------------------


def attribute(stage, seeds, device) -> None:
    from mzr.world.pool import expert_route

    if not seeds:
        print("  (no argmax failures to attribute)")
        return
    solvable = 0
    for s in seeds:
        res, total = expert_route(stage.board_spec(), stage.generator, s, device)
        if res is None:
            print(f"  seed {s}: generator produced no nets")
            continue
        ok = len(res.completed) == total
        solvable += ok
        print(f"  seed {s}: expert {len(res.completed)}/{total} "
              f"-> {'SOLVABLE (policy problem)' if ok else 'expert also failed'}")
    print(f"  {solvable}/{len(seeds)} argmax failures are on boards the expert routes")


# -- main -------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="0", choices=sorted(STAGES))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-boards", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--skip-expert", action="store_true")
    args = p.parse_args()

    torch.manual_seed(0)
    stage = STAGES[args.stage]
    seeds = EVAL_SEEDS[: args.eval_boards]
    dev = args.device
    print(f"stage {args.stage}: {stage.name} | {len(seeds)} eval boards | device {dev}")

    print("\n=== 0. policy ===")
    policy, upd, best = load_policy(args.ckpt, stage, dev)

    print("\n=== 1. non-learned floor (same eval seeds) ===")
    for name, (mean, perfect) in baseline_floor(stage, dev, seeds).items():
        print(f"  {name:<10} completion {mean:.3f}  perfect {perfect:.3f}")

    print("\n=== 2. behaviour trace ===")
    tr = {}
    for det in (True, False):
        arm = "argmax" if det else "sampled"
        t = trace(policy, stage, dev, seeds, det)
        tr[arm] = t
        print(f"  {arm:<8} completion {t['completion']:.3f} perfect {t['perfect']:.3f} "
              f"| steps {t['steps']}")
        print(f"           moved {t['move_rate']:.3f} rejected {t['rej_rate']:.3f} "
              f"contended {t['contended_rate']:.3f} vias/live-step {t['via_rate']:.3f}")
        print(f"           chosen (dir,step) pair jointly UNSAFE: "
              f"{t['joint_unsafe_rate']*100:.2f}% of actions")
        print(f"           longest frozen run {t['worst_frozen']} steps | "
              f"frontiers frozen >=10 steps: {t['frontiers_frozen_10plus']}")
        print(f"           nets done {t['nets_done']} failed {t['nets_failed']}")

    print("\n=== 3. where the deterministic policy's choices concentrate ===")
    print("  (counts over live frontier-steps; stage 0 uses 1 net, 2 layers)")
    for h in HEADS:
        a = tr["argmax"]["head_hist"][h]
        s = tr["sampled"]["head_hist"][h]
        tot_a, tot_s = max(1, sum(a)), max(1, sum(s))
        fa = [round(x / tot_a, 3) for x in a]
        fs = [round(x / tot_s, 3) for x in s]
        print(f"  {h:<10} argmax {fa}")
        print(f"  {'':<10} sample {fs}")

    print("\n=== 4. critic quality ===")
    cq = critic_quality(policy, stage, dev, args.batch, args.rollout)
    for k, v in cq.items():
        print(f"  {k:<32} {v:+.4f}" if isinstance(v, float) else f"  {k:<32} {v}")
    if cq["explained_variance"] <= 0.05:
        print("  -> V(s) explains ~none of the return variance: every advantage")
        print("     PPO sees is noise. The value head reads only the globally")
        print("     mean-pooled field embedding (encoder.py global_proj).")

    print("\n=== 5. argmax failures: board or policy? ===")
    fails = tr["argmax"]["fail_seeds"]
    print(f"  argmax failed on {len(fails)} of {len(seeds)} boards: {fails[:16]}")
    if args.skip_expert:
        print("  (--skip-expert)")
    else:
        attribute(stage, fails[:12], dev)

    print("\n=== summary ===")
    print(f"  argmax {tr['argmax']['completion']:.3f} vs sampled "
          f"{tr['sampled']['completion']:.3f} "
          f"(gap {tr['sampled']['completion']-tr['argmax']['completion']:+.3f})")
    print(f"  critic explained variance {cq['explained_variance']:+.3f}")
    print(f"  argmax joint-unsafe action rate "
          f"{tr['argmax']['joint_unsafe_rate']*100:.2f}%")
    print(f"  argmax longest frozen frontier run {tr['argmax']['worst_frozen']} "
          f"of {tr['argmax']['steps']} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
