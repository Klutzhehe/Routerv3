"""Route quality, and what the policy is actually choosing.

Completion is not a sufficient gate, and this module exists because relying on
it hid two separate failures for three training runs:

* A stage-0 policy read **1.0000 completion on 1000 held-out boards** while
  46.5% of them were double-routed at 2.3x the copper. A net drawn twice is
  connected twice, so completion certified it as perfect.
* A later policy read **0.99 completion** while emitting 38% right-angle
  bends, 1.53x mean copper with a 25x tail, and routes that oscillated in
  place until the step budget ran out.

And one that completion could never have caught at all: the direction head was
collapsed onto `d0` -- "straight down the geodesic gradient" -- on **317 of 317
actions**. The policy had learned step size and via placement and *no
steering*; the geodesic field was doing all of the obstacle avoidance. That is
invisible to every completion-shaped metric, and it is the thing that decides
whether the model can negotiate congestion at stage 1, where deviating from
your own gradient is the entire task.

So: measure what the routes look like and what the policy chooses, at training
time, next to the completion number -- not by hand, three runs later.
"""

from __future__ import annotations

import math

import torch

from mzr.eval.render import _bends
from mzr.world.spec import NUM_ENDS

_SQ2 = math.sqrt(2.0)


def _octile(a, b) -> float:
    """Shortest possible length on an 8-connected grid."""
    dy, dx = abs(a[0] - b[0]), abs(a[1] - b[1])
    lo, hi = min(dy, dx), max(dy, dx)
    return (hi - lo) + _SQ2 * lo


@torch.no_grad()
def route_quality(world) -> dict:
    """Copper ratio, bend histogram and double-routing, from world state.

    `copper_ratio` is laid copper over the octile pad-to-pad distance -- 1.0
    means every net was drawn at exactly its ideal length. The MEDIAN is the
    honest headline: a handful of wandering nets drags the mean badly (measured
    1.000x median against a 1.53x mean), and the gap between the two is itself
    the signal that some routes are pathological.

    `doubled` counts nets where BOTH frontiers ran >= 70% of the pad-to-pad
    distance. Each net grows from both pads and they are meant to meet in the
    middle, so a healthy leg is two runs of about half each; two long runs
    means they mirror-routed past each other and both pad-snapped.
    """
    rv = world.route_v.cpu().numpy()
    rn = world.route_n.cpu().numpy()
    pad = world.net_pad.cpu().numpy()
    valid = world.net_valid.cpu().numpy()
    per_net = world.cfg.max_legs * NUM_ENDS

    ratios, doubled, bends = [], 0, {"straight": 0, "soft": 0, "right_angle": 0}

    for b in range(rv.shape[0]):
        runs: dict[int, list] = {}
        for f in range(world.F):
            n = int(rn[b, f])
            if n < 2:
                continue
            pts = rv[b, f, :n].astype(int).tolist()
            runs.setdefault(f // per_net, []).append(pts)
            for k, v in _bends(pts).items():
                bends[k] += v

        for net in range(world.cfg.max_nets):
            if not bool(valid[b, net]):
                continue
            src = pad[b, net, 0, 0].astype(int).tolist()[1:]
            dst = pad[b, net, 0, 1].astype(int).tolist()[1:]
            ideal = _octile(src, dst)
            if ideal <= 0:
                continue
            legs = runs.get(net, [])
            lens = [sum(_octile(p[1:], q[1:]) for p, q in zip(pl, pl[1:])) for pl in legs]
            ratios.append(sum(lens) / ideal)
            if sum(1 for L in lens if L >= 0.70 * ideal) >= 2:
                doubled += 1

    ratios.sort()
    total_bends = sum(bends.values()) or 1
    return {
        "copper_mean": (sum(ratios) / len(ratios)) if ratios else float("nan"),
        "copper_median": ratios[len(ratios) // 2] if ratios else float("nan"),
        "copper_max": ratios[-1] if ratios else float("nan"),
        "doubled": doubled,
        "right_angle_frac": bends["right_angle"] / total_bends,
        "bends": bends,
    }


@torch.no_grad()
def action_profile(policy, obs) -> dict:
    """What the policy chooses, and how collapsed each head is.

    `dir_d0_frac` is the number that matters. `d0` is defined as "one cell down
    the geodesic gradient", so a policy at d0_frac == 1.0 is a *field
    follower*: it has learned when to step far and when to via, and nothing
    about steering. It can only avoid obstacles the geodesic field already
    avoids, which means it has no mechanism for the one thing stage 1 asks for
    -- leaving a channel because another net needs it more.

    Measured on a gate-passing stage-0 checkpoint: 1.000. Completion was 0.99.
    """
    out = policy.forward(obs)
    live = obs.frontier_mask
    n_live = live.float().sum().clamp_min(1.0)

    ent = {}
    for k, lg in out.logits.items():
        d = torch.distributions.Categorical(logits=lg)
        ent[f"ent_{k}"] = float((d.entropy().detach() * live.float()).sum() / n_live)

    act = policy.act(obs, deterministic=True)["action"]
    d = act["direction"][live]
    n = max(1, d.numel())
    prof = {
        "dir_d0_frac": float((d == 0).sum()) / n,
        "dir_distinct": int(torch.unique(d).numel()),
        "step_mean": float(act["step"][live].float().mean()) if n else float("nan"),
        "via_frac": float((act["layer"][live] > 0).float().mean()) if n else float("nan"),
    }
    prof.update(ent)
    return prof


def quality_verdict(q: dict, prof: dict, *, max_copper: float, max_right_angle: float,
                    min_dir_entropy: float) -> tuple[bool, str]:
    """Does this policy route WELL, not just completely?

    Returned alongside completion so a gate can require both. Kept as a pure
    function so the training loop and any offline eval apply exactly the same
    thresholds -- the moment they diverge, one of them is lying.
    """
    fails = []
    if not (q["copper_median"] <= max_copper):
        fails.append(f"copper_median {q['copper_median']:.3f} > {max_copper}")
    if not (q["right_angle_frac"] <= max_right_angle):
        fails.append(f"right_angle {q['right_angle_frac']:.2f} > {max_right_angle}")
    if q["doubled"]:
        fails.append(f"{q['doubled']} double-routed")
    if not (prof["ent_direction"] >= min_dir_entropy):
        fails.append(
            f"direction head collapsed (entropy {prof['ent_direction']:.3f} < "
            f"{min_dir_entropy}, d0 {prof['dir_d0_frac']:.0%}) -- following the "
            f"field, not steering"
        )
    return (not fails), "; ".join(fails)
