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
    # Imported here, not at module scope: render.py pulls in
    # diagnose_stage0 -> training.run, and run.py imports this module, so a
    # top-level import is a cycle.
    from mzr.eval.render import _bends

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


class ProfileAccumulator:
    """What the policy chose, accumulated ACROSS an episode.

    It has to be accumulated per step, not read at the end. Measured the hard
    way: sampling the profile from the post-rollout observation averages over
    zero live frontiers (they have all retired), which reports d0_frac 0%,
    0 distinct directions and entropy 0.000 for *every* policy -- failing a
    good one and a collapsed one identically. A metric that reads zero for
    everybody is worse than no metric.

    `dir_d0_frac` is the number that matters. `d0` is "one cell down the
    geodesic gradient", so a policy at 1.0 is a FIELD FOLLOWER: it has learned
    when to step far and when to via, and nothing about steering. It can only
    avoid what the geodesic field already avoids -- so it has no mechanism for
    what stage 1 is entirely about, leaving a channel because another net needs
    it more. Measured on a stage-0 checkpoint at 0.99 completion: 317 of 317
    actions were d0.
    """

    def __init__(self) -> None:
        self.n = 0
        self.d0 = 0
        self.dirs: set[int] = set()
        self.step_sum = 0.0
        self.via = 0
        self.ent_sum: dict[str, float] = {}
        self.ent_n = 0

    @torch.no_grad()
    def update(self, policy, obs, action=None) -> None:
        live = obs.frontier_mask
        n_live = int(live.sum())
        if n_live == 0:
            return

        out = policy.forward(obs)
        for k, lg in out.logits.items():
            d = torch.distributions.Categorical(logits=lg)
            e = float((d.entropy().detach() * live.float()).sum() / max(n_live, 1))
            self.ent_sum[k] = self.ent_sum.get(k, 0.0) + e
        self.ent_n += 1

        if action is None:
            action = policy.act(obs, deterministic=True)["action"]
        d = action["direction"][live]
        self.n += int(d.numel())
        self.d0 += int((d == 0).sum())
        self.dirs.update(int(x) for x in torch.unique(d).tolist())
        self.step_sum += float(action["step"][live].float().sum())
        self.via += int((action["layer"][live] > 0).sum())

    def result(self) -> dict:
        n = max(1, self.n)
        out = {
            "dir_d0_frac": self.d0 / n,
            "dir_distinct": len(self.dirs),
            "step_mean": self.step_sum / n,
            "via_frac": self.via / n,
            "actions_seen": self.n,
        }
        m = max(1, self.ent_n)
        for k, v in self.ent_sum.items():
            out[f"ent_{k}"] = v / m
        out.setdefault("ent_direction", 0.0)
        return out


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
    if not prof.get("actions_seen", 0):
        # Nothing was observed; do not manufacture a verdict from an empty set.
        fails.append("no live actions sampled -- profile is empty, not collapsed")
    elif not (prof["ent_direction"] >= min_dir_entropy):
        fails.append(
            f"direction head collapsed (entropy {prof['ent_direction']:.3f} < "
            f"{min_dir_entropy}, d0 {prof['dir_d0_frac']:.0%}) -- following the "
            f"field, not steering"
        )
    return (not fails), "; ".join(fails)
