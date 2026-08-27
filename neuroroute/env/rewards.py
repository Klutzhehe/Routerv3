"""Reward terms.

The shaping term is **potential-based** (`gamma*Phi(s') - Phi(s)` with
`Phi = -geodesic/scale`), which is policy-invariant: it changes how fast the
policy learns but cannot change what the optimal policy is. That property is
why it is safe to make it the dominant dense signal, and it is carried over
unchanged from `docs/RL_PLAN.md` where it is already validated [LIVE].

Two calibration notes recorded from the existing thread, because both cost real
time to discover:

1. **Reward and completion rate are not the same objective.** Measured on a
   24-net board: a random policy scored -330 reward against greedy's -177 and
   still routed *more* nets (9 vs 8) -- straight-line is not the best
   completion strategy under congestion. Track completion.
2. **The collision penalty can make a policy timid.** At 0.5/step, twenty
   colliding steps cancel an entire completion bonus, so a policy learns to
   stand still. It is deliberately lower here relative to the arrival bonus,
   and `rejection` is reported separately so the trade is visible rather than
   buried in a scalar.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neuroroute.env.observation import LENGTH_SCALE
from neuroroute.world.engine import BatchedRouterWorld, StepResult


@dataclass
class RewardConfig:
    gamma: float = 0.99
    #: Weight on the potential-based geodesic shaping term.
    progress: float = 1.0
    #: Flat per-step cost. Makes finishing sooner strictly better.
    step_cost: float = 0.01
    #: Per rejected action. See calibration note 2 above.
    rejection: float = 0.15
    #: Per via. Vias cost area, yield and signal integrity; a free via action
    #: produces a policy that drills instead of routing.
    via: float = 0.30
    #: Per rip-up. More disruptive than a via -- it discards a whole
    #: completed route, not just adds one -- so it needs its own cost:
    #: nothing else stops a *trained* policy from ripping up nets for free
    #: once the untrained-default bias (h_ripup_none) washes out through
    #: learning. Set well below the arrival/completion bonuses so a rip-up
    #: that genuinely unblocks a later net is still clearly worth it.
    ripup: float = 0.50
    arrival: float = 10.0
    failure: float = 4.0
    #: Terminal, per board.
    completion: float = 20.0
    wirelength: float = 0.5
    drc: float = 2.0
    #: Differential pairs.
    pair_gap: float = 4.0
    pair_skew: float = 2.0
    pair_split: float = 0.5
    #: Length-matched groups, applied after the refine phase.
    length_error: float = 4.0
    length_tolerance_cells: float = 2.0
    length_bonus: float = 5.0


def step_reward(res: StepResult, cfg: RewardConfig) -> torch.Tensor:
    """(B, K) per-head reward for one `step()`.

    `res.progress` is already `Phi(s) - Phi(s')` in cells, summed over the
    legs a head owns, so it needs only scaling. Inactive slots get exactly
    zero, never a step cost -- an idle slot must not make standing still
    expensive, or the scheduler learns to keep every slot busy with a bad net
    purely to dodge the penalty.
    """
    active = res.active.float()
    r = cfg.progress * (res.progress / LENGTH_SCALE)
    r = r - cfg.step_cost * active
    r = r - cfg.rejection * res.rejected.float()
    r = r - cfg.via * res.via_placed.float()
    r = r + cfg.arrival * res.arrived.float()
    r = r - cfg.failure * res.exhausted.float()
    return r * active


def terminal_reward(world: BatchedRouterWorld, cfg: RewardConfig) -> tuple[torch.Tensor, dict]:
    """(B,) end-of-episode reward, plus the metrics behind it.

    Every constraint the user actually cares about lands here rather than in
    the per-step term, because all three are properties of a *finished* net:
    a pair's gap error is only meaningful over its whole routed length, a
    group's length target is not known until its longest member is routed, and
    wirelength is not comparable mid-route.
    """
    stats = world.board_stats()
    valid = world.net_valid.float()
    n_valid = valid.sum(dim=1).clamp_min(1.0)

    r = cfg.completion * stats["completion"]

    # Excess wirelength over the straight-line lower bound, averaged per net.
    src = world.net_src[:, :, 0, 1:].float()
    dst = world.net_dst[:, :, 0, 1:].float()
    ideal = torch.linalg.vector_norm(dst - src, dim=-1).clamp_min(1.0)
    routed = world.net_len.sum(dim=-1)
    done = ((world.net_status == 2) & world.net_valid).float()
    detour = ((routed / ideal - 1.0).clamp_min(0.0) * done).sum(dim=1) / n_valid
    r = r - cfg.wirelength * detour

    gap = world.pair_gap_error()
    skew = world.pair_skew() / LENGTH_SCALE
    is_pair = (world.net_kind == 1).float() * valid
    n_pair = is_pair.sum(dim=1).clamp_min(1.0)
    r = r - cfg.pair_gap * (gap * is_pair).sum(dim=1) / n_pair
    r = r - cfg.pair_skew * (skew * is_pair).sum(dim=1) / n_pair
    r = r - cfg.pair_split * stats["split_fraction"]

    len_err = world.group_length_error()
    in_group = (world.net_group >= 0).float() * valid
    n_group = in_group.sum(dim=1).clamp_min(1.0)
    r = r - cfg.length_error * (len_err / LENGTH_SCALE * in_group).sum(dim=1) / n_group
    within = (len_err <= cfg.length_tolerance_cells).float() * in_group
    r = r + cfg.length_bonus * within.sum(dim=1) / n_group

    metrics = {
        "completion": stats["completion"],
        "vias": stats["vias"].float(),
        "ripups": stats["ripups"].float(),
        "wirelength": stats["wirelength"],
        "detour": detour,
        "pair_gap_error": (gap * is_pair).sum(dim=1) / n_pair,
        "pair_skew_cells": (world.pair_skew() * is_pair).sum(dim=1) / n_pair,
        "split_fraction": stats["split_fraction"],
        "length_error_cells": (len_err * in_group).sum(dim=1) / n_group,
        "length_within_tol": within.sum(dim=1) / n_group,
    }
    return r, metrics
