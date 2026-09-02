"""Reward terms for the simultaneous-frontier MDP.

The dense signal is **potential-based shaping** -- ``gamma*Phi(s') - Phi(s)``
with ``Phi = -geodesic/scale`` -- which is policy-invariant: it changes how
fast the policy learns but cannot change what the optimal policy is. That is
why it is safe to make it the dominant term. Carried over from `docs/RL_PLAN.md`
where it is validated [LIVE].

What is new here is the **congestion-delta** term. Simultaneous growth only
works if a frontier is charged for *creating* contention and credited for
*resolving* it -- otherwise every frontier greedily descends its own gradient
into the same channel and the congestion price has nothing to push against.
The term is the per-step change in board-wide present congestion, shared
equally across the board's live frontiers: it is a joint outcome, so it gets a
joint (not per-frontier) attribution.

Calibration notes carried from `neuroroute/`, each of which cost real time:

1. **Reward and completion are not the same objective.** On a 24-net board a
   random policy scored -330 reward against greedy's -177 and still routed
   *more* nets. Track completion, gate on completion, never on reward.
2. **A large rejection/collision penalty makes a policy timid.** At 0.5/step,
   twenty rejected steps cancel a whole completion bonus and the policy learns
   to stand still. Kept low here, and `contended` (arbitrated away by another
   net, no fault of this frontier) is separated from `rejected` (this
   frontier's move was illegal) so the two are not conflated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mzr.env.observation import LENGTH_SCALE
from mzr.world.engine import StepResult


@dataclass
class RewardConfig:
    gamma: float = 0.99
    #: Potential-based geodesic shaping. Raised from 1.0: at the old weight a
    #: step toward the target was worth ~0.02 (a 1-cell geodesic drop / a
    #: LENGTH_SCALE of 32), which the normalised advantage's value-function
    #: noise drowned -- on stage 0 the `direction`-toward-target bias eroded
    #: and argmax stopped pointing at the pad. At 4.0 a good step is ~0.12,
    #: clearly the best local move.
    progress: float = 4.0
    #: When > 0, shape on the **leg's** closing gap instead of each frontier's
    #: distance to its own far pad, and `progress` is ignored.
    #:
    #: Per-frontier shaping pays a leg twice for one corridor: both frontiers
    #: are rewarded for nearing the opposite pad, so the optimal behaviour is
    #: for each to route the whole net. Measured on the gate-clearing stage-0
    #: policy: 6 of 9 boards laid ~2.2x the necessary copper as a closed loop,
    #: two mirror paths between the same pads, while `completion` read 1.000
    #: because the net was connected -- twice. One board had a frontier that
    #: never moved at all.
    #:
    #: The leg gap is `d_src + d_dst - D`: it starts at D and hits zero exactly
    #: when the pair has covered the route between them, so it pays for ground
    #: covered once and pays nothing for a frontier that keeps going after the
    #: work is done.
    leg_progress: float = 0.0
    #: Dense weight on closing the distance to this frontier's **partner** --
    #: the other end of its own leg. Added to `progress`, not a replacement.
    #:
    #: This is the only shaping term whose sign flips on a redundant traverse.
    #: Two frontiers mirror-routing around opposite sides of an obstacle end up
    #: SWAPPING positions, so tip distance runs D -> narrow -> D and the second
    #: half of the detour is charged. `progress` (distance to the far pad) and
    #: `leg_progress` (the leg gap) both keep paying straight through that swap,
    #: which is why neither stopped the loop -- measured, twice.
    #:
    #: Pairing is per-LEG, so it generalises to any net count, and a
    #: differential pair's two legs pair up independently rather than the P leg
    #: chasing the N leg.
    tip_progress: float = 0.0
    #: Flat per-step cost, so finishing sooner is strictly better.
    step_cost: float = 0.01
    #: This frontier's move was illegal. See calibration note 2.
    rejection: float = 0.10
    #: This frontier lost a cell to another net in arbitration -- not its
    #: fault, so a much lighter touch than `rejection`. It is still a small
    #: negative because the policy *can* learn to route where arbitration will
    #: not bite.
    contended: float = 0.02
    #: Per via. Vias cost area, yield and signal integrity; a free via action
    #: yields a policy that drills instead of routing.
    via: float = 0.30
    #: Weight on the per-step change in board-wide present congestion, split
    #: across live frontiers. Negative delta (congestion fell) is a reward.
    congestion_delta: float = 0.30
    #: Per 45-degree octant of bend **beyond the first**, on an accepted move.
    #: A 45-degree turn is free; 90 degrees costs one unit, 135 two, a reversal
    #: three. This is a real fab rule rather than an aesthetic one: a right
    #: angle in copper is an acid trap when the board is etched and an
    #: impedance discontinuity when the trace is driven, which is why layout
    #: practice replaces every 90-degree corner with two 45-degree bends.
    #:
    #: The action space already moves in octants, so this needs no new action
    #: -- only a memory of the previous heading (`world.fr_dir`). Kept small:
    #: it is a preference between equally-valid routes, and must never outweigh
    #: `progress` or the policy will prefer a straight line into a wall over
    #: turning to reach the pad.
    corner: float = 0.08
    #: A leg connected this step -- credited to both its frontiers. Lowered
    #: from 5.0: at 5.0 (x2 frontiers = +10) plus a completion bonus of 20,
    #: the return was dominated by a few large spikes at irregular episode
    #: positions, and the near-zero-init value head could not track it --
    #: vloss swung 7 -> 128 between rollout batches and fed the policy thrash.
    arrival: float = 2.0
    #: A net ran out of budget with a leg still open.
    failure: float = 4.0
    #: Terminal, per board. Lowered from 20.0 with `arrival` -- see above.
    completion: float = 10.0
    wirelength: float = 0.5  # weight on excess routed length over straight-line
    #: Differential pairs (stage 6). Dormant until pairs appear in the netlist.
    pair_gap: float = 4.0
    pair_skew: float = 2.0
    pair_split: float = 0.5
    #: Length-matched groups (stage 7).
    length_error: float = 4.0
    length_tolerance_cells: float = 2.0
    length_bonus: float = 5.0


def step_reward(
    res: StepResult,
    arrived: torch.Tensor,
    cfg: RewardConfig,
) -> torch.Tensor:
    """(B, F) per-frontier reward for one macro-step.

    `res.progress` is already ``Phi(s) - Phi(s')`` in cells per frontier, so it
    needs only scaling. `arrived` (B, F) bool is supplied by the env, which
    diffs `leg_done` across the step -- the engine's `StepResult` reports
    connections only as a board-level count, and credit assignment wants the
    frontier that actually closed the leg.

    Dead frontiers get exactly zero -- never a step cost. A frontier that has
    finished or been retired must not make the policy's "do nothing here" look
    expensive.
    """
    live = res.live.float()

    if cfg.leg_progress > 0.0:
        r = cfg.leg_progress * (res.leg_progress / LENGTH_SCALE)
    else:
        r = cfg.progress * (res.progress / LENGTH_SCALE)
    if cfg.tip_progress > 0.0:
        r = r + cfg.tip_progress * (res.tip_progress / LENGTH_SCALE)
    r = r - cfg.step_cost * live
    r = r - cfg.rejection * res.rejected.float()
    r = r - cfg.contended * res.contended.float()
    r = r - cfg.via * res.via_placed.float()
    # Straight and 45-degree bends are free; charge only the excess octants, so
    # the term prices right angles rather than taxing every turn.
    r = r - cfg.corner * (res.turn - 1.0).clamp_min(0.0)
    r = r + cfg.arrival * arrived.float()

    # Board-wide congestion change, shared equally across this board's live
    # frontiers. `congestion_delta` is (B,); a fall in congestion is a reward.
    n_live = live.sum(dim=1, keepdim=True).clamp_min(1.0)
    r = r - cfg.congestion_delta * (res.congestion_delta.unsqueeze(1) / n_live) * live

    return r * live


def failure_penalty(res: StepResult, cfg: RewardConfig) -> torch.Tensor:
    """(B,) charge for nets abandoned this step. Board-level: a starved net is
    an ordering/negotiation failure, not one frontier's fault."""
    return -cfg.failure * res.nets_failed.float()


def terminal_reward(world, cfg: RewardConfig) -> tuple[torch.Tensor, dict]:
    """(B,) end-of-episode reward, plus the metrics behind it.

    Every constraint the user cares about lands here rather than per-step,
    because each is a property of a *finished* route: a pair's gap error is
    only meaningful over its whole length, a group's length target is not known
    until its longest member is routed, and wirelength is not comparable
    mid-route.
    """
    stats = world.board_stats()
    valid = world.net_valid
    valf = valid.float()
    n_valid = valf.sum(dim=1).clamp_min(1.0)

    r = cfg.completion * stats["completion"]

    # Excess wirelength over the straight-line lower bound, per completed net.
    src = world.net_pad[:, :, 0, 0, 1:].float()
    dst = world.net_pad[:, :, 0, 1, 1:].float()
    ideal = torch.linalg.vector_norm(dst - src, dim=-1).clamp_min(1.0)
    routed = world.net_len.sum(dim=-1)
    done = ((world.net_status == 1) & valid).float()
    detour = ((routed / ideal - 1.0).clamp_min(0.0) * done).sum(dim=1) / n_valid
    r = r - cfg.wirelength * detour

    # --- differential pairs (dormant until pairs exist) -------------------
    is_pair = (world.net_kind == 1).float() * valf
    n_pair = is_pair.sum(dim=1).clamp_min(1.0)
    if bool((is_pair > 0).any()):
        gap = world.pair_gap_error()
        skew = world.pair_skew() / LENGTH_SCALE
        r = r - cfg.pair_gap * (gap * is_pair).sum(dim=1) / n_pair
        r = r - cfg.pair_skew * (skew * is_pair).sum(dim=1) / n_pair
        r = r - cfg.pair_split * (world.net_split * is_pair).sum(dim=1) / n_pair

    # --- length-matched groups (dormant until groups exist) ---------------
    in_group = (world.net_group >= 0).float() * valf
    n_group = in_group.sum(dim=1).clamp_min(1.0)
    if bool((in_group > 0).any()):
        len_err = world.group_length_error()
        r = r - cfg.length_error * (len_err / LENGTH_SCALE * in_group).sum(dim=1) / n_group
        within = (len_err <= cfg.length_tolerance_cells).float() * in_group
        r = r + cfg.length_bonus * within.sum(dim=1) / n_group

    metrics = {
        "completion": stats["completion"],
        "vias": stats["vias"],
        "ripups": stats["ripups"],
        "wirelength": stats["length"],
        "detour": detour,
        "congestion": stats["congestion"],
        "failed": stats["failed"],
    }
    return r, metrics
