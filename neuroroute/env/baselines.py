"""Non-learned baselines.

These exist for the same reason `docs/RL_PLAN.md` insists on B0/B1/B2: a
completion percentage means nothing without a number to beat, and the cheapest
way to discover that an environment is broken is to run a policy whose
behaviour you can predict exactly.

`greedy_action` is the important one. Because direction index 0 is defined as
"down this head's own geodesic gradient", an all-zeros action *is* the greedy
router. That makes it simultaneously:

* the regression check on the whole observation/bearing/snap stack -- if the
  greedy completion number moves after an unrelated refactor, something in
  that stack broke (this is exactly how the line env's 8/24 number was used);
* the score a near-zero-init policy starts at, since near-zero logits sample
  action 0.

`greedy_safe_action` adds one non-learned improvement: take the longest step
the raycast proves is clear, instead of always stepping 1. It costs nothing and
separates "the policy learned to route" from "the policy learned to take big
steps".
"""

from __future__ import annotations

import torch

from neuroroute.env.observation import Observation
from neuroroute.world.spec import NUM_STEPS


def _base(obs: Observation) -> dict[str, torch.Tensor]:
    B, K = obs.head_mask.shape
    z = torch.zeros(B, K, dtype=torch.long, device=obs.head_mask.device)
    return {
        "direction": z.clone(),
        "step": z.clone(),
        "layer": z.clone(),
        "via": z.clone(),
        "width": z.clone(),
        "couple": torch.ones_like(z),
    }


def greedy_action(obs: Observation) -> dict[str, torch.Tensor]:
    """Walk straight down the geodesic gradient, one cell at a time."""
    return _base(obs)


def greedy_safe_action(obs: Observation) -> dict[str, torch.Tensor]:
    """Down the gradient, taking the longest step the raycast proves is clear."""
    act = _base(obs)
    safe = obs.safety[:, :, 0, :]                    # (B, K, NUM_STEPS) for direction 0
    idx = torch.arange(NUM_STEPS, device=safe.device).view(1, 1, -1)
    best = torch.where(safe, idx, torch.full_like(idx, -1)).amax(dim=-1)
    act["step"] = best.clamp_min(0)
    return act


def detour_action(obs: Observation, rng: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Greedy, but when every step in the gradient direction is blocked, turn.

    Picks the nearest-to-gradient direction that has any clear step. This is
    the cheapest possible obstacle response and is the honest bar for "did the
    policy learn anything beyond turning when blocked" -- worth having,
    because `docs/RL_PLAN.md` records a random policy out-completing greedy on
    a dense board purely by wandering around obstacles.
    """
    act = _base(obs)
    safe = obs.safety                                 # (B, K, D, S)
    any_safe = safe.any(dim=-1)                       # (B, K, D)
    D = any_safe.shape[-1]
    # Prefer the smallest turn away from direction 0, breaking ties toward 0.
    turn = torch.arange(D, device=safe.device)
    cost = torch.minimum(turn, D - turn).float().view(1, 1, D)
    cost = torch.where(any_safe, cost, torch.full_like(cost.expand_as(any_safe), 1e6))
    d = cost.argmin(dim=-1)
    act["direction"] = d

    chosen = safe.gather(2, d.view(*d.shape, 1, 1).expand(*d.shape, 1, safe.shape[-1])).squeeze(2)
    idx = torch.arange(safe.shape[-1], device=safe.device).view(1, 1, -1)
    act["step"] = torch.where(chosen, idx, torch.full_like(idx, -1)).amax(dim=-1).clamp_min(0)
    return act


def layer_hop_action(obs: Observation, via_threshold: float = 0.02) -> dict[str, torch.Tensor]:
    """Detour routing, plus a via whenever another layer is meaningfully closer.

    This is the baseline that exists purely to prove the multi-layer machinery
    works end to end -- vias placed, occupancy marked across the span, the
    route still connected through the layer change. `switch_layer()` is
    **0-for-32** against KiCad's PNS router (docs/RL_PLAN.md, Gate A, closed
    after three sessions), so this is the first time anything in this project
    can route on more than one layer at all. A baseline that never places a
    via would leave that completely untested.

    `via_threshold` is in units of LENGTH_SCALE, on the same relative scale as
    `Observation.geo_layer`. It is small on purpose: the geodesic field already
    charges `via_cost` for every layer change, so a negative `geo_layer` entry
    *already means* "closer even after paying for the via". The threshold only
    has to clear bilinear-sampling noise, not re-price the via. An earlier
    0.15 sat just above the 4-cell via cost (4/32 = 0.125) and suppressed
    every legitimate hop -- the baseline placed almost no vias and multi-layer
    boards scored identically to single-layer ones.
    """
    act = detour_action(obs)
    gl = obs.geo_layer                                  # (B, K, L), relative to current
    best = gl.argmin(dim=-1)                            # (B, K)
    gain = -gl.gather(-1, best.unsqueeze(-1)).squeeze(-1)

    cur = obs.head_pos[..., 0]
    worth = (gain > via_threshold) & (best != cur) & obs.head_mask
    act["layer"] = torch.where(worth, best + 1, torch.zeros_like(best))
    return act
