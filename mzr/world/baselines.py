"""Non-learned routers that drive the world directly.

These earn their place three times over, which is why they live in `world/`
rather than in a test file:

1. **The DRC gate** (`scripts/validate_kicad.py`) needs routed boards and no
   trained policy, so the sim-to-real question can be answered before any
   training exists to be wasted on a fiction.
2. **Stage 1's bar.** `mzr/DESIGN.md` section 7 sets the bar at *sequential +
   PathFinder negotiation*, not naive greedy -- the field tried concurrent
   routing and retreated to sequential-plus-negotiation, so that is the thing
   simultaneous growth actually has to beat.
3. **Behaviour-cloning demonstrations.** PRIMAL's authors got a decentralised
   multi-agent policy to work with "demonstrations of an expert planner during
   training, as well as careful reward shaping" -- not pure RL. NeuroRoute was
   pure RL and plateaued at 60-65%. The expert here is free: it is the same
   code the baseline uses.

All of them emit the world's ordinary action tuple, so anything that consumes a
policy consumes these unchanged.

**The zero action is the greedy router.** Direction 0 is egocentric -- it means
"down this frontier's own geodesic gradient" -- so `greedy` is literally all
zeros. That is the property that lets a near-zero-init policy start *at* the
baseline rather than below it, and `verify_world.py` checks it holds.
"""

from __future__ import annotations

import torch

import mzr.world.geometry as geo

Action = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

#: Coarse-field improvement, in coarse cells, required before `layer_hop` will
#: place a via.
#:
#: Must stay small. The geodesic relaxation **already charges `via_cost`** for a
#: crossing, so a layer that reads better after that charge is already worth the
#: via -- this threshold only exists to reject numerical noise. NeuroRoute set
#: it to 0.15, which sat just above the 4-cell via cost expressed in coarse
#: units (4/32 = 0.125) and suppressed *every* legitimate hop: vias went to ~0
#: and 8-layer boards scored identically to single-layer ones.
LAYER_HOP_MARGIN = 1e-3


def _zeros(world) -> torch.Tensor:
    return torch.zeros(
        world.cfg.batch_size, world.F, dtype=torch.long, device=world.device
    )


def greedy(world) -> Action:
    """Step one cell down the geodesic gradient. Never changes layer.

    The floor every learned policy has to clear. On multi-layer boards it
    *cannot* finish a net whose pads sit on different layers, since it never
    places a via -- which is exactly the gap `layer_hop` measures.
    """
    z = _zeros(world)
    return z, z, z, z, z, z


def detour(world) -> Action:
    """Greedy, but taking the longest step available.

    Separates "the policy learned to steer" from "the policy learned to move
    faster": if a trained policy beats greedy but not detour, what it found was
    step length, not geometry.
    """
    z = _zeros(world)
    longest = torch.full_like(z, len(world.tables.step_len) - 1)
    return z, longest, z, z, z, z


def layer_hop_action(world) -> torch.Tensor:
    """(B, F) layer action: hop to whichever layer the geodesic likes best.

    0 = stay; ``j > 0`` = place a via and move to layer ``j - 1``.

    The frontier's cached field is 3-D (`L, h, w`) and its cross-layer
    relaxation already prices a via, so "which layer is cheapest from here" is a
    question the field can answer directly -- no separate heuristic needed.
    """
    B, F = world.cfg.batch_size, world.F
    L = world.num_layers
    M = B * F
    ds = world.cfg.geodesic_downsample

    fld = (world._frontier_field() if world.cfg.copper_seeded
           else world.fr_geo.reshape(M, L, *world._geo_shape).float())
    pos = world.fr_pos.reshape(M, 3)
    cur = pos[:, 0]

    # Cost from this (y, x) on every layer, so the comparison is like-for-like.
    per_layer = []
    for l in range(L):
        lay = torch.full_like(cur, l)
        per_layer.append(geo.sample_coarse(fld, lay, pos[:, 1], pos[:, 2], ds))
    stack = torch.nan_to_num(torch.stack(per_layer, dim=-1), posinf=1e9, nan=1e9)

    best = stack.argmin(dim=-1)
    best_val = stack.gather(1, best.view(M, 1)).squeeze(1)
    cur_val = stack.gather(1, cur.view(M, 1)).squeeze(1)

    hop = (best != cur) & (best_val < cur_val - LAYER_HOP_MARGIN)
    return torch.where(hop, best + 1, torch.zeros_like(best)).view(B, F)


def layer_hop(world) -> Action:
    """Greedy plus "place a via when another layer is closer".

    The strongest non-learned baseline, and the honest bar for anything that
    claims multi-layer routing works: the greedy-to-layer_hop gap is precisely
    the nets whose pads sit on different layers.
    """
    z = _zeros(world)
    return z, z, layer_hop_action(world), z, z, z


BASELINES = {"greedy": greedy, "detour": detour, "layer_hop": layer_hop}


def rollout(world, policy=layer_hop, *, max_steps: int | None = None) -> int:
    """Drive `world` to the end of its episode. Returns macro-steps taken."""
    cap = max_steps if max_steps is not None else world.cfg.max_macro_steps
    n = 0
    while not world.episode_done() and n < cap:
        world.step(*policy(world))
        n += 1
    return n
