"""Observation construction: what the policy actually sees.

Two tensors go to the model, and the split is deliberate:

* **`field`** ``(B, C, L, H, W)`` -- the board. Shared by every frontier on that
  board, so it is encoded **once** per macro-step no matter how many frontiers
  are live. This is what makes simultaneous growth affordable: the expensive
  part of the forward pass does not scale with frontier count.
* **`frontiers`** ``(B, F, D)`` -- per-frontier sensory vector. Exact,
  non-learned local geometry: raycasts, per-(direction, step) safety, geodesic
  lookahead, and price lookahead. Never pooled, never decoded.

**Hand the policy the geometry; do not ask it to reconstruct it.** That is the
clearest finding in this repo's history, and it is load-bearing here. The
raycast sensor took Rejected-Action Rate from 1.51% to 0.40% [LIVE] precisely
because it is recomputed from the occupancy grid on every forward pass rather
than decoded from a learned embedding -- while four separate attempts to decode
comparable information *out of* an embedding all failed (`jepa/` x3,
`models/fast_lookahead.py`).

`price_ray` is this design's own application of that principle, and it is the
one the whole architecture rests on. The congestion price is what makes nets
negotiate instead of race, so the policy must be able to see, per candidate
move, *"this direction is contested"* -- directly, at decision time, in one
forward pass. Leaving that to be inferred from the field encoder would put the
single most important signal behind exactly the kind of bottleneck this repo has
already failed at four times.

Everything metric is divided by `LENGTH_SCALE` so the model's inputs stay
stationary as boards grow. That stationarity is what the 128 -> 1024
generalisation claim rests on, and nothing here has a board-size-dependent
parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mzr.world import geometry as geo
from mzr.world.engine import STATUS_DONE, STATUS_ROUTING
from mzr.world.spec import (
    KIND_DIFF_PAIR,
    NUM_DIRECTIONS,
    NUM_ENDS,
    NUM_KINDS,
    NUM_STEPS,
    OCC_FREE,
    STEP_LENGTHS,
)

#: Field channels. Kept small on purpose -- each is L*H*W floats per board, and
#: the encoder is the memory bottleneck at L=8, H=W=128.
CH_OCCUPIED = 0
CH_KEEPOUT = 1
CH_PAD = 2
CH_POUR = 3
#: The negotiation substrate. Present and history are kept **separate** rather
#: than handed over as PathFinder's single product: they mean different things
#: -- "who wants this now" versus "this has been fought over for a while" -- and
#: collapsing them costs the policy the ability to tell a transient crossing
#: from a genuinely oversubscribed channel.
CH_PRICE_PRESENT = 4
CH_PRICE_HISTORY = 5
#: Copper belonging to nets that are still routing, i.e. still rippable. This is
#: different information from "occupied", and it is what lets the encoder tell a
#: settled obstacle from one that might yet move out of the way -- which is
#: precisely the judgement simultaneous growth requires and sequential routing
#: never has to make.
CH_ROUTING_NET = 6
FIELD_CHANNELS = 7

#: Distance normaliser, in cells.
LENGTH_SCALE = 32.0

#: Extra per-frontier scalars, counted in `frontier_feature_dim`.
NUM_SCALARS = 12


def frontier_feature_dim(num_layers: int) -> int:
    return (
        NUM_DIRECTIONS                    # raycast free distance
        + NUM_DIRECTIONS * NUM_STEPS      # per-(direction, step) safety
        + NUM_DIRECTIONS * NUM_STEPS      # geodesic lookahead per candidate move
        + NUM_DIRECTIONS * NUM_STEPS      # price lookahead per candidate move
        + num_layers                      # current layer one-hot
        + num_layers                      # target layer one-hot
        + num_layers                      # cost-to-go on every layer, from here
        + num_layers                      # is a via to that layer even legal
        + NUM_KINDS                       # net kind one-hot
        + NUM_SCALARS
    )


@dataclass
class Observation:
    field: torch.Tensor          # (B, C, L, H, W) float32
    frontiers: torch.Tensor      # (B, F, D) float32
    #: (B, F, 3) int64. **Must be a clone, never a view** -- `world.fr_pos` is
    #: replaced in place by `step()`, and an aliasing observation made
    #: NeuroRoute's `policy.evaluate()` silently disagree with `policy.act()`,
    #: breaking the PPO ratio with no error anywhere.
    frontier_pos: torch.Tensor
    frontier_mask: torch.Tensor  # (B, F) bool -- alive and routing
    #: (B, F, NUM_DIRECTIONS, NUM_STEPS) bool. Not a model input -- the fixed,
    #: non-learned suppression mask applied to the direction and step logits.
    safety: torch.Tensor
    #: (B, F, L) bool -- whether a via from here to that layer is legal. Applied
    #: as fixed logit suppression exactly like `safety`. Without it an untrained
    #: policy spends most of its actions attempting impossible vias -- measured
    #: at a 92.6% rejected-action rate against a 1.6% baseline.
    via_safe: torch.Tensor
    #: (B, F, L) float32 -- cost-to-go from this (y, x) on **every** layer,
    #: relative to the current one. Negative means that layer is closer. This is
    #: the layer head's whole decision basis, and it is why the geodesic field is
    #: 3-D: a 2-D field cannot express "this layer is blocked here but the one
    #: below is open", which is exactly the question a via answers.
    geo_layer: torch.Tensor
    #: (B, F) int64 -- the egocentric bearing, so a renderer or baseline can
    #: decode what "direction 0" meant on this step.
    bearing: torch.Tensor
    #: (B, F) bool -- is this frontier part of a differential pair? The `couple`
    #: action only means anything for a pair.
    is_pair: torch.Tensor


def stack_observations(obs_list: list[Observation]) -> Observation:
    """Concatenate a list of same-shape observations along the batch axis.

    The PPO update needs `policy.evaluate()` on many stored timesteps. Calling
    it once per timestep runs the (heavy) field encoder that many times; on the
    stage-0 profile that was ~30 s of the ~40 s update. Stacking K timesteps
    into one (K*B, ...) observation and evaluating once collapses that to a
    single encoder pass -- same FLOPs, an order of magnitude less launch
    overhead, and it actually parallelises on the GPU.

    Every field is a plain tensor with a leading batch dim, so this is one
    `torch.cat` per field. The caller is responsible for reshaping the
    (K*B, ...) outputs back to (K, B, ...).
    """
    return Observation(
        field=torch.cat([o.field for o in obs_list], dim=0),
        frontiers=torch.cat([o.frontiers for o in obs_list], dim=0),
        frontier_pos=torch.cat([o.frontier_pos for o in obs_list], dim=0),
        frontier_mask=torch.cat([o.frontier_mask for o in obs_list], dim=0),
        safety=torch.cat([o.safety for o in obs_list], dim=0),
        via_safe=torch.cat([o.via_safe for o in obs_list], dim=0),
        geo_layer=torch.cat([o.geo_layer for o in obs_list], dim=0),
        bearing=torch.cat([o.bearing for o in obs_list], dim=0),
        is_pair=torch.cat([o.is_pair for o in obs_list], dim=0),
    )


def build_field(world) -> torch.Tensor:
    """(B, C, L, H, W) board tensor."""
    occ = world.occ
    B, L, H, W = occ.shape
    out = torch.zeros(
        B, FIELD_CHANNELS, L, H, W, dtype=torch.float32, device=occ.device
    )
    out[:, CH_OCCUPIED] = (occ != OCC_FREE).float()
    out[:, CH_KEEPOUT] = (occ < 0).float()
    out[:, CH_PAD] = (world.static > 0).float()
    out[:, CH_POUR] = world.pour.float()

    price = world.price.observation()          # (B, 2, L, H, W), already scaled
    out[:, CH_PRICE_PRESENT] = price[:, 0]
    out[:, CH_PRICE_HISTORY] = price[:, 1]

    # Copper of nets that have not settled yet. Built by mapping each cell's
    # owner id through a per-net "still routing" table -- one gather, rather
    # than a comparison per net, which at 2000 nets is the difference between
    # affordable and not.
    routing = torch.zeros(B, world.cfg.max_nets + 1, dtype=torch.bool, device=occ.device)
    routing[:, 1:] = world.net_valid & (world.net_status == STATUS_ROUTING)
    ids = occ.long().clamp_min(0).flatten(1)
    out[:, CH_ROUTING_NET] = routing.gather(1, ids).view(B, L, H, W).float()
    return out


def _one_hot(idx: torch.Tensor, n: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(idx.clamp(0, n - 1), n).float()


def build_observation(world) -> Observation:
    """Assemble everything the policy sees for one macro-step."""
    B, F = world.cfg.batch_size, world.F
    L, H, W = world.shape
    N = world.cfg.max_nets
    dev = world.device
    M = B * F
    ds = world.cfg.geodesic_downsample

    f_idx = torch.arange(F, device=dev).view(1, F).expand(B, F)
    net = f_idx // (world.cfg.max_legs * NUM_ENDS)
    b_flat = torch.arange(B, device=dev).view(B, 1).expand(B, F).reshape(M)
    n_flat = net.reshape(M)

    alive = world.fr_alive & (world.net_status.gather(1, net) == STATUS_ROUTING)
    pos = world.fr_pos.reshape(M, 3)
    width = world.net_width.gather(1, net)
    w_flat = width.reshape(M)
    kind = world.net_kind.gather(1, net)
    is_pair = kind == KIND_DIFF_PAIR

    fld = world.fr_geo.reshape(M, L, *world._geo_shape).float()
    tgt = world._target_pad().reshape(M, 3)

    # --- exact local geometry, recomputed every step ------------------------
    free = geo.raycast(
        world.occ, b_flat, pos[:, 0], pos[:, 1], pos[:, 2], n_flat, world.tables, w_flat
    )                                                   # (M, NUM_DIRECTIONS)
    safety = geo.step_safety(free)                      # (M, NUM_DIRECTIONS, NUM_STEPS)
    bearing = geo.bearing_from_field(
        fld, pos[:, 0], pos[:, 1], pos[:, 2], world.tables, ds, free_units=free
    )                                                   # (M,)

    # Rotate the absolute-frame raycast into the egocentric frame. The raycast
    # reports absolute directions while every action resolves relative to the
    # bearing; when those two frames drifted apart in NeuroRoute, the
    # rejected-action rate hit 86% while the baseline believed it was only
    # taking moves the raycast had called safe. After the rotation: 1.6%.
    roll = torch.arange(NUM_DIRECTIONS, device=dev).view(1, -1)
    ego = (bearing.view(M, 1) + roll) % NUM_DIRECTIONS
    free_ego = free.gather(1, ego)
    safety_ego = safety.gather(1, ego.unsqueeze(-1).expand(M, NUM_DIRECTIONS, NUM_STEPS))

    # --- candidate-move lookahead: geodesic and price -----------------------
    steps = world.tables.step_len.view(1, 1, NUM_STEPS)
    unit = world.tables.path[:, 0, 0]                   # (NUM_DIRECTIONS, 2)
    # `ego` already maps each egocentric direction slot to its absolute
    # direction, so index the offset table with it directly -- slot d of the
    # result is then "what happens if I take egocentric direction d".
    sel = unit[ego]                                      # (M, NUM_DIRECTIONS, 2)
    cy = pos[:, 1].view(M, 1, 1) + sel[..., 0:1] * steps
    cx = pos[:, 2].view(M, 1, 1) + sel[..., 1:2] * steps
    lay = pos[:, 0].view(M, 1, 1).expand(M, NUM_DIRECTIONS, NUM_STEPS)

    here = geo.sample_coarse(fld, pos[:, 0], pos[:, 1], pos[:, 2], ds).view(M, 1, 1)
    there = geo.sample_coarse(fld, lay, cy, cx, ds)
    # Improvement, not absolute distance: "does this move get me closer" is the
    # question, and a difference is already scale-free.
    geo_ray = torch.nan_to_num((here - there) * ds, posinf=0.0, neginf=0.0, nan=0.0)
    geo_ray = (geo_ray / LENGTH_SCALE).clamp(-4.0, 4.0)

    price = world.price.field()                          # (B, L, H, W)
    bb = b_flat.view(M, 1, 1).expand(M, NUM_DIRECTIONS, NUM_STEPS)
    inb = (cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)
    price_ray = price[bb, lay.clamp(0, L - 1), cy.clamp(0, H - 1), cx.clamp(0, W - 1)]
    # Out of bounds reads as maximally expensive, not free -- otherwise walking
    # off the board looks like the cheapest move available.
    price_ray = torch.where(inb, price_ray, torch.full_like(price_ray, 8.0))
    price_ray = (price_ray.log1p() / 3.0).clamp(0.0, 2.0)

    # --- per-layer cost-to-go and via legality ------------------------------
    per_layer = []
    for l in range(L):
        ll = torch.full_like(pos[:, 0], l)
        per_layer.append(geo.sample_coarse(fld, ll, pos[:, 1], pos[:, 2], ds))
    stack = torch.nan_to_num(
        torch.stack(per_layer, dim=-1) * ds, posinf=1e6, nan=1e6
    )                                                    # (M, L)
    cur = stack.gather(1, pos[:, 0].view(M, 1))
    geo_layer = ((cur - stack) / LENGTH_SCALE).clamp(-4.0, 4.0)

    if world.spec.layers.through_only:
        lo = torch.zeros(M, dtype=torch.long, device=dev)
        hi = torch.full((M,), L - 1, dtype=torch.long, device=dev)
    else:
        tl = torch.arange(L, device=dev)
        lo = torch.minimum(pos[:, 0:1], tl.view(1, L)).reshape(-1)
        hi = torch.maximum(pos[:, 0:1], tl.view(1, L)).reshape(-1)
    via_safe = torch.zeros(M, L, dtype=torch.bool, device=dev)
    for l in range(L):
        if world.spec.layers.through_only:
            vlo, vhi = lo, hi
        else:
            vlo = torch.minimum(pos[:, 0], torch.full_like(pos[:, 0], l))
            vhi = torch.maximum(pos[:, 0], torch.full_like(pos[:, 0], l))
        vflat, vvalid = geo.via_claims(
            world.occ, b_flat, vlo, vhi, pos[:, 1], pos[:, 2],
            torch.zeros(M, dtype=torch.long, device=dev), world.tables,
        )
        via_safe[:, l] = geo.claims_passable(world.occ, vflat, vvalid, n_flat)
    # Staying put is always "legal"; the layer head reads index 0 as stay.
    via_safe[:, 0] = via_safe[:, 0] | (pos[:, 0] == 0)

    # --- scalars ------------------------------------------------------------
    dist = (here.view(M) / LENGTH_SCALE).clamp(0.0, 8.0)
    dyx = (tgt[:, 1:] - pos[:, 1:]).float()
    straight = dyx.norm(dim=-1) / LENGTH_SCALE
    routed = world.net_len.gather(
        1, net.unsqueeze(-1).expand(B, F, 2)
    ).reshape(M, 2).sum(dim=-1) / LENGTH_SCALE
    steps_used = world.fr_steps.reshape(M).float() / max(1, world.cfg.max_steps_per_frontier)
    partner = (
        world.fr_pos.view(B, N, world.cfg.max_legs, NUM_ENDS, 3)
        .flip(dims=[3])
        .reshape(M, 3)
    )
    to_partner = (partner[:, 1:] - pos[:, 1:]).float().norm(dim=-1) / LENGTH_SCALE
    own_price = price[b_flat, pos[:, 0], pos[:, 1], pos[:, 2]]

    scalars = torch.stack(
        [
            dist,
            torch.log1p(dist),
            straight,
            routed,
            steps_used,
            to_partner.clamp(0.0, 8.0),
            (partner[:, 0] == pos[:, 0]).float(),
            (own_price.log1p() / 3.0).clamp(0.0, 2.0),
            alive.reshape(M).float(),
            (world.net_vias.gather(1, net).reshape(M).float() / 8.0).clamp(0.0, 4.0),
            free_ego[:, 0].float() / float(max(STEP_LENGTHS)),
            torch.full((M,), world.step_count / max(1, world.cfg.max_macro_steps), device=dev),
        ],
        dim=-1,
    )

    feats = torch.cat(
        [
            free_ego.float() / float(max(STEP_LENGTHS)),
            safety_ego.reshape(M, -1).float(),
            geo_ray.reshape(M, -1),
            price_ray.reshape(M, -1),
            _one_hot(pos[:, 0], L),
            _one_hot(tgt[:, 0], L),
            geo_layer,
            via_safe.float(),
            _one_hot(kind.reshape(M), NUM_KINDS),
            scalars,
        ],
        dim=-1,
    )
    # A dead frontier contributes nothing. Zeroing here rather than relying on
    # the mask downstream means a masking bug shows up as "no signal", not as
    # stale geometry from a frontier that stopped ten steps ago.
    feats = feats * alive.reshape(M, 1).float()

    return Observation(
        field=build_field(world),
        frontiers=feats.view(B, F, -1),
        frontier_pos=world.fr_pos.clone(),
        frontier_mask=alive,
        safety=safety_ego.view(B, F, NUM_DIRECTIONS, NUM_STEPS),
        via_safe=via_safe.view(B, F, L),
        geo_layer=geo_layer.view(B, F, L),
        bearing=bearing.view(B, F),
        is_pair=is_pair,
    )
