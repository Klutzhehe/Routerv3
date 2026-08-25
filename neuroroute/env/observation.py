"""Observation construction.

Three tensors go to the model, and the split is deliberate:

* **`field`** ``(B, C, L, H, W)`` -- the board. Shared by every head on that
  board, so it is encoded **once** per step no matter how many heads are
  active. This is what makes `K` simultaneous heads nearly free: the expensive
  part of the forward pass does not scale with `K`.
* **`heads`** ``(B, K, F_head)`` -- per-head sensory vector. Exact, non-learned
  local geometry: raycasts, per-(direction, step) safety, and geodesic
  lookahead. Never pooled, never decoded.
* **`nets`** ``(B, N, F_net)`` -- one token per net, for the scheduler.

The reason the head vector carries so much raw geometry is the single clearest
finding in this repo's history. `models/pcb_encoder.py`'s raycast sensor took
the Rejected-Action Rate from 1.51% to 0.40% [LIVE] precisely because it is
computed fresh from the occupancy grid every forward pass rather than decoded
from a learned embedding -- and four separate attempts to decode comparable
information *from* an embedding all failed (`jepa/`,
`models/fast_lookahead.py`). Hand the policy the geometry; do not ask it to
reconstruct the geometry.

`geo_ray` is new here and is the natural extension of that principle: sampling
the *geodesic* field along each candidate (direction, step) answers "which of
my moves actually gets me closer, going around obstacles" directly, which is
the question `models/analytic_lookahead.py` was answering by replaying the real
environment at ~16x the per-step cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neuroroute.world import geometry as geo
from neuroroute.world.engine import STATUS_ACTIVE, STATUS_DONE, STATUS_PENDING, BatchedRouterWorld
from neuroroute.world.spec import (
    KIND_DIFF_PAIR,
    NUM_DIRECTIONS,
    NUM_KINDS,
    NUM_STEPS,
    OCC_FREE,
    STEP_LENGTHS,
)

#: Field channels. Kept small on purpose -- every one is L*H*W floats per
#: board, and the encoder is the memory bottleneck at L=8, H=W=128.
FIELD_CHANNELS = 6
CH_OCCUPIED = 0
CH_KEEPOUT = 1
CH_PAD = 2
CH_POUR = 3
CH_DEMAND = 4
CH_ACTIVE_NET = 5

#: Distance normaliser, in cells. Everything metric is divided by this so the
#: model's inputs stay stationary as boards get bigger -- the property the
#: 128 -> 1024 generalisation claim rests on.
LENGTH_SCALE = 32.0


def head_feature_dim(num_layers: int) -> int:
    return (
        NUM_DIRECTIONS                    # raycast free distance
        + NUM_DIRECTIONS * NUM_STEPS      # per-(direction, step) safety
        + NUM_DIRECTIONS * NUM_STEPS      # geodesic lookahead per candidate move
        + num_layers                      # current layer one-hot
        + num_layers                      # target layer one-hot
        + num_layers                      # cost-to-go on every layer, from here
        + num_layers                      # is a via to that layer even legal
        + NUM_KINDS                       # net kind one-hot
        + 12                              # scalars, below
    )


def net_feature_dim() -> int:
    return 4 + NUM_KINDS + 6


@dataclass
class Observation:
    field: torch.Tensor       # (B, C, L, H, W) float32
    heads: torch.Tensor       # (B, K, F_head) float32
    head_pos: torch.Tensor    # (B, K, 3) int64 -- primary leg, for latent gather
    head_mask: torch.Tensor   # (B, K) bool
    nets: torch.Tensor        # (B, N, F_net) float32
    net_mask: torch.Tensor    # (B, N) bool -- schedulable (valid & pending)
    #: (B, K, NUM_DIRECTIONS, NUM_STEPS) bool. Not a model input -- it is the
    #: fixed, non-learned action suppression mask applied to the direction and
    #: step logits. See models/heads.py.
    safety: torch.Tensor
    #: (B, K) int64 -- each head's egocentric bearing, so a renderer or a
    #: baseline can decode what "direction 0" meant on this step.
    bearing: torch.Tensor
    #: (B, K) bool -- is this head routing a differential pair? The `couple`
    #: action only means anything for a pair, and the policy masks that
    #: dimension's log-prob on everything else.
    head_is_pair: torch.Tensor
    #: (B, K, L) bool -- whether a via from here to that layer is legal.
    #: Applied as fixed logit suppression on the layer head, exactly like
    #: `safety` is on direction and step. Without it an untrained policy spends
    #: most of its actions attempting vias that cannot possibly fit -- measured
    #: at a 92.6% rejected-action rate against a 1.6% baseline.
    via_safe: torch.Tensor
    #: (B, K, L) float32 -- cost-to-go from the head's (y, x) on **every**
    #: layer, relative to its current layer. Negative means that layer is
    #: closer to the target. This is the layer head's whole decision basis and
    #: it is the reason the geodesic field is 3-D: a 2-D field cannot express
    #: "this layer is blocked here but the one below is open", which is
    #: precisely the question a via answers.
    geo_layer: torch.Tensor


def build_field(world: BatchedRouterWorld, demand: torch.Tensor | None = None) -> torch.Tensor:
    """(B, C, L, H, W) board tensor.

    Channel `CH_ACTIVE_NET` marks copper belonging to nets that are *currently
    being routed*, which is different information from "occupied": it is the
    only channel that changes meaning depending on which heads are live, and it
    is what lets the encoder distinguish a settled obstacle from one that is
    still growing and might yet move out of the way.
    """
    occ = world.occ
    B, L, H, W = occ.shape
    out = torch.zeros(B, FIELD_CHANNELS, L, H, W, dtype=torch.float32, device=occ.device)

    out[:, CH_OCCUPIED] = (occ != OCC_FREE).float()
    out[:, CH_KEEPOUT] = (occ < 0).float()
    out[:, CH_PAD] = (world.static > 0).float()
    out[:, CH_POUR] = world.pour.float()
    if demand is not None:
        out[:, CH_DEMAND] = demand

    active_ids = world.head_net.clamp_min(0) + 1                       # (B, K)
    live = (world.head_net >= 0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    match = (occ.unsqueeze(1) == active_ids.view(B, -1, 1, 1, 1)) & live
    out[:, CH_ACTIVE_NET] = match.any(dim=1).float()
    return out


def build_observation(
    world: BatchedRouterWorld,
    demand: torch.Tensor | None = None,
) -> Observation:
    B, K = world.head_net.shape
    L, H, W = world.shape
    dev = world.device
    M = B * K
    ds = world.cfg.geodesic_downsample

    field = build_field(world, demand)

    bb = torch.arange(B, device=dev).view(B, 1).expand(B, K)
    net = world.head_net.clamp_min(0)
    live = world.head_net >= 0

    pos = world.head_pos[:, :, 0].reshape(M, 3)
    b_i = bb.reshape(M)
    n_i = net.reshape(M)
    wc = world.net_width[bb, net].reshape(M)

    # --- exact local free space -------------------------------------------
    # `raycast` reports *absolute* directions; every consumer downstream --
    # the action heads, the suppression mask, the engine's move resolution --
    # works in the **egocentric** frame where index 0 is the reference bearing.
    # Rotating here, once, is what keeps the two frames from silently drifting
    # apart; they did, and the symptom was an 86% rejected-action rate with a
    # baseline that was supposedly only taking moves the raycast called safe.
    free_abs = geo.raycast(world.occ, b_i, pos[:, 0], pos[:, 1], pos[:, 2], n_i, world.tables, wc)

    fld = world.head_geo[:, :, 0].reshape(M, L, H // ds, W // ds)
    here = world._geo_at(fld, pos)
    bearing = geo.bearing_from_field(
        fld, pos[:, 0], pos[:, 1], pos[:, 2], world.tables, ds, free_units=free_abs
    )

    dir_idx = torch.arange(NUM_DIRECTIONS, device=dev).view(1, -1)
    abs_dir = (bearing.view(M, 1) + dir_idx) % NUM_DIRECTIONS          # (M, D)
    free = free_abs.gather(1, abs_dir)                                 # egocentric
    safety = geo.step_safety(free)                                     # (M, D, S)

    # --- geodesic lookahead per candidate move -----------------------------
    unit = world.tables.path[:, 0, 0]                                  # (D, 2)
    lens = torch.as_tensor(STEP_LENGTHS, device=dev).view(1, 1, NUM_STEPS, 1)
    off = unit[abs_dir]                                                # (M, D, 2)
    cand = pos[:, None, None, 1:] + off[:, :, None, :] * lens          # (M, D, S, 2)

    gl = pos[:, 0].view(M, 1, 1).expand(M, NUM_DIRECTIONS, NUM_STEPS)
    geo_ray = geo.sample_coarse(fld, gl, cand[..., 0], cand[..., 1], ds) * ds

    # --- can a via even be placed here, per target layer -------------------
    # A through via has to be free on EVERY layer at once, so on a populated
    # board most of them are illegal. Handing the policy that fact costs one
    # cheap geometric test per layer and saves it from having to discover it
    # through the reward -- the same argument as the raycast sensor, which is
    # the one mechanism in this repo with a confirmed measured win.
    layers_t = torch.arange(L, device=dev)
    via_ok = []
    for ell in range(L):
        tgt_l = torch.full_like(pos[:, 0], ell)
        if world.spec.layers.through_only:
            lo = torch.zeros_like(tgt_l)
            hi = torch.full_like(tgt_l, L - 1)
        else:
            lo = torch.minimum(pos[:, 0], tgt_l)
            hi = torch.maximum(pos[:, 0], tgt_l)
        legal = geo.check_via(
            world.occ, b_i, lo, hi, pos[:, 1], pos[:, 2],
            torch.zeros_like(tgt_l), n_i, world.tables,
        )
        # Staying put is always available; it is the `layer=0` action, not a
        # via, so the head's own layer is marked unavailable *as a via target*.
        via_ok.append(legal & (tgt_l != pos[:, 0]))
    via_safe = torch.stack(via_ok, dim=-1)                             # (M, L)

    # --- cost-to-go on every layer, at this (y, x) -------------------------
    all_layers = torch.arange(L, device=dev).view(1, L).expand(M, L)
    geo_layer = geo.sample_coarse(
        fld, all_layers, pos[:, 1].view(M, 1).expand(M, L), pos[:, 2].view(M, 1).expand(M, L), ds
    ) * ds
    geo_layer = torch.nan_to_num(geo_layer, posinf=1e6, nan=1e6)
    geo_layer = ((geo_layer - here.view(M, 1)) / LENGTH_SCALE).clamp(-8.0, 8.0)
    geo_ray = torch.nan_to_num(geo_ray, posinf=1e6, nan=1e6)
    # Relative to standing still, so the sign is directly "does this help".
    geo_ray = ((here.view(M, 1, 1) - geo_ray) / LENGTH_SCALE).clamp(-4.0, 4.0)

    # --- scalars ------------------------------------------------------------
    tgt = world.net_dst[b_i, n_i, 0]
    dy = (tgt[:, 1] - pos[:, 1]).float()
    dx = (tgt[:, 2] - pos[:, 2]).float()
    euclid = torch.sqrt(dy * dy + dx * dx)
    routed = world.net_len[b_i, n_i].sum(-1)
    kind = world.net_kind[b_i, n_i]
    steps = world.head_steps.reshape(M).float()
    budget = float(world.cfg.max_steps_per_net)

    onehot = lambda v, n: torch.nn.functional.one_hot(v.clamp(0, n - 1), n).float()  # noqa: E731

    scalars = torch.stack(
        [
            (here / LENGTH_SCALE).clamp(0, 32),
            torch.log1p((here / LENGTH_SCALE).clamp(0, 1e4)),
            euclid / LENGTH_SCALE,
            # Detour ratio: 1.0 is ideal, >1 means the route is wandering.
            (routed / euclid.clamp_min(1.0)).clamp(0, 8),
            steps / budget,
            1.0 - steps / budget,
            free.float().mean(-1) / max(STEP_LENGTHS),
            (free == 0).float().mean(-1),
            (kind == KIND_DIFF_PAIR).float(),
            wc.float() / max(1, world.spec.rules.num_width_classes - 1),
            world.net_vias[b_i, n_i].float() / 8.0,
            torch.isinf(here).float(),
        ],
        dim=-1,
    )
    scalars = torch.nan_to_num(scalars, posinf=32.0, neginf=-32.0)

    heads = torch.cat(
        [
            free.float() / max(STEP_LENGTHS),
            safety.float().reshape(M, -1),
            geo_ray.reshape(M, -1),
            onehot(pos[:, 0], L),
            onehot(tgt[:, 0], L),
            geo_layer,
            via_safe.float(),
            onehot(kind, NUM_KINDS),
            scalars,
        ],
        dim=-1,
    ).view(B, K, -1)
    heads = heads * live.unsqueeze(-1).float()

    # --- net tokens ---------------------------------------------------------
    src = world.net_src[:, :, 0].float()
    dstn = world.net_dst[:, :, 0].float()
    span = torch.linalg.vector_norm(dstn[..., 1:] - src[..., 1:], dim=-1)
    status = world.net_status
    nets = torch.cat(
        [
            src[..., 1:] / float(max(H, W)),
            dstn[..., 1:] / float(max(H, W)),
            torch.nn.functional.one_hot(world.net_kind.clamp(0, NUM_KINDS - 1), NUM_KINDS).float(),
            torch.stack(
                [
                    span / LENGTH_SCALE,
                    (status == STATUS_PENDING).float(),
                    (status == STATUS_ACTIVE).float(),
                    (status == STATUS_DONE).float(),
                    world.net_width.float() / max(1, world.spec.rules.num_width_classes - 1),
                    (world.net_group >= 0).float(),
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )
    nets = nets * world.net_valid.unsqueeze(-1).float()

    return Observation(
        field=field,
        heads=heads,
        # Cloned, not a view. `world.head_pos` is mutated in place by
        # `step()`, so an aliasing observation would silently describe the
        # world *after* the action it was used to choose -- which made
        # `policy.evaluate()` disagree with `policy.act()` on identical inputs
        # and would have shown up in training as a permanently broken PPO
        # importance ratio.
        head_pos=world.head_pos[:, :, 0].clone(),
        head_mask=live,
        nets=nets,
        net_mask=world.pending_mask(),
        safety=safety.view(B, K, NUM_DIRECTIONS, NUM_STEPS),
        bearing=bearing.view(B, K),
        head_is_pair=(world.net_kind[bb, net] == KIND_DIFF_PAIR) & live,
        via_safe=via_safe.view(B, K, L),
        geo_layer=geo_layer.view(B, K, L),
    )
