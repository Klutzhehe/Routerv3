"""Batched lattice geometry: legality, stamping, raycasts, geodesic fields.

Every function here is pure tensor algebra over a batch of boards. There is no
Python loop over nets and no Python loop over cells -- that is the whole point.
`pcbworld/environment.py` rebuilds observations and checks collisions in Python
per step, which caps it at one board per process; the measured consequence is
in `docs/RL_PLAN.md` (0.745 ms/step before caching, 2 workers, `nproc`=2).
Here a step over `B` boards x `K` active heads is a fixed number of gathers.

The trick that makes it possible: the action space has only
``NUM_DIRECTIONS x max(STEP_LENGTHS)`` distinct move shapes, so every cell a
move could ever touch is a **precomputed constant offset** from the head. No
Bresenham at runtime -- just index arithmetic and a gather.

Coordinate convention throughout: ``(layer, y, x)``, matching the
``(B, L, H, W)`` tensor layout so the convolutional encoder needs no
transposes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from neuroroute.world.spec import (
    DIRECTION_VECTORS,
    NUM_DIRECTIONS,
    OCC_FREE,
    STEP_LENGTHS,
    DesignRules,
)

MAX_STEP = max(STEP_LENGTHS)
#: Cells touched per unit of travel: the cell itself plus, on a diagonal, the
#: two corner cells the 45-degree trace passes between. Blocking those is what
#: stops two diagonal traces X-crossing through the same lattice corner --
#: a real short that a naive centre-line-only model would emit as legal.
CELLS_PER_STEP = 3


@dataclass
class GeometryTables:
    """Precomputed constant offsets, built once per device.

    Attributes
    ----------
    path : (NUM_DIRECTIONS, MAX_STEP, CELLS_PER_STEP, 2) int64
        ``path[d, k]`` are the (dy, dx) offsets touched by the ``k+1``-th unit
        of travel in direction ``d``.
    path_mask : (NUM_DIRECTIONS, MAX_STEP, CELLS_PER_STEP) bool
        Which of those entries are real (orthogonal moves use 1 of 3).
    dil : (D, 2) int64
        Lateral dilation offsets, a full ``(2*R+1)^2`` square.
    dil_mask_width : (num_width_classes, D) bool
    dil_mask_via : (num_via_classes, D) bool
        Which dilation offsets apply to each width / via class.
    step_len : (NUM_STEPS,) int64
        STEP_LENGTHS as a tensor, for indexing by step class.
    """

    path: torch.Tensor
    path_mask: torch.Tensor
    dil: torch.Tensor
    dil_mask_width: torch.Tensor
    dil_mask_via: torch.Tensor
    step_len: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.path.device


def build_tables(rules: DesignRules, device: torch.device | str = "cpu") -> GeometryTables:
    """Build the constant offset tables for a rule set. Cheap; call once."""
    device = torch.device(device)

    dirs = torch.as_tensor(DIRECTION_VECTORS, dtype=torch.int64, device=device)  # (8, 2)
    path = torch.zeros(NUM_DIRECTIONS, MAX_STEP, CELLS_PER_STEP, 2, dtype=torch.int64, device=device)
    path_mask = torch.zeros(NUM_DIRECTIONS, MAX_STEP, CELLS_PER_STEP, dtype=torch.bool, device=device)

    for d in range(NUM_DIRECTIONS):
        dy, dx = int(dirs[d, 0]), int(dirs[d, 1])
        diagonal = dy != 0 and dx != 0
        for k in range(MAX_STEP):
            n = k + 1
            path[d, k, 0] = torch.tensor([n * dy, n * dx], device=device)
            path_mask[d, k, 0] = True
            if diagonal:
                # The two lattice cells the 45-degree trace passes between.
                path[d, k, 1] = torch.tensor([n * dy, (n - 1) * dx], device=device)
                path[d, k, 2] = torch.tensor([(n - 1) * dy, n * dx], device=device)
                path_mask[d, k, 1] = True
                path_mask[d, k, 2] = True

    width_radii = torch.as_tensor(rules.width_radii(), dtype=torch.int64, device=device)
    via_radii = torch.as_tensor(rules.via_radii(), dtype=torch.int64, device=device)
    r_max = int(torch.maximum(width_radii.max(), via_radii.max()).item())

    span = torch.arange(-r_max, r_max + 1, dtype=torch.int64, device=device)
    gy, gx = torch.meshgrid(span, span, indexing="ij")
    dil = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)  # (D, 2)
    cheb = dil.abs().amax(dim=-1)  # (D,) Chebyshev radius of each offset

    dil_mask_width = cheb[None, :] <= width_radii[:, None]
    dil_mask_via = cheb[None, :] <= via_radii[:, None]

    step_len = torch.as_tensor(STEP_LENGTHS, dtype=torch.int64, device=device)

    return GeometryTables(
        path=path,
        path_mask=path_mask,
        dil=dil,
        dil_mask_width=dil_mask_width,
        dil_mask_via=dil_mask_via,
        step_len=step_len,
    )


# ---------------------------------------------------------------------------
# Core gather
# ---------------------------------------------------------------------------


def gather_occ(
    occ: torch.Tensor,
    b: torch.Tensor,
    layer: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read `occ` at arbitrary broadcastable indices, treating out-of-bounds as
    keepout rather than wrapping or erroring.

    Returns
    -------
    values : same shape as the broadcast indices, dtype of `occ`
    inbounds : bool, same shape
    """
    _, L, H, W = occ.shape
    inb = (
        (layer >= 0) & (layer < L) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
    )
    ly = layer.clamp(0, L - 1)
    yy = y.clamp(0, H - 1)
    xx = x.clamp(0, W - 1)
    vals = occ[b, ly, yy, xx]
    return vals, inb


def _move_cells(
    tables: GeometryTables,
    y: torch.Tensor,
    x: torch.Tensor,
    direction: torch.Tensor,
    step_class: torch.Tensor,
    width_class: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Absolute cells touched by a batch of moves.

    Returns
    -------
    cy, cx : (M, MAX_STEP, CELLS_PER_STEP, D) int64
        Absolute cell coordinates.
    valid : (M, MAX_STEP, CELLS_PER_STEP, D) bool
        True where the entry is a real cell of a real travel unit for this
        move's step length and width class. Entries past the move's own step
        length are False, which is how one fixed-shape tensor serves all four
        step classes.
    """
    m = y.shape[0]
    off = tables.path[direction]                       # (M, MAX_STEP, C, 2)
    pmask = tables.path_mask[direction]                # (M, MAX_STEP, C)
    dil = tables.dil                                   # (D, 2)
    dmask = tables.dil_mask_width[width_class]         # (M, D)

    cy = y.view(m, 1, 1, 1) + off[..., 0:1] + dil[None, None, None, :, 0]
    cx = x.view(m, 1, 1, 1) + off[..., 1:2] + dil[None, None, None, :, 1]

    n_travel = tables.step_len[step_class]             # (M,)
    k_idx = torch.arange(MAX_STEP, device=y.device).view(1, MAX_STEP, 1, 1)
    within = k_idx < n_travel.view(m, 1, 1, 1)

    valid = within & pmask[..., None] & dmask[:, None, None, :]
    return cy, cx, valid


def check_moves(
    occ: torch.Tensor,
    b: torch.Tensor,
    layer: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    direction: torch.Tensor,
    step_class: torch.Tensor,
    width_class: torch.Tensor,
    net_id: torch.Tensor,
    tables: GeometryTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legality of a batch of `M` proposed moves.

    A cell is passable if it is free, or already owned by this same net (a net
    may cross its own copper -- electrically fine, and penalised as wirelength
    rather than forbidden). Out-of-bounds counts as blocked.

    Returns
    -------
    ok : (M,) bool -- the full move is legal
    free_units : (M,) int64 -- how many units of travel are clear before the
        first blocked one. Kept separate from `ok` so a partial-move variant
        can be tried later without touching this function.
    """
    cy, cx, valid = _move_cells(tables, y, x, direction, step_class, width_class)
    m, S, C, D = cy.shape

    bb = b.view(m, 1, 1, 1).expand(m, S, C, D)
    ll = layer.view(m, 1, 1, 1).expand(m, S, C, D)
    vals, inb = gather_occ(occ, bb, ll, cy, cx)

    own = net_id.view(m, 1, 1, 1) + 1
    passable = ((vals == OCC_FREE) | (vals == own)) & inb
    blocked = valid & ~passable                         # (M, S, C, D)

    blocked_unit = blocked.any(dim=-1).any(dim=-1)      # (M, S)
    ok = ~blocked_unit.any(dim=-1)

    # Index of the first blocked unit, or MAX_STEP if none -- i.e. how many
    # units of travel are clear.
    idx = torch.arange(S, device=occ.device).view(1, S).expand(m, S)
    free_units = torch.where(blocked_unit, idx, torch.full_like(idx, S)).amin(dim=-1)
    return ok, free_units


def stamp_moves(
    occ: torch.Tensor,
    b: torch.Tensor,
    layer: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    direction: torch.Tensor,
    step_class: torch.Tensor,
    width_class: torch.Tensor,
    net_id: torch.Tensor,
    tables: GeometryTables,
    active: torch.Tensor | None = None,
) -> None:
    """Write copper for a batch of moves into `occ`, in place.

    `active` masks which of the `M` moves actually happened. Inactive entries
    are redirected to a clamped no-op write of their own value, so the whole
    batch stays one scatter with no data-dependent shapes.
    """
    cy, cx, valid = _move_cells(tables, y, x, direction, step_class, width_class)
    m, S, C, D = cy.shape

    if active is not None:
        valid = valid & active.view(m, 1, 1, 1)

    _, L, H, W = occ.shape
    inb = (cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)
    write = valid & inb

    bb = b.view(m, 1, 1, 1).expand_as(cy)
    ll = layer.view(m, 1, 1, 1).expand_as(cy)
    vals = (net_id.view(m, 1, 1, 1) + 1).expand_as(cy).to(occ.dtype)

    flat = ((bb * L + ll.clamp(0, L - 1)) * H + cy.clamp(0, H - 1)) * W + cx.clamp(0, W - 1)
    # Only write where the cell is currently free; never clobber another net's
    # copper even if a caller passes an illegal move by mistake.
    cur = occ.view(-1)[flat]
    write = write & (cur == OCC_FREE)
    occ.view(-1)[flat[write]] = vals[write]


def stamp_via(
    occ: torch.Tensor,
    b: torch.Tensor,
    layer_lo: torch.Tensor,
    layer_hi: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    via_class: torch.Tensor,
    net_id: torch.Tensor,
    tables: GeometryTables,
    active: torch.Tensor | None = None,
) -> None:
    """Write a via disc across an inclusive layer span, in place.

    Blind/buried and through vias are the same operation with different spans
    -- see `LayerStack.via_span`.
    """
    m = y.shape[0]
    _, L, H, W = occ.shape
    dil = tables.dil                                   # (D, 2)
    dmask = tables.dil_mask_via[via_class]             # (M, D)

    lay = torch.arange(L, device=occ.device).view(1, L, 1)
    within = (lay >= layer_lo.view(m, 1, 1)) & (lay <= layer_hi.view(m, 1, 1))

    cy = y.view(m, 1, 1) + dil[None, None, :, 0]
    cx = x.view(m, 1, 1) + dil[None, None, :, 1]
    cy = cy.expand(m, L, dil.shape[0])
    cx = cx.expand(m, L, dil.shape[0])

    valid = within & dmask[:, None, :]
    if active is not None:
        valid = valid & active.view(m, 1, 1)
    inb = (cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)
    write = valid & inb

    bb = b.view(m, 1, 1).expand_as(cy)
    ll = lay.expand_as(cy)
    vals = (net_id.view(m, 1, 1) + 1).expand_as(cy).to(occ.dtype)

    flat = ((bb * L + ll) * H + cy.clamp(0, H - 1)) * W + cx.clamp(0, W - 1)
    cur = occ.view(-1)[flat]
    write = write & (cur == OCC_FREE)
    occ.view(-1)[flat[write]] = vals[write]


def check_via(
    occ: torch.Tensor,
    b: torch.Tensor,
    layer_lo: torch.Tensor,
    layer_hi: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    via_class: torch.Tensor,
    net_id: torch.Tensor,
    tables: GeometryTables,
) -> torch.Tensor:
    """Whether a via disc would be legal across the span. (M,) bool."""
    m = y.shape[0]
    _, L, H, W = occ.shape
    dil = tables.dil
    dmask = tables.dil_mask_via[via_class]

    lay = torch.arange(L, device=occ.device).view(1, L, 1)
    within = (lay >= layer_lo.view(m, 1, 1)) & (lay <= layer_hi.view(m, 1, 1))

    cy = (y.view(m, 1, 1) + dil[None, None, :, 0]).expand(m, L, dil.shape[0])
    cx = (x.view(m, 1, 1) + dil[None, None, :, 1]).expand(m, L, dil.shape[0])

    bb = b.view(m, 1, 1).expand_as(cy)
    ll = lay.expand_as(cy)
    vals, inb = gather_occ(occ, bb, ll, cy, cx)

    own = net_id.view(m, 1, 1) + 1
    passable = ((vals == OCC_FREE) | (vals == own)) & inb
    valid = within & dmask[:, None, :]
    return ~(valid & ~passable).any(dim=-1).any(dim=-1)


# ---------------------------------------------------------------------------
# Raycast sensor
# ---------------------------------------------------------------------------


def raycast(
    occ: torch.Tensor,
    b: torch.Tensor,
    layer: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    net_id: torch.Tensor,
    tables: GeometryTables,
    width_class: torch.Tensor | None = None,
) -> torch.Tensor:
    """Free travel distance per direction, in units, capped at MAX_STEP.

    This is the direct descendant of `models/pcb_encoder.py::_raycast_sensor`
    -- the one mechanism in this repo with a confirmed measured win
    (Rejected-Action Rate 1.51% -> 0.40% [LIVE], see
    docs/WORLD_MODEL_SPATIAL_DESIGN.md's addendum). It is recomputed from the
    live occupancy grid every forward pass, never decoded from a learned
    representation, so it cannot degrade the way a learned bias can.

    Two things are fixed here that the raster version could not fix:

    1. **Bearing exactness.** The raster sensor cast rays along the *raw*
       geodesic gradient while the env moved along an *EMA-smoothed* one, and
       the resulting half-bucket mismatch was diagnosed as the cause of the
       residual false negatives (`NEIGHBOR_SUPPRESSION_FRACTION` exists purely
       to paper over it). Here the ray and the move use the *same* constant
       offset table, so the reading is exact by construction, not approximately
       aligned.
    2. **Width awareness.** A direction clear for a narrow trace can be blocked
       for a wide one. Passing `width_class` casts the ray at the trace's real
       dilated footprint.

    Returns
    -------
    (M, NUM_DIRECTIONS) int64 in [0, MAX_STEP].
    """
    m = y.shape[0]
    if width_class is None:
        width_class = torch.zeros_like(y)

    dirs = torch.arange(NUM_DIRECTIONS, device=occ.device)
    dd = dirs.view(1, NUM_DIRECTIONS).expand(m, NUM_DIRECTIONS).reshape(-1)
    rep = lambda t: t.view(m, 1).expand(m, NUM_DIRECTIONS).reshape(-1)  # noqa: E731

    longest = torch.full_like(dd, len(STEP_LENGTHS) - 1)
    _, free_units = check_moves(
        occ,
        rep(b),
        rep(layer),
        rep(y),
        rep(x),
        dd,
        longest,
        rep(width_class),
        rep(net_id),
        tables,
    )
    return free_units.view(m, NUM_DIRECTIONS)


def step_safety(free_units: torch.Tensor) -> torch.Tensor:
    """Per-(direction, step-class) safety mask from a raycast reading.

    ``free_units`` is how far the ray travelled before blocking; a step class
    of length `n` is safe iff ``free_units >= n``. This is the
    ``dist_safe`` tensor from `models/pcb_encoder.py`, and its whole reason for
    existing is that per-direction granularity is too coarse: a direction clear
    for 2 cells but blocked at 8 reads "fine" at direction level while still
    colliding at the distance the action actually tries.

    Returns (M, NUM_DIRECTIONS, NUM_STEPS) bool.
    """
    lens = torch.as_tensor(STEP_LENGTHS, device=free_units.device).view(1, 1, -1)
    return free_units.unsqueeze(-1) >= lens


# ---------------------------------------------------------------------------
# Geodesic cost-to-go
# ---------------------------------------------------------------------------


def geodesic_field(
    blocked: torch.Tensor,
    target_layer: torch.Tensor,
    target_y: torch.Tensor,
    target_x: torch.Tensor,
    *,
    iterations: int = 96,
    via_cost: float = 4.0,
    downsample: int = 4,
    upsample: bool = True,
) -> torch.Tensor:
    """Obstacle-aware, **multi-layer** cost-to-go, batched.

    `pcbworld/congestion.py::compute_geodesic_distance_field` establishes both
    halves of this in the raster thread [LIVE]: an obstacle-aware field beats a
    Euclidean one (it routes *around* things, so potential-based shaping does
    not pull the head into a wall), and computing it on a downsampled grid then
    upsampling is accurate enough for shaping while being far cheaper. Both are
    kept. What is new is the **layer dimension**: relaxation propagates across
    layers at `via_cost`, so the field answers "is it cheaper to go around on
    this layer or drop through to the next" -- which is exactly the question a
    via action asks, and is unanswerable in a 2-D field.

    Parameters
    ----------
    blocked : (M, L, H, W) bool -- cells this net may not enter.
    target_* : (M,) int64 -- goal cell.

    upsample : bool
        When False, return the coarse ``(M, L, H//ds, W//ds)`` field instead of
        upsampling it. The engine stores the coarse form -- at ``ds=4`` that is
        16x less memory per active head, which is what makes caching a field
        per head affordable at ``B=64, K=8, L=8`` -- and upsamples only the
        planes an observation actually needs.

    Returns
    -------
    (M, L, H, W) float32 cost-to-go in cell units, `inf` where unreachable.
    Coarse ``(M, L, h, w)`` when ``upsample=False``; note the coarse field is
    in *coarse* cell units, so multiply by `downsample` to compare with the
    upsampled one.
    """
    m, L, H, W = blocked.shape
    ds = max(1, int(downsample))
    h, w = max(1, H // ds), max(1, W // ds)

    # A coarse cell is blocked only if *every* fine cell in it is blocked, so
    # the coarse field never claims a corridor is closed when a gap exists.
    if ds > 1:
        coarse = (
            F.max_pool2d(
                (~blocked).reshape(m * L, 1, H, W).float(), kernel_size=ds, stride=ds
            )
            .reshape(m, L, h, w)
            .le(0.5)
        )
    else:
        coarse = blocked

    dist = torch.full((m, L, h, w), float("inf"), device=blocked.device, dtype=torch.float32)

    ty = (target_y // ds).clamp(0, h - 1)
    tx = (target_x // ds).clamp(0, w - 1)
    tl = target_layer.clamp(0, L - 1)
    idx = torch.arange(m, device=blocked.device)
    dist[idx, tl, ty, tx] = 0.0

    neg_inf = -1e9
    for _ in range(iterations):
        # In-plane 8-connected relaxation via max-pool on the negated field.
        nd = torch.where(torch.isinf(dist), torch.full_like(dist, neg_inf), -dist)
        pooled = F.max_pool2d(nd.reshape(m * L, 1, h, w), 3, stride=1, padding=1)
        cand = -pooled.reshape(m, L, h, w) + 1.0

        # Cross-layer relaxation: a via costs `via_cost` cells of travel.
        if L > 1:
            up = torch.full_like(dist, float("inf"))
            dn = torch.full_like(dist, float("inf"))
            up[:, :-1] = dist[:, 1:] + via_cost
            dn[:, 1:] = dist[:, :-1] + via_cost
            cand = torch.minimum(cand, torch.minimum(up, dn))

        cand = torch.where(coarse, torch.full_like(cand, float("inf")), cand)
        new = torch.minimum(dist, cand)
        new[idx, tl, ty, tx] = 0.0
        if torch.equal(new, dist):
            dist = new
            break
        dist = new

    if not upsample:
        return dist

    if ds > 1:
        finite = torch.isfinite(dist)
        filled = torch.where(finite, dist, torch.zeros_like(dist))
        up = F.interpolate(filled.reshape(m * L, 1, h, w), size=(H, W), mode="bilinear", align_corners=False)
        reach = F.interpolate(finite.reshape(m * L, 1, h, w).float(), size=(H, W), mode="bilinear", align_corners=False)
        out = torch.where(reach > 0.25, up / reach.clamp_min(1e-6), torch.full_like(up, float("inf")))
        dist = out.reshape(m, L, H, W) * ds

    return dist


def descent_direction(
    field: torch.Tensor,
    layer: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    tables: GeometryTables,
) -> torch.Tensor:
    """Which of the `NUM_DIRECTIONS` most reduces the geodesic field.

    This defines the **egocentric action frame**: direction index 0 is
    remapped to this bearing, so "walk straight at the target, around
    obstacles" is action 0 on every board regardless of pose. That property is
    what lets a near-zero-init policy start *at* the greedy baseline
    (`docs/HANDOVER.md`) rather than below it, and it is why board-pose
    generalisation never has to be learned.

    Returns (M,) int64 in [0, NUM_DIRECTIONS).
    """
    m = y.shape[0]
    unit = tables.path[:, 0, 0, :]  # (NUM_DIRECTIONS, 2) single-unit offsets
    cy = y.view(m, 1) + unit[None, :, 0]
    cx = x.view(m, 1) + unit[None, :, 1]

    _, L, H, W = field.shape
    inb = (cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)
    bb = torch.arange(m, device=field.device).view(m, 1).expand(m, NUM_DIRECTIONS)
    ll = layer.view(m, 1).expand(m, NUM_DIRECTIONS)
    vals = field[bb, ll.clamp(0, L - 1), cy.clamp(0, H - 1), cx.clamp(0, W - 1)]
    vals = torch.where(inb, vals, torch.full_like(vals, float("inf")))
    vals = torch.nan_to_num(vals, nan=float("inf"), posinf=1e9)
    return vals.argmin(dim=-1)


# ---------------------------------------------------------------------------
# General segments (arbitrary endpoints)
# ---------------------------------------------------------------------------
#
# The offset tables above only cover the NUM_DIRECTIONS x STEP_LENGTHS moves
# the connect phase can make, which is what keeps `step()` a pure gather. Two
# things need arbitrary endpoints and cannot use them:
#
#   * the final snap onto a target pad, from wherever the head stopped;
#   * the refine phase, which drags polyline vertices to arbitrary cells.
#
# Both are far off the hot path (once per net, and once per refine action),
# so they get a general sampled-line implementation instead.

#: Samples per general segment. A segment is checked at this many points along
#: its length; `2 * MAX_STEP` oversamples a max-length move by 2x, so no cell
#: on a 45-degree line can be skipped.
SEGMENT_SAMPLES = 2 * MAX_STEP + 1


def _segment_cells(
    y0: torch.Tensor,
    x0: torch.Tensor,
    y1: torch.Tensor,
    x1: torch.Tensor,
    width_class: torch.Tensor,
    tables: GeometryTables,
    samples: int = SEGMENT_SAMPLES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cells touched by arbitrary segments, dilated by width class.

    **Includes the diagonal corner guards**, matching `_move_cells`. This is
    not cosmetic symmetry -- it is a real clearance rule, and omitting it here
    produced actual KiCad DRC failures.

    A 45-degree trace passes *between* two lattice cells. Any cell at
    perpendicular distance ``1/sqrt(2)`` from its centre line is only
    ``pitch/sqrt(2) - width = 0.283 - 0.2 = 0.083 mm`` from its copper edge,
    well inside a 0.2 mm clearance. `_move_cells` blocks those two cells at
    every step of a diagonal move, which is why lattice moves are legal.

    The snap-to-pad segment and the refine-phase vertex drag both go through
    *this* function instead, and without the guards they laid diagonal copper
    with no clearance reservation beside it. Another net then legally occupied
    a guard cell, and KiCad reported the pair at **0.0828 mm against a 0.2 mm
    rule** -- exactly the predicted number. Found by
    `neuroroute/scripts/validate_kicad.py`; it does not reproduce on every
    board, which is precisely why the check has to be run rather than reasoned
    about.

    Returns cy, cx, valid -- each (M, samples * 3, D).
    """
    m = y0.shape[0]
    t = torch.linspace(0.0, 1.0, samples, device=y0.device).view(1, samples)
    sy = torch.round(y0.view(m, 1) + (y1 - y0).view(m, 1) * t).long()
    sx = torch.round(x0.view(m, 1) + (x1 - x0).view(m, 1) * t).long()

    # Guard offsets, one axis each, pointing back along the direction of
    # travel -- identical to the (k, k-1) / (k-1, k) pair `_move_cells` uses.
    step_y = torch.sign((y1 - y0).view(m, 1))
    step_x = torch.sign((x1 - x0).view(m, 1))
    diagonal = (step_y != 0) & (step_x != 0)

    cy = torch.cat([sy, sy, sy - step_y], dim=1)
    cx = torch.cat([sx, sx - step_x, sx], dim=1)
    real = torch.cat(
        [
            torch.ones_like(sy, dtype=torch.bool),
            diagonal.expand_as(sy),
            diagonal.expand_as(sy),
        ],
        dim=1,
    )

    dil = tables.dil
    dmask = tables.dil_mask_width[width_class]  # (M, D)
    cy = cy[:, :, None] + dil[None, None, :, 0]
    cx = cx[:, :, None] + dil[None, None, :, 1]
    valid = real[:, :, None] & dmask[:, None, :]
    return cy, cx, valid


def check_segments(
    occ: torch.Tensor,
    b: torch.Tensor,
    layer: torch.Tensor,
    y0: torch.Tensor,
    x0: torch.Tensor,
    y1: torch.Tensor,
    x1: torch.Tensor,
    width_class: torch.Tensor,
    net_id: torch.Tensor,
    tables: GeometryTables,
) -> torch.Tensor:
    """(M,) bool -- whether each arbitrary segment is fully passable."""
    m = y0.shape[0]
    cy, cx, valid = _segment_cells(y0, x0, y1, x1, width_class, tables)
    bb = b.view(m, 1, 1).expand_as(cy)
    ll = layer.view(m, 1, 1).expand_as(cy)
    vals, inb = gather_occ(occ, bb, ll, cy, cx)
    own = net_id.view(m, 1, 1) + 1
    passable = ((vals == OCC_FREE) | (vals == own)) & inb
    return ~(valid & ~passable).any(dim=-1).any(dim=-1)


def stamp_segments(
    occ: torch.Tensor,
    b: torch.Tensor,
    layer: torch.Tensor,
    y0: torch.Tensor,
    x0: torch.Tensor,
    y1: torch.Tensor,
    x1: torch.Tensor,
    width_class: torch.Tensor,
    net_id: torch.Tensor,
    tables: GeometryTables,
    active: torch.Tensor | None = None,
    erase: bool = False,
) -> None:
    """Write (or, with `erase=True`, clear) arbitrary segments, in place.

    Erase only clears cells this net owns, so removing one net can never punch
    a hole in another's copper -- the invariant a rip-up action depends on.
    """
    m = y0.shape[0]
    _, L, H, W = occ.shape
    cy, cx, valid = _segment_cells(y0, x0, y1, x1, width_class, tables)
    if active is not None:
        valid = valid & active.view(m, 1, 1)
    inb = (cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)

    bb = b.view(m, 1, 1).expand_as(cy)
    ll = layer.view(m, 1, 1).expand_as(cy)
    flat = ((bb * L + ll.clamp(0, L - 1)) * H + cy.clamp(0, H - 1)) * W + cx.clamp(0, W - 1)

    own = (net_id.view(m, 1, 1) + 1).expand_as(cy).to(occ.dtype)
    cur = occ.view(-1)[flat]
    if erase:
        write = valid & inb & (cur == own)
        occ.view(-1)[flat[write]] = torch.zeros((), dtype=occ.dtype, device=occ.device)
    else:
        write = valid & inb & (cur == OCC_FREE)
        occ.view(-1)[flat[write]] = own[write]


def upsample_field(coarse: torch.Tensor, height: int, width: int, downsample: int) -> torch.Tensor:
    """Coarse geodesic field -> full lattice resolution, in fine-cell units.

    Unreachable (`inf`) cells are handled by upsampling a reachability mask
    alongside the values and re-masking, rather than letting `inf` contaminate
    the bilinear filter and blow out an entire neighbourhood.
    """
    m, L, h, w = coarse.shape
    finite = torch.isfinite(coarse)
    filled = torch.where(finite, coarse, torch.zeros_like(coarse))
    up = F.interpolate(
        filled.reshape(m * L, 1, h, w), size=(height, width), mode="bilinear", align_corners=False
    )
    reach = F.interpolate(
        finite.reshape(m * L, 1, h, w).float(), size=(height, width), mode="bilinear", align_corners=False
    )
    out = torch.where(reach > 0.25, up / reach.clamp_min(1e-6), torch.full_like(up, float("inf")))
    return out.reshape(m, L, height, width) * float(downsample)


def sample_coarse(
    coarse: torch.Tensor,
    layer: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    downsample: int,
) -> torch.Tensor:
    """Bilinearly sample a coarse field at **fine** lattice coordinates.

    Nearest-neighbour indexing into a 4x-downsampled field makes the geodesic
    piecewise-constant: every cell in a 4x4 block reads the same value, so the
    gradient between adjacent fine cells is zero and a descent direction
    computed from it is arbitrary. `pcbworld/environment.py` hit exactly this
    and solved it the same way (`_bilinear_sample` feeding `_geo_descent_dir`).

    Unreachable cells are `inf`; they are excluded from the interpolation by
    weight rather than allowed to poison the whole neighbourhood, and a sample
    with no reachable corner returns `inf`.

    Returns the sampled value in **coarse** units (multiply by `downsample` for
    fine-cell units), shape = broadcast of the index tensors.
    """
    m, L, h, w = coarse.shape
    ds = float(downsample)
    fy = (y.float() + 0.5) / ds - 0.5
    fx = (x.float() + 0.5) / ds - 0.5

    y0 = torch.floor(fy)
    x0 = torch.floor(fx)
    wy = (fy - y0).unsqueeze(-1)
    wx = (fx - x0).unsqueeze(-1)
    y0 = y0.long()
    x0 = x0.long()

    idx = torch.arange(m, device=coarse.device).view(*([m] + [1] * (y.dim() - 1)))
    idx = idx.expand_as(y)
    ll = layer.clamp(0, L - 1)

    vals = []
    wts = []
    for dy, dx, wgt in (
        (0, 0, (1 - wy) * (1 - wx)),
        (0, 1, (1 - wy) * wx),
        (1, 0, wy * (1 - wx)),
        (1, 1, wy * wx),
    ):
        v = coarse[idx, ll, (y0 + dy).clamp(0, h - 1), (x0 + dx).clamp(0, w - 1)]
        vals.append(v)
        wts.append(wgt.squeeze(-1))

    stack_v = torch.stack(vals, dim=-1)
    stack_w = torch.stack(wts, dim=-1)
    finite = torch.isfinite(stack_v)
    stack_w = stack_w * finite.float()
    denom = stack_w.sum(dim=-1)
    num = (torch.where(finite, stack_v, torch.zeros_like(stack_v)) * stack_w).sum(dim=-1)
    return torch.where(denom > 1e-6, num / denom.clamp_min(1e-6), torch.full_like(num, float("inf")))


def bearing_from_field(
    coarse: torch.Tensor,
    layer: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    tables: GeometryTables,
    downsample: int,
    free_units: torch.Tensor | None = None,
    lookahead: int | None = None,
) -> torch.Tensor:
    """The egocentric reference bearing: which direction to call "index 0".

    Two departures from a plain gradient descent, both deliberate:

    * **Sampled at a real lookahead distance**, not at the adjacent cell. On a
      field stored at `downsample=4`, adjacent fine cells are inside the same
      coarse cell and carry no gradient at all; sampling one coarse cell out
      recovers it.
    * **Block-aware.** A direction whose very first cell is occupied is
      demoted, so index 0 is "the best direction I can actually move in", not
      "the best direction if walls did not exist". This is what makes a
      near-zero-init policy start *at* a working greedy router rather than at
      one that walks into pads -- the whole justification for the egocentric
      frame in `docs/HANDOVER.md`. The demotion is additive and finite, so if
      every direction is blocked the ranking falls back to pure geodesic
      rather than becoming arbitrary.

    Returns (M,) int64 in [0, NUM_DIRECTIONS).
    """
    m = y.shape[0]
    la = lookahead if lookahead is not None else max(1, downsample)
    unit = tables.path[:, 0, 0]                       # (NUM_DIRECTIONS, 2)
    cy = y.view(m, 1) + unit[None, :, 0] * la
    cx = x.view(m, 1) + unit[None, :, 1] * la
    ll = layer.view(m, 1).expand(m, NUM_DIRECTIONS)

    vals = sample_coarse(coarse, ll, cy, cx, downsample)
    vals = torch.nan_to_num(vals, nan=1e9, posinf=1e9)

    if free_units is not None:
        # Finite penalty: enough to lose to any reachable open direction,
        # small enough that an all-blocked head still ranks by geodesic.
        vals = vals + (free_units <= 0).float() * 1e6
    return vals.argmin(dim=-1)


# ---------------------------------------------------------------------------
# Claim indices: cells a proposed action would write, as flat offsets
# ---------------------------------------------------------------------------
#
# The stamp_* functions above check-and-write in one call, which is correct for
# a single actor. It is NOT correct for `K` heads acting in the same batched
# step: each one's legality was decided against the occupancy as it was
# *before* the step, so two heads can both be told "clear" for the same cell
# and then both write it. The scatter has a single winner, the loser's head
# advances anyway, and its route ends up with a hole in it that nothing
# detects until a flood fill much later.
#
# So the engine needs the cells *before* deciding: compute claims, resolve who
# gets what, then write. These functions return flat indices into `occ` plus a
# validity mask; `resolve_claims` picks winners; `write_claims` commits.

def _flatten(shape, b, layer, y, x):
    _, L, H, W = shape
    return ((b * L + layer.clamp(0, L - 1)) * H + y.clamp(0, H - 1)) * W + x.clamp(0, W - 1)


def move_claims(
    occ: torch.Tensor,
    b: torch.Tensor,
    layer: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    direction: torch.Tensor,
    step_class: torch.Tensor,
    width_class: torch.Tensor,
    tables: GeometryTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat indices and validity of the cells a lateral move would occupy.
    Both (M, X)."""
    m = y.shape[0]
    _, L, H, W = occ.shape
    cy, cx, valid = _move_cells(tables, y, x, direction, step_class, width_class)
    inb = (cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)
    bb = b.view(m, 1, 1, 1).expand_as(cy)
    ll = layer.view(m, 1, 1, 1).expand_as(cy)
    flat = _flatten(occ.shape, bb, ll, cy, cx)
    return flat.reshape(m, -1), (valid & inb).reshape(m, -1)


def via_claims(
    occ: torch.Tensor,
    b: torch.Tensor,
    layer_lo: torch.Tensor,
    layer_hi: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    via_class: torch.Tensor,
    tables: GeometryTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat indices and validity of the cells a via disc would occupy."""
    m = y.shape[0]
    _, L, H, W = occ.shape
    dil = tables.dil
    dmask = tables.dil_mask_via[via_class]
    lay = torch.arange(L, device=occ.device).view(1, L, 1)
    within = (lay >= layer_lo.view(m, 1, 1)) & (lay <= layer_hi.view(m, 1, 1))

    cy = (y.view(m, 1, 1) + dil[None, None, :, 0]).expand(m, L, dil.shape[0])
    cx = (x.view(m, 1, 1) + dil[None, None, :, 1]).expand(m, L, dil.shape[0])
    inb = (cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)
    bb = b.view(m, 1, 1).expand_as(cy)
    ll = lay.expand_as(cy)
    flat = _flatten(occ.shape, bb, ll, cy, cx)
    valid = within & dmask[:, None, :] & inb
    return flat.reshape(m, -1), valid.reshape(m, -1)


def segment_claims(
    occ: torch.Tensor,
    b: torch.Tensor,
    layer: torch.Tensor,
    y0: torch.Tensor,
    x0: torch.Tensor,
    y1: torch.Tensor,
    x1: torch.Tensor,
    width_class: torch.Tensor,
    tables: GeometryTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat indices and validity of the cells an arbitrary segment covers."""
    m = y0.shape[0]
    _, L, H, W = occ.shape
    cy, cx, valid = _segment_cells(y0, x0, y1, x1, width_class, tables)
    inb = (cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)
    bb = b.view(m, 1, 1).expand_as(cy)
    ll = layer.view(m, 1, 1).expand_as(cy)
    flat = _flatten(occ.shape, bb, ll, cy, cx)
    return flat.reshape(m, -1), (valid & inb).reshape(m, -1)


def claims_passable(
    occ: torch.Tensor,
    flat: torch.Tensor,
    valid: torch.Tensor,
    net_id: torch.Tensor,
) -> torch.Tensor:
    """(M,) bool -- every claimed cell is free or already this net's."""
    cur = occ.view(-1)[flat]
    own = (net_id.view(-1, 1) + 1).to(occ.dtype)
    passable = (cur == OCC_FREE) | (cur == own)
    return ~(valid & ~passable).any(dim=-1)


def resolve_claims(
    occ: torch.Tensor,
    claim_buf: torch.Tensor,
    flats: list[torch.Tensor],
    valids: list[torch.Tensor],
    net_ids: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Decide which of several simultaneous claimants may write.

    Claims are arbitrated **by net**, not by claimant: the two legs of one
    differential pair travel a cell apart and routinely claim overlapping
    corner cells, but they are the same net, so that is not a conflict. Two
    different nets claiming one cell is.

    The winner of a contested cell is the lowest net id, which makes the
    outcome deterministic -- scatter order is not.

    Conservative by one degree: a claimant that loses any single cell is
    rejected entirely, and a claimant rejected that way still counts as having
    won the cells it took. That can reject a third claimant unnecessarily. It
    is rare (it needs three heads adjacent in one step), and the failure mode
    is a wasted step rather than broken copper -- the right side to err on.

    Returns one (M,) bool per claim group.
    """
    if not flats:
        return []
    BIG = torch.iinfo(torch.int32).max
    claim_buf.fill_(BIG)

    only_free = []
    for flat, valid in zip(flats, valids):
        only_free.append(valid & (occ.view(-1)[flat] == OCC_FREE))

    for flat, free, nid in zip(flats, only_free, net_ids):
        ids = nid.view(-1, 1).expand_as(flat).to(torch.int32)
        claim_buf.scatter_reduce_(0, flat[free], ids[free], reduce="amin", include_self=True)

    out = []
    for flat, free, nid in zip(flats, only_free, net_ids):
        winner = claim_buf[flat]
        lost = free & (winner != nid.view(-1, 1).to(torch.int32))
        out.append(~lost.any(dim=-1))
    return out


def write_claims(
    occ: torch.Tensor,
    flat: torch.Tensor,
    valid: torch.Tensor,
    net_id: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Commit claimed cells to `occ`, never overwriting existing copper."""
    write = valid & active.view(-1, 1)
    write = write & (occ.view(-1)[flat] == OCC_FREE)
    vals = (net_id.view(-1, 1) + 1).expand_as(flat).to(occ.dtype)
    occ.view(-1)[flat[write]] = vals[write]
