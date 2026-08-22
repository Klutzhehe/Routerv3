"""The line-geometry observation: the board as a set of segments, in a frame
that points at the target.

This is the tensor contract between the routing env and the policy, and the
single source of truth for feature order -- the same job pcbworld/agents/cfp/
spec.py did for the (now parked) raster design. Pure numpy: no pcbnew, no
pcbworld_pns_bridge, so it imports and runs anywhere and is fully unit
testable without Colab. That is deliberate -- it is the piece most likely to
carry subtle bugs, and the piece least able to afford a Colab round trip to
find them.

## Why segments rather than a raster

`get_board_geometry()` already returns segments with exact endpoints, so
there is no rasterizer to write (that was the parked CFP design's build-order
step 2, never started). Two measurements decided it:

  - The CFP canvas encoder was 71% of a forward pass -- dense convolutions
    over a mostly-empty binary image.
  - 256 px over a 50 mm board is 0.195 mm/px, against a design clearance of
    0.2 mm. One pixel is one clearance: the margin that decides whether a
    route is legal is SUB-PIXEL in that representation. Segment endpoints are
    exact.

Unrouted nets are lines too -- a pending net is a straight pad-to-pad
segment. Feeding routed copper as solid lines and pending nets as "ghost"
lines, distinguished by one flag, lets the policy see where future nets still
need to go. That is most of what CFP's separate "reserve plane" existed to
provide, for one one-hot bit.

## The frame

Origin at the routing head, +x pointing at the target pad, lengths divided by
`length_scale`. Rotating or translating the whole board leaves this
representation unchanged, so a policy trained on one board pose generalises
to every other pose without ever seeing one. With no training data and RL
only, sample efficiency is the whole game and this is the largest free win
available.

It also sets the action space's zero point: because +x IS the target
direction, a mean-zero policy walks straight at the pad. Training starts at
the greedy baseline rather than below it -- the property the CFP design
wanted from its zero-init flat field, obtained here by choosing coordinates
well instead of by architecture.

## geodesic_dist

`dist_to_target` is the straight line. The reward's potential is the shortest
obstacle-free path (see env/geodesic.py), and those differ by exactly the
amount an obstacle is in the way -- so a critic given only the straight line
is being asked to predict returns from a feature blind to what generates
them. This carries the quantity the potential is actually built from.

## base_heading_cos / base_heading_sin

The env does not always turn from the target bearing. While the head is
colliding it turns from its OWN previous heading instead, so the trace
contours along an obstacle rather than being yanked back into it. That makes
the base heading part of the environment state -- and it was state the policy
could not see, so on every board where a collision actually happened the
agent was steering in a frame it had no way to observe. Worse, the turn noise
stops being self-correcting there: off the target bearing it re-aims every
step, but off its own previous heading it random-walks, and at sigma = 0.30
(27 deg/step) the heading is uniform after ~44 of a 120-step budget.

These two features are the fix: the unit vector of the base heading the NEXT
action will turn from, expressed in this frame. Not colliding -> the base IS
the target bearing -> (1, 0). Colliding -> the accumulated heading, and the
policy can finally compute the turn that clears the obstacle.

Storing the effective base rather than the raw previous heading is deliberate:
the raw value plus head_collides is the same information, but it would make
the network learn a conditional to recover the quantity that actually decides
where the step lands. This is the minimal sufficient statistic for the
action's effect, which is what an observation should carry.

## Two details that are bugs if skipped

  - Endpoint order is canonicalised AFTER the transform, so a segment's
    feature vector never depends on which end the bridge happened to report
    first. Without this the policy sees two different vectors for identical
    geometry.
  - Nearest-K is by point-to-SEGMENT distance, not endpoint distance. A long
    track passing 0.3 mm from the head matters more than a distant track that
    happens to have a nearby endpoint, and endpoint distance ranks them
    backwards.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

MM = 1_000_000  # KiCad internal units are nm

# Segment kinds, one-hot in this order.
KIND_TRACK = 0    # committed copper
KIND_PAD = 1      # a pad, as a degenerate segment (see pad_to_segment)
KIND_EDGE = 2     # board outline
KIND_GHOST = 3    # an unrouted net's straight pad-to-pad line
NUM_KINDS = 4

GLOBAL_FEATURES: tuple[str, ...] = (
    "dist_to_target",     # / length_scale
    "log_dist",           # log1p of the above -- keeps far targets in range
    "steps_remaining",    # fraction of the step budget left
    "detour_ratio",       # routed length / straight-line distance, 1.0 = ideal
    "head_layer",
    "head_collides",
    "target_layer",
    "length_slack",       # / length_scale; 0 until the length-tuning stage
    "base_heading_cos",   # see below -- the frame the NEXT action turns from
    "base_heading_sin",
    "geodesic_dist",      # shortest OBSTACLE-FREE distance / length_scale
)
NUM_GLOBAL = len(GLOBAL_FEATURES)

# Index lookup so callers and tests never hard-code a position.
GLOBAL_INDEX: dict[str, int] = {name: i for i, name in enumerate(GLOBAL_FEATURES)}

SEGMENT_FEATURES: tuple[str, ...] = (
    "x1", "y1", "x2", "y2",   # local frame, / length_scale
    "width",                  # / length_scale
    "is_track", "is_pad", "is_edge", "is_ghost",
    "same_net",
    "same_layer",
    "valid",                  # 0 for padding rows -- the mask, carried inline
)
NUM_SEGMENT_FEATURES = len(SEGMENT_FEATURES)


@dataclasses.dataclass(frozen=True)
class LineObsConfig:
    """Sizing and normalisation. `length_scale` divides every distance, so
    features stay O(1): 10 mm is roughly a fifth of a 50 mm board, which puts
    a typical net's remaining distance in [0, 5] and a clearance-scale gap at
    0.02."""

    k_nearest: int = 32
    length_scale: float = 10.0 * MM
    max_steps: int = 80

    @property
    def flat_size(self) -> int:
        return NUM_GLOBAL + self.k_nearest * NUM_SEGMENT_FEATURES


@dataclasses.dataclass
class Segment:
    """One obstacle in board coordinates, before the local transform."""

    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    kind: int
    net: str
    layer: int


def pad_to_segment(pad, kind: int = KIND_PAD) -> Segment:
    """A pad as a degenerate segment: both endpoints at its centre, width set
    to its larger dimension.

    Keeping exactly one item type in the encoder is worth more than modelling
    a pad's rectangle exactly -- the policy sees "a round obstacle of this
    size here", which is what matters for avoidance, and the encoder needs no
    second code path."""
    return Segment(
        x1=float(pad.x),
        y1=float(pad.y),
        x2=float(pad.x),
        y2=float(pad.y),
        width=float(max(pad.size_x, pad.size_y)),
        kind=kind,
        net=pad.net,
        layer=getattr(pad, "layer_top", 0),
    )


def track_to_segment(track) -> Segment:
    return Segment(
        x1=float(track.x1),
        y1=float(track.y1),
        x2=float(track.x2),
        y2=float(track.y2),
        width=float(track.width),
        kind=KIND_TRACK,
        net=track.net,
        layer=track.layer,
    )


def ghost_segment(start, end, net: str, layer: int = 0, width: float = 0.0) -> Segment:
    """An unrouted net, as the straight line it will have to span."""
    return Segment(
        x1=float(start[0]), y1=float(start[1]),
        x2=float(end[0]), y2=float(end[1]),
        width=width, kind=KIND_GHOST, net=net, layer=layer,
    )


def board_segments(geometry, *, include_edges: bool = True) -> list[Segment]:
    """Committed copper and board outline from a get_board_geometry() result.

    Deliberately does NOT include pads -- the env adds those itself, because
    which pads count as obstacles depends on the net being routed (its own
    two pads are targets, not obstacles) and this function has no way to know
    that. Vias are skipped for the same reason a pad is a degenerate segment:
    a via at the head's own layer transition is not an obstacle to itself,
    and the env owns that distinction.
    """
    segments = [track_to_segment(t) for t in geometry.tracks]
    if include_edges:
        for edge in geometry.board_edge:
            segments.append(
                Segment(
                    x1=float(edge.x1), y1=float(edge.y1),
                    x2=float(edge.x2), y2=float(edge.y2),
                    width=float(edge.width), kind=KIND_EDGE, net="", layer=-1,
                )
            )
    return segments


def point_segment_distance(
    px: float, py: float, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray
) -> np.ndarray:
    """Distance from one point to each of N segments, vectorised.

    Degenerate (zero-length) segments -- every pad -- fall out correctly:
    the clamped projection parameter is 0 and this reduces to point-to-point
    distance, so pads need no special case."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    # np.divide with where= leaves t at 0 for zero-length segments instead of
    # emitting a divide-by-zero warning and a nan.
    t = np.zeros_like(len_sq, dtype=np.float64)
    np.divide(((px - x1) * dx + (py - y1) * dy), len_sq, out=t, where=len_sq > 0)
    np.clip(t, 0.0, 1.0, out=t)
    return np.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _local_frame(
    head: tuple[float, float], target: tuple[float, float]
) -> tuple[float, float, float, float]:
    """(hx, hy, cos, sin) for the transform into the head-at-origin,
    +x-toward-target frame.

    When the head is exactly on the target the direction is undefined; the
    identity rotation is used, which is harmless because the episode ends on
    that same step."""
    hx, hy = head
    dx, dy = target[0] - hx, target[1] - hy
    dist = float(np.hypot(dx, dy))
    if dist < 1e-9:
        return hx, hy, 1.0, 0.0
    return hx, hy, dx / dist, dy / dist


def to_local(
    xs: np.ndarray, ys: np.ndarray, hx: float, hy: float, cos: float, sin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate by -theta about the head, where theta is the target bearing."""
    tx = xs - hx
    ty = ys - hy
    return tx * cos + ty * sin, -tx * sin + ty * cos


def build_observation(
    segments: list[Segment],
    *,
    head: tuple[float, float],
    target: tuple[float, float],
    head_layer: int,
    target_layer: int,
    own_net: str,
    steps_taken: int,
    routed_length: float,
    straight_line_length: float,
    head_collides: bool,
    config: LineObsConfig,
    length_slack: float = 0.0,
    base_heading: float | None = None,
    geodesic_dist: float | None = None,
) -> np.ndarray:
    """The flat observation vector: globals, then k_nearest segment rows.

    Flat rather than a Dict space so the existing PPO baseline and any
    gymnasium vector wrapper take it unchanged; the policy reshapes the tail
    to (k_nearest, NUM_SEGMENT_FEATURES). The `valid` column carries the mask
    inline for the same reason -- one array, one Box, no second space to keep
    in sync.
    """
    scale = config.length_scale
    obs = np.zeros(config.flat_size, dtype=np.float32)

    dist = float(np.hypot(target[0] - head[0], target[1] - head[1]))
    obs[0] = dist / scale
    obs[1] = np.log1p(dist / scale)
    obs[2] = max(0.0, 1.0 - steps_taken / max(1, config.max_steps))
    obs[3] = routed_length / straight_line_length if straight_line_length > 0 else 1.0
    obs[4] = float(head_layer)
    obs[5] = float(bool(head_collides))
    obs[6] = float(target_layer)
    obs[7] = length_slack / scale

    # The frame the next turn is applied from, as a unit vector in THIS frame.
    # None means "the target bearing", which is +x here, i.e. (1, 0).
    if base_heading is None:
        obs[8], obs[9] = 1.0, 0.0
    else:
        bearing = math.atan2(target[1] - head[1], target[0] - head[0])
        delta = base_heading - bearing
        obs[8], obs[9] = math.cos(delta), math.sin(delta)

    # The distance the reward is actually shaped on. dist_to_target is the
    # straight line, which is what the potential USED to use; the critic has
    # to predict returns driven by the obstacle-free distance, and cannot do
    # that from a feature that ignores the obstacles. None means no field was
    # built, in which case the two coincide by definition.
    obs[10] = (dist if geodesic_dist is None else geodesic_dist) / scale

    if not segments:
        return obs

    x1 = np.fromiter((s.x1 for s in segments), dtype=np.float64, count=len(segments))
    y1 = np.fromiter((s.y1 for s in segments), dtype=np.float64, count=len(segments))
    x2 = np.fromiter((s.x2 for s in segments), dtype=np.float64, count=len(segments))
    y2 = np.fromiter((s.y2 for s in segments), dtype=np.float64, count=len(segments))

    distances = point_segment_distance(head[0], head[1], x1, y1, x2, y2)
    k = min(config.k_nearest, len(segments))
    # argpartition then sort just the K kept: O(n) rather than sorting every
    # segment on a dense board, and the policy still sees them nearest-first.
    nearest = np.argpartition(distances, k - 1)[:k]
    nearest = nearest[np.argsort(distances[nearest])]

    hx, hy, cos, sin = _local_frame(head, target)
    lx1, ly1 = to_local(x1[nearest], y1[nearest], hx, hy, cos, sin)
    lx2, ly2 = to_local(x2[nearest], y2[nearest], hx, hy, cos, sin)

    # Canonical endpoint order, AFTER the transform: identical geometry must
    # produce an identical row regardless of which end came first.
    #
    # The x-tie test needs a tolerance, not `==`. A segment perpendicular to
    # the head->target line has both endpoints at the same local x, so the
    # tie-break on y is what orders it -- but an exact comparison only holds
    # when no rotation was applied. Rotate the identical board and the two x
    # values differ in the last bit, `lx1 > lx2` decides instead of the y
    # tie-break, and the row comes out with its endpoints swapped. Caught by
    # the rigid-transform invariance test, which is exactly the property that
    # makes this representation worth using.
    #
    # Tolerance is relative to the coordinate magnitude, so this stays
    # correct whatever units the caller works in: at nm scale it lands around
    # a hundredth of a nanometre -- comfortably above float64 round-off
    # (~1e-8 nm here) and comfortably below KiCad's 1 nm resolution, so it can
    # never merge two genuinely distinct coordinates.
    eps = 1e-9 * np.maximum(np.abs(lx1), np.abs(lx2))
    swap = np.where(np.abs(lx1 - lx2) <= eps, ly1 > ly2, lx1 > lx2)
    lx1, lx2 = np.where(swap, lx2, lx1), np.where(swap, lx1, lx2)
    ly1, ly2 = np.where(swap, ly2, ly1), np.where(swap, ly1, ly2)

    rows = obs[NUM_GLOBAL:].reshape(config.k_nearest, NUM_SEGMENT_FEATURES)
    for row, idx, ax1, ay1, ax2, ay2 in zip(rows, nearest, lx1, ly1, lx2, ly2):
        seg = segments[idx]
        row[0] = ax1 / scale
        row[1] = ay1 / scale
        row[2] = ax2 / scale
        row[3] = ay2 / scale
        row[4] = seg.width / scale
        row[5 + seg.kind] = 1.0
        row[9] = float(seg.net == own_net and seg.net != "")
        row[10] = float(seg.layer == head_layer)
        row[11] = 1.0  # valid

    return obs
