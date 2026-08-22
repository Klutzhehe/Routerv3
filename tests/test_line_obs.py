"""Exercises the line-geometry observation.

This module is pure numpy with no bridge dependency, so unlike most of
pcbworld/env/* it can be tested properly rather than just smoke-tested --
and it should be, because its invariants (frame canonicalisation, endpoint
ordering, nearest-K by the right metric) are the kind that fail silently:
the policy still trains, just on a representation that quietly encodes the
same geometry two different ways.
"""

from __future__ import annotations

import math

import numpy as np

from pcbworld.env.line_obs import (
    KIND_GHOST,
    KIND_PAD,
    KIND_TRACK,
    MM,
    NUM_GLOBAL,
    NUM_SEGMENT_FEATURES,
    LineObsConfig,
    Segment,
    build_observation,
    point_segment_distance,
)

CFG = LineObsConfig(k_nearest=4, length_scale=10.0 * MM, max_steps=80)


def _seg(x1, y1, x2, y2, *, kind=KIND_TRACK, net="other", layer=0, width=0.25 * MM):
    return Segment(x1=x1, y1=y1, x2=x2, y2=y2, width=width, kind=kind, net=net, layer=layer)


def _obs(segments, head=(0.0, 0.0), target=(20.0 * MM, 0.0), **kwargs):
    defaults = dict(
        head_layer=0,
        target_layer=0,
        own_net="net_0",
        steps_taken=0,
        routed_length=0.0,
        straight_line_length=20.0 * MM,
        head_collides=False,
        config=CFG,
    )
    defaults.update(kwargs)
    return build_observation(segments, head=head, target=target, **defaults)


def _rows(obs):
    return obs[NUM_GLOBAL:].reshape(CFG.k_nearest, NUM_SEGMENT_FEATURES)


# -- the frame -----------------------------------------------------------


def test_target_always_lies_on_the_positive_x_axis():
    """The whole action space depends on this: +x IS the target direction, so
    a mean-zero policy walks straight at the pad."""
    for angle in (0.0, math.pi / 3, math.pi, -2.1):
        tx = 15.0 * MM * math.cos(angle)
        ty = 15.0 * MM * math.sin(angle)
        # A segment sitting exactly on the head->target line must land on the
        # local +x axis whatever the board-space bearing is.
        mid = (tx / 2, ty / 2)
        obs = _obs([_seg(mid[0], mid[1], mid[0], mid[1], kind=KIND_PAD)], target=(tx, ty))
        row = _rows(obs)[0]
        assert row[1] == 0.0 or abs(row[1]) < 1e-6, f"angle {angle}: y={row[1]}"
        assert row[0] > 0, f"angle {angle}: target-side segment landed at x={row[0]}"


def test_rigidly_transforming_the_whole_board_leaves_the_observation_unchanged():
    """Translation + rotation invariance is the sample-efficiency argument for
    this representation. If it does not hold, it is just a coordinate list."""
    segments = [
        _seg(5.0 * MM, 2.0 * MM, 9.0 * MM, 3.0 * MM),
        _seg(12.0 * MM, -4.0 * MM, 12.0 * MM, 4.0 * MM, net="net_9"),
        _seg(1.0 * MM, 1.0 * MM, 1.0 * MM, 1.0 * MM, kind=KIND_PAD),
    ]
    head, target = (0.0, 0.0), (20.0 * MM, 0.0)
    base = _obs(segments, head=head, target=target)

    theta, shift = 0.7, (123.0 * MM, -45.0 * MM)
    cos, sin = math.cos(theta), math.sin(theta)

    def move(x, y):
        return (x * cos - y * sin + shift[0], x * sin + y * cos + shift[1])

    moved = [
        Segment(*move(s.x1, s.y1), *move(s.x2, s.y2), s.width, s.kind, s.net, s.layer)
        for s in segments
    ]
    transformed = _obs(moved, head=move(*head), target=move(*target))

    np.testing.assert_allclose(base, transformed, atol=1e-4)


def test_endpoint_order_is_canonical_so_identical_geometry_gives_identical_rows():
    forward = _obs([_seg(5.0 * MM, 2.0 * MM, 9.0 * MM, 3.0 * MM)])
    reversed_ = _obs([_seg(9.0 * MM, 3.0 * MM, 5.0 * MM, 2.0 * MM)])
    np.testing.assert_allclose(forward, reversed_, atol=1e-6)


# -- nearest-K selection -------------------------------------------------


def test_point_segment_distance_measures_to_the_segment_not_its_endpoints():
    """A long track passing close by outranks a distant track with a nearby
    endpoint -- endpoint distance ranks those two backwards."""
    x1 = np.array([-100.0 * MM]); y1 = np.array([1.0 * MM])
    x2 = np.array([100.0 * MM]); y2 = np.array([1.0 * MM])
    d = point_segment_distance(0.0, 0.0, x1, y1, x2, y2)
    assert math.isclose(d[0], 1.0 * MM, rel_tol=1e-9)


def test_degenerate_segments_reduce_to_point_distance_without_nan():
    x = np.array([3.0 * MM]); y = np.array([4.0 * MM])
    d = point_segment_distance(0.0, 0.0, x, y, x.copy(), y.copy())
    assert math.isclose(d[0], 5.0 * MM, rel_tol=1e-9)
    assert not np.isnan(d).any()


def test_nearest_k_keeps_the_closest_and_orders_them_by_distance():
    far = _seg(40.0 * MM, 40.0 * MM, 41.0 * MM, 41.0 * MM, net="far")
    near = _seg(1.0 * MM, 0.0, 1.0 * MM, 0.0, kind=KIND_PAD, net="near")
    mid = _seg(5.0 * MM, 0.0, 5.0 * MM, 0.0, kind=KIND_PAD, net="mid")
    obs = _obs([far, mid, near])
    rows = _rows(obs)

    # Rows are nearest-first, in local coords where the head is the origin.
    assert rows[0][0] < rows[1][0] < rows[2][0]
    assert rows[0][11] == 1.0 and rows[2][11] == 1.0


def test_more_segments_than_k_drops_the_far_ones():
    segments = [
        _seg(d * MM, 0.0, d * MM, 0.0, kind=KIND_PAD, net=f"n{d}") for d in range(1, 12)
    ]
    rows = _rows(_obs(segments))
    assert (rows[:, 11] == 1.0).all(), "all K rows should be filled"
    # The kept set must be the four nearest, i.e. local x of 1..4 mm.
    np.testing.assert_allclose(sorted(rows[:, 0]), [0.1, 0.2, 0.3, 0.4], atol=1e-5)


def test_padding_rows_are_zero_and_marked_invalid():
    rows = _rows(_obs([_seg(2.0 * MM, 0.0, 3.0 * MM, 0.0)]))
    assert rows[0][11] == 1.0
    assert (rows[1:, 11] == 0.0).all()
    assert (rows[1:] == 0.0).all(), "padding must be all-zero, not just masked"


def test_empty_board_returns_globals_with_no_valid_rows():
    obs = _obs([])
    assert obs.shape == (CFG.flat_size,)
    assert (_rows(obs)[:, 11] == 0.0).all()


# -- feature encoding ----------------------------------------------------


def test_kind_one_hot_and_ownership_flags():
    own = _seg(2.0 * MM, 0.0, 3.0 * MM, 0.0, net="net_0", layer=0)
    ghost = _seg(2.5 * MM, 1.0 * MM, 4.0 * MM, 1.0 * MM, kind=KIND_GHOST, net="net_5", layer=2)
    rows = _rows(_obs([own, ghost], head_layer=0, own_net="net_0"))

    assert rows[0][5] == 1.0 and rows[0][9] == 1.0 and rows[0][10] == 1.0  # track, own net, same layer
    assert rows[1][8] == 1.0 and rows[1][9] == 0.0 and rows[1][10] == 0.0  # ghost, other net, other layer


def test_unnamed_segments_are_never_counted_as_the_agents_own_net():
    """Board edges carry net="" -- matching own_net="" would flag the whole
    outline as the agent's own copper."""
    edge = _seg(2.0 * MM, 0.0, 3.0 * MM, 0.0, net="")
    rows = _rows(_obs([edge], own_net=""))
    assert rows[0][9] == 0.0


def test_globals_carry_distance_progress_and_collision():
    obs = _obs(
        [],
        head=(0.0, 0.0),
        target=(20.0 * MM, 0.0),
        steps_taken=40,
        routed_length=30.0 * MM,
        straight_line_length=20.0 * MM,
        head_collides=True,
        head_layer=1,
        target_layer=0,
    )
    assert math.isclose(obs[0], 2.0, rel_tol=1e-6)          # 20mm / 10mm scale
    assert math.isclose(obs[1], math.log1p(2.0), rel_tol=1e-6)
    assert math.isclose(obs[2], 0.5, rel_tol=1e-6)          # 40 of 80 steps used
    assert math.isclose(obs[3], 1.5, rel_tol=1e-6)          # 30mm routed for a 20mm span
    assert obs[4] == 1.0 and obs[5] == 1.0 and obs[6] == 0.0


def test_head_on_target_does_not_produce_nan():
    """The bearing is undefined at zero distance; the episode ends on that
    same step, but the observation still has to be finite."""
    obs = _obs([_seg(1.0 * MM, 1.0 * MM, 2.0 * MM, 2.0 * MM)], head=(5.0 * MM, 5.0 * MM), target=(5.0 * MM, 5.0 * MM))
    assert np.isfinite(obs).all()


def test_observation_is_float32_and_the_advertised_size():
    obs = _obs([_seg(1.0 * MM, 0.0, 2.0 * MM, 0.0)])
    assert obs.dtype == np.float32
    assert obs.shape == (NUM_GLOBAL + CFG.k_nearest * NUM_SEGMENT_FEATURES,)
