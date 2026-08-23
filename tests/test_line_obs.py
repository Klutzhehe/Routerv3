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
import pytest

from pcbworld.env.line_obs import (
    GLOBAL_INDEX,
    KIND_EDGE,
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


# -- base heading --------------------------------------------------------


def test_base_heading_defaults_to_the_target_bearing():
    """None means "the next turn is measured from the target bearing", and the
    target bearing is +x in this frame -- so the unit vector is (1, 0)."""
    obs = _obs([])
    ci, si = GLOBAL_INDEX["base_heading_cos"], GLOBAL_INDEX["base_heading_sin"]
    assert obs[ci] == pytest.approx(1.0)
    assert obs[si] == pytest.approx(0.0)


def test_base_heading_is_relative_to_the_target_bearing():
    ci, si = GLOBAL_INDEX["base_heading_cos"], GLOBAL_INDEX["base_heading_sin"]

    # Head at origin, target on +x, so the bearing is 0 and the offset is the
    # absolute heading.
    obs = _obs([], base_heading=math.radians(30.0))
    assert math.degrees(math.atan2(obs[si], obs[ci])) == pytest.approx(30.0, abs=1e-4)

    # Same geometry rotated 90 degrees: the target bearing is now +y, and a
    # heading 30 degrees off it must produce the identical feature pair.
    obs_rot = _obs(
        [], target=(0.0, 20.0 * MM), base_heading=math.radians(120.0)
    )
    assert math.degrees(math.atan2(obs_rot[si], obs_rot[ci])) == pytest.approx(30.0, abs=1e-4)


def test_base_heading_is_a_unit_vector_for_any_angle():
    ci, si = GLOBAL_INDEX["base_heading_cos"], GLOBAL_INDEX["base_heading_sin"]
    for deg in (-179.0, -90.0, -1.0, 0.0, 1.0, 90.0, 179.0, 359.0):
        obs = _obs([], base_heading=math.radians(deg))
        assert math.hypot(obs[ci], obs[si]) == pytest.approx(1.0, abs=1e-5)


def test_base_heading_survives_a_rigid_transform_of_the_whole_board():
    """The frame's whole point: rotate board, head, target and heading together
    and every feature must be unchanged."""
    ci, si = GLOBAL_INDEX["base_heading_cos"], GLOBAL_INDEX["base_heading_sin"]
    seg = _seg(5.0 * MM, 1.0 * MM, 15.0 * MM, 1.0 * MM)
    base = _obs([seg], base_heading=math.radians(40.0))

    theta = math.radians(37.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def rot(x, y):
        return x * cos_t - y * sin_t, x * sin_t + y * cos_t

    rx1, ry1 = rot(seg.x1, seg.y1)
    rx2, ry2 = rot(seg.x2, seg.y2)
    rt = rot(20.0 * MM, 0.0)
    rotated = _obs(
        [_seg(rx1, ry1, rx2, ry2)],
        head=(0.0, 0.0),
        target=rt,
        base_heading=math.radians(40.0) + theta,
    )

    assert rotated[ci] == pytest.approx(base[ci], abs=1e-5)
    assert rotated[si] == pytest.approx(base[si], abs=1e-5)
    assert np.allclose(rotated, base, atol=1e-4)


# -- clearance -----------------------------------------------------------


def test_clearance_measures_room_to_the_obstacle_edge_not_its_centre():
    """Each obstacle's half-width is subtracted, so this is the room a trace
    centre actually has."""
    from pcbworld.env.line_obs import nearest_obstacle_gap

    pad = _seg(10 * MM, 0.0, 10 * MM, 0.0, kind=KIND_PAD, width=2.0 * MM)
    # 5mm from the centre of a 2mm-wide pad leaves 4mm to its edge
    assert nearest_obstacle_gap(5 * MM, 0.0, [pad]) == pytest.approx(4.0 * MM, rel=1e-6)
    # inside the footprint reads negative
    assert nearest_obstacle_gap(10.5 * MM, 0.0, [pad]) < 0.0


def test_clearance_ignores_ghosts_and_edges():
    """A ghost is a plan, not copper -- reporting contact with one would warn
    about something that does not exist."""
    from pcbworld.env.line_obs import nearest_obstacle_gap

    ghost = _seg(1 * MM, 0.0, 1 * MM, 5 * MM, kind=KIND_GHOST, width=0.25 * MM)
    edge = _seg(0.0, 0.0, 0.0, 30 * MM, kind=KIND_EDGE, width=0.1 * MM)
    assert not np.isfinite(nearest_obstacle_gap(5 * MM, 0.0, [ghost, edge]))

    real = _seg(6 * MM, -5 * MM, 6 * MM, 5 * MM, kind=KIND_TRACK, width=0.25 * MM)
    assert np.isfinite(nearest_obstacle_gap(5 * MM, 0.0, [ghost, edge, real]))


def test_clearance_features_land_in_the_observation():
    ci = GLOBAL_INDEX["clearance_now"]
    ca = GLOBAL_INDEX["clearance_ahead"]

    obs = _obs([], clearance_now=2.0 * MM, clearance_ahead=0.1 * MM)
    assert obs[ci] == pytest.approx(0.2)     # 2mm / 10mm length_scale
    assert obs[ca] == pytest.approx(0.01)

    # an empty board reports "plenty of room" rather than a huge number
    wide = _obs([], clearance_now=float("inf"), clearance_ahead=None)
    assert wide[ci] == 3.0 and wide[ca] == 3.0

    # inside copper reads negative, and stays bounded
    tight = _obs([], clearance_now=-50.0 * MM, clearance_ahead=-0.02 * MM)
    assert tight[ci] == pytest.approx(-1.0)
    assert tight[ca] == pytest.approx(-0.002)


def test_geodesic_direction_is_relative_to_the_target_bearing():
    """Like base_heading, it is the TURN to make, not a compass heading, so it
    survives rotating the whole board."""
    import math as _m

    ci, si = GLOBAL_INDEX["geo_dir_cos"], GLOBAL_INDEX["geo_dir_sin"]

    obs = _obs([], geodesic_direction=_m.radians(35.0))
    assert _m.degrees(_m.atan2(obs[si], obs[ci])) == pytest.approx(35.0, abs=1e-4)

    rotated = _obs([], target=(0.0, 20.0 * MM), geodesic_direction=_m.radians(125.0))
    assert _m.degrees(_m.atan2(rotated[si], rotated[ci])) == pytest.approx(35.0, abs=1e-4)

    # no field -> the best guess is the target bearing, which is +x here
    assert _obs([])[ci] == pytest.approx(1.0)
    assert _obs([])[si] == pytest.approx(0.0)
