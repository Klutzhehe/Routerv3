"""Exercises the obstacle-free cost-to-go field.

Pure numpy like the module it tests, so this runs anywhere. The properties
worth pinning are the ones that fail silently: a field that is subtly wrong
still trains, just toward the wrong thing, and the symptom shows up 300k
steps later as a completion rate that will not move.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pcbworld.env.geodesic import GeodesicConfig, GeodesicField
from pcbworld.env.line_obs import (
    KIND_EDGE,
    KIND_GHOST,
    KIND_PAD,
    KIND_TRACK,
    MM,
    Segment,
)

CFG = GeodesicConfig()


def _seg(x1, y1, x2, y2, *, kind=KIND_TRACK, width=0.25 * MM, net="other"):
    return Segment(x1=x1, y1=y1, x2=x2, y2=y2, width=width, kind=kind, net=net, layer=0)


def _pad(x, y, size=0.5 * MM):
    return _seg(x, y, x, y, kind=KIND_PAD, width=size)


def _field(segments, head=(0.0, 0.0), target=(20.0 * MM, 0.0), config=CFG):
    return GeodesicField.build(segments, head=head, target=target, config=config)


# -- the basic contract --------------------------------------------------


def test_open_space_costs_the_straight_line():
    f = _field([])
    # 8-connectivity on a grid overestimates a Euclidean path slightly, and
    # the cell centre never lands exactly on the query point; a couple of
    # cells of slack is the discretisation, not an error.
    assert f.cost_to_go(0.0, 0.0) == pytest.approx(20.0 * MM, rel=0.05)
    assert f.reachable


def test_cost_falls_as_the_target_is_approached():
    f = _field([])
    costs = [f.cost_to_go(x * MM, 0.0) for x in range(0, 20, 2)]
    assert all(b < a for a, b in zip(costs, costs[1:])), costs


def test_an_obstacle_on_the_line_makes_the_route_longer():
    blocked = _field([_pad(10.0 * MM, 0.0, size=2.0 * MM)])
    clear = _field([])
    assert blocked.cost_to_go(0.0, 0.0) > clear.cost_to_go(0.0, 0.0)
    assert blocked.reachable


def test_a_sealed_target_is_reported_unreachable():
    """The caller falls back to straight-line on this rather than treating it
    as a penalty -- an unreachable pad is a fact about the board."""
    box = [
        _seg(18 * MM, -2 * MM, 22 * MM, -2 * MM),
        _seg(18 * MM, 2 * MM, 22 * MM, 2 * MM),
        _seg(18 * MM, -2 * MM, 18 * MM, 2 * MM),
        _seg(22 * MM, -2 * MM, 22 * MM, 2 * MM),
    ]
    f = _field(box)
    assert not f.reachable
    assert not math.isfinite(f.cost_to_go(0.0, 0.0))


# -- the property the whole change exists for ----------------------------


def test_going_around_an_obstacle_beats_going_into_it():
    """THE load-bearing test.

    Under the old straight-line potential the per-step reward was +0.0545 to
    drive at the obstacle against +0.0050 to contour, so every step of a
    correct detour scored worse than the wrong move and 600k steps of PPO
    never found one. The field has to invert that ordering.
    """
    wall = _seg(3 * MM, -3 * MM, 3 * MM, 3 * MM, width=0.5 * MM)
    f = _field([wall])

    here = f.cost_to_go(0.0, 0.0)
    toward = here - f.cost_to_go(0.5 * MM, 0.0)
    sideways = here - f.cost_to_go(0.0, 0.5 * MM)
    away = here - f.cost_to_go(-0.5 * MM, 0.0)

    assert sideways > toward, (
        f"contouring gains {sideways:.0f} nm against {toward:.0f} nm for driving "
        "into the wall; the potential still rewards the wrong move"
    )
    assert away < 0.0, "retreating from the target must still cost"


def test_the_field_routes_around_rather_than_through():
    """A point just past the obstacle's edge must be cheaper than one squarely
    behind it, or the gradient points into copper."""
    wall = _seg(5 * MM, -4 * MM, 5 * MM, 4 * MM, width=0.5 * MM)
    f = _field([wall])
    behind_wall = f.cost_to_go(4.0 * MM, 0.0)
    beside_wall = f.cost_to_go(4.0 * MM, 5.0 * MM)
    assert beside_wall < behind_wall


# -- what counts as an obstacle ------------------------------------------


def test_the_board_outline_is_a_boundary_not_a_wall_across_the_middle():
    """Regression for a diagonal wall through every board.

    generate_board.py emits the outline as a SHAPE_T_RECT and the bridge
    reports a rect by two opposite corners, so rasterising that "segment" as
    a line lays a wall from corner to corner -- which on a 35mm board crosses
    most nets and made a simple 18mm route measure 37mm. Edges have to be
    taken as a bounding box.
    """
    diagonal_rect = _seg(0.0, 0.0, 35 * MM, 35 * MM, kind=KIND_EDGE, width=0.1 * MM)
    # A route along y=10mm crosses the corner-to-corner diagonal at x=10mm.
    with_outline = _field(
        [diagonal_rect], head=(2 * MM, 10 * MM), target=(20 * MM, 10 * MM)
    )
    without = _field([], head=(2 * MM, 10 * MM), target=(20 * MM, 10 * MM))

    assert with_outline.reachable
    assert with_outline.cost_to_go(2 * MM, 10 * MM) == pytest.approx(
        without.cost_to_go(2 * MM, 10 * MM), rel=0.05
    )


def test_outside_the_outline_is_blocked():
    outline = _seg(0.0, 0.0, 30 * MM, 30 * MM, kind=KIND_EDGE, width=0.1 * MM)
    f = _field([outline], head=(2 * MM, 2 * MM), target=(25 * MM, 25 * MM))
    assert math.isfinite(f.cost_to_go(15 * MM, 15 * MM))
    assert not math.isfinite(f.cost_to_go(-5 * MM, 15 * MM))


def test_unrouted_nets_do_not_block_the_plan():
    """A ghost is a plan, not copper. Planning around one would make a target
    unreachable whenever two nets want the same corridor, which is most of
    what the later curriculum stages are about."""
    ghost = _seg(10 * MM, -30 * MM, 10 * MM, 30 * MM, kind=KIND_GHOST, width=0.25 * MM)
    assert _field([ghost]).cost_to_go(0.0, 0.0) == pytest.approx(
        _field([]).cost_to_go(0.0, 0.0), rel=0.01
    )


def test_a_head_inside_an_obstacle_still_gets_a_finite_cost():
    """The head spends 10-45% of its steps inside copper -- PNS marks the
    collision and advances anyway -- so this is the common case. Left at inf
    the potential would switch to the straight-line fallback and back, putting
    a reward spike on each switch."""
    f = _field([_pad(10 * MM, 0.0, size=2.0 * MM)])
    assert math.isfinite(f.cost_to_go(10 * MM, 0.0))
    assert f.cost_to_go(10 * MM, 0.0) > 0.0


# -- mechanics -----------------------------------------------------------


def test_the_relaxation_converges_to_the_same_answer_as_a_slow_reference():
    """Guards the directional sweep order against the one-cell-per-pass
    formulation it replaced for speed. Bellman-Ford converges under any edge
    order, so these must agree exactly."""
    segs = [_pad(6 * MM, 1 * MM, size=2 * MM), _seg(12 * MM, -4 * MM, 12 * MM, 2 * MM)]
    cfg = GeodesicConfig(cell_nm=1.0 * MM)
    f = GeodesicField.build(segs, head=(0.0, 0.0), target=(18 * MM, 0.0), config=cfg)

    cx = f.origin_x + (np.arange(f.nx) + 0.5) * f.cell
    cy = f.origin_y + (np.arange(f.ny) + 0.5) * f.cell
    blocked = GeodesicField._blocked_mask(segs, *np.meshgrid(cx, cy), cfg)
    ti, tj = f._cell_of(18 * MM, 0.0)
    hi, hj = f._cell_of(0.0, 0.0)
    blocked[ti, tj] = False
    blocked[hi, hj] = False

    ref = np.full(blocked.shape, np.inf, dtype=np.float32)
    ref[ti, tj] = 0.0
    ortho, diag = np.float32(f.cell), np.float32(math.sqrt(2.0) * f.cell)
    ny, nx = blocked.shape
    for _ in range(4 * (nx + ny)):
        padded = np.full((ny + 2, nx + 2), np.inf, dtype=np.float32)
        padded[1:-1, 1:-1] = ref
        best = np.minimum.reduce([
            padded[0:-2, 1:-1] + ortho, padded[2:, 1:-1] + ortho,
            padded[1:-1, 0:-2] + ortho, padded[1:-1, 2:] + ortho,
            padded[0:-2, 0:-2] + diag, padded[0:-2, 2:] + diag,
            padded[2:, 0:-2] + diag, padded[2:, 2:] + diag,
        ])
        np.minimum(best, ref, out=best)
        best[blocked] = np.inf
        best[ti, tj] = 0.0
        if np.array_equal(best, ref):
            break
        ref = best

    swept = GeodesicField._relax(blocked, ti, tj, f.cell)
    free = np.isfinite(ref)
    assert np.allclose(swept[free], ref[free]), "sweep order changed the answer"


def test_interpolation_makes_the_potential_smooth_within_a_cell():
    """Nearest-cell sampling would make the potential a staircase, and every
    step landing inside the current cell would earn exactly zero shaping."""
    f = _field([])
    xs = np.arange(0.0, 5.0 * MM, 0.1 * MM)
    costs = [f.cost_to_go(x, 0.0) for x in xs]
    steps = np.diff(costs)
    assert np.all(steps < 0), "cost must fall monotonically toward the target"
    assert len(set(np.round(costs, 3))) > len(xs) // 2, "field is quantised to cells"


def test_a_coarse_board_falls_back_to_fewer_cells():
    cfg = GeodesicConfig(cell_nm=0.5 * MM, max_cells_per_axis=32)
    f = _field([], head=(0.0, 0.0), target=(200 * MM, 0.0), config=cfg)
    assert f.nx <= 40 and f.ny <= 40
    assert f.cell > 0.5 * MM, "should have coarsened rather than built a huge grid"


def test_building_the_field_is_deterministic():
    segs = [_pad(8 * MM, 1 * MM, size=2 * MM)]
    a = _field(segs).cost
    b = _field(segs).cost
    assert np.array_equal(np.nan_to_num(a, posinf=-1), np.nan_to_num(b, posinf=-1))
