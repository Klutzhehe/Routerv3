"""Unit tests for pcbworld.viz.render_board.

Pure Python + matplotlib, no pcbnew/bridge dependency -- these run anywhere,
same as pcbworld/agents/cfp's tests. Fixtures use the exact field names
bindings.cpp exposes (verified by reading bindings.cpp directly, not
assumed) so a passing test here means the renderer works against the real
bridge's actual output shape, not just against itself.
"""

from __future__ import annotations

from collections import namedtuple

import matplotlib

matplotlib.use("Agg")  # headless -- no display in a test run
import matplotlib.pyplot as plt
import pytest

from pcbworld.viz.render_board import _net_color, render_board

MM = 1_000_000

TrackSegment = namedtuple(
    "TrackSegment", ["x1", "y1", "x2", "y2", "width", "layer", "net", "is_arc"]
)
ViaGeom = namedtuple(
    "ViaGeom", ["x", "y", "diameter", "drill", "layer_top", "layer_bottom", "net"]
)
PadGeom = namedtuple(
    "PadGeom", ["x", "y", "size_x", "size_y", "layer_top", "layer_bottom", "net", "pad_name"]
)
EdgeShape = namedtuple("EdgeShape", ["shape_type", "x1", "y1", "x2", "y2", "width"])
BoardGeometry = namedtuple("BoardGeometry", ["tracks", "vias", "pads", "zones", "courtyards", "board_edge"])
NetPad = namedtuple("NetPad", ["net", "pad_name", "x", "y", "layer"])


def _empty_geometry() -> BoardGeometry:
    return BoardGeometry([], [], [], [], [], [])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")  # matplotlib accumulates open figures across tests otherwise


def test_empty_geometry_does_not_crash():
    ax = render_board(_empty_geometry())
    assert ax is not None


def test_board_edge_rect_is_drawn():
    geometry = BoardGeometry([], [], [], [], [], [EdgeShape("rect", 0, 0, 50 * MM, 50 * MM, 0)])
    ax = render_board(geometry)
    assert len(ax.patches) == 1


def test_board_edge_circle_uses_start_end_as_center_radius():
    # x1,y1 = center; x2,y2 = a point on the circumference (see
    # pns_bridge.cpp's comment: EdgeShape has no explicit radius field).
    geometry = BoardGeometry([], [], [], [], [], [EdgeShape("circle", 10 * MM, 10 * MM, 15 * MM, 10 * MM, 0)])
    ax = render_board(geometry)
    circle = ax.patches[0]
    assert circle.center == (10.0, 10.0)
    assert circle.radius == pytest.approx(5.0)


def test_tracks_are_plotted_between_their_endpoints():
    track = TrackSegment(0, 0, 20 * MM, 0, 250_000, 0, "net_0", False)
    geometry = BoardGeometry([track], [], [], [], [], [])
    ax = render_board(geometry)
    assert len(ax.lines) == 1
    xdata, ydata = ax.lines[0].get_data()
    assert list(xdata) == [0.0, 20.0]
    assert list(ydata) == [0.0, 0.0]


def test_via_draws_outer_circle_and_drill_hole():
    via = ViaGeom(10 * MM, 10 * MM, 600_000, 300_000, 0, 2, "net_0")
    geometry = BoardGeometry([], [via], [], [], [], [])
    ax = render_board(geometry)
    assert len(ax.patches) == 2  # via body + drill hole
    radii = sorted(p.radius for p in ax.patches)
    assert radii == [pytest.approx(0.15), pytest.approx(0.3)]  # drill=0.3mm dia, via=0.6mm dia


def test_pads_field_and_net_pads_fallback_both_render():
    geometry_with_pads = BoardGeometry(
        [], [], [PadGeom(0, 0, 500_000, 500_000, 0, 0, "net_0", "J1:1")], [], [], []
    )
    ax = render_board(geometry_with_pads)
    assert len(ax.patches) == 1

    geometry_without_pads = _empty_geometry()
    fallback_pads = [NetPad("net_0", "J1:1", 0, 0, -1)]
    ax2 = render_board(geometry_without_pads, net_pads=fallback_pads)
    assert len(ax2.patches) == 1


def test_net_color_is_deterministic_and_distinguishes_nets():
    assert _net_color("net_0") == _net_color("net_0")
    assert _net_color("net_0") != _net_color("net_1")


def test_net_color_handles_empty_net_gracefully():
    # Board-edge/keepout items and unrouted pads can have net == "" --
    # must not crash or collide visually with a real net by accident.
    assert _net_color("") == (0.55, 0.55, 0.55)


def test_y_axis_is_inverted_to_match_kicad_convention():
    ax = render_board(_empty_geometry())
    bottom, top = ax.get_ylim()
    assert bottom > top  # inverted: KiCad Y grows downward, matplotlib's default is upward


def test_title_reports_track_and_via_counts_by_default():
    geometry = BoardGeometry(
        [TrackSegment(0, 0, 1 * MM, 0, 250_000, 0, "net_0", False)],
        [ViaGeom(0, 0, 600_000, 300_000, 0, 2, "net_0")],
        [], [], [], [],
    )
    ax = render_board(geometry)
    assert "1 track segment" in ax.get_title()
    assert "1 via" in ax.get_title()
