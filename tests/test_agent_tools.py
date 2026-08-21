"""Tests for the validated agent tool surface.

Unlike tests/test_*_route_env.py, most of what is checked here is real:
argument validation, unit handling, error taxonomy and the deviation report
are pure Python decisions made BEFORE or AFTER the router call, so a stub
bridge does not weaken them. The stub is deliberately hostile where it
matters -- DeviatingBridge reproduces the one behaviour measured on the
real router that a naive tool layer would hide (push() returning True while
the head goes somewhere else).
"""

from __future__ import annotations

from collections import namedtuple

import pytest

from pcbworld.agent.tools import ErrorCode, RouterTools

MM = 1_000_000

NetPad = namedtuple("NetPad", ["net", "pad_name", "x", "y", "layer"])
Candidate = namedtuple("Candidate", ["id", "x", "y", "kind", "net"])
DRCViolation = namedtuple("DRCViolation", ["error_code", "message", "severity", "x", "y"])
HeadGeometry = namedtuple(
    "HeadGeometry", ["active", "segments", "vias", "end_x", "end_y", "layer", "length"]
)
DesignRules = namedtuple(
    "DesignRules",
    [
        "track_width",
        "via_diameter",
        "via_drill",
        "clearance",
        "min_track_width",
        "min_via_diameter",
        "min_via_drill",
        "min_hole_to_hole",
    ],
)


class StubBridge:
    """Accepts everything and puts the head exactly where asked."""

    def __init__(self, pads=None):
        self._pads = pads if pads is not None else [
            NetPad("net_0", "J1:1", 5 * MM, 5 * MM, -1),
            NetPad("net_0", "J2:1", 25 * MM, 5 * MM, -1),
            NetPad("net_1", "J3:1", 5 * MM, 15 * MM, -1),
            NetPad("net_1", "J4:1", 25 * MM, 15 * MM, -1),
        ]
        self.head = (5 * MM, 5 * MM)
        self.routing = False
        self.collides = False
        self.fix_ok = True
        self.push_ok = True
        self.committed = []
        self.ripped = []
        self.violations = []

    def net_pads(self):
        return self._pads

    def query_hover_items(self, x, y, layer=0, slop_radius=0):
        return [Candidate(1, x, y, "pad", "net_0")]

    def start_route(self, x, y, item_id, layer):
        self.routing = True
        self.head = (x, y)
        return True

    def push(self, x, y, item_id=-1):
        if not self.push_ok:
            return False
        self.head = (x, y)
        return True

    def fix(self, x, y, item_id=-1, force_finish=False, force_commit=False):
        return self.fix_ok

    def commit_routing(self):
        self.committed.append(self.head)
        self.routing = False

    def stop_routing(self):
        self.routing = False

    def run_drc(self):
        return self.violations

    def rip_up(self, net):
        self.ripped.append(net)
        return 3

    def get_head_geometry(self):
        return HeadGeometry(self.routing, [], [], self.head[0], self.head[1], 0, 0.0)

    def head_collides(self):
        return self.collides

    def get_design_rules(self):
        return DesignRules(
            250_000, 600_000, 300_000, 200_000, 150_000, 600_000, 300_000, 250_000
        )


class DeviatingBridge(StubBridge):
    """push() returns True but the head lands 2mm off.

    This is the measured real-router behaviour (push accepted 72/72 while
    fix rejected ~67%) that the tool layer exists to surface.
    """

    def push(self, x, y, item_id=-1):
        self.head = (x + 2 * MM, y)
        return True


def make_tools(bridge=None, **kwargs):
    return RouterTools(bridge or StubBridge(), 50.0, 50.0, **kwargs)


# -- argument validation, before the router is ever touched ----------------


def test_route_to_without_start_route_says_so():
    result = make_tools().route_to(10.0, 10.0)
    assert not result.ok
    assert result.error_code == ErrorCode.NO_ROUTE_IN_PROGRESS


def test_out_of_bounds_names_the_legal_range_and_the_unit():
    tools = make_tools()
    tools.start_route("net_0")
    result = tools.route_to(10_000_000.0, 5.0)

    assert not result.ok
    assert result.error_code == ErrorCode.OUT_OF_BOUNDS
    # The mm/nm confusion is the likeliest catastrophic model error, so the
    # message has to name the unit explicitly.
    assert "MILLIMETRES" in result.message
    assert "50.000" in result.message


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_coordinates_rejected(bad):
    tools = make_tools()
    tools.start_route("net_0")
    result = tools.route_to(bad, 5.0)
    assert not result.ok
    assert result.error_code == ErrorCode.BAD_COORDINATE


def test_step_longer_than_the_cap_is_refused_with_the_cap_stated():
    tools = make_tools(max_step_mm=4.0)
    tools.start_route("net_0")
    result = tools.route_to(25.0, 5.0)

    assert not result.ok
    assert result.error_code == ErrorCode.STEP_TOO_LONG
    assert "4.000" in result.message


def test_zero_length_move_refused():
    tools = make_tools()
    tools.start_route("net_0")
    result = tools.route_to(5.0, 5.0)
    assert not result.ok
    assert result.error_code == ErrorCode.ZERO_LENGTH_MOVE


def test_unknown_net_suggests_near_misses():
    result = make_tools().start_route("net_O")  # letter O, not zero
    assert not result.ok
    assert result.error_code == ErrorCode.UNKNOWN_NET
    assert "net_0" in result.message


def test_double_start_route_is_refused():
    tools = make_tools()
    tools.start_route("net_0")
    result = tools.start_route("net_1")
    assert not result.ok
    assert result.error_code == ErrorCode.ROUTE_ALREADY_ACTIVE


def test_rerouting_a_routed_net_points_at_rip_up():
    tools = make_tools()
    tools.start_route("net_0")
    tools.finish_route()
    result = tools.start_route("net_0")

    assert not result.ok
    assert result.error_code == ErrorCode.NET_ALREADY_ROUTED
    assert "rip_up" in result.message


# -- the silent-success case ----------------------------------------------


def test_deviation_is_reported_even_though_push_succeeded():
    tools = make_tools(DeviatingBridge())
    tools.start_route("net_0")
    result = tools.route_to(10.0, 5.0)

    assert result.ok  # the router did accept it
    assert result.warnings, "a 2mm deviation must not be reported as a clean move"
    assert "2.000" in result.to_model()
    assert result.data["head_mm"][0] == pytest.approx(12.0)


def test_small_deviation_within_tolerance_is_not_flagged():
    tools = make_tools(deviation_tolerance_mm=5.0)
    tools.start_route("net_0")
    result = tools.route_to(10.0, 5.0)
    assert result.ok
    assert not result.warnings


def test_collision_is_surfaced_on_an_accepted_move():
    bridge = StubBridge()
    bridge.collides = True
    tools = make_tools(bridge)
    tools.start_route("net_0")
    result = tools.route_to(10.0, 5.0)

    assert result.ok
    assert any("colliding" in w for w in result.warnings)


def test_missing_head_readback_degrades_loudly_not_silently():
    # A bridge built before the head-state bindings existed. Composed rather
    # than subclassed from StubBridge on purpose -- inheriting and deleting
    # the attributes just re-exposes the parent's versions.
    class LegacyBridge:
        def __init__(self):
            self._inner = StubBridge()

        net_pads = property(lambda self: self._inner.net_pads)
        query_hover_items = property(lambda self: self._inner.query_hover_items)
        start_route = property(lambda self: self._inner.start_route)
        push = property(lambda self: self._inner.push)
        fix = property(lambda self: self._inner.fix)
        commit_routing = property(lambda self: self._inner.commit_routing)
        stop_routing = property(lambda self: self._inner.stop_routing)
        run_drc = property(lambda self: self._inner.run_drc)

    bridge = LegacyBridge()
    tools = make_tools(bridge)
    assert not tools.has_head_readback
    tools.start_route("net_0")
    result = tools.route_to(10.0, 5.0)

    assert result.ok
    assert any("read-back unavailable" in w for w in result.warnings)


# -- failure reporting ----------------------------------------------------


def test_finish_route_failure_distinguishes_collision_from_distance():
    bridge = StubBridge()
    bridge.fix_ok = False
    bridge.collides = True
    tools = make_tools(bridge)
    tools.start_route("net_0")
    result = tools.finish_route()

    assert not result.ok
    assert result.error_code == ErrorCode.HEAD_COLLIDES
    assert "rip_up" in result.message


def test_finish_route_failure_when_far_from_target_says_so():
    bridge = StubBridge()
    bridge.fix_ok = False
    tools = make_tools(bridge)
    tools.start_route("net_0")
    result = tools.finish_route()

    assert not result.ok
    assert result.error_code == ErrorCode.NOT_AT_TARGET
    assert "from the target pad" in result.message


def test_push_rejection_suggests_a_via():
    bridge = StubBridge()
    bridge.push_ok = False
    tools = make_tools(bridge)
    tools.start_route("net_0")
    result = tools.route_to(10.0, 5.0)

    assert not result.ok
    assert result.error_code == ErrorCode.ROUTER_REJECTED
    assert "via" in result.message


def test_check_drc_returns_records_not_a_count():
    bridge = StubBridge()
    bridge.violations = [
        DRCViolation(5, "Clearance 0.06mm < 0.2mm required", "error", 12 * MM, 8 * MM)
    ]
    result = make_tools(bridge).check_drc()

    rendered = result.to_model()
    assert "0.06mm" in rendered  # the actual numbers survive
    assert "12.000" in rendered  # and the location


def test_rip_up_of_an_unrouted_net_warns_that_nothing_changed():
    class NoopRipUp(StubBridge):
        def rip_up(self, net):
            return 0

    result = make_tools(NoopRipUp()).rip_up("net_0")
    assert result.ok
    assert result.data["removed_items"] == 0
    assert result.warnings


def test_board_info_exposes_the_via_minimum_the_agent_must_respect():
    result = make_tools().get_board_info()
    assert result.data["min_via_diameter_mm"] == pytest.approx(0.6)


# -- unit handling --------------------------------------------------------


def test_coordinates_go_out_in_mm_and_reach_the_bridge_in_nm():
    bridge = StubBridge()
    tools = make_tools(bridge)
    tools.start_route("net_0")
    tools.route_to(10.0, 5.0)

    assert bridge.head == (10 * MM, 5 * MM)


def test_list_nets_reports_mm_coordinates():
    rendered = make_tools().list_nets().to_model()
    assert "(5.000, 5.000)" in rendered
    assert "(25.000, 5.000)" in rendered
