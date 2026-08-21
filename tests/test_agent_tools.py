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
TrackSegment = namedtuple("TrackSegment", ["x1", "y1", "x2", "y2", "width", "layer", "net", "is_arc"])
ViaGeom = namedtuple("ViaGeom", ["x", "y", "diameter", "drill", "layer_top", "layer_bottom", "net"])
PadGeom = namedtuple("PadGeom", ["x", "y", "size_x", "size_y", "layer_top", "layer_bottom", "net", "pad_name"])
EdgeShape = namedtuple("EdgeShape", ["shape_type", "x1", "y1", "x2", "y2", "width"])
BoardGeometry = namedtuple("BoardGeometry", ["tracks", "vias", "pads", "zones", "courtyards", "board_edge"])
HeadGeometry = namedtuple(
    "HeadGeometry", ["active", "segments", "vias", "end_x", "end_y", "layer", "length"]
)
HeadObstacle = namedtuple("HeadObstacle", ["found", "net", "kind", "x", "y"])
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
        self.layer = 0
        self.routing = False
        self.collides = False
        # get_head_obstacle() detail, consulted only when self.collides is
        # True. Defaults to "same net as whatever's active" (the case a
        # live Colab run actually produced -- head_collides() firing
        # against the route's own target pad on a completely empty board)
        # since that is the scenario RouterTools' _collision_message() most
        # needs correct test coverage for.
        self.obstacle_net = "net_0"
        self.obstacle_kind = "pad"
        self.obstacle_xy = (25 * MM, 5 * MM)
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
        self.layer = layer if layer >= 0 else 0
        return True

    def push(self, x, y, item_id=-1):
        if not self.push_ok:
            return False
        self.head = (x, y)
        return True

    def toggle_via_placement(self):
        self.layer = 1 if self.layer == 0 else 0

    def switch_layer(self, layer):
        self.layer = layer
        return True

    def fix(self, x, y, item_id=-1, force_finish=False, force_commit=False):
        return self.fix_ok

    def commit_routing(self):
        self.committed.append(self.head)
        self.routing = False
        self.layer = 0

    def stop_routing(self):
        self.routing = False
        self.layer = 0

    def run_drc(self):
        return self.violations

    def rip_up(self, net):
        self.ripped.append(net)
        return 3

    def get_head_geometry(self):
        return HeadGeometry(self.routing, [], [], self.head[0], self.head[1], self.layer, 0.0)

    def head_collides(self):
        return self.collides

    def get_head_obstacle(self):
        if not self.collides:
            return HeadObstacle(found=False, net="", kind="", x=0, y=0)
        return HeadObstacle(
            found=True,
            net=self.obstacle_net,
            kind=self.obstacle_kind,
            x=self.obstacle_xy[0],
            y=self.obstacle_xy[1],
        )

    def get_design_rules(self):
        return DesignRules(
            250_000, 600_000, 300_000, 200_000, 150_000, 600_000, 300_000, 250_000
        )

    def get_board_geometry(self):
        pads_geom = [
            PadGeom(p.x, p.y, 500_000, 500_000, 0, 1, p.net, p.pad_name)
            for p in self._pads
        ]
        return BoardGeometry(
            tracks=[],
            vias=[],
            pads=pads_geom,
            zones=[],
            courtyards=[],
            board_edge=[EdgeShape("segment", 0, 0, 50 * MM, 50 * MM, 100_000)],
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


def test_step_a_hair_over_the_cap_from_display_rounding_is_still_accepted():
    """Live Colab run (Qwen3-4B): the model read a displayed head position
    rounded to 3 decimals, computed "displayed + 8.0" as its next waypoint,
    and got rejected -- "that move is 8.000mm but the limit is 8.000mm per
    call" -- both numbers rendered identically, because the TRUE head
    position (fuller precision than what to_model() ever shows) made the
    real step a few microns over 8.0mm. Not raw float epsilon: 32.064 -
    24.064 == 8.0 exactly in Python. The tolerance exists specifically to
    absorb this gap between what a model can see (rounded text) and what
    the check compares (full precision) -- reproduced here directly with a
    head position that displays as a round number but sits a few microns
    below it, matching the live case."""
    # A pad position that DISPLAYS as "24.064" (3-decimal rounding) but
    # sits a few microns below it -- start_route() reads its position from
    # net_pads(), so the precise coordinate has to live there, not in
    # bridge.head directly (start_route() overwrites head from the pad).
    bridge = StubBridge(
        pads=[
            NetPad("net_0", "J1:1", int(24.0637 * MM), 5 * MM, -1),
            NetPad("net_0", "J2:1", 60 * MM, 5 * MM, -1),
        ]
    )
    tools = make_tools(bridge, max_step_mm=8.0)
    tools.start_route("net_0")

    # A model computing "24.064 + 8.0" from the displayed value -- a real
    # step of ~8.0003mm from the TRUE head position, previously rejected.
    result = tools.route_to(32.064, 5.0)
    assert result.ok

    # A step meaningfully over the limit (not just display-rounding noise)
    # must still be refused -- the tolerance is a few microns, not a
    # loophole.
    tools2 = make_tools(max_step_mm=8.0)
    tools2.start_route("net_0")
    result2 = tools2.route_to(35.0, 5.0)  # 30mm real step
    assert not result2.ok
    assert result2.error_code == ErrorCode.STEP_TOO_LONG


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


def test_same_net_collision_warns_it_may_be_the_own_target_pad():
    """A live Colab run found head_collides() firing on the very first net
    routed against a completely empty board, at the instant the head
    reached its own target pad -- get_head_obstacle() reported the
    colliding item's net as the SAME net being routed. RouterTools must
    tell an agent that is a likely-benign self-touch, not "rip something
    up", or it will send the agent chasing a blocker that may not exist."""
    bridge = StubBridge()
    bridge.collides = True
    bridge.obstacle_net = "net_0"  # matches the net being routed below
    tools = make_tools(bridge)
    tools.start_route("net_0")
    result = tools.route_to(10.0, 5.0)

    assert result.ok
    message = " ".join(result.warnings)
    assert "OWN net" in message
    assert "not a real obstacle" in message
    # The message correctly says "not... needs to be ripped up" -- a
    # negation, not an instruction. What must NOT appear is an unqualified
    # imperative telling the agent to go rip something up.
    assert "rip_up(" not in message
    assert "call rip_up" not in message.lower()


def test_different_net_collision_names_the_actual_blocking_net():
    bridge = StubBridge()
    bridge.collides = True
    bridge.obstacle_net = "net_1"  # a DIFFERENT net than the one being routed
    bridge.obstacle_kind = "segment"
    tools = make_tools(bridge)
    tools.start_route("net_0")
    result = tools.route_to(10.0, 5.0)

    assert result.ok
    message = " ".join(result.warnings)
    assert "net_1" in message
    assert "segment" in message
    assert "OWN net" not in message


def test_finish_route_same_net_collision_does_not_tell_agent_to_rip_up():
    """finish_route()'s HEAD_COLLIDES message must not blanket-instruct
    rip_up() any more -- for the same-net case there is nothing routed yet
    to rip up (fix() just failed, nothing was committed)."""
    bridge = StubBridge()
    bridge.fix_ok = False
    bridge.collides = True
    bridge.obstacle_net = "net_0"
    tools = make_tools(bridge)
    tools.start_route("net_0")
    result = tools.finish_route()

    assert not result.ok
    assert result.error_code == ErrorCode.HEAD_COLLIDES
    assert "finish_route() again" in result.message or "OWN net" in result.message
    assert "not a real obstacle" in result.message


def test_finish_route_different_net_collision_still_names_the_blocker():
    bridge = StubBridge()
    bridge.fix_ok = False
    bridge.collides = True
    bridge.obstacle_net = "net_1"
    tools = make_tools(bridge)
    tools.start_route("net_0")
    result = tools.finish_route()

    assert not result.ok
    assert result.error_code == ErrorCode.HEAD_COLLIDES
    assert "net_1" in result.message


def test_collision_detail_degrades_gracefully_without_get_head_obstacle():
    """A bridge with head_collides() but not get_head_obstacle() (an older
    compiled build) must still work -- just with the generic message, not
    a crash. Composed rather than subclassed-and-deleted, same reasoning as
    LegacyBridge above: get_head_obstacle is a class method, not an
    instance attribute, so `del` on the instance wouldn't remove it."""

    class NoObstacleDetailBridge:
        def __init__(self):
            self._inner = StubBridge()
            self._inner.collides = True

        net_pads = property(lambda self: self._inner.net_pads)
        query_hover_items = property(lambda self: self._inner.query_hover_items)
        start_route = property(lambda self: self._inner.start_route)
        push = property(lambda self: self._inner.push)
        fix = property(lambda self: self._inner.fix)
        commit_routing = property(lambda self: self._inner.commit_routing)
        stop_routing = property(lambda self: self._inner.stop_routing)
        run_drc = property(lambda self: self._inner.run_drc)
        get_head_geometry = property(lambda self: self._inner.get_head_geometry)
        head_collides = property(lambda self: self._inner.head_collides)
        get_design_rules = property(lambda self: self._inner.get_design_rules)
        # get_head_obstacle deliberately NOT forwarded here.

    bridge = NoObstacleDetailBridge()
    tools = make_tools(bridge)
    assert not tools.has_obstacle_detail
    assert tools.has_collision_readback  # head_collides() IS available

    tools.start_route("net_0")
    result = tools.route_to(10.0, 5.0)

    assert result.ok
    assert any("colliding with something" in w for w in result.warnings)


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
    # A genuine different-net blocker, not the default same-net collision --
    # this test's intent is a real obstacle, which is the one case where
    # naming rip_up() as the fix is still correct. See
    # test_finish_route_same_net_collision_does_not_tell_agent_to_rip_up
    # for the other branch, added after a live Colab run showed
    # head_collides() firing against a route's own target pad.
    bridge.obstacle_net = "net_1"
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


# -- via and layer switching ----------------------------------------------


def test_place_via_and_switch_layer_without_active_route_rejected():
    tools = make_tools()
    res1 = tools.place_via()
    assert not res1.ok
    assert res1.error_code == ErrorCode.NO_ROUTE_IN_PROGRESS

    res2 = tools.switch_to_layer(1)
    assert not res2.ok
    assert res2.error_code == ErrorCode.NO_ROUTE_IN_PROGRESS


def test_switch_to_layer_validates_layer_number():
    tools = make_tools()
    tools.start_route("net_0")

    bad_layer_res = tools.switch_to_layer(2)
    assert not bad_layer_res.ok
    assert bad_layer_res.error_code == ErrorCode.OUT_OF_BOUNDS
    assert "0" in bad_layer_res.message and "1" in bad_layer_res.message

    bad_type_res = tools.switch_to_layer("F_Cu")  # type: ignore
    assert not bad_type_res.ok
    assert bad_type_res.error_code == ErrorCode.OUT_OF_BOUNDS


def test_place_via_and_switch_to_layer_toggle_layer():
    bridge = StubBridge()
    tools = make_tools(bridge)
    tools.start_route("net_0")

    assert bridge.layer == 0
    res_via = tools.place_via()
    assert res_via.ok
    assert bridge.layer == 1
    assert res_via.data["layer"] == 1

    res_switch = tools.switch_to_layer(0)
    assert res_switch.ok
    assert bridge.layer == 0
    assert res_switch.data["layer"] == 0


def test_via_violating_design_rules_rejected_with_actionable_error():
    class IllegalViaBridge(StubBridge):
        def get_design_rules(self):
            # via diameter (0.4mm) is below board minimum (0.6mm)
            return DesignRules(
                250_000, 400_000, 300_000, 200_000, 150_000, 600_000, 300_000, 250_000
            )

    tools = make_tools(IllegalViaBridge())
    tools.start_route("net_0")

    res = tools.place_via()
    assert not res.ok
    assert res.error_code == ErrorCode.VIOLATES_DESIGN_RULE
    assert "0.400" in res.message
    assert "0.600" in res.message
    assert "get_board_info()" in res.message

    res2 = tools.switch_to_layer(1)
    assert not res2.ok
    assert res2.error_code == ErrorCode.VIOLATES_DESIGN_RULE


def test_finish_route_on_back_layer_refused():
    bridge = StubBridge()
    tools = make_tools(bridge)
    tools.start_route("net_0")
    tools.route_to(10.0, 5.0)
    tools.switch_to_layer(1)

    # Head is on layer 1 (B_Cu), finish should refuse
    result = tools.finish_route()
    assert not result.ok
    assert "layer 0" in result.message
    assert "switch_to_layer(0)" in result.message

    # Switching back to layer 0 allows finish
    tools.switch_to_layer(0)
    ok_res = tools.finish_route()
    assert ok_res.ok

