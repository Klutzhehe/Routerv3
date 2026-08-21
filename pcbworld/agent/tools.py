"""Validated tool surface an LLM agent drives the router through.

Every function here wraps a pcbworld_pns_bridge call in three phases:

    validate  -> reject a bad argument WITHOUT touching the router, and say
                 what was wrong, what was received, and what is legal
    execute   -> the actual bridge call
    verify    -> read back what the router ACTUALLY did and report any
                 disagreement with what was asked

The third phase is the point of this module. `push()` is `ROUTER::Move()`,
and a Colab measurement (scripts/measure_waypoint_fidelity.py, three runs)
found it returning True for 72/72 net-attempts while `fix()` then rejected
~67% of those same routes. A tool layer that forwards that bare `True`
would tell the agent its move succeeded, let it stack nine more moves on
top, and surface the problem only at the end with no way to attribute it.
Silent success is the failure mode this module exists to prevent; a loud
rejection was never the hard part.

Units: this API is in **millimetres, as floats**, not KiCad's internal
nanometres. Every board dimension a human or a model reasons about is in
mm, and mm/nm confusion is the single likeliest way for a model to emit a
catastrophically wrong coordinate (50mm and 50000000nm are the same edge of
the same board). Conversion happens here, once, and every returned
coordinate is mm too so nothing round-trips through the wrong unit.

Nothing here imports pcbnew. It takes an already-constructed bridge object,
so it is agnostic about who built it -- the real pcbworld_pns_bridge in
Colab, or a fixture. See ROADMAP.md's process-isolation constraint.
"""

from __future__ import annotations

import dataclasses
import difflib
import math
from typing import Any

MM = 1_000_000  # KiCad internal units are nm

# How far the committed head may sit from the point that was requested
# before the result is reported as a deviation rather than a clean move.
# 2x a default 0.25mm track width, matching the fidelity bar
# docs/AI_ARCHITECTURE.md sets ("median deviation under ~2x track pitch").
DEFAULT_DEVIATION_TOLERANCE_MM = 0.5

# Cap on a single route_to() move. This is the "max length so it can't
# straight-line to the finish" idea: bounded steps force the agent to
# commit incrementally and see the consequence of each move, instead of
# emitting one hop to the target and learning nothing about what went
# wrong in between.
DEFAULT_MAX_STEP_MM = 8.0

SNAP_RADIUS_NM = int(0.5 * MM)


class ErrorCode:
    """Stable machine-readable error kinds.

    Small models are noticeably better at recovering when the error names a
    category they can pattern-match than when it is only prose, so every
    failure carries one of these AND a sentence.
    """

    NO_ROUTE_IN_PROGRESS = "NO_ROUTE_IN_PROGRESS"
    ROUTE_ALREADY_ACTIVE = "ROUTE_ALREADY_ACTIVE"
    UNKNOWN_NET = "UNKNOWN_NET"
    NET_ALREADY_ROUTED = "NET_ALREADY_ROUTED"
    BAD_COORDINATE = "BAD_COORDINATE"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    STEP_TOO_LONG = "STEP_TOO_LONG"
    ZERO_LENGTH_MOVE = "ZERO_LENGTH_MOVE"
    VIOLATES_DESIGN_RULE = "VIOLATES_DESIGN_RULE"
    ROUTER_REJECTED = "ROUTER_REJECTED"
    HEAD_DEVIATED = "HEAD_DEVIATED"
    HEAD_COLLIDES = "HEAD_COLLIDES"
    NOT_AT_TARGET = "NOT_AT_TARGET"
    NO_ITEM_AT_POINT = "NO_ITEM_AT_POINT"


@dataclasses.dataclass
class ToolResult:
    """What every tool returns. Never a bare bool.

    `ok` False always comes with `error_code` and a `message` that names the
    offending value and the legal alternative. `ok` True may STILL carry
    `warnings` -- that is the deviation case: the router accepted the move
    but did not do exactly what was asked, which the agent has to know
    before it builds anything on top.
    """

    ok: bool
    message: str
    error_code: str | None = None
    data: dict[str, Any] = dataclasses.field(default_factory=dict)
    warnings: list[str] = dataclasses.field(default_factory=list)

    def to_model(self) -> str:
        """Compact text form fed back to the LLM.

        Text rather than JSON on purpose: a 4B-class model at Q4 spends
        fewer tokens and makes fewer parse errors on `KEY: value` lines than
        on nested JSON, and none of this payload is deep enough to need
        structure.
        """
        head = "OK" if self.ok else f"ERROR [{self.error_code}]"
        lines = [f"{head}: {self.message}"]

        for key, value in self.data.items():
            lines.append(f"  {key}: {_fmt(value)}")

        for warning in self.warnings:
            lines.append(f"  WARNING: {warning}")

        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def _mm(nm: float) -> float:
    return nm / MM


def _nm(mm: float) -> int:
    return int(round(mm * MM))


class RouterTools:
    """Stateful tool surface over one bridge instance and one board.

    Tracks its own notion of the active route because the bridge does not
    expose one: calling push() with no route in progress is a silent no-op
    at the C++ level (PNS_BRIDGE::Push returns false only when m_router is
    null), which would look to an agent like a legitimate rejection rather
    than a call-order mistake. Distinguishing those two is worth the
    bookkeeping.
    """

    def __init__(
        self,
        bridge,
        board_width_mm: float,
        board_height_mm: float,
        max_step_mm: float = DEFAULT_MAX_STEP_MM,
        deviation_tolerance_mm: float = DEFAULT_DEVIATION_TOLERANCE_MM,
    ) -> None:
        self.bridge = bridge
        self.board_width_mm = board_width_mm
        self.board_height_mm = board_height_mm
        self.max_step_mm = max_step_mm
        self.deviation_tolerance_mm = deviation_tolerance_mm

        self._active_net: str | None = None
        self._target_xy_nm: tuple[int, int] | None = None
        self._target_item_id: int = -1
        self._requested_mm: tuple[float, float] | None = None
        self._routed_nets: set[str] = set()

        # get_head_geometry()/head_collides() are new C++ that has never been
        # compiled (see the commit that added them). Probe once rather than
        # assuming: without them this layer still works, it just cannot
        # report deviation -- which is a degraded mode worth naming out loud
        # instead of crashing an agent run halfway through a board.
        self.has_head_readback = hasattr(bridge, "get_head_geometry")
        self.has_collision_readback = hasattr(bridge, "head_collides")

    # -- introspection -----------------------------------------------------

    def get_board_info(self) -> ToolResult:
        """Board size, design rules, and the legal ranges every other tool
        validates against. The agent should call this first -- it is what
        stops it inventing a via size the board will refuse."""
        data: dict[str, Any] = {
            "board_width_mm": self.board_width_mm,
            "board_height_mm": self.board_height_mm,
            "max_step_mm": self.max_step_mm,
        }

        rules = self._design_rules()
        if rules is not None:
            data.update(
                {
                    "track_width_mm": _mm(rules.track_width),
                    "via_diameter_mm": _mm(rules.via_diameter),
                    "via_drill_mm": _mm(rules.via_drill),
                    "clearance_mm": _mm(rules.clearance),
                    "min_track_width_mm": _mm(rules.min_track_width),
                    "min_via_diameter_mm": _mm(rules.min_via_diameter),
                    "min_via_drill_mm": _mm(rules.min_via_drill),
                }
            )

        return ToolResult(ok=True, message="board info", data=data)

    def list_nets(self) -> ToolResult:
        """Every net with its pad coordinates and whether it is routed yet."""
        by_net: dict[str, list] = {}
        for pad in self.bridge.net_pads():
            if pad.net:
                by_net.setdefault(pad.net, []).append(pad)

        lines = []
        for net in sorted(by_net):
            pads = by_net[net]
            coords = "; ".join(f"({_mm(p.x):.3f}, {_mm(p.y):.3f})" for p in pads)
            state = "routed" if net in self._routed_nets else "unrouted"
            lines.append(f"{net} [{state}] pads: {coords}")

        return ToolResult(
            ok=True,
            message=f"{len(by_net)} net(s)",
            data={"nets": "\n    " + "\n    ".join(lines) if lines else "(none)"},
        )

    # -- routing -----------------------------------------------------------

    def start_route(self, net: str) -> ToolResult:
        """Opens a routing session on `net`, from its first pad toward its
        second."""
        if self._active_net is not None:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.ROUTE_ALREADY_ACTIVE,
                message=(
                    f"already routing {self._active_net!r}. Call finish_route() "
                    f"or abandon_route() before starting another net."
                ),
            )

        pads_by_net: dict[str, list] = {}
        for pad in self.bridge.net_pads():
            if pad.net:
                pads_by_net.setdefault(pad.net, []).append(pad)

        if net not in pads_by_net:
            # Name the near-misses. A small model that typo'd a net name
            # recovers immediately from this and flounders without it.
            close = difflib.get_close_matches(net, sorted(pads_by_net), n=3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            return ToolResult(
                ok=False,
                error_code=ErrorCode.UNKNOWN_NET,
                message=f"no net named {net!r} on this board.{hint}",
                data={"valid_nets": sorted(pads_by_net)},
            )

        if net in self._routed_nets:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.NET_ALREADY_ROUTED,
                message=(
                    f"{net!r} is already routed. Call rip_up({net!r}) first if "
                    f"you want to reroute it."
                ),
            )

        pads = pads_by_net[net]
        if len(pads) < 2:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.UNKNOWN_NET,
                message=f"{net!r} has {len(pads)} pad(s); routing needs at least 2.",
            )

        start_pad, target_pad = pads[0], pads[1]

        start_id = self._pad_item_id(start_pad.x, start_pad.y)
        if start_id is None:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.NO_ITEM_AT_POINT,
                message=(
                    f"no routable item found at {net!r}'s start pad "
                    f"({_mm(start_pad.x):.3f}, {_mm(start_pad.y):.3f})."
                ),
            )

        target_id = self._pad_item_id(target_pad.x, target_pad.y)
        if target_id is None:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.NO_ITEM_AT_POINT,
                message=(
                    f"no routable item found at {net!r}'s target pad "
                    f"({_mm(target_pad.x):.3f}, {_mm(target_pad.y):.3f})."
                ),
            )

        if not self.bridge.start_route(start_pad.x, start_pad.y, start_id, 0):
            return ToolResult(
                ok=False,
                error_code=ErrorCode.ROUTER_REJECTED,
                message=(
                    f"the router refused to start a route at {net!r}'s start pad. "
                    f"The pad may already be occupied by other copper."
                ),
            )

        self._active_net = net
        self._target_xy_nm = (target_pad.x, target_pad.y)
        self._target_item_id = target_id
        self._requested_mm = (_mm(start_pad.x), _mm(start_pad.y))

        return ToolResult(
            ok=True,
            message=f"routing {net!r}",
            data={
                "start_mm": (_mm(start_pad.x), _mm(start_pad.y)),
                "target_mm": (_mm(target_pad.x), _mm(target_pad.y)),
                "distance_to_target_mm": self._distance_to_target_mm(),
            },
        )

    def route_to(self, x_mm: float, y_mm: float) -> ToolResult:
        """Moves the routing head toward (x_mm, y_mm).

        Validates the point is finite, on the board, and within max_step_mm
        of the current head, THEN pushes, THEN reads the head back to check
        the router actually went where it was told.
        """
        if self._active_net is None:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.NO_ROUTE_IN_PROGRESS,
                message="no route in progress. Call start_route(net) first.",
            )

        bad = self._validate_point(x_mm, y_mm)
        if bad is not None:
            return bad

        head_before = self._head_position_mm()
        step_mm = math.hypot(x_mm - head_before[0], y_mm - head_before[1])

        if step_mm < 1e-6:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.ZERO_LENGTH_MOVE,
                message=(
                    f"the head is already at ({x_mm:.3f}, {y_mm:.3f}). "
                    f"Pick a different point."
                ),
            )

        if step_mm > self.max_step_mm:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.STEP_TOO_LONG,
                message=(
                    f"that move is {step_mm:.3f}mm but the limit is "
                    f"{self.max_step_mm:.3f}mm per call. Route there in several "
                    f"shorter steps."
                ),
                data={
                    "head_mm": head_before,
                    "requested_mm": (x_mm, y_mm),
                    "max_step_mm": self.max_step_mm,
                },
            )

        accepted = self.bridge.push(_nm(x_mm), _nm(y_mm), -1)
        self._requested_mm = (x_mm, y_mm)

        if not accepted:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.ROUTER_REJECTED,
                message=(
                    f"the router refused to move to ({x_mm:.3f}, {y_mm:.3f}). "
                    f"Something is in the way -- try a different direction, or "
                    f"place a via and cross on the other layer."
                ),
                data={"head_mm": self._head_position_mm()},
            )

        # -- verify: did it actually go there? ---------------------------
        head_after = self._head_position_mm()
        data: dict[str, Any] = {
            "head_mm": head_after,
            "distance_to_target_mm": self._distance_to_target_mm(),
        }
        warnings: list[str] = []

        if self.has_head_readback:
            deviation = math.hypot(head_after[0] - x_mm, head_after[1] - y_mm)
            data["deviation_mm"] = deviation
            if deviation > self.deviation_tolerance_mm:
                warnings.append(
                    f"the router moved the head to ({head_after[0]:.3f}, "
                    f"{head_after[1]:.3f}), {deviation:.3f}mm from the "
                    f"({x_mm:.3f}, {y_mm:.3f}) you asked for. It routed around "
                    f"something. Plan your next move from where the head "
                    f"actually is."
                )
        else:
            warnings.append(
                "head read-back unavailable on this bridge build, so the "
                "reported head position is the point requested, not the point "
                "the router actually reached."
            )

        if self.has_collision_readback and self.bridge.head_collides():
            warnings.append(
                "the head is currently colliding with something. finish_route() "
                "will fail while this is true -- move away or rip up whatever is "
                "in the way."
            )
            data["head_collides"] = True

        return ToolResult(
            ok=True,
            message=f"head moved to ({head_after[0]:.3f}, {head_after[1]:.3f})",
            data=data,
            warnings=warnings,
        )

    def place_via(self) -> ToolResult:
        """Toggles via placement at the current head position to change copper layers."""
        if self._active_net is None:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.NO_ROUTE_IN_PROGRESS,
                message="no route in progress. Call start_route(net) first.",
            )

        rule_err = self._validate_via_rules()
        if rule_err is not None:
            return rule_err

        if not hasattr(self.bridge, "toggle_via_placement"):
            return ToolResult(
                ok=False,
                error_code=ErrorCode.ROUTER_REJECTED,
                message="this bridge build has no toggle_via_placement() support.",
            )

        self.bridge.toggle_via_placement()

        head_after = self._head_position_mm()
        current_layer = 0
        if self.has_head_readback:
            head = self.bridge.get_head_geometry()
            if head.active:
                current_layer = head.layer

        warnings = []
        if self.has_collision_readback and self.bridge.head_collides():
            warnings.append(
                "the head is currently colliding with something after placing via."
            )

        return ToolResult(
            ok=True,
            message=f"placed via at ({head_after[0]:.3f}, {head_after[1]:.3f}), now on layer {current_layer}",
            data={
                "head_mm": head_after,
                "layer": current_layer,
                "distance_to_target_mm": self._distance_to_target_mm(),
            },
            warnings=warnings,
        )

    def switch_to_layer(self, layer: int) -> ToolResult:
        """Switches the active routing layer to 0 (F_Cu) or 1 (B_Cu), placing a via."""
        if self._active_net is None:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.NO_ROUTE_IN_PROGRESS,
                message="no route in progress. Call start_route(net) first.",
            )

        if not isinstance(layer, int) or isinstance(layer, bool) or layer not in (0, 1):
            return ToolResult(
                ok=False,
                error_code=ErrorCode.OUT_OF_BOUNDS,
                message=f"layer must be 0 (F_Cu) or 1 (B_Cu), got {layer!r}.",
            )

        rule_err = self._validate_via_rules()
        if rule_err is not None:
            return rule_err

        if not hasattr(self.bridge, "switch_layer"):
            return ToolResult(
                ok=False,
                error_code=ErrorCode.ROUTER_REJECTED,
                message="this bridge build has no switch_layer() support.",
            )

        accepted = self.bridge.switch_layer(layer)
        if accepted is False:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.ROUTER_REJECTED,
                message=f"the router refused to switch to layer {layer}.",
                data={"head_mm": self._head_position_mm()},
            )

        head_after = self._head_position_mm()
        warnings = []
        if self.has_collision_readback and self.bridge.head_collides():
            warnings.append("the head is currently colliding with something.")

        return ToolResult(
            ok=True,
            message=f"switched to layer {layer}",
            data={
                "head_mm": head_after,
                "layer": layer,
                "distance_to_target_mm": self._distance_to_target_mm(),
            },
            warnings=warnings,
        )

    def finish_route(self) -> ToolResult:
        """Attempts to land the route on its target pad and commit it."""
        if self._active_net is None:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.NO_ROUTE_IN_PROGRESS,
                message="no route in progress. Call start_route(net) first.",
            )

        if self.has_head_readback:
            head = self.bridge.get_head_geometry()
            if head.active and head.layer != 0:
                return ToolResult(
                    ok=False,
                    error_code=ErrorCode.ROUTER_REJECTED,
                    message=(
                        f"route head is on layer {head.layer} (B_Cu), but SMD pads are on "
                        f"layer 0 (F_Cu). Switch back to layer 0 with switch_to_layer(0) "
                        f"before finishing."
                    ),
                    data={
                        "head_mm": self._head_position_mm(),
                        "layer": head.layer,
                    },
                )

        net = self._active_net
        target = self._target_xy_nm
        assert target is not None  # set together with _active_net

        fixed = self.bridge.fix(target[0], target[1], self._target_item_id, True, True)

        if not fixed:
            # The measured-common case. Say what is true rather than just
            # "failed", because the agent's next move depends entirely on
            # WHY: too far away is a routing problem, a collision is a
            # rip-up problem.
            distance = self._distance_to_target_mm()
            colliding = self.has_collision_readback and self.bridge.head_collides()

            if colliding:
                reason = (
                    "the head is colliding with existing copper at the target. "
                    "Use check_drc() to see what, then rip_up() that net and "
                    "route it later."
                )
                code = ErrorCode.HEAD_COLLIDES
            elif distance > self.deviation_tolerance_mm:
                reason = (
                    f"the head is still {distance:.3f}mm from the target pad. "
                    f"Route closer before finishing."
                )
                code = ErrorCode.NOT_AT_TARGET
            else:
                reason = (
                    "the router refused to connect even though the head is at "
                    "the pad -- something is occupying the pad's clearance."
                )
                code = ErrorCode.ROUTER_REJECTED

            return ToolResult(
                ok=False,
                error_code=code,
                message=f"could not finish {net!r}: {reason}",
                data={
                    "head_mm": self._head_position_mm(),
                    "target_mm": (_mm(target[0]), _mm(target[1])),
                    "distance_to_target_mm": distance,
                },
            )

        self.bridge.commit_routing()
        self._routed_nets.add(net)
        self._clear_route()

        return ToolResult(ok=True, message=f"{net!r} routed and committed")

    def abandon_route(self) -> ToolResult:
        """Drops the in-progress route without committing it."""
        if self._active_net is None:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.NO_ROUTE_IN_PROGRESS,
                message="no route in progress; nothing to abandon.",
            )

        net = self._active_net
        self.bridge.stop_routing()
        self._clear_route()
        return ToolResult(ok=True, message=f"abandoned the route on {net!r}")

    def rip_up(self, net: str) -> ToolResult:
        """Removes a routed net's copper so its space can be reused."""
        if self._active_net is not None:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.ROUTE_ALREADY_ACTIVE,
                message=(
                    f"finish or abandon the route on {self._active_net!r} before "
                    f"ripping up another net."
                ),
            )

        if not hasattr(self.bridge, "rip_up"):
            return ToolResult(
                ok=False,
                error_code=ErrorCode.ROUTER_REJECTED,
                message="this bridge build has no rip_up() support.",
            )

        removed = self.bridge.rip_up(net)
        self._routed_nets.discard(net)

        if removed == 0:
            return ToolResult(
                ok=True,
                message=f"{net!r} had no copper to remove.",
                data={"removed_items": 0},
                warnings=[
                    "nothing changed -- either that net was never routed, or the "
                    "name is wrong. Check list_nets()."
                ],
            )

        return ToolResult(
            ok=True,
            message=f"ripped up {net!r}",
            data={"removed_items": removed},
        )

    def check_drc(self) -> ToolResult:
        """Full DRC violation records -- message, position, severity.

        Deliberately returns the records, not a count. Every env in this
        repo collapses run_drc() to `sum(1 for v in violations if v.severity
        == "error")`, which throws away the clearance numbers and the
        location, i.e. everything needed to act on it.
        """
        violations = self.bridge.run_drc()
        errors = [v for v in violations if v.severity == "error"]

        if not errors:
            return ToolResult(ok=True, message="DRC clean, 0 errors")

        lines = [
            f"[{v.severity}] {v.message} at ({_mm(v.x):.3f}, {_mm(v.y):.3f})"
            for v in errors[:20]
        ]
        if len(errors) > 20:
            lines.append(f"... and {len(errors) - 20} more")

        return ToolResult(
            ok=True,
            message=f"{len(errors)} DRC error(s)",
            data={"violations": "\n    " + "\n    ".join(lines)},
        )

    # -- internals ---------------------------------------------------------

    def _design_rules(self):
        if not hasattr(self.bridge, "get_design_rules"):
            return None
        return self.bridge.get_design_rules()

    def _validate_via_rules(self) -> ToolResult | None:
        rules = self._design_rules()
        if rules is None:
            return None
        if hasattr(rules, "min_via_diameter") and hasattr(rules, "via_diameter"):
            if rules.via_diameter < rules.min_via_diameter:
                return ToolResult(
                    ok=False,
                    error_code=ErrorCode.VIOLATES_DESIGN_RULE,
                    message=(
                        f"configured via diameter ({_mm(rules.via_diameter):.3f}mm) is "
                        f"below board minimum ({_mm(rules.min_via_diameter):.3f}mm). "
                        f"Check get_board_info()."
                    ),
                    data={
                        "via_diameter_mm": _mm(rules.via_diameter),
                        "min_via_diameter_mm": _mm(rules.min_via_diameter),
                    },
                )
        if hasattr(rules, "min_via_drill") and hasattr(rules, "via_drill"):
            if rules.via_drill < rules.min_via_drill:
                return ToolResult(
                    ok=False,
                    error_code=ErrorCode.VIOLATES_DESIGN_RULE,
                    message=(
                        f"configured via drill ({_mm(rules.via_drill):.3f}mm) is "
                        f"below board minimum ({_mm(rules.min_via_drill):.3f}mm). "
                        f"Check get_board_info()."
                    ),
                    data={
                        "via_drill_mm": _mm(rules.via_drill),
                        "min_via_drill_mm": _mm(rules.min_via_drill),
                    },
                )
        return None

    def _clear_route(self) -> None:
        self._active_net = None
        self._target_xy_nm = None
        self._target_item_id = -1
        self._requested_mm = None

    def _pad_item_id(self, x_nm: int, y_nm: int) -> int | None:
        """Resolves a pad to a router item id, preferring an actual 'pad'
        hit. query_hover_items() also returns tracks/vias near the point,
        and handing fix() an unrelated track's id makes it refuse for
        reasons that look like a collision and are not (see
        measure_waypoint_fidelity.py's _pick_pad_candidate)."""
        candidates = self.bridge.query_hover_items(
            x_nm, y_nm, layer=0, slop_radius=SNAP_RADIUS_NM
        )
        if not candidates:
            return None

        for candidate in candidates:
            if candidate.kind == "pad":
                return candidate.id

        return candidates[0].id

    def _head_position_mm(self) -> tuple[float, float]:
        """Where the head actually is, preferring the router's own answer
        over the point we asked for."""
        if self.has_head_readback:
            head = self.bridge.get_head_geometry()
            if head.active:
                return (_mm(head.end_x), _mm(head.end_y))

        return self._requested_mm or (0.0, 0.0)

    def _distance_to_target_mm(self) -> float:
        if self._target_xy_nm is None:
            return 0.0
        head = self._head_position_mm()
        return math.hypot(
            head[0] - _mm(self._target_xy_nm[0]),
            head[1] - _mm(self._target_xy_nm[1]),
        )

    def _validate_point(self, x_mm: float, y_mm: float) -> ToolResult | None:
        """Argument checks that must happen BEFORE the router is touched."""
        for name, value in (("x_mm", x_mm), ("y_mm", y_mm)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return ToolResult(
                    ok=False,
                    error_code=ErrorCode.BAD_COORDINATE,
                    message=f"{name} must be a number in millimetres, got {value!r}.",
                )
            if not math.isfinite(float(value)):
                return ToolResult(
                    ok=False,
                    error_code=ErrorCode.BAD_COORDINATE,
                    message=f"{name} must be finite, got {value!r}.",
                )

        if not (0.0 <= x_mm <= self.board_width_mm) or not (
            0.0 <= y_mm <= self.board_height_mm
        ):
            return ToolResult(
                ok=False,
                error_code=ErrorCode.OUT_OF_BOUNDS,
                message=(
                    f"({x_mm:.3f}, {y_mm:.3f}) is outside the board. Valid range "
                    f"is x 0-{self.board_width_mm:.3f}mm, "
                    f"y 0-{self.board_height_mm:.3f}mm. Note these are "
                    f"MILLIMETRES, not nanometres."
                ),
            )

        return None


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_board_info",
            "description": "Returns board dimensions and design rules (track width, via sizes, clearance limits in mm).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_nets",
            "description": "Lists all nets on the board, their pad coordinates (in mm), and routing status.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_route",
            "description": "Opens a routing session for a given net, starting at its first pad.",
            "parameters": {
                "type": "object",
                "properties": {
                    "net": {
                        "type": "string",
                        "description": "The exact name of the net to route.",
                    }
                },
                "required": ["net"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_to",
            "description": "Moves the active routing head toward (x_mm, y_mm) on the board.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_mm": {
                        "type": "number",
                        "description": "Target X coordinate in millimetres.",
                    },
                    "y_mm": {
                        "type": "number",
                        "description": "Target Y coordinate in millimetres.",
                    },
                },
                "required": ["x_mm", "y_mm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_via",
            "description": "Places a via at the current head position to toggle between layer 0 (F_Cu) and layer 1 (B_Cu).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_to_layer",
            "description": "Switches the active routing layer (0 for F_Cu, 1 for B_Cu), placing a via at current position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer": {
                        "type": "integer",
                        "description": "Layer index: 0 for F_Cu (top), 1 for B_Cu (bottom).",
                    }
                },
                "required": ["layer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_route",
            "description": "Connects the routing head to the target pad and commits the route. Head must be on layer 0 (F_Cu).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abandon_route",
            "description": "Drops the in-progress route without committing it.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rip_up",
            "description": "Removes all copper traces and vias for a previously routed net.",
            "parameters": {
                "type": "object",
                "properties": {
                    "net": {
                        "type": "string",
                        "description": "The exact name of the net to rip up.",
                    }
                },
                "required": ["net"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_drc",
            "description": "Runs Design Rule Checking across the board and returns all clearance/connectivity violations.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
