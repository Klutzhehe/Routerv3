"""Model D / Phase 3: Bulk Single-Ended Router with Reservation Zone Avoidance.

Routes general non-critical nets (GPIO, SPI, I2C, control, power/reset) using fast
obstacle and meander-reservation avoidance pathfinding.
"""

from __future__ import annotations

import math
from typing import List, Dict, Tuple, Optional, Any

from pcbworld.hierarchical.bridge_util import describe_contact, pad_candidate, push_path
from pcbworld.hierarchical.specs import (
    ReservationZone,
    RouteResult,
    PipelinePhase,
    PadInfo,
)

MODE_ROUTE_SINGLE = 1


class BulkRouter:
    """Routes bulk single-ended nets while steering clear of reserved meander zones."""

    def __init__(self, bridge: Any):
        self.bridge = bridge

    def route_net(
        self,
        net_name: str,
        pads: List[PadInfo],
        reservation_zones: List[ReservationZone],
        layer: int = 0,
        planner: Optional[Any] = None,
        obstacles: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> RouteResult:
        """Routes a single-ended bulk net, actively detour-routing around reservation zones."""
        if len(pads) < 2:
            return RouteResult(
                net_name=net_name,
                success=False,
                phase=PipelinePhase.PHASE_3_BULK_ROUTING,
                error_message=f"Net {net_name} has fewer than 2 pads",
            )

        start_pad = pads[0]
        target_pad = pads[1]

        # Try routing on preferred layer, then fallback to bottom layer (layer 31/1) if blocked
        candidate_layers = [layer]
        if layer == 0:
            candidate_layers.append(31)

        last_error = ""

        for curr_layer in candidate_layers:
            # 1. Compute collision-free detour waypoints
            # `is not None`, not truthiness: an empty obstacle list is a
            # board with nothing in the way, which the planner answers
            # with [] in about a millisecond. Treating it as "no planner"
            # silently downgraded to the reservation-zone-only detour.
            if planner is not None:
                waypoints = planner.plan_corridor(
                    (start_pad.x, start_pad.y),
                    (target_pad.x, target_pad.y),
                    obstacles,
                    reservation_zones=reservation_zones,
                    is_diff_pair=False,
                    layer=curr_layer,
                )
            else:
                waypoints = self._compute_detour_waypoints(
                    start_pad.x, start_pad.y, target_pad.x, target_pad.y, curr_layer, reservation_zones
                )

            # 2. Configure Bridge
            if hasattr(self.bridge, "set_mode"):
                self.bridge.set_mode(MODE_ROUTE_SINGLE)

            start_id = pad_candidate(self.bridge, start_pad.x, start_pad.y, curr_layer)
            target_id = pad_candidate(self.bridge, target_pad.x, target_pad.y, curr_layer)

            started = self.bridge.start_route(start_pad.x, start_pad.y, start_id, curr_layer)
            if not started:
                last_error = f"start_route failed on layer {curr_layer}"
                continue

            # 3. Push the corridor, noting where (if anywhere) it clipped
            contact = push_path(self.bridge, waypoints, (target_pad.x, target_pad.y))

            # 4. Fix & Commit
            fixed = self.bridge.fix(target_pad.x, target_pad.y, target_id, True, True)
            if fixed:
                self.bridge.commit_routing()
                wirelength_nm = int(math.hypot(target_pad.x - start_pad.x, target_pad.y - start_pad.y))
                if hasattr(self.bridge, "get_head_geometry"):
                    hg = self.bridge.get_head_geometry()
                    if hg and hasattr(hg, "length") and hg.length > 0:
                        wirelength_nm = int(hg.length)

                return RouteResult(
                    net_name=net_name,
                    success=True,
                    phase=PipelinePhase.PHASE_3_BULK_ROUTING,
                    wirelength_nm=wirelength_nm,
                    num_segments=len(waypoints) + 1,
                    drc_clean=True,
                )
            else:
                if hasattr(self.bridge, "stop_routing"):
                    self.bridge.stop_routing()
                detail = describe_contact(contact)
                last_error = (
                    f"fix() failed on layer {curr_layer}"
                    + (f": {detail}" if detail else " with a clean head")
                )

        return RouteResult(
            net_name=net_name,
            success=False,
            phase=PipelinePhase.PHASE_3_BULK_ROUTING,
            error_message=last_error or f"Routing failed for {net_name}",
        )

    def _compute_detour_waypoints(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        layer: int,
        reservation_zones: List[ReservationZone],
    ) -> List[Tuple[int, int]]:
        """Calculates intermediate waypoints around any active reservation zone."""
        waypoints: List[Tuple[int, int]] = []
        margin_nm = 500_000  # 0.5 mm clearance around reservation envelope

        for zone in reservation_zones:
            if not zone.active or zone.layer != layer:
                continue

            if zone.intersects_segment(x1, y1, x2, y2, layer):
                x_min, y_min, x_max, y_max = zone.bbox_nm

                # Pick shortest detour (above/below or left/right)
                detour_options = [
                    # Top detour
                    ((x1 + x2) // 2, y_max + margin_nm),
                    # Bottom detour
                    ((x1 + x2) // 2, y_min - margin_nm),
                    # Left detour
                    (x_min - margin_nm, (y1 + y2) // 2),
                    # Right detour
                    (x_max + margin_nm, (y1 + y2) // 2),
                ]

                # Pick option that adds minimal total detour distance
                best_pt = min(
                    detour_options,
                    key=lambda pt: math.hypot(pt[0] - x1, pt[1] - y1) + math.hypot(x2 - pt[0], y2 - pt[1]),
                )
                waypoints.append(best_pt)

        return waypoints
