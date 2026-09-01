"""Model C / Phase 2: Synchronous Bus Routing & Meander Reservation Manager.

Routes baseline tracks for all nets in a synchronous group (e.g. DDR byte lane),
calculates the length delta ΔL relative to the longest reference net, and synthesizes
spatial ReservationZone bounding boxes to safeguard room for Phase 4 meander expansion.
"""

from __future__ import annotations

import math
from typing import List, Dict, Tuple, Optional, Any

from pcbworld.hierarchical.bridge_util import pad_candidate
from pcbworld.hierarchical.specs import (
    LengthGroupSpec,
    ReservationZone,
    RouteResult,
    PipelinePhase,
    PadInfo,
)

MODE_ROUTE_SINGLE = 1
MODE_TUNE_SINGLE = 3


class BusBundleRouter:
    """Routes synchronous bus groups and creates meander space reservation envelopes."""

    def __init__(
        self,
        bridge: Any,
        default_max_amplitude_nm: int = 2_500_000,  # 2.5 mm
        default_meander_spacing_nm: int = 1_000_000, # 1.0 mm
    ):
        self.bridge = bridge
        self.default_max_amplitude_nm = default_max_amplitude_nm
        self.default_meander_spacing_nm = default_meander_spacing_nm

    def route_group_baseline(
        self,
        spec: LengthGroupSpec,
        net_to_pads: Dict[str, List[PadInfo]],
        planner: Optional[Any] = None,
        obstacles: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[List[RouteResult], List[ReservationZone]]:
        """Routes baseline straight tracks for all member nets in the length group,
        and constructs ReservationZone bounding boxes for shorter traces."""
        results: List[RouteResult] = []
        reservation_zones: List[ReservationZone] = []

        if hasattr(self.bridge, "set_mode"):
            self.bridge.set_mode(MODE_ROUTE_SINGLE)

        # 1. Route each member net straight baseline
        for net_name in spec.member_nets:
            pads = net_to_pads.get(net_name, [])
            if len(pads) < 2:
                results.append(
                    RouteResult(
                        net_name=net_name,
                        success=False,
                        phase=PipelinePhase.PHASE_2_BUS_RESERVATION,
                        error_message=f"Length group member {net_name} has fewer than 2 pads",
                    )
                )
                continue

            start_pad = pads[0]
            target_pad = pads[1]

            start_id = pad_candidate(self.bridge, start_pad.x, start_pad.y, spec.assigned_layer)
            target_id = pad_candidate(self.bridge, target_pad.x, target_pad.y, spec.assigned_layer)

            started = self.bridge.start_route(start_pad.x, start_pad.y, start_id, spec.assigned_layer)
            if not started:
                results.append(
                    RouteResult(
                        net_name=net_name,
                        success=False,
                        phase=PipelinePhase.PHASE_2_BUS_RESERVATION,
                        error_message=f"start_route failed for {net_name}",
                    )
                )
                continue

            # Push intermediate avoidance waypoints if planner is provided
            # See bulk_router: an empty obstacle list means an empty
            # board, not an absent planner.
            if planner is not None:
                waypoints = planner.plan_corridor((start_pad.x, start_pad.y), (target_pad.x, target_pad.y), obstacles, layer=spec.assigned_layer)
                for wx, wy in waypoints:
                    self.bridge.push(wx, wy)

            self.bridge.push(target_pad.x, target_pad.y)
            fixed = self.bridge.fix(target_pad.x, target_pad.y, target_id, True, True)
            if not fixed:
                if hasattr(self.bridge, "stop_routing"):
                    self.bridge.stop_routing()
                results.append(
                    RouteResult(
                        net_name=net_name,
                        success=False,
                        phase=PipelinePhase.PHASE_2_BUS_RESERVATION,
                        error_message=f"fix() failed for {net_name}",
                    )
                )
                continue

            self.bridge.commit_routing()

            # Measure baseline routed length
            length_nm = int(math.hypot(target_pad.x - start_pad.x, target_pad.y - start_pad.y))
            if hasattr(self.bridge, "get_head_geometry"):
                hg = self.bridge.get_head_geometry()
                if hg and hasattr(hg, "length") and hg.length > 0:
                    length_nm = int(hg.length)

            spec.routed_lengths[net_name] = length_nm
            results.append(
                RouteResult(
                    net_name=net_name,
                    success=True,
                    phase=PipelinePhase.PHASE_2_BUS_RESERVATION,
                    wirelength_nm=length_nm,
                    num_segments=1,
                    drc_clean=True,
                )
            )

        # 2. Determine reference length L_ref
        if spec.routed_lengths:
            max_len = max(spec.routed_lengths.values())
            spec.target_length_nm = max_len

            # 3. Create Reservation Zones for nets needing meander growth
            for net_name, length_nm in spec.routed_lengths.items():
                delta_l = max_len - length_nm
                if delta_l > spec.target_tolerance_nm:
                    # Space needed: bounding box around trace midpoint
                    pads = net_to_pads.get(net_name, [])
                    if len(pads) >= 2:
                        p1, p2 = pads[0], pads[1]
                        mid_x = (p1.x + p2.x) // 2
                        mid_y = (p1.y + p2.y) // 2

                        # Estimate reservation footprint
                        # Meander expands perpendicular to the trace
                        dx = p2.x - p1.x
                        dy = p2.y - p1.y
                        track_len = math.hypot(dx, dy)
                        if track_len > 0:
                            # Width along trace ~ meander span, Height perp ~ amplitude
                            span_along_track = min(track_len * 0.8, delta_l * 0.8)
                            half_span = int(span_along_track / 2)
                            half_amp = self.default_max_amplitude_nm

                            x_min = int(mid_x - max(half_span, half_amp))
                            x_max = int(mid_x + max(half_span, half_amp))
                            y_min = int(mid_y - max(half_span, half_amp))
                            y_max = int(mid_y + max(half_span, half_amp))

                            zone = ReservationZone(
                                zone_id=f"res_{spec.group_id}_{net_name}",
                                owner_net=net_name,
                                bbox_nm=(x_min, y_min, x_max, y_max),
                                layer=spec.assigned_layer,
                                created_in_phase=PipelinePhase.PHASE_2_BUS_RESERVATION,
                                active=True,
                            )
                            reservation_zones.append(zone)

        return results, reservation_zones
