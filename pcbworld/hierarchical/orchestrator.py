"""Master Pipeline Orchestrator for Hierarchical Multi-Model PCB Routing.

Executes Phases 0 through 4 end-to-end:
- Phase 0: Netlist classification & schedule planning
- Phase 1: High-Speed Diff-Pairs (PCIe/Ethernet)
- Phase 2: Synchronous Bus Baseline & Meander Reservation (DDR)
- Phase 3: Bulk Single-Ended Routing (GPIO/Control)
- Phase 3.5: Negotiated Congestion & Rip-Up Arbitration
- Phase 4: Active Meander Expansion & Final DRC Polish
"""

from __future__ import annotations

import math
from typing import List, Dict, Tuple, Optional, Any, Set

from pcbworld.hierarchical.bridge_util import pad_candidate
from pcbworld.hierarchical.specs import (
    NetTier,
    PipelinePhase,
    PadInfo,
    DiffPairSpec,
    LengthGroupSpec,
    ReservationZone,
    RouteResult,
    HierarchicalPipelineReport,
)
from pcbworld.hierarchical.scheduler import ConstraintScheduler
from pcbworld.hierarchical.diff_pair_router import DiffPairRouter, MODE_TUNE_SINGLE, MODE_TUNE_DIFF_PAIR_SKEW
from pcbworld.hierarchical.bus_bundle_router import BusBundleRouter
from pcbworld.hierarchical.bulk_router import BulkRouter
from pcbworld.hierarchical.ripup_arbitrator import RipUpArbitrator
from pcbworld.hierarchical.spatial_corridor_planner import SpatialCorridorPlanner
from pcbworld.hierarchical.escape_router import ConcurrentEscapeRouter, EscapeStub
from pcbworld.env.line_obs import KIND_PAD, Segment, board_segments, pad_to_segment


class HierarchicalOrchestrator:
    """End-to-end orchestrator for the 5-phase hierarchical PCB routing pipeline."""

    def __init__(
        self,
        bridge: Any,
        scheduler: Optional[ConstraintScheduler] = None,
        max_ripup_iterations: int = 5,
        planner: Optional[SpatialCorridorPlanner] = None,
        escape_router: Optional[ConcurrentEscapeRouter] = None,
    ):
        self.bridge = bridge
        self.scheduler = scheduler or ConstraintScheduler()
        self.planner = planner or SpatialCorridorPlanner()
        self.escape_router = escape_router or ConcurrentEscapeRouter()
        self.diff_router = DiffPairRouter(bridge)
        self.bus_router = BusBundleRouter(bridge)
        self.bulk_router = BulkRouter(bridge)
        self.arbitrator = RipUpArbitrator(bridge)
        self.max_ripup_iterations = max_ripup_iterations

    def obstacle_segments(self, own_nets: Optional[Set[str]] = None) -> List[Segment]:
        """Everything currently in the way, as real segment geometry.

        Read fresh from `get_board_geometry()` (~0.13 ms) rather than
        accumulated in a list. The accumulating version had two defects that
        between them explain most of this pipeline's obstacle-avoidance
        failures:

        * It stored each track's axis-aligned BOUNDING BOX. A 45-degree trace
          across a 20 mm span became a 20x20 mm solid block, so the nets this
          pipeline deliberately routes last were planning around a board that
          was mostly fictional copper.

        * It re-appended EVERY track on the board after EVERY successful net,
          so the list grew quadratically -- 24 nets left ~300 duplicate boxes,
          each contributing eight nav nodes to a visibility graph whose search
          is quadratic in nodes.

        `own_nets` are the endpoints of the route being planned: a pad the
        route has to LAND on is a target, not an obstacle, and leaving it in
        makes its own destination unreachable.
        """
        own = own_nets or set()
        geometry = self.bridge.get_board_geometry()
        segments = board_segments(geometry)
        for pad in geometry.pads:
            if pad.net not in own:
                segments.append(pad_to_segment(pad, kind=KIND_PAD))
        return segments

    def route_board(self) -> HierarchicalPipelineReport:
        """Executes the full 5-phase hierarchical routing pipeline on the loaded board."""
        report = HierarchicalPipelineReport()

        # -------------------------------------------------------------
        # PHASE 0: Netlist Analysis & Stackup Planning
        # -------------------------------------------------------------
        raw_pads = self.bridge.net_pads() if hasattr(self.bridge, "net_pads") else []
        diff_specs, len_specs, analog_nets, bulk_nets, net_to_pads = self.scheduler.analyze_board(raw_pads)

        report.total_nets = len(net_to_pads)
        report.diff_pairs_total = len(diff_specs)
        report.length_groups_total = len(len_specs)
        report.bulk_nets_total = len(analog_nets) + len(bulk_nets)

        net_tiers: Dict[str, NetTier] = {}
        for dp in diff_specs:
            net_tiers[dp.p_net] = NetTier.DIFF_PAIR
            net_tiers[dp.n_net] = NetTier.DIFF_PAIR
        for lg in len_specs:
            for m in lg.member_nets:
                net_tiers[m] = NetTier.LENGTH_GROUP
        for a in analog_nets:
            net_tiers[a] = NetTier.SENSITIVE_ANALOG
        for b in bulk_nets:
            net_tiers[b] = NetTier.BULK_DIGITAL

        committed_nets: Set[str] = set()

        # Obstacles are read from the board when a route is planned, not
        # accumulated here -- see obstacle_segments(). A pad's real size comes
        # from get_board_geometry(); the 0.5 mm radius this used to assume was
        # wrong in both directions on a board with mixed footprints.

        # -------------------------------------------------------------
        # PHASE 0.5: Concurrent Pin Escape (Dense Cluster Fanout)
        # -------------------------------------------------------------
        all_pad_objs: List[PadInfo] = []
        target_lookup: Dict[str, Tuple[int, int]] = {}
        for net, pads in net_to_pads.items():
            all_pad_objs.extend(pads)
            if len(pads) >= 2:
                target_lookup[net] = (pads[1].x, pads[1].y)

        escape_map = self.escape_router.escape_all_dense_clusters(all_pad_objs, target_lookup=target_lookup)

        # -------------------------------------------------------------
        # PHASE 1: High-Speed Serial Routing (Differential Pairs)
        # -------------------------------------------------------------
        p1_results = []
        for dp in diff_specs:
            if len(dp.p_pads) >= 2:
                p1, p2 = dp.p_pads[0], dp.p_pads[1]
                waypoints = self.planner.plan_corridor(
                    (p1.x, p1.y),
                    (p2.x, p2.y),
                    obstacles=self.obstacle_segments({dp.p_net, dp.n_net}),
                    is_diff_pair=True,
                    layer=dp.assigned_layer,
                )
            else:
                waypoints = []

            res = self.diff_router.route_pair(dp, waypoints=waypoints)
            p1_results.append(res)
            if res.success:
                committed_nets.add(dp.p_net)
                committed_nets.add(dp.n_net)
                report.diff_pairs_routed += 1
                report.routed_nets += 2

        report.phase_results[PipelinePhase.PHASE_1_DIFF_PAIR] = p1_results

        # -------------------------------------------------------------
        # PHASE 2: Synchronous Bus Routing & Meander Reservation
        # -------------------------------------------------------------
        p2_results = []
        all_reservation_zones: List[ReservationZone] = []
        for lg in len_specs:
            group_results, zones = self.bus_router.route_group_baseline(
                lg, net_to_pads, planner=self.planner,
                obstacles=self.obstacle_segments(set(lg.member_nets)),
            )
            p2_results.extend(group_results)
            all_reservation_zones.extend(zones)
            for res in group_results:
                if res.success:
                    committed_nets.add(res.net_name)
                    report.routed_nets += 1

        report.phase_results[PipelinePhase.PHASE_2_BUS_RESERVATION] = p2_results

        # -------------------------------------------------------------
        # PHASE 3: Bulk Single-Ended Routing (Avoiding Reservations)
        # -------------------------------------------------------------
        p3_results = []
        unrouted_bulk = list(analog_nets) + list(bulk_nets)
        ripup_iter = 0

        while unrouted_bulk and ripup_iter <= self.max_ripup_iterations:
            net_name = unrouted_bulk.pop(0)
            pads = net_to_pads.get(net_name, [])
            res = self.bulk_router.route_net(
                net_name,
                pads,
                all_reservation_zones,
                layer=0,
                planner=self.planner,
                obstacles=self.obstacle_segments({net_name}),
            )

            if res.success:
                committed_nets.add(net_name)
                report.routed_nets += 1
                report.bulk_nets_routed += 1
                p3_results.append(res)
            else:
                # Phase 3.5: Negotiated Congestion & Rip-Up
                ripup_iter += 1
                report.ripup_count += 1
                victim = self.arbitrator.select_victim(
                    failed_net=net_name,
                    failed_tier=net_tiers.get(net_name, NetTier.BULK_DIGITAL),
                    committed_nets=committed_nets,
                    net_tiers=net_tiers,
                )
                if victim:
                    self.arbitrator.execute_ripup(victim)
                    committed_nets.discard(victim)
                    report.routed_nets -= 1
                    if victim in analog_nets or victim in bulk_nets:
                        report.bulk_nets_routed -= 1
                    # Re-queue both victim and failed net
                    unrouted_bulk.append(victim)
                    unrouted_bulk.append(net_name)
                else:
                    p3_results.append(res)

        report.phase_results[PipelinePhase.PHASE_3_BULK_ROUTING] = p3_results

        # -------------------------------------------------------------
        # PHASE 4: Active Meander Expansion & Final Length Tuning Polish
        # -------------------------------------------------------------
        p4_results = []
        # Deactivate reservation zones as we are now expanding meanders directly into them
        for zone in all_reservation_zones:
            zone.active = False

        for lg in len_specs:
            if not lg.target_length_nm:
                continue

            for member_net in lg.member_nets:
                current_len = lg.routed_lengths.get(member_net, 0)
                delta_l = lg.target_length_nm - current_len

                if delta_l > lg.target_tolerance_nm:
                    # Re-open trace and tune length with native PNS meander placer
                    pads = net_to_pads.get(member_net, [])
                    if len(pads) >= 2:
                        start_pad, target_pad = pads[0], pads[1]
                        if hasattr(self.bridge, "set_mode"):
                            self.bridge.set_mode(MODE_TUNE_SINGLE)
                        if hasattr(self.bridge, "set_target_length"):
                            self.bridge.set_target_length(lg.target_length_nm)

                        start_id = pad_candidate(self.bridge, start_pad.x, start_pad.y, lg.assigned_layer)
                        target_id = pad_candidate(self.bridge, target_pad.x, target_pad.y, lg.assigned_layer)

                        started = self.bridge.start_route(start_pad.x, start_pad.y, start_id, lg.assigned_layer)
                        if started:
                            self.bridge.push(target_pad.x, target_pad.y)
                            fixed = self.bridge.fix(target_pad.x, target_pad.y, target_id, True, True)
                            if fixed:
                                self.bridge.commit_routing()
                                p4_results.append(
                                    RouteResult(
                                        net_name=member_net,
                                        success=True,
                                        phase=PipelinePhase.PHASE_4_MEANDER_TUNING,
                                        wirelength_nm=lg.target_length_nm,
                                        length_mismatch_nm=0,
                                        drc_clean=True,
                                    )
                                )
                                continue

                    p4_results.append(
                        RouteResult(
                            net_name=member_net,
                            success=False,
                            phase=PipelinePhase.PHASE_4_MEANDER_TUNING,
                            error_message="Meander tuning failed",
                        )
                    )
                else:
                    p4_results.append(
                        RouteResult(
                            net_name=member_net,
                            success=True,
                            phase=PipelinePhase.PHASE_4_MEANDER_TUNING,
                            wirelength_nm=current_len,
                            length_mismatch_nm=delta_l,
                            drc_clean=True,
                        )
                    )
            report.length_groups_tuned += 1

        report.phase_results[PipelinePhase.PHASE_4_MEANDER_TUNING] = p4_results

        # -------------------------------------------------------------
        # FINAL DRC VALIDATION
        # -------------------------------------------------------------
        if hasattr(self.bridge, "run_drc"):
            drc_violations = self.bridge.run_drc()
            report.drc_violations = len(drc_violations) if drc_violations else 0

        report.all_clean = (report.routed_nets == report.total_nets) and (report.drc_violations == 0)
        return report
