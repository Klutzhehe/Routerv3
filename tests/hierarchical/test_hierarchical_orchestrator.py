"""End-to-end unit tests for the Hierarchical PCB Routing Pipeline."""

import pytest
from tests.fake_bridge import FakePNSBridge, NetPad, MM
from pcbworld.hierarchical.orchestrator import HierarchicalOrchestrator
from pcbworld.hierarchical.specs import PipelinePhase


def test_hierarchical_orchestrator_end_to_end():
    # Construct a mixed test board fixture:
    # 1 Diff Pair (diffpair_0_P, diffpair_0_N)
    # 1 Length-Matched Group with 2 members (lengthgrp_0_0, lengthgrp_0_1)
    # 2 Bulk Nets (net_gpio_0, net_gpio_1)
    nets = [
        # Diff pair 0
        NetPad("diffpair_0_P", "J1:1", 0, 0, 0),
        NetPad("diffpair_0_P", "J2:1", 20 * MM, 0, 0),
        NetPad("diffpair_0_N", "J1:2", 0, 1 * MM, 0),
        NetPad("diffpair_0_N", "J2:2", 20 * MM, 1 * MM, 0),
        # Length group 0 (DDR Byte Lane)
        NetPad("lengthgrp_0_0", "U1:1", 0, 10 * MM, 0),
        NetPad("lengthgrp_0_0", "U2:1", 25 * MM, 10 * MM, 0),  # 25 mm
        NetPad("lengthgrp_0_1", "U1:2", 0, 15 * MM, 0),
        NetPad("lengthgrp_0_1", "U2:2", 20 * MM, 15 * MM, 0),  # 20 mm (requires 5 mm tuning)
        # Bulk digital nets
        NetPad("net_gpio_0", "G1:1", 0, 25 * MM, 0),
        NetPad("net_gpio_0", "G2:1", 20 * MM, 25 * MM, 0),
        NetPad("net_gpio_1", "G1:2", 0, 30 * MM, 0),
        NetPad("net_gpio_1", "G2:2", 20 * MM, 30 * MM, 0),
    ]

    bridge = FakePNSBridge(nets=nets)
    orchestrator = HierarchicalOrchestrator(bridge=bridge)

    report = orchestrator.route_board()

    # Verify report statistics
    assert report.total_nets == 6
    assert report.diff_pairs_total == 1
    assert report.diff_pairs_routed == 1
    assert report.length_groups_total == 1
    assert report.length_groups_tuned == 1
    assert report.bulk_nets_total == 2
    assert report.bulk_nets_routed == 2
    assert report.routed_nets == 6

    # Verify Phase results
    assert len(report.phase_results[PipelinePhase.PHASE_1_DIFF_PAIR]) == 1
    assert report.phase_results[PipelinePhase.PHASE_1_DIFF_PAIR][0].success

    assert len(report.phase_results[PipelinePhase.PHASE_2_BUS_RESERVATION]) == 2
    assert all(r.success for r in report.phase_results[PipelinePhase.PHASE_2_BUS_RESERVATION])

    assert len(report.phase_results[PipelinePhase.PHASE_3_BULK_ROUTING]) == 2
    assert all(r.success for r in report.phase_results[PipelinePhase.PHASE_3_BULK_ROUTING])

    assert len(report.phase_results[PipelinePhase.PHASE_4_MEANDER_TUNING]) == 2
    assert all(r.success for r in report.phase_results[PipelinePhase.PHASE_4_MEANDER_TUNING])
