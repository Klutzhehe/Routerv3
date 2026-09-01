"""Unit tests for individual routers: DiffPairRouter, BusBundleRouter, BulkRouter, RipUpArbitrator."""

import pytest
from tests.fake_bridge import FakePNSBridge, NetPad, MM
from pcbworld.hierarchical.specs import (
    PadInfo,
    DiffPairSpec,
    LengthGroupSpec,
    ReservationZone,
    NetTier,
    PipelinePhase,
)
from pcbworld.hierarchical.diff_pair_router import DiffPairRouter
from pcbworld.hierarchical.bus_bundle_router import BusBundleRouter
from pcbworld.hierarchical.bulk_router import BulkRouter
from pcbworld.hierarchical.ripup_arbitrator import RipUpArbitrator


def test_diff_pair_router():
    pads = [
        NetPad("diffpair_0_P", "J1:1", 0, 0, 0),
        NetPad("diffpair_0_P", "J2:1", 20 * MM, 0, 0),
        NetPad("diffpair_0_N", "J1:2", 0, 1 * MM, 0),
        NetPad("diffpair_0_N", "J2:2", 20 * MM, 1 * MM, 0),
    ]
    bridge = FakePNSBridge(nets=pads)
    router = DiffPairRouter(bridge)

    spec = DiffPairSpec(
        pair_id="diffpair_0",
        p_net="diffpair_0_P",
        n_net="diffpair_0_N",
        p_pads=[PadInfo("diffpair_0_P", "J1:1", 0, 0, 0), PadInfo("diffpair_0_P", "J2:1", 20 * MM, 0, 0)],
        n_pads=[PadInfo("diffpair_0_N", "J1:2", 0, 1 * MM, 0), PadInfo("diffpair_0_N", "J2:2", 20 * MM, 1 * MM, 0)],
    )

    res = router.route_pair(spec)
    assert res.success
    assert res.net_name == "diffpair_0"
    assert res.phase == PipelinePhase.PHASE_1_DIFF_PAIR


def test_bus_bundle_router_and_reservation():
    pads = [
        NetPad("lengthgrp_0_0", "U1:1", 0, 10 * MM, 0),
        NetPad("lengthgrp_0_0", "U2:1", 30 * MM, 10 * MM, 0),  # 30 mm
        NetPad("lengthgrp_0_1", "U1:2", 0, 15 * MM, 0),
        NetPad("lengthgrp_0_1", "U2:2", 20 * MM, 15 * MM, 0),  # 20 mm -> delta = 10 mm
    ]
    bridge = FakePNSBridge(nets=pads)
    router = BusBundleRouter(bridge)

    spec = LengthGroupSpec(
        group_id="0",
        member_nets=["lengthgrp_0_0", "lengthgrp_0_1"],
        reference_net="lengthgrp_0_0",
        target_tolerance_nm=250_000,
    )
    net_to_pads = {
        "lengthgrp_0_0": [PadInfo("lengthgrp_0_0", "U1:1", 0, 10 * MM, 0), PadInfo("lengthgrp_0_0", "U2:1", 30 * MM, 10 * MM, 0)],
        "lengthgrp_0_1": [PadInfo("lengthgrp_0_1", "U1:2", 0, 15 * MM, 0), PadInfo("lengthgrp_0_1", "U2:2", 20 * MM, 15 * MM, 0)],
    }

    results, zones = router.route_group_baseline(spec, net_to_pads)
    assert len(results) == 2
    assert all(r.success for r in results)
    assert spec.target_length_nm == 30 * MM
    # Expect 1 reservation zone created for lengthgrp_0_1
    assert len(zones) == 1
    assert zones[0].owner_net == "lengthgrp_0_1"
    assert zones[0].active


def test_bulk_router_detour():
    pads = [
        NetPad("net_gpio", "G1:1", 0, 15 * MM, 0),
        NetPad("net_gpio", "G2:1", 30 * MM, 15 * MM, 0),
    ]
    bridge = FakePNSBridge(nets=pads)
    router = BulkRouter(bridge)

    # Active reservation zone directly on the straight path (y = 15 mm)
    zone = ReservationZone(
        zone_id="res_0",
        owner_net="other_net",
        bbox_nm=(10 * MM, 12 * MM, 20 * MM, 18 * MM),
        layer=0,
        active=True,
    )

    net_pads = [PadInfo("net_gpio", "G1:1", 0, 15 * MM, 0), PadInfo("net_gpio", "G2:1", 30 * MM, 15 * MM, 0)]
    res = router.route_net("net_gpio", net_pads, [zone], layer=0)
    assert res.success
    # Waypoints added for detour
    assert res.num_segments >= 2


def test_ripup_arbitrator():
    bridge = FakePNSBridge()
    arbitrator = RipUpArbitrator(bridge, max_ripup_per_net=2)

    committed_nets = {"diffpair_0_P", "lengthgrp_0_0", "net_gpio_0"}
    net_tiers = {
        "diffpair_0_P": NetTier.DIFF_PAIR,
        "lengthgrp_0_0": NetTier.LENGTH_GROUP,
        "net_gpio_0": NetTier.BULK_DIGITAL,
    }

    # If a high-priority DiffPair fails, it should pick the lowest-priority net (net_gpio_0) as victim
    victim = arbitrator.select_victim("diffpair_1_P", NetTier.DIFF_PAIR, committed_nets, net_tiers)
    assert victim == "net_gpio_0"

    # If Bulk digital fails, it cannot rip up DiffPair or LengthGroup (only equal/lower priority)
    victim_for_bulk = arbitrator.select_victim("net_gpio_1", NetTier.BULK_DIGITAL, {"diffpair_0_P", "lengthgrp_0_0"}, net_tiers)
    assert victim_for_bulk is None
