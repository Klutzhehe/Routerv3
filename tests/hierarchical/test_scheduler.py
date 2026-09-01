"""Unit tests for Model A: ConstraintScheduler."""

import pytest
from pcbworld.hierarchical.scheduler import ConstraintScheduler
from pcbworld.hierarchical.specs import PadInfo, NetTier


def test_scheduler_classification():
    scheduler = ConstraintScheduler()

    # Create dummy pad fixtures
    pads = [
        # Diff pair 0
        PadInfo(net="diffpair_0_P", pad_name="J1:1", x=0, y=0, layer=0),
        PadInfo(net="diffpair_0_P", pad_name="J2:1", x=20_000_000, y=0, layer=0),
        PadInfo(net="diffpair_0_N", pad_name="J1:2", x=0, y=1_000_000, layer=0),
        PadInfo(net="diffpair_0_N", pad_name="J2:2", x=20_000_000, y=1_000_000, layer=0),
        # Length group 0 (DDR Byte 0)
        PadInfo(net="lengthgrp_0_0", pad_name="U1:1", x=10_000_000, y=10_000_000, layer=0),
        PadInfo(net="lengthgrp_0_0", pad_name="U2:1", x=30_000_000, y=10_000_000, layer=0),
        PadInfo(net="lengthgrp_0_1", pad_name="U1:2", x=10_000_000, y=12_000_000, layer=0),
        PadInfo(net="lengthgrp_0_1", pad_name="U2:2", x=25_000_000, y=12_000_000, layer=0),
        # Sensitive analog
        PadInfo(net="analog_rf_in", pad_name="A1:1", x=0, y=30_000_000, layer=0),
        PadInfo(net="analog_rf_in", pad_name="A2:1", x=10_000_000, y=30_000_000, layer=0),
        # Bulk digital
        PadInfo(net="net_gpio_0", pad_name="G1:1", x=0, y=40_000_000, layer=0),
        PadInfo(net="net_gpio_0", pad_name="G2:1", x=15_000_000, y=40_000_000, layer=0),
    ]

    diff_specs, len_specs, analog_nets, bulk_nets, net_to_pads = scheduler.analyze_board(pads)

    assert len(diff_specs) == 1
    assert diff_specs[0].pair_id == "diffpair_0"
    assert diff_specs[0].p_net == "diffpair_0_P"
    assert diff_specs[0].n_net == "diffpair_0_N"

    assert len(len_specs) == 1
    assert len_specs[0].group_id == "0"
    assert len_specs[0].member_nets == ["lengthgrp_0_0", "lengthgrp_0_1"]
    assert len_specs[0].reference_net == "lengthgrp_0_0"

    assert analog_nets == ["analog_rf_in"]
    assert bulk_nets == ["net_gpio_0"]
    assert len(net_to_pads) == 6
