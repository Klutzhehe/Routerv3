"""Unit tests for ConcurrentEscapeRouter."""

import pytest
from pcbworld.hierarchical.specs import PadInfo
from pcbworld.hierarchical.escape_router import ConcurrentEscapeRouter


def test_concurrent_escape_router():
    router = ConcurrentEscapeRouter(
        cluster_threshold_nm=3_000_000,
        escape_distance_nm=2_000_000,
        step_size_nm=500_000,
        min_cluster_size=3,
    )

    # 4 dense pads in a square grid (BGA-like cluster centered around (10mm, 10mm))
    dense_pads = [
        PadInfo("net_bga_0", "U1:1", 9_000_000, 9_000_000, 0),
        PadInfo("net_bga_1", "U1:2", 11_000_000, 9_000_000, 0),
        PadInfo("net_bga_2", "U1:3", 9_000_000, 11_000_000, 0),
        PadInfo("net_bga_3", "U1:4", 11_000_000, 11_000_000, 0),
        # Isolated pad far away
        PadInfo("net_iso", "J1:1", 40_000_000, 40_000_000, 0),
    ]

    clusters = router.identify_clusters(dense_pads)
    assert len(clusters) == 1
    assert len(clusters[0]) == 4

    escape_map = router.escape_all_dense_clusters(dense_pads)
    assert len(escape_map) == 4
    assert "net_bga_0:U1:1" in escape_map
    assert "net_iso:J1:1" not in escape_map  # Isolated pad not in dense cluster

    stub_0 = escape_map["net_bga_0:U1:1"]
    # Bottom-left pad should fan out toward lower-left
    assert stub_0.escape_x < stub_0.orig_x
    assert stub_0.escape_y < stub_0.orig_y
    assert stub_0.step_count == 4
