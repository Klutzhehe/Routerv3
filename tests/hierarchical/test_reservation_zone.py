"""Unit tests for ReservationZone spatial intersection."""

from pcbworld.hierarchical.specs import ReservationZone, PipelinePhase


def test_reservation_zone_containment_and_intersection():
    zone = ReservationZone(
        zone_id="res_0",
        owner_net="net_test",
        bbox_nm=(10_000_000, 10_000_000, 20_000_000, 20_000_000),
        layer=0,
        active=True,
    )

    # Point containment
    assert zone.contains_point(15_000_000, 15_000_000, layer=0)
    assert not zone.contains_point(5_000_000, 15_000_000, layer=0)
    assert not zone.contains_point(15_000_000, 15_000_000, layer=1)  # Wrong layer

    # Segment intersection
    # Line passing through the zone
    assert zone.intersects_segment(0, 15_000_000, 30_000_000, 15_000_000, layer=0)
    # Line completely outside
    assert not zone.intersects_segment(0, 0, 5_000_000, 5_000_000, layer=0)
    # Line on another layer
    assert not zone.intersects_segment(0, 15_000_000, 30_000_000, 15_000_000, layer=1)

    # Deactivated zone
    zone.active = False
    assert not zone.contains_point(15_000_000, 15_000_000, layer=0)
    assert not zone.intersects_segment(0, 15_000_000, 30_000_000, 15_000_000, layer=0)
