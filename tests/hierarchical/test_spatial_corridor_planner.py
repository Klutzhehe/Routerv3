"""What the corridor planner has to get right about obstacles.

These pin behaviour, not implementation: every assertion is about the
geometry of the corridor that comes out, measured against the real segment
geometry of what is in the way. The previous planner passed no such test
because there was none -- it had unit tests for the scheduler, the routers
and the rip-up arbitrator, and none at all for the component whose whole job
is avoiding things.
"""

import math

import pytest

from pcbworld.env.line_obs import KIND_TRACK, Segment
from pcbworld.hierarchical.spatial_corridor_planner import (
    DIFF_PAIR_EXTRA_NM,
    SpatialCorridorPlanner,
    bbox_to_segment,
)
from pcbworld.hierarchical.specs import ReservationZone

MM = 1_000_000

# Track half-width (125 um) plus the design clearance (200 um). A corridor
# closer than this to copper is illegal, and PNS refuses to commit it.
LEGAL_GAP_NM = 325_000


def track(x1, y1, x2, y2, net="other", width=0.25 * MM) -> Segment:
    return Segment(
        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
        width=float(width), kind=KIND_TRACK, net=net, layer=0,
    )


def _point_seg_gap(px, py, seg: Segment) -> float:
    dx, dy = seg.x2 - seg.x1, seg.y2 - seg.y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, ((px - seg.x1) * dx + (py - seg.y1) * dy) / len_sq))
    cx, cy = seg.x1 + t * dx, seg.y1 + t * dy
    return math.hypot(px - cx, py - cy) - seg.width / 2.0


def min_gap(polyline, segments, samples: int = 60) -> float:
    """Closest the corridor ever comes to any obstacle's edge.

    Sampled along each leg rather than at the waypoints: a corridor whose
    waypoints are both clear can still cut a corner through copper between
    them, and that is the failure this is looking for."""
    worst = float("inf")
    for (ax, ay), (bx, by) in zip(polyline, polyline[1:]):
        for i in range(samples + 1):
            t = i / samples
            px, py = ax + t * (bx - ax), ay + t * (by - ay)
            for seg in segments:
                worst = min(worst, _point_seg_gap(px, py, seg))
    return worst


def corridor(planner, start, target, obstacles, **kw):
    """The full pushed polyline: start, the planner's waypoints, target."""
    return [start] + planner.plan_corridor(start, target, obstacles, **kw) + [target]


# -- the basics ---------------------------------------------------------


def test_an_unobstructed_route_gets_no_waypoints():
    """A detour nobody needs is wirelength nobody asked for."""
    planner = SpatialCorridorPlanner()
    assert planner.plan_corridor((2 * MM, 10 * MM), (20 * MM, 10 * MM), obstacles=[]) == []


def test_a_wall_across_the_route_is_routed_around_not_through():
    planner = SpatialCorridorPlanner()
    wall = [track(11 * MM, 2 * MM, 11 * MM, 18 * MM)]

    path = corridor(planner, (2 * MM, 10 * MM), (20 * MM, 10 * MM), wall)

    assert len(path) > 2, "should have produced at least one waypoint"
    assert min_gap(path, wall) >= LEGAL_GAP_NM, "corridor clips the wall"


def test_the_corridor_clears_the_gap_between_two_obstacles():
    """Two pads with a wide-enough channel between them: go through it."""
    planner = SpatialCorridorPlanner()
    obstacles = [
        track(11 * MM, 0 * MM, 11 * MM, 6 * MM),
        track(11 * MM, 14 * MM, 11 * MM, 20 * MM),
    ]

    path = corridor(planner, (2 * MM, 10 * MM), (20 * MM, 10 * MM), obstacles)

    assert min_gap(path, obstacles) >= LEGAL_GAP_NM
    # An 8 mm channel centred on the route is comfortably passable, so the
    # plan should take it rather than going the long way round either end.
    assert max(abs(y - 10 * MM) for _, y in path) < 4 * MM


# -- the bug this planner was rewritten for -----------------------------


def test_a_diagonal_track_does_not_block_its_whole_bounding_box():
    """The failure the bbox planner could not avoid.

    A 45-degree trace from (10,10) to (30,30) has a 20x20 mm bounding box.
    Treating that box as solid marks a fifth of a 50 mm board unroutable --
    and this pipeline routes the important nets FIRST, so every later net is
    planning around the boxes of the earlier ones.

    The route here runs well clear of the trace but squarely through its box.
    """
    planner = SpatialCorridorPlanner()
    diagonal = [track(10 * MM, 10 * MM, 30 * MM, 30 * MM)]

    start, target = (12 * MM, 26 * MM), (24 * MM, 38 * MM)

    # Precondition: the straight line really is legal -- it is parallel to the
    # trace, 10 mm away -- so any detour at all would be manufactured.
    assert min_gap([start, target], diagonal) >= LEGAL_GAP_NM

    # ... and it really does cross the bounding box, so a bbox planner sees it
    # as blocked.
    assert 10 * MM <= start[0] <= 30 * MM and 10 * MM <= start[1] <= 30 * MM

    assert planner.plan_corridor(start, target, obstacles=diagonal) == []


def test_a_start_inside_another_pads_clearance_still_gets_a_corridor():
    """In a dense cluster a pad sits inside its neighbour's inflated
    footprint. The visibility graph then has no legal first edge -- every
    move out of the start clips the box it is standing in -- and falls
    through to the straight line, so exactly the crowded case that needed a
    detour got none."""
    planner = SpatialCorridorPlanner()
    neighbour = track(2 * MM, 10.4 * MM, 2 * MM, 10.4 * MM, width=1.0 * MM)
    wall = track(11 * MM, 2 * MM, 11 * MM, 18 * MM)
    obstacles = [neighbour, wall]

    path = corridor(planner, (2 * MM, 10 * MM), (20 * MM, 10 * MM), obstacles)

    assert len(path) > 2, "a start inside copper still needs a plan"
    assert min_gap(path, [wall]) >= LEGAL_GAP_NM


# -- inputs and margins --------------------------------------------------


def test_legacy_bounding_box_obstacles_are_still_accepted():
    """The orchestrator now passes Segments, but bus_bundle_router and any
    caller outside this repo may still pass boxes."""
    planner = SpatialCorridorPlanner()
    box = (10 * MM, 8 * MM, 12 * MM, 12 * MM)

    waypoints = planner.plan_corridor((2 * MM, 10 * MM), (20 * MM, 10 * MM), obstacles=[box])

    assert waypoints, "a box squarely in the way must still produce a detour"
    assert min_gap(
        [(2 * MM, 10 * MM)] + waypoints + [(20 * MM, 10 * MM)], [bbox_to_segment(box)]
    ) >= LEGAL_GAP_NM


def test_bbox_to_segment_covers_the_rectangle_it_replaces():
    seg = bbox_to_segment((10 * MM, 8 * MM, 12 * MM, 12 * MM))
    # Spine along the long (y) axis, thickness the short (x) side.
    assert seg.x1 == seg.x2 == 11 * MM
    assert (seg.y1, seg.y2) == (8 * MM, 12 * MM)
    assert seg.width == 2 * MM
    # The rectangle's own centre is inside the capsule.
    assert _point_seg_gap(11 * MM, 10 * MM, seg) < 0


def test_a_reservation_zone_is_avoided_like_copper():
    planner = SpatialCorridorPlanner()
    zone = ReservationZone(
        zone_id="z0", owner_net="lengthgrp_0_1",
        bbox_nm=(9 * MM, 6 * MM, 13 * MM, 14 * MM), layer=0,
    )

    waypoints = planner.plan_corridor(
        (2 * MM, 10 * MM), (20 * MM, 10 * MM), obstacles=[], reservation_zones=[zone]
    )

    assert waypoints, "an active reservation zone across the route must be detoured"
    path = [(2 * MM, 10 * MM)] + waypoints + [(20 * MM, 10 * MM)]
    assert min_gap(path, [bbox_to_segment(zone.bbox_nm)]) >= LEGAL_GAP_NM


def test_an_inactive_reservation_zone_is_ignored():
    """Phase 4 deactivates the zones precisely so meanders can expand into
    them; a planner that kept avoiding them would leave the meander nowhere
    to go."""
    planner = SpatialCorridorPlanner()
    zone = ReservationZone(
        zone_id="z0", owner_net="lengthgrp_0_1",
        bbox_nm=(9 * MM, 6 * MM, 13 * MM, 14 * MM), layer=0, active=False,
    )
    assert planner.plan_corridor(
        (2 * MM, 10 * MM), (20 * MM, 10 * MM), obstacles=[], reservation_zones=[zone]
    ) == []


def test_a_diff_pair_is_planned_with_a_wider_margin_than_a_single_track():
    """A coupled pair is two tracks and the gap between them. A corridor
    planned at single-ended width fits the centreline and not the pair."""
    planner = SpatialCorridorPlanner()
    obstacles = [
        track(11 * MM, 0 * MM, 11 * MM, 8.6 * MM),
        track(11 * MM, 11.4 * MM, 11 * MM, 20 * MM),
    ]
    start, target = (2 * MM, 10 * MM), (20 * MM, 10 * MM)

    single = corridor(planner, start, target, obstacles)
    pair = corridor(planner, start, target, obstacles, is_diff_pair=True)

    single_gap = min_gap(single, obstacles)
    pair_gap = min_gap(pair, obstacles)

    assert pair_gap >= single_gap, "the pair's corridor must not be tighter"
    assert pair_gap >= LEGAL_GAP_NM + DIFF_PAIR_EXTRA_NM * 0.5


# -- the plan has to be followable --------------------------------------


def test_the_plan_declines_a_channel_too_tight_to_be_followed():
    """The margin is legal clearance PLUS the follower's tracking error.

    Measured on 25 Colab boards: a plan at the 325 um legal minimum threads
    corridors where the correct path is legal and everything beside it is
    not, the head tracks it to within about half a step, and PNS then refuses
    the whole route. 700 um was the swept peak. So a channel that is legal but
    only just should be declined, not taken.
    """
    planner = SpatialCorridorPlanner()
    obstacles = [
        track(11 * MM, 0 * MM, 11 * MM, 9.6 * MM),
        track(11 * MM, 10.4 * MM, 11 * MM, 20 * MM),
    ]

    path = corridor(planner, (2 * MM, 10 * MM), (20 * MM, 10 * MM), obstacles)

    assert max(abs(y - 10 * MM) for _, y in path) > 4 * MM, (
        "took a 0.8 mm channel a 0.5 mm-step head cannot hold a line in"
    )


def test_waypoints_are_corners_not_one_per_cell():
    """The field yields a point per 0.5 mm cell. Pushing all of them would
    hand PNS hundreds of collinear waypoints per net."""
    planner = SpatialCorridorPlanner()
    wall = [track(11 * MM, 2 * MM, 11 * MM, 18 * MM)]

    waypoints = planner.plan_corridor((2 * MM, 10 * MM), (20 * MM, 10 * MM), wall)

    assert 0 < len(waypoints) <= 8, f"expected a handful of corners, got {len(waypoints)}"


def test_the_corridor_is_legal_on_randomised_boards():
    """The property that decides nets, checked over many boards at once.

    Every individual case above pins one mechanism. This pins the outcome:
    whatever the geometry, the corridor this planner emits must be legal
    everywhere along its length, because PNS refuses a route that touched
    anything and one illegal micron loses the whole net.

    It is also the test that catches margin arithmetic. The simplifier is
    allowed to move the corridor, and the only budget it has is what the
    field planned in excess of legal -- get that wrong and corridors come out
    a few microns under, which no hand-written scenario is likely to hit but
    which shows up immediately over a hundred random ones.
    """
    import random

    rng = random.Random(11)
    planner = SpatialCorridorPlanner()
    checked = 0

    for _ in range(100):
        start = (rng.uniform(3 * MM, 47 * MM), rng.uniform(3 * MM, 47 * MM))
        target = (rng.uniform(3 * MM, 47 * MM), rng.uniform(3 * MM, 47 * MM))
        if math.hypot(target[0] - start[0], target[1] - start[1]) < 15 * MM:
            continue

        obstacles = []
        for _ in range(rng.randint(3, 8)):
            x1, y1 = rng.uniform(0, 50 * MM), rng.uniform(0, 50 * MM)
            angle, length = rng.uniform(0, 2 * math.pi), rng.uniform(6 * MM, 20 * MM)
            seg = track(x1, y1, x1 + length * math.cos(angle), y1 + length * math.sin(angle))
            # The route has to be able to leave and arrive; copper sitting on
            # an endpoint is a bad scenario, not a hard one.
            if min(_point_seg_gap(*start, seg), _point_seg_gap(*target, seg)) < 1 * MM:
                continue
            obstacles.append(seg)

        if not obstacles:
            continue

        checked += 1
        path = corridor(planner, start, target, obstacles)
        gap = min_gap(path, obstacles)
        assert gap >= LEGAL_GAP_NM, (
            f"corridor came within {gap / MM:.4f} mm of copper "
            f"(legal is {LEGAL_GAP_NM / MM} mm) routing {start} -> {target}"
        )

    assert checked > 50, f"scenario generator produced only {checked} usable boards"
