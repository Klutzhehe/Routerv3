"""Spatial Corridor & Waypoint Planner for Pure AI Routing (RM_MARK_OBSTACLES).

Computes collision-free intermediate waypoints around pads, committed tracks,
and reservation zones without relying on KiCad's automatic push/shove.

## Why this plans on a field rather than a visibility graph

The first version inflated every obstacle to an axis-aligned bounding box and
ran A* over the corners. Three things go wrong with that on a real board, and
they compound:

  1. **A diagonal track becomes a filled square.** A 45-degree trace from one
     corner of a 30 mm region to the other has a 30x30 mm bounding box, so a
     single routed diagonal marks a fifth of the board unroutable. The
     hierarchical pipeline routes the important nets FIRST and then has to
     thread the rest past them -- which is exactly the case where every
     earlier net's bbox is standing in the way of every later one. The
     ordering that is supposed to help is what makes this worst.

  2. **Corners are not where the gap is.** The graph can only turn at an
     inflated box's corners, so a corridor between two obstacles is invisible
     unless a corner happens to sit in it.

  3. **A start inside any inflated box has no legal first edge.** Every
     segment out of it clips the box it is standing in, `_line_collides` is
     true for all of them, and A* falls through to the straight line -- i.e.
     precisely the case that needed a detour silently gets none. In a dense
     cluster a pad is routinely within a neighbour's inflated footprint.

The cost-to-go field in `pcbworld/env/geodesic.py` has none of these
properties: obstacles are capsules around their true segment geometry, the
plan can turn anywhere, and a source inside copper still gets a finite value
(`_fill_blocked`). It is also the same field the RL env shapes its reward on,
at the same measured inflation, so the analytic and learned routers plan
against one definition of "in the way" rather than two.

The visibility-graph search is kept as the fallback for when the field
reports the target unreachable -- that is a statement about the inflated
plan, not about the board, and a tighter path is better than no path.
"""

from __future__ import annotations

import math
import heapq
from typing import List, Tuple, Dict, Set, Optional, Any, Sequence

import numpy as np

from pcbworld.env.geodesic import GeodesicConfig, GeodesicField
from pcbworld.env.line_obs import KIND_EDGE, KIND_GHOST, KIND_TRACK, Segment
from pcbworld.hierarchical.specs import ReservationZone, PadInfo

# Cell size for the planning grid. 0.5 mm matches the RL env's step, which is
# the resolution the follower can actually act on -- see GeodesicConfig.
PLAN_CELL_NM = 500_000.0

# Legal clearance (325 um) plus the margin an imperfect follower needs. Swept
# on 25 Colab boards; 700 um was the peak (see GeodesicConfig's docstring).
# Not a tuning knob to reach for first -- grid resolution scored worse at
# every setting.
PLAN_INFLATION_NM = 700_000.0

# A coupled pair occupies two tracks and their gap, so it needs its own
# half-width on top of the single-ended margin: 2*200um width + 150um gap
# leaves ~275 um of extra half-width over a 250 um single track.
DIFF_PAIR_EXTRA_NM = 300_000.0

# Track half-width (125 um) plus the design clearance (200 um): the margin
# below which PNS refuses the route. This is a hard legality threshold, not a
# planning margin -- see ACCEPT_MARGIN_NM.
LEGAL_GAP_NM = 325_000.0

# What a straight line has to clear before this planner declines to plan.
#
# The field's 700 um inflation is deliberately larger than legal, because it
# was calibrated for the RL head: a 0.5 mm fixed step at a continuous heading
# tracks a plan to within about half a step. THIS planner's follower is
# different -- the router pushes to the exact waypoint -- so holding it to the
# RL head's margin makes it detour around lines it could have taken. Measured
# on 200 randomised scenarios: 70 genuinely needed a detour and the field
# alone produced 170.
#
# So the straight line is judged against the real geometry at legal clearance
# plus one cell of slack, and only the ones that actually fail go to the
# field. That is a measurement of the board, not a second margin to tune.
ACCEPT_MARGIN_NM = 500_000.0


def bbox_to_segment(
    bbox: Tuple[int, int, int, int], net: str = "", layer: int = 0
) -> Segment:
    """A rectangular keep-out as one capsule segment along its long axis.

    A `Segment` with width w covers a stadium of radius w/2 about its spine,
    so a spine along the rectangle's long axis with width equal to its SHORT
    side covers everything except the four corners -- and the planner's own
    700 um inflation covers those. One segment instead of a rasterised grid
    keeps the obstacle list short enough that building the field stays the
    ~20 ms it is measured at.
    """
    x1, y1, x2, y2 = bbox
    width_x = abs(x2 - x1)
    width_y = abs(y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    if width_x >= width_y:
        spine = (min(x1, x2), cy, max(x1, x2), cy)
        thickness = width_y
    else:
        spine = (cx, min(y1, y2), cx, max(y1, y2))
        thickness = width_x

    return Segment(
        x1=float(spine[0]), y1=float(spine[1]), x2=float(spine[2]), y2=float(spine[3]),
        width=float(thickness), kind=KIND_TRACK, net=net, layer=layer,
    )


def _as_segments(obstacles: Sequence[Any]) -> List[Segment]:
    """Accept either real Segments or legacy (xmin, ymin, xmax, ymax) boxes.

    Callers that can reach `get_board_geometry()` should pass Segments -- a
    bbox of a diagonal track is a lie about where the copper is, and the
    whole reason this planner exists is that the lie was expensive.
    """
    out: List[Segment] = []
    for item in obstacles or ():
        if isinstance(item, Segment):
            out.append(item)
        else:
            out.append(bbox_to_segment(tuple(item)))
    return out


def polyline_clearance(
    points: Sequence[Tuple[float, float]],
    segments: Sequence[Segment],
    sample_nm: float = 250_000.0,
) -> float:
    """Closest a pushed polyline ever comes to an obstacle's edge, in nm.

    Sampled ALONG each leg, not just at its ends: two clear waypoints with
    copper between them is the corridor failure that matters, and endpoint
    checks are blind to it. Negative means the line is inside a footprint.

    Ghosts and the board outline are excluded for the same reasons
    `nearest_obstacle_gap` excludes them: an unrouted net is a plan rather
    than copper, and the outline is a boundary rather than something to keep
    clear of by this margin.
    """
    usable = [s for s in segments if s.kind not in (KIND_EDGE, KIND_GHOST)]
    if not usable or len(points) < 2:
        return float("inf")

    x1 = np.fromiter((s.x1 for s in usable), dtype=np.float64, count=len(usable))
    y1 = np.fromiter((s.y1 for s in usable), dtype=np.float64, count=len(usable))
    x2 = np.fromiter((s.x2 for s in usable), dtype=np.float64, count=len(usable))
    y2 = np.fromiter((s.y2 for s in usable), dtype=np.float64, count=len(usable))
    half = np.fromiter((s.width for s in usable), dtype=np.float64, count=len(usable)) / 2.0

    px: List[float] = []
    py: List[float] = []
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        n = max(1, int(math.ceil(math.hypot(bx - ax, by - ay) / sample_nm)))
        t = np.linspace(0.0, 1.0, n + 1)
        px.extend(ax + t * (bx - ax))
        py.extend(ay + t * (by - ay))

    p = np.asarray(px)[:, None]
    q = np.asarray(py)[:, None]

    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(len_sq > 0, ((p - x1) * dx + (q - y1) * dy) / np.where(len_sq > 0, len_sq, 1.0), 0.0)
    np.clip(t, 0.0, 1.0, out=t)
    gap = np.hypot(p - (x1 + t * dx), q - (y1 + t * dy)) - half
    return float(gap.min())


class SpatialCorridorPlanner:
    """Plans collision-free obstacle-avoidance waypoints for pure validator mode."""

    # Headings sampled per step. 16 gives 22.5-degree granularity, which the
    # RDP simplification smooths back out; finer sampling costs a field
    # lookup each and buys resolution the pushed corridor does not keep.
    _RING_SAMPLES = 16

    def __init__(
        self,
        default_clearance_nm: int = 250_000,
        inflation_nm: float = PLAN_INFLATION_NM,
        cell_nm: float = PLAN_CELL_NM,
        max_plan_steps: int = 400,
    ):
        # Retained for the visibility-graph fallback, which still reasons in
        # boxes; the field path uses `inflation_nm`.
        self.default_clearance_nm = default_clearance_nm
        self.inflation_nm = inflation_nm
        self.cell_nm = cell_nm
        self.max_plan_steps = max_plan_steps

    # -- public API ------------------------------------------------------

    def plan_corridor(
        self,
        start_xy: Tuple[int, int],
        target_xy: Tuple[int, int],
        obstacles: Sequence[Any],
        reservation_zones: Optional[List[ReservationZone]] = None,
        is_diff_pair: bool = False,
        layer: int = 0,
    ) -> List[Tuple[int, int]]:
        """Intermediate waypoints from start to target, avoiding everything.

        Returns [] when the straight line is already clear -- the caller then
        pushes directly, which is both cheaper and shorter than any detour.
        """
        start = (float(start_xy[0]), float(start_xy[1]))
        target = (float(target_xy[0]), float(target_xy[1]))

        segments = _as_segments(obstacles)
        for zone in reservation_zones or ():
            if zone.active and zone.layer == layer:
                segments.append(bbox_to_segment(zone.bbox_nm, net=zone.owner_net, layer=layer))

        inflation = self.inflation_nm + (DIFF_PAIR_EXTRA_NM if is_diff_pair else 0.0)

        # Does the straight line actually clear everything? Answered against
        # the real geometry rather than the inflated field, because the field
        # is deliberately pessimistic for a follower this planner does not
        # have (see ACCEPT_MARGIN_NM). Skipping this made the planner detour
        # 170 times on 200 scenarios where 70 needed it.
        accept = LEGAL_GAP_NM + (ACCEPT_MARGIN_NM - LEGAL_GAP_NM) * (2.0 if is_diff_pair else 1.0)
        if polyline_clearance([start, target], segments) >= accept:
            return []

        # The grid has to be big enough to hold a way AROUND the obstacles in
        # it. GeodesicField.build takes the bounding box of the segment
        # ENDPOINTS and adds margin_nm -- it does not know about each
        # segment's own width -- so with the default 2 mm margin a single wide
        # obstacle (a reservation zone becomes a 4 mm capsule) blocks every
        # row of the grid, the target reads as unreachable, and this quietly
        # drops through to the visibility fallback. The margin therefore has
        # to clear the widest obstacle's half-width plus the inflation, with a
        # couple of cells to turn in.
        #
        # In the orchestrator's own calls the board outline is in `segments`
        # and bounds the grid properly; this matters for callers that pass
        # bare obstacles, which includes every legacy bbox caller.
        widest = max((seg.width for seg in segments), default=0.0)
        margin = inflation + widest / 2.0 + 2.0 * self.cell_nm
        config = GeodesicConfig(
            cell_nm=self.cell_nm,
            inflation_nm=inflation,
            margin_nm=max(GeodesicConfig.margin_nm, margin),
        )

        field = GeodesicField.build(segments, head=start, target=target, config=config)
        if not field.reachable:
            # The inflated plan has no route. That is a statement about the
            # margin, not about the board, so fall back to the tighter
            # box-visibility search rather than returning nothing.
            return self._visibility_fallback(start_xy, target_xy, obstacles,
                                             reservation_zones, is_diff_pair, layer)

        # The simplifier may move the corridor by up to `epsilon`, and the
        # only margin it is allowed to spend is what the field planned in
        # excess of legal. At one cell (500 um) against 375 um of spare it
        # could spend more than there was: measured, a corridor traced at
        # 700 um came out at 318 um, six microns under legal, and PNS refuses
        # that route as surely as a direct hit.
        spare = max(0.0, inflation - LEGAL_GAP_NM)
        epsilon = min(self.cell_nm, 0.5 * spare)

        path = self._simplify(self.descent_path(field, start, target), epsilon)
        path = self._string_pull(path, segments, accept)
        return [(int(round(x)), int(round(y))) for x, y in path[1:-1]]

    def descent_path(
        self,
        field: GeodesicField,
        start: Tuple[float, float],
        target: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        """Walk downhill on the field from start to target.

        One step per cell, toward the cheapest point on a ring of that radius
        -- the same query the RL env's `geo_dir` feature uses, so the analytic
        plan and the learned policy's guidance are the same curve rather than
        two approximations of it.

        Two guards, both load-bearing:

        **Blocked candidates are rejected.** `_fill_blocked` gives cells
        inside copper a finite cost on purpose, so the RL head -- which is
        inside an obstacle on 10-45% of its steps -- gets a smooth potential
        instead of a cliff. For a PATH those values are a shortcut straight
        through the obstacle, and the plan will take it: a reservation zone
        4 mm wide was crossed at its own centreline because the filled cost
        there was lower than going around. Only the mask can say no.

        **The cost must strictly decrease.** Bilinear interpolation near a
        wall is not guaranteed monotone, and without this a plan can rock
        between two cells for the whole step budget.
        """
        step = field.cell
        pos = start
        path = [pos]
        best = field.cost_to_go(*pos)

        # Leaving the start is allowed to be blocked -- a pad in a dense
        # cluster sits inside its neighbour's inflated footprint, and
        # refusing to move would fail exactly the crowded case that needs a
        # plan. Once out, the corridor stays out.
        escaped = not field.is_blocked(*start)

        for _ in range(self.max_plan_steps):
            if math.hypot(target[0] - pos[0], target[1] - pos[1]) <= step:
                break

            nxt, cost = None, best
            for i in range(self._RING_SAMPLES):
                theta = 2.0 * math.pi * i / self._RING_SAMPLES
                cand = (pos[0] + step * math.cos(theta), pos[1] + step * math.sin(theta))
                if escaped and field.is_blocked(*cand):
                    continue
                c = field.cost_to_go(*cand)
                if math.isfinite(c) and c < cost:
                    nxt, cost = cand, c

            if nxt is None:
                break
            pos, best = nxt, cost
            path.append(pos)
            if not escaped and not field.is_blocked(*pos):
                escaped = True

        path.append(target)
        return path

    # -- internals -------------------------------------------------------

    @classmethod
    def _simplify(
        cls, path: List[Tuple[float, float]], epsilon: float = PLAN_CELL_NM
    ) -> List[Tuple[float, float]]:
        """Ramer-Douglas-Peucker: keep only the points the corridor turns at.

        The field gives one point per cell, so a straight run arrives as a
        hundred collinear waypoints -- and `descent_direction` samples a ring
        of 16, so a "straight" run is really a 22.5-degree zig-zag about the
        true heading. A heading-delta filter keeps every one of those kinks;
        RDP measures deviation from the CHORD instead, so quantisation noise
        below the tolerance collapses and only real corners survive.

        Epsilon is chosen by the caller, and it is a safety parameter rather
        than a cosmetic one: whatever it is, the simplified corridor can sit
        that much closer to copper than the traced one did. See
        `plan_corridor`, which derives it from the margin the field actually
        planned in excess of legal clearance.
        """
        if len(path) <= 2:
            return list(path)

        start, end = path[0], path[-1]
        dx, dy = end[0] - start[0], end[1] - start[1]
        chord = math.hypot(dx, dy)

        worst_i, worst_d = 0, -1.0
        for i, (px, py) in enumerate(path[1:-1], start=1):
            if chord <= 0.0:
                d = math.hypot(px - start[0], py - start[1])
            else:
                # Perpendicular distance to the chord.
                d = abs(dy * (px - start[0]) - dx * (py - start[1])) / chord
            if d > worst_d:
                worst_i, worst_d = i, d

        if worst_d <= epsilon:
            return [start, end]

        left = cls._simplify(path[: worst_i + 1], epsilon)
        right = cls._simplify(path[worst_i:], epsilon)
        return left[:-1] + right

    @staticmethod
    def _string_pull(
        path: List[Tuple[float, float]],
        segments: Sequence[Segment],
        accept: float,
    ) -> List[Tuple[float, float]]:
        """Drop every waypoint the corridor can legally cut the corner past.

        The field plans at a margin wider than legal, so its corridor hugs a
        safe envelope rather than the shortest legal path -- it rounds an
        obstacle at 700 um where 325 um would have committed. Pulling the
        string taut against the REAL geometry recovers the wirelength without
        giving up any legality: a shortcut is only taken when the direct leg
        measures clear at `accept`, which is the same threshold the
        straight-line pre-check uses.

        Greedy forward scan, farthest-first: from each kept point, take the
        most distant later point the line still clears. O(n^2) legs on a
        handful of corners, which is nothing beside building the field.
        """
        if len(path) <= 2:
            return list(path)

        kept = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if polyline_clearance([path[i], path[j]], segments) >= accept:
                    break
                j -= 1
            kept.append(path[j])
            i = j
        return kept

    def _visibility_fallback(
        self,
        start_xy: Tuple[int, int],
        target_xy: Tuple[int, int],
        obstacles: Sequence[Any],
        reservation_zones: Optional[List[ReservationZone]],
        is_diff_pair: bool,
        layer: int,
    ) -> List[Tuple[int, int]]:
        """The original inflated-bbox visibility search.

        Only reached when the field says the target is unreachable at the
        follow margin. Its weaknesses are documented at the top of this file;
        it is here because a tight path beats no path.
        """
        x1, y1 = start_xy
        x2, y2 = target_xy

        padding = self.default_clearance_nm if not is_diff_pair else int(self.default_clearance_nm * 1.5)
        padded_boxes: List[Tuple[int, int, int, int]] = []

        for item in obstacles or ():
            if isinstance(item, Segment):
                ox1, oy1 = min(item.x1, item.x2), min(item.y1, item.y2)
                ox2, oy2 = max(item.x1, item.x2), max(item.y1, item.y2)
                half = item.width / 2.0
                ox1, oy1, ox2, oy2 = ox1 - half, oy1 - half, ox2 + half, oy2 + half
            else:
                ox1, oy1, ox2, oy2 = item

            # Skip a box the route has to start or end inside -- otherwise
            # every edge out of that endpoint clips it and the search has no
            # legal first move.
            if (ox1 <= x1 <= ox2 and oy1 <= y1 <= oy2) or (ox1 <= x2 <= ox2 and oy1 <= y2 <= oy2):
                continue
            padded_boxes.append((ox1 - padding, oy1 - padding, ox2 + padding, oy2 + padding))

        if reservation_zones:
            for zone in reservation_zones:
                if zone.active and zone.layer == layer:
                    zx1, zy1, zx2, zy2 = zone.bbox_nm
                    padded_boxes.append((zx1 - padding, zy1 - padding, zx2 + padding, zy2 + padding))

        if not self._line_collides(x1, y1, x2, y2, padded_boxes):
            return []

        nav_nodes = [(x1, y1), (x2, y2)]
        offset = int(padding * 1.2)
        for (bx1, by1, bx2, by2) in padded_boxes:
            nav_nodes.extend([
                (bx1 - offset, by1 - offset),
                (bx1 - offset, by2 + offset),
                (bx2 + offset, by1 - offset),
                (bx2 + offset, by2 + offset),
                ((bx1 + bx2) // 2, by2 + offset),
                ((bx1 + bx2) // 2, by1 - offset),
                (bx1 - offset, (by1 + by2) // 2),
                (bx2 + offset, (by1 + by2) // 2),
            ])

        path = self._astar_search((x1, y1), (x2, y2), nav_nodes, padded_boxes)
        if len(path) > 2:
            return [(int(px), int(py)) for px, py in path[1:-1]]

        for (bx1, by1, bx2, by2) in padded_boxes:
            if self._segment_intersects_box(x1, y1, x2, y2, bx1, by1, bx2, by2):
                mid_x = (x1 + x2) // 2
                detour_y = by2 + offset if abs(y1 - by2) < abs(y1 - by1) else by1 - offset
                return [(int(mid_x), int(detour_y))]

        return []

    def _line_collides(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        boxes: List[Tuple[int, int, int, int]],
    ) -> bool:
        """Checks if line (x1, y1)-(x2, y2) collides with any bounding box."""
        for (bx1, by1, bx2, by2) in boxes:
            if self._segment_intersects_box(x1, y1, x2, y2, bx1, by1, bx2, by2):
                return True
        return False

    @staticmethod
    def _segment_intersects_box(
        x1: float, y1: float, x2: float, y2: float,
        bx1: float, by1: float, bx2: float, by2: float
    ) -> bool:
        """Fast Liang-Barsky line clipping / box intersection."""
        dx = x2 - x1
        dy = y2 - y1

        p = [-dx, dx, -dy, dy]
        q = [x1 - bx1, bx2 - x1, y1 - by1, by2 - y1]

        u1 = 0.0
        u2 = 1.0

        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return False
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    if t > u2:
                        return False
                    if t > u1:
                        u1 = t
                else:
                    if t < u1:
                        return False
                    if t < u2:
                        u2 = t

        return u1 <= u2

    def _astar_search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        nodes: List[Tuple[int, int]],
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int]]:
        """A* search over visibility graph."""
        open_set: List[Tuple[float, float, Tuple[int, int], List[Tuple[int, int]]]] = []
        heapq.heappush(open_set, (math.hypot(goal[0] - start[0], goal[1] - start[1]), 0.0, start, [start]))
        visited: Set[Tuple[int, int]] = set()

        while open_set:
            f, cost, curr, path = heapq.heappop(open_set)
            if curr in visited:
                continue
            visited.add(curr)

            if curr == goal:
                return path

            if curr != goal and not self._line_collides(curr[0], curr[1], goal[0], goal[1], boxes):
                heapq.heappush(open_set, (
                    cost + math.hypot(goal[0] - curr[0], goal[1] - curr[1]),
                    cost + math.hypot(goal[0] - curr[0], goal[1] - curr[1]),
                    goal,
                    path + [goal],
                ))

            for neighbor in nodes:
                if neighbor in visited:
                    continue
                dist = math.hypot(neighbor[0] - curr[0], neighbor[1] - curr[1])
                if dist > 50_000_000:  # Prune far nodes (>50mm)
                    continue
                if not self._line_collides(curr[0], curr[1], neighbor[0], neighbor[1], boxes):
                    new_cost = cost + dist
                    heuristic = math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                    heapq.heappush(open_set, (new_cost + heuristic, new_cost, neighbor, path + [neighbor]))

        return [start, goal]
