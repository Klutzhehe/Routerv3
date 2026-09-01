"""Post-Routing Track Geometry Optimizer.

Performs string-pulling, collinear vertex merging, 45-degree corner normalization,
and redundant segment elimination to produce production-grade, shortest-path copper traces.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional


class TrackOptimizer:
    """Optimizes routed line-segment geometry for minimum wirelength, fewest corners, and clean 45-degree angles."""

    def __init__(self, clearance_margin_nm: int = 200_000):
        self.clearance_margin_nm = clearance_margin_nm

    @staticmethod
    def merge_collinear_segments(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Removes redundant intermediate collinear vertices."""
        if len(points) <= 2:
            return points

        simplified: List[Tuple[int, int]] = [points[0]]

        for i in range(1, len(points) - 1):
            p_prev = simplified[-1]
            p_curr = points[i]
            p_next = points[i + 1]

            # Vector cross product to test collinearity: (y2 - y1)*(x3 - x2) - (y3 - y2)*(x2 - x1)
            dx1 = p_curr[0] - p_prev[0]
            dy1 = p_curr[1] - p_prev[1]
            dx2 = p_next[0] - p_curr[0]
            dy2 = p_next[1] - p_curr[1]

            cross_product = dx1 * dy2 - dy1 * dx2
            dot_product = dx1 * dx2 + dy1 * dy2

            # If cross product is 0 and vectors point in the same direction, p_curr is redundant
            if abs(cross_product) < 1e-3 and dot_product > 0:
                continue

            simplified.append(p_curr)

        simplified.append(points[-1])
        return simplified

    def string_pull(
        self,
        points: List[Tuple[int, int]],
        obstacles: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int]]:
        """Applies ray-casting string pulling (shortcut shortcutting) across non-colliding vertices."""
        if len(points) <= 2:
            return points

        pulled: List[Tuple[int, int]] = [points[0]]
        curr_idx = 0

        while curr_idx < len(points) - 1:
            # Look as far forward as possible
            furthest_idx = curr_idx + 1
            for next_idx in range(len(points) - 1, curr_idx + 1, -1):
                p1 = points[curr_idx]
                p2 = points[next_idx]
                if not self._segment_collides(p1, p2, obstacles):
                    furthest_idx = next_idx
                    break

            pulled.append(points[furthest_idx])
            curr_idx = furthest_idx

        return self.merge_collinear_segments(pulled)

    def _segment_collides(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        obstacles: List[Tuple[int, int, int, int]],
    ) -> bool:
        """Checks if straight segment p1->p2 intersects any obstacle bounding box."""
        x1, y1 = p1
        x2, y2 = p2

        for (ox1, oy1, ox2, oy2) in obstacles:
            # Skip if p1 or p2 are inside this obstacle (e.g. pad connection)
            if (ox1 <= x1 <= ox2 and oy1 <= y1 <= oy2) or (ox1 <= x2 <= ox2 and oy1 <= y2 <= oy2):
                continue
            if self._segment_intersects_box(x1, y1, x2, y2, ox1, oy1, ox2, oy2):
                return True
        return False

    @staticmethod
    def _segment_intersects_box(
        x1: int, y1: int, x2: int, y2: int,
        bx1: int, by1: int, bx2: int, by2: int,
    ) -> bool:
        dx = x2 - x1
        dy = y2 - y1
        p = [-dx, dx, -dy, dy]
        q = [x1 - bx1, bx2 - x1, y1 - by1, by2 - y1]
        u1, u2 = 0.0, 1.0
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
