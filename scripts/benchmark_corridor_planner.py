"""Old bbox-visibility planner vs new geodesic planner, on the same scenarios.

Run:  python scripts/benchmark_corridor_planner.py

No bridge needed -- both planners are pure geometry, so this runs anywhere
and does not cost a Colab round.

"Old" is reproduced exactly: axis-aligned bounding boxes of each obstacle
segment (what the orchestrator used to accumulate), fed to the visibility
search that is still present as the fallback. "New" is plan_corridor() on the
real segment geometry.

Metric is the one that decides a net: is the pushed corridor legal everywhere
along its length, at track half-width + design clearance? PNS refuses a route
that touched anything, so a corridor with one illegal millimetre is a failed
net regardless of how good the rest of it is.
"""

import math
import random
import sys
import time

sys.path.insert(0, ".")

from pcbworld.env.line_obs import KIND_TRACK, Segment
from pcbworld.hierarchical.spatial_corridor_planner import SpatialCorridorPlanner

MM = 1_000_000
LEGAL_GAP_NM = 325_000
BOARD = 50 * MM


def point_seg_gap(px, py, seg):
    dx, dy = seg.x2 - seg.x1, seg.y2 - seg.y1
    len_sq = dx * dx + dy * dy
    t = 0.0 if len_sq == 0 else max(0.0, min(1.0, ((px - seg.x1) * dx + (py - seg.y1) * dy) / len_sq))
    cx, cy = seg.x1 + t * dx, seg.y1 + t * dy
    return math.hypot(px - cx, py - cy) - seg.width / 2.0


def min_gap(path, segments, samples=40):
    worst = float("inf")
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        n = max(2, min(samples, int(math.hypot(bx - ax, by - ay) / (0.25 * MM))))
        for i in range(n + 1):
            t = i / n
            px, py = ax + t * (bx - ax), ay + t * (by - ay)
            for seg in segments:
                worst = min(worst, point_seg_gap(px, py, seg))
    return worst


def path_len(path):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])) or 1.0


def bbox_of(seg):
    half = seg.width / 2.0
    return (
        min(seg.x1, seg.x2) - half, min(seg.y1, seg.y2) - half,
        max(seg.x1, seg.x2) + half, max(seg.y1, seg.y2) + half,
    )


def make_scenario(rng):
    """A start, a target, and some committed copper between them."""
    while True:
        start = (rng.uniform(3 * MM, 47 * MM), rng.uniform(3 * MM, 47 * MM))
        target = (rng.uniform(3 * MM, 47 * MM), rng.uniform(3 * MM, 47 * MM))
        if math.hypot(target[0] - start[0], target[1] - start[1]) > 15 * MM:
            break

    segments = []
    for _ in range(rng.randint(3, 8)):
        # Mostly diagonal traces: the case a bounding box misrepresents most.
        x1, y1 = rng.uniform(0, BOARD), rng.uniform(0, BOARD)
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(6 * MM, 20 * MM)
        seg = Segment(
            x1=x1, y1=y1,
            x2=x1 + length * math.cos(angle), y2=y1 + length * math.sin(angle),
            width=0.25 * MM, kind=KIND_TRACK, net=f"n{rng.randint(0, 99)}", layer=0,
        )
        # An obstacle sitting on an endpoint is not an obstacle, it is a bug
        # in the scenario -- the route has to be able to leave and arrive.
        if point_seg_gap(*start, seg) < 1 * MM or point_seg_gap(*target, seg) < 1 * MM:
            continue
        segments.append(seg)
    return start, target, segments


def main(trials=200, seed=7):
    rng = random.Random(seed)
    planner = SpatialCorridorPlanner()

    stats = {
        "old": {"legal": 0, "detoured": 0, "time": 0.0, "ratio": []},
        "new": {"legal": 0, "detoured": 0, "time": 0.0, "ratio": []},
    }
    needed = 0
    scenarios = 0

    for _ in range(trials):
        start, target, segments = make_scenario(rng)
        if not segments:
            continue
        scenarios += 1

        straight_blocked = min_gap([start, target], segments) < LEGAL_GAP_NM
        needed += int(straight_blocked)

        # OLD: bounding boxes + visibility search (the shipped behaviour).
        boxes = [bbox_of(s) for s in segments]
        t0 = time.perf_counter()
        old_wp = planner._visibility_fallback(start, target, boxes, None, False, 0)
        stats["old"]["time"] += time.perf_counter() - t0
        old_path = [start] + list(old_wp) + [target]
        stats["old"]["legal"] += int(min_gap(old_path, segments) >= LEGAL_GAP_NM)
        stats["old"]["detoured"] += int(bool(old_wp))
        stats["old"]["ratio"].append(path_len(old_path) / path_len([start, target]))

        # NEW: real geometry + cost-to-go field.
        t0 = time.perf_counter()
        new_wp = planner.plan_corridor(start, target, segments)
        stats["new"]["time"] += time.perf_counter() - t0
        new_path = [start] + list(new_wp) + [target]
        stats["new"]["legal"] += int(min_gap(new_path, segments) >= LEGAL_GAP_NM)
        stats["new"]["detoured"] += int(bool(new_wp))
        stats["new"]["ratio"].append(path_len(new_path) / path_len([start, target]))

    print(f"scenarios: {scenarios}   straight line already illegal in: {needed}")
    print()
    print(f"{'planner':>8} | {'legal corridor':>17} | {'detoured':>8} | {'wirelen':>7} | {'ms/net':>7}")
    print("-" * 62)
    for name in ("old", "new"):
        s = stats[name]
        print(
            f"{name:>8} | {s['legal']:>6}/{scenarios} ({100*s['legal']/scenarios:5.1f}%) "
            f"| {s['detoured']:>8} | {sum(s['ratio'])/len(s['ratio']):6.3f}x | {1000*s['time']/scenarios:7.2f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(trials=args.trials, seed=args.seed)
