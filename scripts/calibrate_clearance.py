"""What clearance does PNS actually enforce? Measured, not assumed.

Everything about the plan's margin has rested on one unverified number. The
geodesic field blocks a cell within `inflation_nm + width/2` of an obstacle,
and `inflation_nm` was set to 325 um from first principles: the routed track's
half-width (125) plus the design clearance (200). Nobody ever checked that
against the router.

The check matters because a simulation built on that number predicted a large
win from widening the margin -- blocked nets going from 22% to 61% committed
clean, overall 52% to 76% -- and the real oracle did not move by a single net
(80/124 before, 80/124 after). A model that confident and that wrong is
wrong about its inputs.

So this asks PNS directly. It walks nets with a = 0 and, at every step,
records two things:

  - `head_collides()`, the router's own verdict
  - the distance from the head to the nearest obstacle in the env's own
    obstacle list, minus that obstacle's half-width

Bucketing the verdict by the distance gives the router's effective clearance
as a measured curve. The distance where collision probability crosses 50% is
the number `inflation_nm` should have been all along. If that lands near
325 um the model was right and the discrepancy is elsewhere; if it lands at
1.5 mm, every margin argument so far was calibrated against the wrong scale.

It also reports the head-to-obstacle distance at the moment fix() refuses,
which is the same question asked at the only point that decides completion.

    python scripts/calibrate_clearance.py /content/curriculum_dataset/stage1_basics
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbworld.env.line_obs import KIND_EDGE, point_segment_distance  # noqa: E402
from pcbworld.env.line_route_env import LineRouteEnv  # noqa: E402

MM = 1_000_000


def nearest_obstacle_gap(pos, obstacles) -> float:
    """Distance from the head to the nearest obstacle EDGE, in nm.

    Each obstacle's own half-width is subtracted, so this is the gap a trace
    centre has to fit in -- the same quantity `inflation_nm` is compared
    against. Negative means the head is inside the obstacle's footprint.
    """
    if not obstacles:
        return float("inf")
    usable = [o for o in obstacles if o.kind != KIND_EDGE]
    if not usable:
        return float("inf")
    x1 = np.fromiter((o.x1 for o in usable), dtype=np.float64, count=len(usable))
    y1 = np.fromiter((o.y1 for o in usable), dtype=np.float64, count=len(usable))
    x2 = np.fromiter((o.x2 for o in usable), dtype=np.float64, count=len(usable))
    y2 = np.fromiter((o.y2 for o in usable), dtype=np.float64, count=len(usable))
    w = np.fromiter((o.width for o in usable), dtype=np.float64, count=len(usable))
    d = point_segment_distance(pos[0], pos[1], x1, y1, x2, y2) - w / 2.0
    return float(d.min())


def walk(board_path: str, max_steps_per_net: int = 120):
    env = LineRouteEnv(
        board_path,
        step_size_nm=500_000,
        snap_radius_nm=400_000,
        max_steps_per_net=max_steps_per_net,
    )
    env.reset()

    gaps: list[float] = []
    hits: list[bool] = []
    refusal_gaps: list[float] = []
    seen_refusals: set[str] = set()

    guard, limit = 0, len(env._nets) * max_steps_per_net * 3
    while guard < limit:
        _obs, _r, terminated, _t, info = env.step(np.array([0.0], dtype=np.float32))
        guard += 1

        gap = nearest_obstacle_gap(env._pos, env._obstacles)
        if math.isfinite(gap):
            gaps.append(gap)
            hits.append(bool(info.get("collides", False)))

        for net, detail in info.get("fix_refusals", {}).items():
            if net not in seen_refusals:
                seen_refusals.add(net)
                if math.isfinite(gap):
                    refusal_gaps.append(gap)

        if terminated:
            break
    return gaps, hits, refusal_gaps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("boards", help=".kicad_pcb file or a directory of them")
    args = ap.parse_args()

    if os.path.isdir(args.boards):
        paths = sorted(
            os.path.join(args.boards, f)
            for f in os.listdir(args.boards)
            if f.endswith(".kicad_pcb")
        )
    else:
        paths = [args.boards]

    gaps: list[float] = []
    hits: list[bool] = []
    refusals: list[float] = []
    for i, path in enumerate(paths):
        g, h, r = walk(path)
        gaps.extend(g)
        hits.extend(h)
        refusals.extend(r)
        print(f"[{i + 1:2d}/{len(paths)}] {Path(path).name}", flush=True)

    g = np.array(gaps)
    h = np.array(hits, dtype=bool)

    print("\n" + "=" * 74)
    print("        WHAT CLEARANCE DOES PNS ACTUALLY ENFORCE?")
    print("=" * 74)
    print(f"steps sampled {len(g)}   head_collides() true on {h.mean() * 100:.1f}%")
    print()
    print("  head-to-obstacle gap  |  steps  | head_collides() says yes")
    print("  " + "-" * 60)
    edges = [-np.inf, 0, 100_000, 200_000, 325_000, 500_000, 700_000,
             1_000_000, 1_500_000, 2_000_000, np.inf]
    labels = ["inside footprint", "0.0-0.1mm", "0.1-0.2mm", "0.2-0.325mm",
              "0.325-0.5mm", "0.5-0.7mm", "0.7-1.0mm", "1.0-1.5mm",
              "1.5-2.0mm", "over 2.0mm"]
    crossing = None
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        m = (g >= lo) & (g < hi)
        if not m.any():
            print(f"  {label:<21} |      0  |    --")
            continue
        rate = h[m].mean()
        print(f"  {label:<21} | {m.sum():6d}  | {rate * 100:5.1f}%")
        if crossing is None and rate < 0.5 and lo >= 0:
            crossing = lo

    print()
    if crossing is not None:
        print(f"  collision probability drops below 50% beyond ~{crossing / 1000:.0f} um")
        print(f"  the field currently blocks within 700 um  (was 325 um)")
    else:
        print("  collision probability never drops below 50% in range --")
        print("  PNS is reporting collisions far outside any modelled clearance,")
        print("  which would mean head_collides() is not what the field models.")

    if refusals:
        r = np.array(refusals)
        print()
        print(f"  gap at the moment fix() refused (n={len(r)}):")
        print(f"    median {np.median(r) / 1000:7.1f} um   "
              f"p10 {np.percentile(r, 10) / 1000:7.1f}   "
              f"p90 {np.percentile(r, 90) / 1000:7.1f}")
        print(f"    refused while INSIDE an obstacle footprint: "
              f"{(r < 0).sum()} of {len(r)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
