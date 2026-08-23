"""Why do ~36% of stage-1 nets never reach their pad?

Four training runs and three reward designs later, the oracle settled what
this is NOT. Steering by the cost-to-go field perfectly, with no network and
no exploration, scored 64.52% (80/124) against the greedy straight line's
63.71% (79/124) -- one net, on 124. And in all three modes every single
failure was "never reached the pad", with zero fix() refusals.

Two facts do not fit the story everyone has been telling:

  - The budget is not tight. 120 steps x 0.5 mm is 60 mm of travel; the
    longest possible net on a 35 mm board is 49.5 mm, and the geodesic
    overhead is a median 1.073x. A moving head arrives with room to spare.
  - When a head does arrive, fix() has never once refused it.

So a net that no policy can route is a net whose head is not going anywhere,
and the thing every mode shares is what happens before the first step:
_pad_candidate() finding the endpoints and start_route() taking charge.

This walks every net on every board with a = 0 and records, per net, whether
the router ever started, what pad ids it was handed, how far the head
actually travelled, and how much of the gap it closed. If the failures line
up with route_never_started, then no amount of RL was ever going to move
them and four rounds of reward engineering were aimed at the wrong thing.

    python scripts/diagnose_net_failures.py /content/curriculum_dataset/stage1_basics
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbworld.env.line_route_env import LineRouteEnv  # noqa: E402

MM = 1_000_000


def diagnose_board(board_path: str, max_steps_per_net: int = 120) -> list[dict]:
    env = LineRouteEnv(
        board_path,
        step_size_nm=500_000,
        snap_radius_nm=400_000,
        max_steps_per_net=max_steps_per_net,
    )
    env.reset()

    rows: list[dict] = []
    seen = -1
    guard = 0
    limit = len(env._nets) * max_steps_per_net * 3

    while guard < limit:
        # Snapshot each net exactly once, at the step where it becomes current:
        # _start_ok and the pad ids are set by _begin_net and are the whole
        # point of this script.
        if env._net_index != seen and env._net_index < len(env._nets):
            seen = env._net_index
            rows.append(
                {
                    "board": Path(board_path).name,
                    "net": env._nets[env._net_index],
                    "started": bool(env._start_ok),
                    "start_pad_id": int(env._start_pad_id),
                    "target_pad_id": int(env._target_id),
                    "straight_mm": env._straight_len / MM,
                    "field": env._field is not None,
                }
            )

        _obs, _r, terminated, _t, info = env.step(np.array([0.0], dtype=np.float32))
        guard += 1
        if terminated:
            break

    completed = set(info["completed"])
    reasons = info.get("failure_reasons", {})
    progress = info.get("failure_progress", {})
    travel = info.get("failure_travel_nm", {})
    for r in rows:
        r["completed"] = r["net"] in completed
        r["reason"] = reasons.get(r["net"], "" if r["completed"] else "unknown")
        r["progress"] = progress.get(r["net"], 1.0 if r["completed"] else float("nan"))
        r["travel_mm"] = travel.get(r["net"], float("nan")) / MM
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("boards", help=".kicad_pcb file or a directory of them")
    ap.add_argument("--max-steps-per-net", type=int, default=120)
    args = ap.parse_args()

    if os.path.isdir(args.boards):
        paths = sorted(
            os.path.join(args.boards, f)
            for f in os.listdir(args.boards)
            if f.endswith(".kicad_pcb")
        )
    else:
        paths = [args.boards]

    rows: list[dict] = []
    for i, path in enumerate(paths):
        rows.extend(diagnose_board(path, args.max_steps_per_net))
        print(f"[{i + 1:2d}/{len(paths)}] {Path(path).name}", flush=True)

    n = len(rows)
    done = [r for r in rows if r["completed"]]
    failed = [r for r in rows if not r["completed"]]
    never = [r for r in failed if not r["started"]]
    moved = [r for r in failed if r["started"]]

    print("\n" + "=" * 78)
    print("                 WHY NETS FAIL -- PER-NET BREAKDOWN")
    print("=" * 78)
    print(f"nets examined              {n}")
    print(f"  completed                {len(done):4d}  ({len(done) / n * 100:5.1f}%)")
    print(f"  failed                   {len(failed):4d}  ({len(failed) / n * 100:5.1f}%)")
    print()
    print("OF THE FAILURES:")
    print(f"  router never started     {len(never):4d}"
          f"  ({len(never) / max(len(failed), 1) * 100:5.1f}%)  <- NO policy can move these")
    print(f"  started but never arrived{len(moved):4d}"
          f"  ({len(moved) / max(len(failed), 1) * 100:5.1f}%)  <- a real steering problem")

    if never:
        bad_start = sum(1 for r in never if r["start_pad_id"] < 0)
        bad_target = sum(1 for r in never if r["target_pad_id"] < 0)
        print()
        print("  never-started nets, pad lookup:")
        print(f"    start pad id  == -1 (not found)  {bad_start:4d} of {len(never)}")
        print(f"    target pad id == -1 (not found)  {bad_target:4d} of {len(never)}")
        print(f"    both ids found, start_route() still refused "
              f"{sum(1 for r in never if r['start_pad_id'] >= 0 and r['target_pad_id'] >= 0):4d}")

    if moved:
        prog = np.array([r["progress"] for r in moved], dtype=float)
        trav = np.array([r["travel_mm"] for r in moved], dtype=float)
        need = np.array([r["straight_mm"] for r in moved], dtype=float)
        print()
        print("  started-but-failed nets:")
        print(f"    distance closed   median {np.nanmedian(prog) * 100:5.1f}%   "
              f"p90 {np.nanpercentile(prog, 90) * 100:5.1f}%")
        print(f"    head travelled    median {np.nanmedian(trav):5.1f} mm  "
              f"against a {np.nanmedian(need):5.1f} mm straight line")
        print(f"    budget allows            {args.max_steps_per_net * 0.5:5.1f} mm")
        stalled = int(np.sum(trav < 1.0))
        print(f"    barely moved at all (<1 mm travelled)  {stalled} of {len(moved)}")

    print()
    lengths_ok = np.array([r["straight_mm"] for r in done], dtype=float)
    lengths_no = np.array([r["straight_mm"] for r in failed], dtype=float)
    if len(lengths_ok) and len(lengths_no):
        print(f"net length, completed  median {np.median(lengths_ok):5.1f} mm")
        print(f"net length, failed     median {np.median(lengths_no):5.1f} mm")
    print("=" * 78)


if __name__ == "__main__":
    main()
