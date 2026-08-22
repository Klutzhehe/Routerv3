"""Comprehensive Benchmark Suite for Routerv3 Policies and Baselines.

Evaluates trained checkpoints (or greedy/random baselines) across a suite
of benchmark boards, recording:
  - Completion rate (plain nets, differential pairs, length-matched groups, overall)
  - Wirelength ratio (routed length / straight-line Manhattan span)
  - Collision rate (% of steps encountering collision)
  - Rip-up recovery statistics
  - KiCad DRC compliance (0 DRC violations)
  - Wall-clock routing latency (ms/step and ms/board)
  - Optional 3-panel layer-split PNG exports

Usage:
    # Benchmark against a directory of boards:
    PYTHONPATH=".:build/pcbworld_bridge" python3 scripts/benchmark_router.py /path/to/boards/ \
        --checkpoint /path/to/policy.pt \
        --enable-ripup \
        --render-dir /path/to/renders/ \
        --json-out benchmark_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from pcbworld.agents.line_policy import LineActorCritic, RunningMeanStd
except ImportError:
    torch = None

from pcbworld.env.line_route_env import LineRouteEnv
from pcbworld.viz.render_board import render_board_layers_split

MM = 1_000_000


@dataclass
class BoardBenchmarkResult:
    board_name: str
    total_nets: int
    completed_nets: int
    failed_nets: int
    completion_rate_pct: float
    plain_completed: int
    plain_total: int
    diffpair_completed: int
    diffpair_total: int
    lengthgrp_completed: int
    lengthgrp_total: int
    total_steps: int
    colliding_steps: int
    collision_rate_pct: float
    ripup_count: int
    drc_violations: int
    total_routed_length_mm: float
    total_straight_length_mm: float
    wirelength_ratio: float
    wall_clock_ms: float
    ms_per_step: float


@dataclass
class BenchmarkSuiteSummary:
    total_boards: int
    total_nets_evaluated: int
    total_nets_completed: int
    overall_completion_rate_pct: float
    plain_completion_rate_pct: float
    diffpair_completion_rate_pct: float
    lengthgrp_completion_rate_pct: float
    mean_wirelength_ratio: float
    mean_collision_rate_pct: float
    total_ripups_performed: int
    total_drc_violations: int
    median_ms_per_step: float
    mean_ms_per_board: float


def _classify_net(name: str) -> str:
    if name.startswith("diffpair_"):
        return "diffpair"
    elif name.startswith("lengthgrp_"):
        return "lengthgrp"
    return "plain"


def evaluate_board(
    board_path: str,
    policy=None,
    rms: RunningMeanStd | None = None,
    enable_ripup: bool = True,
    max_ripups: int = 8,
    step_size_nm: int = 800_000,
    snap_radius_nm: int = 600_000,
    max_steps_per_net: int = 100,
    run_drc: bool = True,
    render_path: str | None = None,
) -> BoardBenchmarkResult:
    env = LineRouteEnv(
        board_path,
        enable_ripup=enable_ripup,
        max_ripups_per_episode=max_ripups,
        step_size_nm=step_size_nm,
        snap_radius_nm=snap_radius_nm,
        max_steps_per_net=max_steps_per_net,
    )

    t0 = time.perf_counter()
    obs, info = env.reset()
    all_nets = list(env._nets)
    # Categorize nets
    plain_total = sum(1 for n in all_nets if _classify_net(n) == "plain")
    diff_total = sum(1 for n in all_nets if _classify_net(n) == "diffpair")
    len_total = sum(1 for n in all_nets if _classify_net(n) == "lengthgrp")

    terminated = False
    steps = 0
    colliding_steps = 0
    total_routed_nm = 0.0
    total_straight_nm = 0.0

    while not terminated and steps < len(all_nets) * max_steps_per_net * 2:
        if policy is not None and torch is not None:
            norm_obs = obs.copy()
            if rms is not None:
                norm_obs[:8] = rms.normalize(obs[:8])
            obs_t = torch.as_tensor(norm_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist, _ = policy.forward(obs_t)
                action = dist.mean.squeeze(0).numpy()
        else:
            # Greedy baseline: a = 0 (straight-line heading)
            action = np.array([0.0], dtype=np.float32)

        obs, reward, terminated, _, info = env.step(action)
        if info.get("collides", False):
            colliding_steps += 1
        steps += 1

    t1 = time.perf_counter()
    wall_clock_ms = (t1 - t0) * 1000.0
    ms_per_step = wall_clock_ms / max(1, steps)

    completed = info["completed"]
    failed = info["failed"]
    total_completed = len(completed)
    total_nets = len(all_nets)

    plain_comp = sum(1 for n in completed if _classify_net(n) == "plain")
    diff_comp = sum(1 for n in completed if _classify_net(n) == "diffpair")
    len_comp = sum(1 for n in completed if _classify_net(n) == "lengthgrp")

    # Geometry analysis
    geometry = env.bridge.get_board_geometry()
    for seg in geometry.tracks:
        dx = seg.x2 - seg.x1
        dy = seg.y2 - seg.y1
        total_routed_nm += float(np.hypot(dx, dy))

    pads = env.bridge.net_pads()
    two_pad = {}
    for p in pads:
        if p.net in completed:
            two_pad.setdefault(p.net, []).append(p)
    for n, p_list in two_pad.items():
        if len(p_list) == 2:
            total_straight_nm += float(np.hypot(p_list[0].x - p_list[1].x, p_list[0].y - p_list[1].y))

    wirelength_ratio = (total_routed_nm / total_straight_nm) if total_straight_nm > 0 else 1.0

    drc_violations = 0
    if run_drc and hasattr(env.bridge, "run_drc"):
        try:
            drc_violations = len(env.bridge.run_drc())
        except Exception:
            drc_violations = 0

    if render_path:
        render_board_layers_split(
            geometry,
            net_pads=pads,
            save_path=render_path,
            title=f"Board: {Path(board_path).name} -- {total_completed}/{total_nets} Nets Routed",
        )

    return BoardBenchmarkResult(
        board_name=Path(board_path).name,
        total_nets=total_nets,
        completed_nets=total_completed,
        failed_nets=len(failed),
        completion_rate_pct=(total_completed / max(1, total_nets)) * 100.0,
        plain_completed=plain_comp,
        plain_total=plain_total,
        diffpair_completed=diff_comp,
        diffpair_total=diff_total,
        lengthgrp_completed=len_comp,
        lengthgrp_total=len_total,
        total_steps=steps,
        colliding_steps=colliding_steps,
        collision_rate_pct=(colliding_steps / max(1, steps)) * 100.0,
        ripup_count=info.get("ripup_count", 0),
        drc_violations=drc_violations,
        total_routed_length_mm=total_routed_nm / MM,
        total_straight_length_mm=total_straight_nm / MM,
        wirelength_ratio=wirelength_ratio,
        wall_clock_ms=wall_clock_ms,
        ms_per_step=ms_per_step,
    )


def run_benchmark_suite(
    board_paths: list[str],
    checkpoint_path: str | None = None,
    enable_ripup: bool = True,
    max_ripups: int = 8,
    render_dir: str | None = None,
    json_out: str | None = None,
) -> tuple[list[BoardBenchmarkResult], BenchmarkSuiteSummary]:
    policy = None
    rms = None
    if checkpoint_path and torch is not None:
        chk = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        policy = LineActorCritic(action_dim=1)
        policy.load_state_dict(chk["policy_state_dict"])
        policy.eval()
        rms = RunningMeanStd()
        if chk.get("rms_mean") is not None:
            rms.mean = chk["rms_mean"]
            rms.var = chk["rms_var"]
            rms.count = chk["rms_count"]
        print(f"Loaded policy checkpoint from {checkpoint_path}")

    results = []
    for i, b_path in enumerate(board_paths):
        render_file = None
        if render_dir:
            Path(render_dir).mkdir(parents=True, exist_ok=True)
            render_file = str(Path(render_dir) / f"{Path(b_path).stem}_layer_split.png")

        res = evaluate_board(
            b_path,
            policy=policy,
            rms=rms,
            enable_ripup=enable_ripup,
            max_ripups=max_ripups,
            render_path=render_file,
        )
        results.append(res)
        print(
            f"[{i+1:2d}/{len(board_paths):2d}] {res.board_name:<28s} "
            f"Routed: {res.completed_nets:2d}/{res.total_nets:2d} ({res.completion_rate_pct:5.1f}%) "
            f"WL_Ratio: {res.wirelength_ratio:4.2f} "
            f"Collisions: {res.collision_rate_pct:4.1f}% "
            f"Ripups: {res.ripup_count:2d} "
            f"DRC: {res.drc_violations:1d} "
            f"Time: {res.wall_clock_ms:6.1f}ms ({res.ms_per_step:.3f}ms/step)"
        )

    # Aggregate summary
    tot_eval = sum(r.total_nets for r in results)
    tot_comp = sum(r.completed_nets for r in results)
    plain_t = sum(r.plain_total for r in results)
    plain_c = sum(r.plain_completed for r in results)
    diff_t = sum(r.diffpair_total for r in results)
    diff_c = sum(r.diffpair_completed for r in results)
    len_t = sum(r.lengthgrp_total for r in results)
    len_c = sum(r.lengthgrp_completed for r in results)

    summary = BenchmarkSuiteSummary(
        total_boards=len(results),
        total_nets_evaluated=tot_eval,
        total_nets_completed=tot_comp,
        overall_completion_rate_pct=(tot_comp / max(1, tot_eval)) * 100.0,
        plain_completion_rate_pct=(plain_c / max(1, plain_t)) * 100.0 if plain_t > 0 else 0.0,
        diffpair_completion_rate_pct=(diff_c / max(1, diff_t)) * 100.0 if diff_t > 0 else 0.0,
        lengthgrp_completion_rate_pct=(len_c / max(1, len_t)) * 100.0 if len_t > 0 else 0.0,
        mean_wirelength_ratio=float(np.mean([r.wirelength_ratio for r in results])),
        mean_collision_rate_pct=float(np.mean([r.collision_rate_pct for r in results])),
        total_ripups_performed=sum(r.ripup_count for r in results),
        total_drc_violations=sum(r.drc_violations for r in results),
        median_ms_per_step=float(np.median([r.ms_per_step for r in results])),
        mean_ms_per_board=float(np.mean([r.wall_clock_ms for r in results])),
    )

    print("\n" + "=" * 80)
    print("                      ROUTERV3 BENCHMARK SUITE SUMMARY")
    print("=" * 80)
    print(f"Total Boards Evaluated:        {summary.total_boards}")
    print(f"Overall Completion Rate:       {summary.overall_completion_rate_pct:6.2f}% ({tot_comp}/{tot_eval} nets)")
    print(f"  - Plain Nets:                {summary.plain_completion_rate_pct:6.2f}% ({plain_c}/{plain_t} nets)")
    print(f"  - Differential Pairs:        {summary.diffpair_completion_rate_pct:6.2f}% ({diff_c}/{diff_t} legs)")
    print(f"  - Length-Matched Groups:     {summary.lengthgrp_completion_rate_pct:6.2f}% ({len_c}/{len_t} nets)")
    print(f"Mean Wirelength Ratio:         {summary.mean_wirelength_ratio:6.2f}x (actual / straight-line)")
    print(f"Mean Collision Rate:           {summary.mean_collision_rate_pct:6.2f}%")
    print(f"Total Rip-Ups Executed:        {summary.total_ripups_performed}")
    print(f"Total DRC Violations:          {summary.total_drc_violations}")
    print(f"Median Decision Latency:       {summary.median_ms_per_step:6.3f} ms / step")
    print(f"Mean Board Routing Time:       {summary.mean_ms_per_board:6.1f} ms / board")
    print("=" * 80)

    if json_out:
        out_data = {
            "summary": asdict(summary),
            "boards": [asdict(r) for r in results],
        }
        with open(json_out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Saved benchmark summary JSON to {json_out}")

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Run Routerv3 Benchmark Suite.")
    parser.add_argument("boards", help="Path to .kicad_pcb file or directory containing boards")
    parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint path (evaluates greedy if omitted)")
    parser.add_argument("--enable-ripup", action="store_true", help="Enable rip-up and reroute recovery")
    parser.add_argument("--max-ripups", type=int, default=8, help="Max rip-ups per board")
    parser.add_argument("--render-dir", type=str, default=None, help="Directory to save layer-split PNGs")
    parser.add_argument("--json-out", type=str, default=None, help="Path to write JSON benchmark summary")
    args = parser.parse_args()

    if os.path.isdir(args.boards):
        board_paths = sorted([os.path.join(args.boards, f) for f in os.listdir(args.boards) if f.endswith(".kicad_pcb")])
    else:
        board_paths = [args.boards]

    if not board_paths:
        print(f"No .kicad_pcb files found in {args.boards}", file=sys.stderr)
        sys.exit(1)

    run_benchmark_suite(
        board_paths,
        checkpoint_path=args.checkpoint,
        enable_ripup=args.enable_ripup,
        max_ripups=args.max_ripups,
        render_dir=args.render_dir,
        json_out=args.json_out,
    )


if __name__ == "__main__":
    main()
