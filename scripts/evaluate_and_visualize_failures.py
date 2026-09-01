"""1,000-Board Evaluation Scanner & Failure Visualizer.

Iterates across 1,000 boards, identifies all routing failures or DRC violations,
and generates high-resolution Matplotlib / SVG visual diagrams showing the routed tracks,
unrouted airwires, and congestion points.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import pcbworld_pns_bridge as bridge_mod
    HAVE_REAL_BRIDGE = True
except ImportError:
    HAVE_REAL_BRIDGE = False
    from tests.fake_bridge import FakePNSBridge as bridge_mod

from pcbworld.hierarchical.orchestrator import HierarchicalOrchestrator
from pcbworld.hierarchical.specs import PipelinePhase, HierarchicalPipelineReport


def render_board_failure_diagram(
    bridge: Any,
    report: HierarchicalPipelineReport,
    output_png: Path,
    board_width_mm: float = 50.0,
    board_height_mm: float = 50.0,
):
    """Renders a detailed visual layout of octilinear routed copper tracks, pads, and unrouted airwires."""
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_facecolor("#181c24")  # Dark CAD theme

    # 1. Draw Board Perimeter
    board_rect = patches.Rectangle(
        (0, 0), board_width_mm, board_height_mm,
        linewidth=2, edgecolor="#50fa7b", facecolor="#1e222b"
    )
    ax.add_patch(board_rect)

    # 2. Extract Real Board Geometry Tracks & Vias
    geom = bridge.get_board_geometry() if hasattr(bridge, "get_board_geometry") else None
    seen_nets = set()

    if geom and hasattr(geom, "tracks"):
        for t in geom.tracks:
            seen_nets.add(t.net)
            # High-speed diff pair = Cyan, Bus = Green, Bulk = Purple
            color = "#00f0ff" if "diff" in t.net.lower() else ("#50fa7b" if "clk" in t.net.lower() or "d" in t.net.lower() else "#bd93f9")
            ax.plot(
                [t.x1 / 1e6, t.x2 / 1e6], [t.y1 / 1e6, t.y2 / 1e6],
                color=color, linewidth=max(1.5, t.width / 1e6 * 4.0),
                solid_capstyle="round", alpha=0.9, zorder=3
            )

        for v in getattr(geom, "vias", []):
            ax.add_patch(patches.Circle((v.x / 1e6, v.y / 1e6), v.diameter / 1e6 / 2, color="#f1fa8c", alpha=0.9, zorder=4))
            ax.add_patch(patches.Circle((v.x / 1e6, v.y / 1e6), v.drill / 1e6 / 2, color="#181c24", zorder=5))

    # 3. Extract Pads
    pads = bridge.net_pads() if hasattr(bridge, "net_pads") else []
    net_to_pads: Dict[str, List[Any]] = {}
    for p in pads:
        net_to_pads.setdefault(p.net, []).append(p)
        px_mm, py_mm = p.x / 1e6, p.y / 1e6
        pad_circ = patches.Circle((px_mm, py_mm), 0.45, facecolor="#f1fa8c", edgecolor="#ffb86c", linewidth=1.5, zorder=6)
        ax.add_patch(pad_circ)
        ax.text(px_mm + 0.6, py_mm + 0.6, p.pad_name, color="#8be9fd", fontsize=7, zorder=7, alpha=0.9)

    # 4. Draw Unrouted Airwires (Red Dashed)
    for net_name, net_pads in net_to_pads.items():
        if net_name not in seen_nets and len(net_pads) >= 2:
            p1, p2 = net_pads[0], net_pads[1]
            ax.plot(
                [p1.x / 1e6, p2.x / 1e6], [p1.y / 1e6, p2.y / 1e6],
                color="#ff5555", linestyle="--", linewidth=2.0, alpha=0.85, zorder=5
            )
            mid_x = (p1.x + p2.x) / 2e6
            mid_y = (p1.y + p2.y) / 2e6
            ax.scatter([mid_x], [mid_y], color="#ff5555", marker="x", s=120, linewidths=2.5, zorder=8)
            ax.text(mid_x + 0.6, mid_y, f"UNROUTED: {net_name}", color="#ff5555", fontsize=8, fontweight="bold", zorder=9)

    # 5. Draw DRC Violations (Yellow Triangle Markers)
    if hasattr(bridge, "run_drc"):
        drc_violations = bridge.run_drc()
        for v in drc_violations:
            if v.x != 0 or v.y != 0:
                ax.scatter(
                    [v.x / 1e6], [v.y / 1e6],
                    color="#ffb86c", marker="^", s=120, edgecolors="#ff5555", linewidths=1.5, zorder=9
                )

    ax.set_xlim(-2, board_width_mm + 2)
    ax.set_ylim(-2, board_height_mm + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)", color="#f8f8f2", fontsize=11)
    ax.set_ylabel("Y (mm)", color="#f8f8f2", fontsize=11)
    ax.tick_params(colors="#f8f8f2")

    title = f"Board Analysis | Routed: {report.routed_nets}/{report.total_nets} | DRC Errors: {report.drc_violations}"
    ax.set_title(title, color="#f8f8f2", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def scan_and_visualize_boards(
    board_dir: Path,
    output_vis_dir: Path,
    stages: Optional[List[int]] = None,
    boards_per_stage: int = 10,
    render_all: bool = True,
    max_visualizations: int = 20,
) -> Dict[str, Any]:
    """Scans selected stages, evaluates them with the hierarchical router, and saves wiring diagrams."""
    bridge = bridge_mod.PNSBridge() if hasattr(bridge_mod, "PNSBridge") else bridge_mod()
    orchestrator = HierarchicalOrchestrator(bridge=bridge)

    target_stages = stages or [1, 2, 3]
    output_vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Scanning Stages {target_stages} ({boards_per_stage} boards/stage) in {board_dir}...")

    total_scanned = 0
    clean_boards = 0
    failed_boards = 0
    vis_count = 0
    t_start = time.time()

    print("="*80)
    print(f"{'Stage':<10} | {'Board File':<20} | {'Status':<8} | {'Nets':<8} | {'DiffPairs':<10} | {'DRC':<5} | {'Time':<7}")
    print("="*80)

    for st in target_stages:
        stage_dir = board_dir / f"stage{st}"
        if not stage_dir.exists():
            print(f"⚠️ Stage directory {stage_dir} not found. Skipping.")
            continue

        b_files = sorted(list(stage_dir.glob("*.kicad_pcb")))[:boards_per_stage]
        for bfile in b_files:
            total_scanned += 1
            t_b0 = time.time()
            bridge.load_board(str(bfile))
            if hasattr(bridge, "set_collision_mode") and hasattr(bridge_mod, "RM_MARK_OBSTACLES"):
                bridge.set_collision_mode(bridge_mod.RM_MARK_OBSTACLES)

            report = orchestrator.route_board()
            elapsed_ms = (time.time() - t_b0) * 1000

            status = "✅ CLEAN" if report.all_clean else "⚠️ FAIL"
            if report.all_clean:
                clean_boards += 1
            else:
                failed_boards += 1

            # Render wiring layout (either failure or clean wiring)
            if vis_count < max_visualizations and (render_all or not report.all_clean):
                vis_count += 1
                prefix = "routed" if report.all_clean else "fail"
                out_png = output_vis_dir / f"{prefix}_stage{st}_{bfile.stem}.png"
                render_board_failure_diagram(bridge, report, out_png)

            print(f"Stage {st:<4d} | {bfile.stem:<20s} | {status:<8s} | {report.routed_nets:>2d}/{report.total_nets:<2d}   | {report.diff_pairs_routed:>2d}/{report.diff_pairs_total:<2d}       | {report.drc_violations:<5d} | {elapsed_ms:>5.1f}ms")

    total_time = time.time() - t_start
    clean_pct = (clean_boards / max(1, total_scanned)) * 100.0

    print("="*80)
    print(f"📊 SELECTIVE SCAN SUMMARY (Stages {target_stages}):")
    print(f"   Total Boards Evaluated : {total_scanned}")
    print(f"   100% Clean Boards      : {clean_boards} ({clean_pct:.1f}%)")
    print(f"   Boards with Issues     : {failed_boards} ({100.0 - clean_pct:.1f}%)")
    print(f"   Wiring Visualizations  : {vis_count} saved to {output_vis_dir}")
    print(f"   Total Scan Time        : {total_time:.2f}s ({total_time/max(1, total_scanned)*1000:.1f} ms/board)")
    print("="*80)

    return {
        "total_scanned": total_scanned,
        "clean_boards": clean_boards,
        "failed_boards": failed_boards,
        "vis_count": vis_count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selective Stage Router Evaluation & Wiring Visualizer")
    parser.add_argument("--board-dir", type=Path, default=Path("/content/boards"))
    parser.add_argument("--vis-dir", type=Path, default=Path("/content/wiring_vis"))
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3], help="Stages to evaluate (e.g. 1 2 3)")
    parser.add_argument("--boards-per-stage", type=int, default=5, help="Number of boards per stage to evaluate")
    parser.add_argument("--render-all", action="store_true", default=True, help="Render both clean and failed wiring")
    args = parser.parse_args()

    scan_and_visualize_boards(
        args.board_dir,
        args.vis_dir,
        stages=args.stages,
        boards_per_stage=args.boards_per_stage,
        render_all=args.render_all,
    )
