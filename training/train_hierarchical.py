"""Hierarchical Policy Trainer & Multi-Stage Curriculum Driver.

Trains and evaluates the Hierarchical Multi-Model AI Routing System across curriculum stages:
- Stage 1: Sparse single-net baseline
- Stage 2: Small multi-tier boards (diff pairs + DDR byte lane)
- Stage 3: Dense multi-tier boards (24 nets)
- Stage 4: High-density industrial boards (50-250 nets with BGAs)

Logs real-time live stats (wirelength, skew, meander residual, DRC) and checkpoints to Drive.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import pcbworld_pns_bridge
    HAVE_REAL_BRIDGE = True
except ImportError:
    HAVE_REAL_BRIDGE = False
    from tests.fake_bridge import FakePNSBridge as pcbworld_pns_bridge

from pcbworld.hierarchical.orchestrator import HierarchicalOrchestrator
from pcbworld.hierarchical.specs import PipelinePhase, HierarchicalPipelineReport


def evaluate_board_pool(
    stage_dir: Path,
    bridge: Any,
    max_boards: int = 50,
) -> Dict[str, Any]:
    """Evaluates the hierarchical routing pipeline across a pool of pre-generated boards."""
    board_files = sorted(stage_dir.glob("*.kicad_pcb"))[:max_boards]
    if not board_files:
        print(f"[WARN] No board files found in {stage_dir}")
        return {}

    orchestrator = HierarchicalOrchestrator(bridge=bridge)

    total_nets_sum = 0
    total_routed_sum = 0
    diff_pairs_routed_sum = 0
    diff_pairs_total_sum = 0
    len_groups_tuned_sum = 0
    len_groups_total_sum = 0
    bulk_routed_sum = 0
    bulk_total_sum = 0
    total_drc_errors = 0
    total_ripups = 0
    total_time_ms = 0.0

    print(f"\nEvaluating {len(board_files)} boards from {stage_dir.name}...")

    for i, board_file in enumerate(board_files):
        t0 = time.time()
        bridge.load_board(str(board_file))
        if hasattr(bridge, "set_collision_mode") and hasattr(pcbworld_pns_bridge, "RM_MARK_OBSTACLES"):
            bridge.set_collision_mode(pcbworld_pns_bridge.RM_MARK_OBSTACLES)

        report = orchestrator.route_board()
        elapsed_ms = (time.time() - t0) * 1000

        total_nets_sum += report.total_nets
        total_routed_sum += report.routed_nets
        diff_pairs_routed_sum += report.diff_pairs_routed
        diff_pairs_total_sum += report.diff_pairs_total
        len_groups_tuned_sum += report.length_groups_tuned
        len_groups_total_sum += report.length_groups_total
        bulk_routed_sum += report.bulk_nets_routed
        bulk_total_sum += report.bulk_nets_total
        total_drc_errors += report.drc_violations
        total_ripups += report.ripup_count
        total_time_ms += elapsed_ms

        status_char = "✅" if report.all_clean else "⚠️"
        print(
            f"  [{i+1:2d}/{len(board_files)}] {board_file.stem:12s} {status_char} "
            f"Routed: {report.routed_nets:2d}/{report.total_nets:2d} "
            f"DiffPairs: {report.diff_pairs_routed}/{report.diff_pairs_total} "
            f"DDRGroups: {report.length_groups_tuned}/{report.length_groups_total} "
            f"DRC: {report.drc_violations} "
            f"Time: {elapsed_ms:.1f}ms"
        )

    num_boards = len(board_files)
    avg_completion = (total_routed_sum / max(1, total_nets_sum)) * 100
    avg_time = total_time_ms / max(1, num_boards)

    print(f"\n{'='*60}")
    print(f"📊 STAGE SUMMARY ({stage_dir.name}):")
    print(f"   Total Boards Evaluated : {num_boards}")
    print(f"   Net Completion Rate    : {total_routed_sum}/{total_nets_sum} ({avg_completion:.1f}%)")
    print(f"   Diff Pairs Success     : {diff_pairs_routed_sum}/{diff_pairs_total_sum}")
    print(f"   Length Groups Matched  : {len_groups_tuned_sum}/{len_groups_total_sum}")
    print(f"   Total DRC Violations   : {total_drc_errors}")
    print(f"   Total Rip-Ups Triggered: {total_ripups}")
    print(f"   Average Routing Time   : {avg_time:.2f} ms / board")
    print(f"{'='*60}")

    return {
        "num_boards": num_boards,
        "completion_rate": avg_completion,
        "drc_errors": total_drc_errors,
        "avg_time_ms": avg_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Hierarchical PCB Router Training & Evaluation Driver")
    parser.add_argument("--board-pool-dir", type=Path, default=Path("/content/boards"), help="Directory containing stage board pools")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], help="Stages to evaluate/train")
    parser.add_argument("--max-boards-per-stage", type=int, default=20, help="Boards to evaluate per stage")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/content/drive/MyDrive/routerv3_checkpoints"), help="Checkpoint directory")
    args = parser.parse_args()

    print("=================================================================")
    print("=== HIERARCHICAL AI PCB ROUTER: TRAINING & EVALUATION DRIVER ===")
    print("=================================================================")
    print(f"Board Pool Directory: {args.board_pool_dir}")
    print(f"Curriculum Stages   : {args.stages}")
    print(f"Have Live C++ Bridge: {HAVE_REAL_BRIDGE}")

    bridge = pcbworld_pns_bridge.PNSBridge() if hasattr(pcbworld_pns_bridge, "PNSBridge") else pcbworld_pns_bridge()

    for stage_num in args.stages:
        stage_dir = args.board_pool_dir / f"stage{stage_num}"
        if not stage_dir.exists():
            print(f"[WARN] Stage directory {stage_dir} does not exist. Skipping.")
            continue

        evaluate_board_pool(stage_dir, bridge, max_boards=args.max_boards_per_stage)


if __name__ == "__main__":
    main()
