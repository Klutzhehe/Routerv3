"""Live Benchmark & Colab Runner for Hierarchical Multi-Model PCB Routing.

Follows ROADMAP.md and AGENTS.md stage markers:
=== STAGE: ... ===
Runs Phases 0-4 end-to-end against pcbworld_pns_bridge with live stats output.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import pcbworld_pns_bridge
    HAVE_BRIDGE = True
except ImportError:
    HAVE_BRIDGE = False
    from tests.fake_bridge import FakePNSBridge as pcbworld_pns_bridge

from pcbworld.hierarchical.orchestrator import HierarchicalOrchestrator
from pcbworld.hierarchical.specs import PipelinePhase


def main():
    parser = argparse.ArgumentParser(description="Run live hierarchical routing pipeline benchmark")
    parser.add_argument("--board", type=str, default="board_hierarchical.kicad_pcb", help="Path to .kicad_pcb board file")
    parser.add_argument("--num-nets", type=int, default=10, help="Number of plain bulk nets")
    parser.add_argument("--num-diff-pairs", type=int, default=2, help="Number of differential pairs")
    parser.add_argument("--num-length-groups", type=int, default=2, help="Number of length-matched groups")
    parser.add_argument("--group-size", type=int, default=4, help="Members per length-matched group")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for board generation")
    args = parser.parse_args()

    print("=================================================================")
    print("=== STAGE 0: Generating Multi-Tier Synthetic Board Subprocess ===")
    print("=================================================================")
    gen_script = os.path.join(os.path.dirname(__file__), "..", "pcbworld", "data", "generate_board.py")
    gen_cmd = [
        sys.executable,
        gen_script,
        "--output", args.board,
        "--num-nets", str(args.num_nets),
        "--num-diff-pairs", str(args.num_diff_pairs),
        "--num-length-matched-groups", str(args.num_length_groups),
        "--length-matched-group-size", str(args.group_size),
        "--seed", str(args.seed),
    ]

    t0 = time.time()
    try:
        subprocess.check_call(gen_cmd)
        print(f"[PASS] Board generated successfully: {args.board} in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"[WARN] Board generation via subprocess: {e} (using existing board if present)")

    print("\n=================================================================")
    print("=== STAGE 1: Initializing PNS Bridge & Hierarchical Router ===")
    print("=================================================================")
    bridge = pcbworld_pns_bridge.PNSBridge() if hasattr(pcbworld_pns_bridge, "PNSBridge") else pcbworld_pns_bridge()
    loaded = bridge.load_board(args.board)
    print(f"Board loaded into bridge: {loaded} (HAVE_REAL_BRIDGE={HAVE_BRIDGE})")

    orchestrator = HierarchicalOrchestrator(bridge=bridge)

    print("\n=================================================================")
    print("=== STAGE 2: Executing Hierarchical 5-Phase Routing Pipeline ===")
    print("=================================================================")
    start_time = time.time()
    report = orchestrator.route_board()
    elapsed = time.time() - start_time

    print(f"\nPipeline Finished in {elapsed * 1000:.2f} ms")
    print(f"Total Nets in Netlist : {report.total_nets}")
    print(f"Total Nets Routed     : {report.routed_nets} / {report.total_nets} ({report.routed_nets / max(1, report.total_nets) * 100:.1f}%)")
    print(f"Diff Pairs Routed     : {report.diff_pairs_routed} / {report.diff_pairs_total}")
    print(f"Length Groups Tuned   : {report.length_groups_tuned} / {report.length_groups_total}")
    print(f"Bulk Nets Routed      : {report.bulk_nets_routed} / {report.bulk_nets_total}")
    print(f"Rip-Up Arbitrations   : {report.ripup_count}")

    print("\n--- Phase 1: High-Speed Diff-Pairs ---")
    for r in report.phase_results.get(PipelinePhase.PHASE_1_DIFF_PAIR, []):
        status = "PASS" if r.success else "FAIL"
        print(f"  [{status}] Pair {r.net_name:20s} | Length: {r.wirelength_nm / 1e6:6.2f} mm | Skew: {r.skew_nm / 1e6:5.3f} mm")

    print("\n--- Phase 2: Synchronous Bus Baseline & Reservation ---")
    for r in report.phase_results.get(PipelinePhase.PHASE_2_BUS_RESERVATION, []):
        status = "PASS" if r.success else "FAIL"
        print(f"  [{status}] Bus Net {r.net_name:16s} | Baseline: {r.wirelength_nm / 1e6:6.2f} mm")

    print("\n--- Phase 3: Bulk Single-Ended Routing ---")
    for r in report.phase_results.get(PipelinePhase.PHASE_3_BULK_ROUTING, []):
        status = "PASS" if r.success else "FAIL"
        print(f"  [{status}] Bulk Net {r.net_name:15s} | Length: {r.wirelength_nm / 1e6:6.2f} mm | Segments: {r.num_segments}")

    print("\n--- Phase 4: Meander Expansion & Final Length Tuning ---")
    for r in report.phase_results.get(PipelinePhase.PHASE_4_MEANDER_TUNING, []):
        status = "PASS" if r.success else "FAIL"
        print(f"  [{status}] Tuned Net {r.net_name:14s} | Final Length: {r.wirelength_nm / 1e6:6.2f} mm | ΔL Residual: {r.length_mismatch_nm / 1e6:5.3f} mm")

    print("\n=================================================================")
    print("=== STAGE 3: Saving Board & Full DRC Engine Sign-Off ===")
    print("=================================================================")
    if hasattr(bridge, "save_board"):
        saved = bridge.save_board(args.board)
        print(f"Board saved back to disk: {saved}")

    print(f"DRC Violations Reported: {report.drc_violations}")
    if report.drc_violations == 0:
        print("[PASS] Full board DRC is 100% CLEAN!")
    else:
        print(f"[WARN] Board has {report.drc_violations} DRC violations")

    print("\n=================================================================")
    print(f"=== PIPELINE STATUS: {'SUCCESS [100% ROUTED & DRC CLEAN]' if report.all_clean else 'COMPLETE'} ===")
    print("=================================================================")


if __name__ == "__main__":
    main()
