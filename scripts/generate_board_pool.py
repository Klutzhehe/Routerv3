#!/usr/bin/env python3
"""Pre-generate board pool for curriculum training.

Runs as a separate process (uses system pcbnew, not the bridge).
Generates N boards per stage with different seeds.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGE_CONFIGS = {
    1: {"num_nets": 1, "num_diff_pairs": 0, "num_length_groups": 0, "group_size": 2},
    2: {"num_nets": 8, "num_diff_pairs": 0, "num_length_groups": 0, "group_size": 2},
    3: {"num_nets": 24, "num_diff_pairs": 0, "num_length_groups": 0, "group_size": 2},
    4: {"num_nets": 8, "num_diff_pairs": 4, "num_length_groups": 0, "group_size": 2},
    5: {"num_nets": 6, "num_diff_pairs": 2, "num_length_groups": 2, "group_size": 3},
    6: {"num_nets": 16, "num_diff_pairs": 4, "num_length_groups": 2, "group_size": 4},
    7: {"num_nets": 40, "num_diff_pairs": 6, "num_length_groups": 3, "group_size": 4},
    8: {"num_nets": 100, "num_diff_pairs": 12, "num_length_groups": 4, "group_size": 8},
}


def generate_board(
    board_path: Path,
    num_nets: int,
    num_diff_pairs: int,
    num_length_groups: int,
    group_size: int,
    seed: int,
    board_width: float = 50.0,
    board_height: float = 50.0,
) -> bool:
    """Generate a single board using pcbworld.data.generate_board."""
    cmd = [
        sys.executable,
        "-m", "pcbworld.data.generate_board",
        str(board_path),
        "--num-nets", str(num_nets),
        "--seed", str(seed),
        "--width-mm", str(board_width),
        "--height-mm", str(board_height),
    ]

    if num_diff_pairs > 0:
        cmd.extend(["--num-diff-pairs", str(num_diff_pairs)])

    if num_length_groups > 0:
        cmd.extend([
            "--num-length-matched-groups", str(num_length_groups),
            "--length-matched-group-size", str(group_size),
        ])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Pre-generate board pool for curriculum")
    parser.add_argument("--output-dir", type=Path, default=Path("/content/boards"), help="Output directory")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], help="Stages to generate")
    parser.add_argument("--pool-size", type=int, default=200, help="Boards per stage")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for generation")
    parser.add_argument("--board-width", type=float, default=50.0, help="Board width (mm)")
    parser.add_argument("--board-height", type=float, default=50.0, help="Board height (mm)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Generating board pool")
    print(f"   Output: {args.output_dir}")
    print(f"   Stages: {args.stages}")
    print(f"   Pool size: {args.pool_size} per stage")
    print(f"   Board size: {args.board_width}x{args.board_height} mm")

    total_generated = 0
    total_failed = 0

    for stage in args.stages:
        if stage not in STAGE_CONFIGS:
            print(f"⚠️  Unknown stage {stage}, skipping")
            continue

        config = STAGE_CONFIGS[stage]
        stage_dir = args.output_dir / f"stage{stage}"
        stage_dir.mkdir(exist_ok=True)

        print(f"\n  Stage {stage}: {config['num_nets']} nets, "
              f"{config['num_diff_pairs']} diff pairs, "
              f"{config['num_length_groups']} length groups")

        for i in range(args.pool_size):
            seed = args.base_seed + stage * 10000 + i
            board_path = stage_dir / f"seed{seed}.kicad_pcb"

            if board_path.exists():
                print(f"    [{i+1}/{args.pool_size}] Exists: seed{seed}")
                total_generated += 1
                continue

            print(f"    [{i+1}/{args.pool_size}] Generating seed{seed}...", end=" ", flush=True)
            success = generate_board(
                board_path,
                config["num_nets"],
                config["num_diff_pairs"],
                config["num_length_groups"],
                config.get("group_size", 2),
                seed,
                args.board_width,
                args.board_height,
            )

            if success:
                print("✅")
                total_generated += 1
            else:
                print("❌")
                total_failed += 1
                board_path.unlink(missing_ok=True)

    print(f"\n{'='*50}")
    print(f"📊 Generation Complete")
    print(f"   Success: {total_generated}")
    print(f"   Failed:  {total_failed}")
    print(f"{'='*50}")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()