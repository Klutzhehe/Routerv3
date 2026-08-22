"""Generates a 3-stage curriculum board dataset for progressive RL training.

Stage 1: Simple obstacle avoidance (2-4 plain nets)
Stage 2: Corridor navigation and crossing traffic (5-8 plain nets)
Stage 3: Full production density (6-10 plain nets + 2 diff pairs + 2 length groups)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def generate_curriculum_dataset(base_dir: str, num_boards_per_stage: int = 20):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    stage1_dir = base_path / "stage1_basics"
    stage2_dir = base_path / "stage2_corridors"
    stage3_dir = base_path / "stage3_production"

    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)
    stage3_dir.mkdir(parents=True, exist_ok=True)

    # Locate generate_board.py
    script_path = Path(__file__).resolve().parent.parent / "pcbworld" / "data" / "generate_board.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find generate_board.py at {script_path}")

    print(f"Generating Stage 1 (Basics: 4-6 nets) in {stage1_dir}...")
    for i in range(num_boards_per_stage):
        num_nets = 4 + (i % 3)
        out_file = stage1_dir / f"board_s1_{i:03d}.kicad_pcb"
        cmd = [
            sys.executable,
            str(script_path),
            str(out_file),
            "--num-nets", str(num_nets),
            "--num-diff-pairs", "0",
            "--num-length-matched-groups", "0",
            "--seed", str(1000 + i),
            "--width-mm", "35",
            "--height-mm", "35",
            "--pad-type", "smd",
        ]
        subprocess.run(cmd, check=True)

    print(f"Generating Stage 2 (Corridors: 7-10 nets) in {stage2_dir}...")
    for i in range(num_boards_per_stage):
        num_nets = 7 + (i % 4)
        out_file = stage2_dir / f"board_s2_{i:03d}.kicad_pcb"
        cmd = [
            sys.executable,
            str(script_path),
            str(out_file),
            "--num-nets", str(num_nets),
            "--num-diff-pairs", "0",
            "--num-length-matched-groups", "0",
            "--seed", str(2000 + i),
            "--width-mm", "35",
            "--height-mm", "35",
            "--pad-type", "smd",
        ]
        subprocess.run(cmd, check=True)

    print(f"Generating Stage 3 (Full Production: 8 nets + 2 diff pairs + 2 length groups) in {stage3_dir}...")
    for i in range(num_boards_per_stage):
        out_file = stage3_dir / f"board_s3_{i:03d}.kicad_pcb"
        cmd = [
            sys.executable,
            str(script_path),
            str(out_file),
            "--num-nets", "8",
            "--num-diff-pairs", "2",
            "--num-length-matched-groups", "2",
            "--seed", str(3000 + i),
            "--width-mm", "35",
            "--height-mm", "35",
            "--pad-type", "smd",
        ]

        subprocess.run(cmd, check=True)


    print(f"Successfully generated 3-stage curriculum dataset in {base_dir}!")


def main():
    parser = argparse.ArgumentParser(description="Generate progressive curriculum dataset for Routerv3.")
    parser.add_argument("output_dir", help="Directory to save the curriculum board stages")
    parser.add_argument("--boards-per-stage", type=int, default=20, help="Number of boards per curriculum stage")
    args = parser.parse_args()

    generate_curriculum_dataset(args.output_dir, args.boards_per_stage)


if __name__ == "__main__":
    main()
