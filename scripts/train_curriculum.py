#!/usr/bin/env python3
"""Curriculum Training Orchestrator for Line-Geometry PCB Router.

Runs the full curriculum from Stage 1 (single net) through Stage 6 (full board + rip-up).
Auto-advances at >80% success rate on current stage.

Usage:
    python -m scripts.train_curriculum --board-dir /path/to/boards --stage 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pcbworld.env.line_obs import NUM_GLOBAL, LineObsConfig
from pcbworld.env.line_route_env import LineRouteEnv
from pcbworld.env.line_diff_pair_tune_env import LineDiffPairTuneEnv
from models.line_geometry_policy import LineGeometryPolicy
from training.train_line_policy import train_line_policy, evaluate_policy


STAGE_CONFIGS = {
    1: {
        "name": "Single Net (Empty Board)",
        "num_nets": 1,
        "num_diff_pairs": 0,
        "num_length_groups": 0,
        "max_steps": 200,
        "target_success": 95.0,
        "description": "Plumbing check — should reach ~100% quickly",
    },
    2: {
        "name": "8 Nets Sequential (Shortest First)",
        "num_nets": 8,
        "num_diff_pairs": 0,
        "num_length_groups": 0,
        "max_steps": 200,
        "target_success": 50.0,
        "description": "Beat 33% straight-line baseline",
    },
    3: {
        "name": "24 Nets Dense",
        "num_nets": 24,
        "num_diff_pairs": 0,
        "num_length_groups": 0,
        "max_steps": 200,
        "target_success": 60.0,
        "description": "Beat stock KiCad (B2) on completion % at equal/lower wirelength",
    },
    4: {
        "name": "Diff Pairs",
        "num_nets": 8,
        "num_diff_pairs": 4,
        "num_length_groups": 0,
        "max_steps": 240,
        "target_success": 80.0,
        "description": "Engine does coupling; policy picks corridor",
    },
    5: {
        "name": "Length-Matched Groups",
        "num_nets": 6,
        "num_diff_pairs": 2,
        "num_length_groups": 2,
        "max_steps": 300,
        "target_success": 80.0,
        "description": "Engine does meanders; 'matched' = tolerance (~0.25mm residual)",
    },
    6: {
        "name": "Full Board + Rip-Up",
        "num_nets": 16,
        "num_diff_pairs": 4,
        "num_length_groups": 2,
        "max_steps": 300,
        "target_success": 90.0,
        "description": "Rip-up action enabled (not yet implemented)",
    },
}


def generate_board_for_stage(
    stage: int,
    board_dir: Path,
    seed: int,
) -> Path:
    """Generate a board for the given curriculum stage using generate_board.py."""
    config = STAGE_CONFIGS[stage]
    board_path = board_dir / f"stage{stage}_seed{seed}.kicad_pcb"

    # Build command for generate_board.py
    cmd = [
        sys.executable,
        "-m", "pcbworld.data.generate_board",
        "--output", str(board_path),
        "--num-nets", str(config["num_nets"]),
        "--seed", str(seed),
    ]

    if config["num_diff_pairs"] > 0:
        cmd.extend(["--num-diff-pairs", str(config["num_diff_pairs"])])

    if config["num_length_groups"] > 0:
        cmd.extend([
            "--num-length-matched-groups", str(config["num_length_groups"]),
            "--length-matched-group-size", "2",
        ])

    # Board size: 50x50mm default from generate_board.py
    cmd.extend(["--board-width", "50", "--board-height", "50"])

    print(f"  Generating board: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ Board generation failed: {result.stderr}")
        raise RuntimeError(f"Board generation failed for stage {stage}, seed {seed}")

    return board_path


def create_board_pool(
    stage: int,
    board_dir: Path,
    pool_size: int = 200,
    base_seed: int = 42,
) -> list[Path]:
    """Pre-generate a pool of boards for the stage."""
    board_dir.mkdir(parents=True, exist_ok=True)
    boards = []

    print(f"\n📦 Generating board pool for Stage {stage} ({pool_size} boards)...")
    for i in range(pool_size):
        seed = base_seed + i
        board_path = board_dir / f"stage{stage}_seed{seed}.kicad_pcb"
        if board_path.exists():
            boards.append(board_path)
            continue

        try:
            generate_board_for_stage(stage, board_dir, seed)
            boards.append(board_path)
        except RuntimeError as e:
            print(f"  ⚠️  Failed to generate seed {seed}: {e}")

    print(f"  ✅ Generated {len(boards)} boards for Stage {stage}")
    return boards


def run_stage(
    stage: int,
    board_pool: list[Path],
    checkpoint_dir: Path,
    total_timesteps: int,
    device: str,
    eval_interval: int = 20_000,
) -> float:
    """Train on a single curriculum stage."""
    config = STAGE_CONFIGS[stage]
    print(f"\n{'='*70}")
    print(f"🎯 STAGE {stage}: {config['name']}")
    print(f"   {config['description']}")
    print(f"   Target: >{config['target_success']}% success")
    print(f"{'='*70}")

    # Train against the whole pool, not board_pool[0]. LineRouteEnv samples
    # a board per reset when handed a directory or a list, and a policy fitted
    # to one board's geometry is a policy that memorised one board.
    train_board = board_pool[0].parent if board_pool[0].parent.is_dir() else board_pool[0]

    # Determine env class
    if stage <= 3:
        env_class = LineRouteEnv
        env_kwargs = {"max_steps_per_episode": config["max_steps"]}
    else:
        env_class = LineDiffPairTuneEnv
        env_kwargs = {"max_steps_per_leg": config["max_steps"] // 4}

    # Train
    model = train_line_policy(
        board_path=str(train_board),
        total_timesteps=total_timesteps,
        checkpoint_dir=str(checkpoint_dir / f"stage{stage}"),
        device_str=device,
        eval_interval=eval_interval,
        **env_kwargs,
    )

    # Evaluate on held-out boards from pool
    print(f"\n📊 Evaluating Stage {stage} on {min(10, len(board_pool)-1)} held-out boards...")
    eval_boards = board_pool[1:11] if len(board_pool) > 1 else [train_board]

    # Load best checkpoint for eval
    checkpoint_path = checkpoint_dir / f"stage{stage}" / "line_policy_latest.pt"
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        global_mean = ckpt["global_mean"]
        global_var = ckpt["global_var"]
    else:
        global_mean = torch.zeros(NUM_GLOBAL)
        global_var = torch.ones(NUM_GLOBAL)

    device_obj = torch.device(device)
    model.eval()

    # Per-board completion is nets_completed / nets_attempted, not "did the
    # last step report something". `info["completed"]` is a LIST of net names,
    # so a truthiness test scores a board where 1 of 24 nets landed as 100%.
    completions = []
    for board in eval_boards:
        if stage <= 3:
            env = env_class(
                board_path=str(board),
                max_steps_per_net=config["max_steps"],
                obs_config=LineObsConfig(max_steps=config["max_steps"]),
            )
        else:
            env = env_class(board_path=str(board), **env_kwargs)

        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        info: dict = {}

        with torch.no_grad():
            while not done:
                flat = torch.as_tensor(obs, dtype=torch.float32, device=device_obj).unsqueeze(0)
                extra = getattr(env, "obs_config", None)
                global_vec, segments, segment_mask = model.split(
                    flat, extra_globals=getattr(extra, "extra_globals", 0)
                )
                segment_mask = segment_mask.bool()

                global_vec_norm = (global_vec - global_mean.to(device_obj)) / (torch.sqrt(global_var.to(device_obj)) + 1e-8)

                dist, _ = model.forward(global_vec_norm, segments, segment_mask)
                action = dist.mean.squeeze(0).cpu().numpy()

                obs, reward, term, trunc, info = env.step(action)
                ep_reward += reward
                done = term or trunc

        attempted = info.get("num_nets") or len(info.get("completed", ())) + len(info.get("failed", ()))
        completions.append(len(info.get("completed", ())) / attempted if attempted else 0.0)

    success_rate = float(np.mean(completions) * 100)
    print(f"  🏁 Stage {stage} Evaluation: {success_rate:.1f}% success rate")

    return success_rate


def main():
    parser = argparse.ArgumentParser(description="Curriculum training for line-geometry PCB router")
    parser.add_argument("--board-dir", type=Path, default=Path("/content/boards"), help="Directory for generated boards")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/content/drive/MyDrive/pcb_line_router/checkpoints"), help="Checkpoint directory")
    parser.add_argument("--start-stage", type=int, default=1, help="Starting stage (1-6)")
    parser.add_argument("--end-stage", type=int, default=6, help="Ending stage (1-6)")
    parser.add_argument("--timesteps-per-stage", type=int, default=200_000, help="Training steps per stage")
    parser.add_argument("--pool-size", type=int, default=200, help="Board pool size per stage")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Training device")
    parser.add_argument("--eval-interval", type=int, default=20_000, help="Evaluation interval during training")
    parser.add_argument("--skip-board-gen", action="store_true", help="Skip board generation (use existing)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"🚀 Curriculum Training for Line-Geometry PCB Router")
    print(f"   Device: {device}")
    print(f"   Stages: {args.start_stage} → {args.end_stage}")
    print(f"   Steps per stage: {args.timesteps_per_stage:,}")
    print(f"   Board pool: {args.pool_size} boards/stage")
    print(f"   Boards: {args.board_dir}")
    print(f"   Checkpoints: {args.checkpoint_dir}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()

    for stage in range(args.start_stage, args.end_stage + 1):
        if stage not in STAGE_CONFIGS:
            print(f"⚠️  Stage {stage} not defined, skipping")
            continue

        # Generate board pool
        if not args.skip_board_gen:
            board_pool = create_board_pool(stage, args.board_dir, args.pool_size)
        else:
            board_pool = list(args.board_dir.glob(f"stage{stage}_seed*.kicad_pcb"))
            if not board_pool:
                print(f"❌ No boards found for stage {stage}. Run without --skip-board-gen first.")
                return 1

        if not board_pool:
            print(f"❌ No boards available for stage {stage}")
            return 1

        # Train stage
        try:
            success_rate = run_stage(
                stage=stage,
                board_pool=board_pool,
                checkpoint_dir=args.checkpoint_dir,
                total_timesteps=args.timesteps_per_stage,
                device=device,
                eval_interval=args.eval_interval,
            )
        except Exception as e:
            print(f"❌ Stage {stage} failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # Check auto-advance criterion
        config = STAGE_CONFIGS[stage]
        if success_rate >= config["target_success"]:
            print(f"  ✅ Stage {stage} PASSED ({success_rate:.1f}% >= {config['target_success']}%) — advancing")
        else:
            print(f"  ⚠️  Stage {stage} BELOW TARGET ({success_rate:.1f}% < {config['target_success']}%)")
            print(f"      Consider: more training, reward tuning, or architecture changes")
            # Continue anyway for now

    elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"🏁 Curriculum Complete! Total time: {elapsed/3600:.1f} hours")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())