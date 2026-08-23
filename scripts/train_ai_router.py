"""Command-Line Interface to Train PCBRouterNet (New AI Router Architecture).

Supports:
  --stage 1 (Single net baseline target >95%)
  --stage 2 (Single net + obstacles)
  --stage 3 (Multi-net spatial competition)
"""

import argparse
import os
import sys
import torch

from training.train import train_single_net_policy
from training.evaluation import evaluate_policy
from pcbworld.environment import PCBRouterEnv
from models.router_policy import PCBRouterNet


def main():
    parser = argparse.ArgumentParser(description="Train AI PCB Router Platform (PCBRouterNet)")
    parser.add_argument("--timesteps", type=int, default=30_000, help="Total training steps")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="Curriculum stage (1=single net, 2=obstacles, 3=multinet)")
    parser.add_argument("--checkpoint-dir", type=str, default="/content/drive/MyDrive/pcb_ai_router/checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for evaluation")
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # --stage was accepted but never actually changed the environment --
    # every stage silently trained/evaluated on 1 net, 0 obstacles. Wired
    # here so stage 2/3 runs actually exercise obstacles / multiple nets.
    STAGE_CONFIG = {
        1: dict(num_nets=1, num_obstacles=0),
        2: dict(num_nets=1, num_obstacles=6),
        3: dict(num_nets=4, num_obstacles=6),
    }
    stage_cfg = STAGE_CONFIG[args.stage]

    if args.eval_only:
        print(f"Loading checkpoint from: {args.checkpoint}")
        model = PCBRouterNet(in_channels=10, action_dim=96, d_model=256, num_transformer_layers=2, num_heads=4)
        if args.checkpoint and os.path.exists(args.checkpoint):
            chk = torch.load(args.checkpoint, map_location=device_str, weights_only=False)
            model.load_state_dict(chk["model_state_dict"])
        model.to(device_str)
        evaluate_policy(model, num_eval_episodes=50, device=device_str, **stage_cfg)
        return

    print("=" * 80)
    print(f"   STARTING MILESTONE {args.stage}: AI PCB ROUTER PLATFORM TRAINING")
    print(f"   Architecture: 10-Channel Grid + CNN-Transformer Policy (PCBRouterNet)")
    print(f"   Stage config: {stage_cfg}")
    print(f"   Device: {device_str.upper()} | Steps: {args.timesteps:,}")
    print("=" * 80)

    model = train_single_net_policy(
        total_timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        device_str=device_str,
        **stage_cfg,
    )

    print("\nRunning post-training benchmark evaluation...")
    evaluate_policy(model, num_eval_episodes=50, device=device_str, **stage_cfg)


if __name__ == "__main__":
    main()
