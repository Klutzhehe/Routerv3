"""Command-Line Interface to Train PCBRouterNet (New AI Router Architecture).

Supports:
  --stage 1 (Single net, empty board -- baseline, target ~100%)
  --stage 2 (Single net + obstacles)
  --stage 3 (Multi-net + obstacles, via/layer still off)
  --stage 4 (Multi-net + obstacles + via/layer)

One new axis of difficulty per stage, deliberately -- see docs/RL_PLAN.md's
Gate A and the stage-1 postmortem in this repo's history for what stacking
more than one at a time costs to debug.
"""

import argparse
import os
import sys
import torch

from training.train import train_single_net_policy
from training.evaluation import evaluate_policy
from pcbworld.environment import PCBRouterEnv
from models.router_policy import PCBRouterNet

# One new axis of difficulty per stage. enable_layer_via stays False through
# stage 3 -- board_generator.py sets tgt_layer = src_layer for these stages,
# so the correct answer is "never touch layer/via", and exposing that
# dimension before the core skill exists just gives the policy a free way to
# fail (measured: ~75% of steps toggled it in an undertrained eval, before it
# was disabled). Multi-net (stage 3) and via/layer (stage 4) are kept
# separate for the same reason: stacking two unvalidated axes at once is how
# a failure stops being attributable to either one.
#
# Module-level (not inside main()) so scripts/render_episode.py can share it
# instead of re-declaring a second copy that quietly drifts out of sync.
STAGE_CONFIG = {
    1: dict(num_nets=1, num_obstacles=0, enable_layer_via=False),
    2: dict(num_nets=1, num_obstacles=6, enable_layer_via=False),
    3: dict(num_nets=4, num_obstacles=6, enable_layer_via=False),
    4: dict(num_nets=4, num_obstacles=6, enable_layer_via=True),
}


def action_dim_for_stage(stage_cfg: dict) -> int:
    return 96 if stage_cfg["enable_layer_via"] else 24


def main():
    parser = argparse.ArgumentParser(description="Train AI PCB Router Platform (PCBRouterNet)")
    parser.add_argument("--timesteps", type=int, default=30_000, help="Total training steps")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4], help="Curriculum stage (1=single net, 2=+obstacles, 3=+multinet, 4=+via/layer)")
    parser.add_argument("--max-steps", type=int, default=120, help="Per-net step budget (max_steps_per_net) -- how many moves the head gets to reach one target before that net times out. Applies to both training and the post-training eval, so they stay comparable.")
    parser.add_argument("--max-net-restarts", type=int, default=0, help="On top of retrying single steps from the same stuck position (always on), restart the WHOLE net from its source pad this many times before giving up, once every local alternative from a jam has been exhausted. 0 (default) disables this -- a jammed net fails immediately, the original behavior.")
    parser.add_argument("--max-no-progress-steps", type=int, default=20, help="Give up (or restart, if --max-net-restarts is set) after this many CONSECUTIVE steps without the head's cost-to-go improving -- catches a head oscillating between two valid, non-colliding cells, which never trips the collision-based jam check at all since neither move is ever rejected.")
    parser.add_argument("--target-steps-per-net", type=float, default=None, help="If set, --timesteps becomes a safety cap instead of the stopping rule: training keeps going (up to that cap) until the rolling average Steps/net for SUCCESSFUL nets drops to this value or below, at --target-success-rate or higher. Use this to train for efficiency, not just completion.")
    parser.add_argument("--target-success-rate", type=float, default=0.95, help="Success rate required (in addition to --target-steps-per-net) before training is allowed to stop early. Ignored unless --target-steps-per-net is set.")
    parser.add_argument("--checkpoint-dir", type=str, default="/content/drive/MyDrive/pcb_ai_router/checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--eval-seed-offset", type=int, default=9000, help="First board seed of the num_eval_episodes-sized block used for the post-training/--eval-only benchmark. Every stage-2 result so far has been measured against the default (9000-9049) -- pass a different offset (e.g. 20000) to check generalization against boards that specific block was never explicitly re-tuned against.")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for evaluation")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Resume TRAINING from this checkpoint's weights instead of a fresh random init (optimizer state and entropy schedule still restart fresh). Use to fine-tune an already-trained checkpoint under a new env config, e.g. turning --max-net-restarts on after it was trained with restarts off.")
    parser.add_argument("--ent-coef-final", type=float, default=0.001, help="Entropy coefficient at the END of training (decays linearly from --ent-coef over --timesteps). Lower means the policy commits harder to whatever it currently prefers by the end of the run -- e.g. always taking the largest step distance, which is efficient in open space but too coarse to thread a tight gap that only a smaller step can fit through. Raise this if late-training failures cluster on boards whose only solution needs sustained small, careful steps rather than the default large stride.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient at the START of training, decaying to --ent-coef-final over --timesteps.")
    parser.add_argument("--failure-penalty", type=float, default=1000.0, help="One-time penalty added when a net times out or exhausts its restarts without connecting (see RewardCalculator.failure_penalty). Default (1000.0) matches connect_bonus for a symmetric structure. Raising it strengthens the terminal signal a failing trajectory backpropagates via GAE, at the cost of pushing the policy toward more caution generally -- untested above the default as of this flag's introduction.")
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    stage_cfg = STAGE_CONFIG[args.stage]
    action_dim = action_dim_for_stage(stage_cfg)

    if args.eval_only:
        print(f"Loading checkpoint from: {args.checkpoint}")
        model = PCBRouterNet(in_channels=10, action_dim=action_dim, d_model=256, num_transformer_layers=2, num_heads=4)
        if args.checkpoint and os.path.exists(args.checkpoint):
            chk = torch.load(args.checkpoint, map_location=device_str, weights_only=False)
            model.load_state_dict(chk["model_state_dict"])
        model.to(device_str)
        evaluate_policy(model, num_eval_episodes=50, max_steps_per_net=args.max_steps, max_net_restarts=args.max_net_restarts, max_no_progress_steps=args.max_no_progress_steps, eval_seed_offset=args.eval_seed_offset, device=device_str, **stage_cfg)
        return

    print("=" * 80)
    print(f"   STARTING MILESTONE {args.stage}: AI PCB ROUTER PLATFORM TRAINING")
    print(f"   Architecture: 10-Channel Grid + CNN-Transformer Policy (PCBRouterNet)")
    print(f"   Stage config: {stage_cfg}  |  max_steps_per_net: {args.max_steps}")
    print(f"   Device: {device_str.upper()} | Steps: {args.timesteps:,}")
    print("=" * 80)

    model = train_single_net_policy(
        total_timesteps=args.timesteps,
        max_steps_per_net=args.max_steps,
        max_net_restarts=args.max_net_restarts,
        max_no_progress_steps=args.max_no_progress_steps,
        target_steps_per_net=args.target_steps_per_net,
        target_success_rate=args.target_success_rate,
        checkpoint_dir=args.checkpoint_dir,
        device_str=device_str,
        init_checkpoint=args.init_checkpoint,
        ent_coef=args.ent_coef,
        ent_coef_final=args.ent_coef_final,
        failure_penalty=args.failure_penalty,
        **stage_cfg,
    )

    print("\nRunning post-training benchmark evaluation...")
    evaluate_policy(model, num_eval_episodes=50, max_steps_per_net=args.max_steps, max_net_restarts=args.max_net_restarts, max_no_progress_steps=args.max_no_progress_steps, eval_seed_offset=args.eval_seed_offset, device=device_str, **stage_cfg)


if __name__ == "__main__":
    main()
