"""Live Interactive Curriculum Trainer for Routerv3 in Colab.

Provides:
- 3-stage progressive curriculum training (Basics -> Corridors -> Production).
- Live in-place ASCII status dashboard with real-time ETA, steps/sec, and stats.
- Live 4-panel matplotlib training graph updating dynamically in the notebook.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

from pcbworld.agents.line_policy import LineActorCritic, RunningMeanStd
from pcbworld.agents.ppo_baseline import PPOConfig, collect_rollout, ppo_update, save_checkpoint
from pcbworld.env.line_obs import NUM_GLOBAL
from pcbworld.env.line_route_env import LineRouteEnv
from scripts.generate_curriculum_boards import generate_curriculum_dataset


def plot_live_curves(history: dict[str, list], output_path: str | Path | None = None):
    if len(history["steps"]) < 2:
        return

    steps = history["steps"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=100)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Mean Reward
    ax = axes[0, 0]
    ax.plot(steps, history["reward"], color="#1f77b4", lw=2, marker="o", markersize=3)
    ax.set_title("Mean Episode Reward", fontsize=11, fontweight="bold")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    # 2. Net Completion Rate %
    ax = axes[0, 1]
    ax.plot(steps, history["completion"], color="#2ca02c", lw=2, marker="s", markersize=3)
    ax.set_title("Net Completion Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Completion %")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # 3. Policy Loss
    ax = axes[1, 0]
    ax.plot(steps, history["policy_loss"], color="#d62728", lw=2)
    ax.set_title("Policy Loss (PPO Clip)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # 4. Value Loss
    ax = axes[1, 1]
    ax.plot(steps, history["value_loss"], color="#9467bd", lw=2)
    ax.set_title("Value Loss (MSE)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")

    plt.close(fig)



def train_curriculum_live(
    dataset_dir: str = "/content/curriculum_dataset",
    checkpoint_dir: str = "/content/drive/MyDrive/routerv3_curriculum",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    boards_per_stage: int = 25,
    stage_limit: int | None = None,
):
    dataset_path = Path(dataset_dir)
    stage1_dir = dataset_path / "stage1_basics"
    stage2_dir = dataset_path / "stage2_corridors"
    stage3_dir = dataset_path / "stage3_production"

    if not (stage1_dir.exists() and stage2_dir.exists() and stage3_dir.exists()):
        print(f"Generating curriculum dataset in {dataset_dir}...")
        generate_curriculum_dataset(dataset_dir, boards_per_stage)

    stages = [
        {
            "name": "Stage 1: Basics (4-6 Nets, Real Obstacles)",
            "board_dir": str(stage1_dir),
            "timesteps": 80_000,
            "enable_ripup": False,
            "max_ripups": 0,
        },
        {
            "name": "Stage 2: Corridors (7-10 Nets, Dense Traffic)",
            "board_dir": str(stage2_dir),
            "timesteps": 60_000,
            "enable_ripup": False,
            "max_ripups": 0,
        },
        {
            "name": "Stage 3: Full Production (8 Plain Nets + Diff Pairs & Length Groups)",
            "board_dir": str(stage3_dir),
            "timesteps": 100_000,
            "enable_ripup": True,
            "max_ripups": 6,
        },
    ]


    if stage_limit is not None:
        stages = stages[:stage_limit]

    total_curriculum_steps = sum(s["timesteps"] for s in stages)
    print(f"Starting Curriculum Training on device: {device.upper()} (Total Steps: {total_curriculum_steps:,})")

    policy = None
    rms = None
    optimizer = None
    cumulative_steps = 0
    start_time = time.time()

    history = {
        "steps": [],
        "reward": [],
        "completion": [],
        "policy_loss": [],
        "value_loss": [],
    }

    chk_base = Path(checkpoint_dir)
    chk_base.mkdir(parents=True, exist_ok=True)
    plot_path = chk_base / "curriculum_training_curves.png"

    for stage_idx, stage_info in enumerate(stages, 1):
        stage_name = stage_info["name"]
        board_dir = stage_info["board_dir"]
        stage_timesteps = stage_info["timesteps"]
        stage_chk_dir = chk_base / f"stage{stage_idx}"
        stage_chk_dir.mkdir(parents=True, exist_ok=True)

        env = LineRouteEnv(
            board_dir,
            step_size_nm=500_000,
            snap_radius_nm=400_000,
            max_steps_per_net=120,
            enable_ripup=stage_info["enable_ripup"],
            max_ripups_per_episode=stage_info["max_ripups"],
        )


        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))

        if policy is None:
            policy = LineActorCritic(action_dim=action_dim).to(device)
            rms = RunningMeanStd(shape=(NUM_GLOBAL,))
            optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        cfg = PPOConfig(
            total_timesteps=stage_timesteps,
            rollout_steps=512,
            epochs=4,
            minibatch_size=64,
            learning_rate=3e-4,
            checkpoint_dir=str(stage_chk_dir),
            checkpoint_interval=5_000,
            device=device,
        )

        obs, _info = env.reset()
        stage_steps = 0
        last_chk_step = 0

        while stage_steps < stage_timesteps:
            t0 = time.time()
            buffer, obs, last_value, rollout_info = collect_rollout(
                env, policy, obs, cfg.rollout_steps, cfg.device, rms=rms
            )
            stats = ppo_update(policy, optimizer, buffer, last_value, cfg)
            t1 = time.time()

            stage_steps += cfg.rollout_steps
            cumulative_steps += cfg.rollout_steps

            episode_rewards = rollout_info["episode_rewards"]
            mean_reward = float(np.mean(episode_rewards)) if episode_rewards else float("nan")
            comp = rollout_info["completed_nets"]
            failed = rollout_info["failed_nets"]
            comp_rate = (comp / (comp + failed) * 100.0) if (comp + failed) > 0 else 0.0


            # Record history for live plotting
            history["steps"].append(cumulative_steps)
            history["reward"].append(mean_reward if not np.isnan(mean_reward) else (history["reward"][-1] if history["reward"] else 0.0))
            history["completion"].append(comp_rate)
            history["policy_loss"].append(stats.get("policy_loss", 0.0))
            history["value_loss"].append(stats.get("value_loss", 0.0))

            # Compute timing & ETA
            elapsed = time.time() - start_time
            sps = cumulative_steps / max(1e-5, elapsed)
            remaining_steps = max(0, total_curriculum_steps - cumulative_steps)
            eta_sec = remaining_steps / max(1e-5, sps)

            short_stage = stage_name.split(":")[0].strip()
            comp_str = f"Completion: {comp_rate:5.1f}% ({comp:2d}/{comp+failed:2d})"
            loss_str = f"P-Loss: {stats.get('policy_loss', 0.0):.4f} | V-Loss: {stats.get('value_loss', 0.0):.4f}"
            timing_str = f"{sps:.1f} sps | ETA: {int(eta_sec//60):02d}m{int(eta_sec%60):02d}s"

            print(
                f"  [{short_stage}] Step {stage_steps:6d}/{stage_timesteps:6d} | "
                f"{comp_str} | Reward: {mean_reward:6.2f} | {loss_str} | {timing_str}",
                flush=True
            )

            # Update saved curves at checkpoint interval
            if stage_steps - last_chk_step >= cfg.checkpoint_interval or stage_steps >= stage_timesteps:
                plot_live_curves(history, plot_path)


            if stage_steps - last_chk_step >= cfg.checkpoint_interval or stage_steps >= stage_timesteps:
                save_checkpoint(
                    stage_chk_dir,
                    stage_steps,
                    policy,
                    optimizer,
                    rms,
                    cfg,
                    {"mean_reward": mean_reward, "completion_rate": comp_rate, **stats},
                )
                # Also save master latest checkpoint
                save_checkpoint(
                    chk_base,
                    cumulative_steps,
                    policy,
                    optimizer,
                    rms,
                    cfg,
                    {"mean_reward": mean_reward, "completion_rate": comp_rate, **stats},
                )
                last_chk_step = stage_steps

    print("\n✅ All 3 Curriculum Stages Completed Successfully!")
    print(f"Final Model Saved to: {chk_base / 'policy_latest.pt'}")
    return policy


def main():
    parser = argparse.ArgumentParser(description="Live curriculum training for Routerv3.")
    parser.add_argument("--dataset-dir", type=str, default="/content/curriculum_dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="/content/drive/MyDrive/routerv3_curriculum")
    parser.add_argument("--boards-per-stage", type=int, default=25)
    args = parser.parse_args()

    train_curriculum_live(
        dataset_dir=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
        boards_per_stage=args.boards_per_stage,
    )


if __name__ == "__main__":
    main()
