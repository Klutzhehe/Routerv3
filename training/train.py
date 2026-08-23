"""PPO & Actor-Critic Curriculum Trainer for AI PCB Router Platform.

Trains PCBRouterNet across progressive stages:
- Stage 1: Single Net (Basics)
- Stage 2: Single Net + Obstacles
- Stage 3: Multi-Net Dense Routing
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pcbworld.environment import PCBRouterEnv
from models.router_policy import PCBRouterNet
from training.replay_buffer import RolloutBuffer


def train_single_net_policy(
    total_timesteps: int = 50_000,
    rollout_steps: int = 512,
    epochs: int = 4,
    minibatch_size: int = 64,
    lr: float = 3e-4,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    checkpoint_dir: str = "/content/drive/MyDrive/pcb_ai_router/checkpoints",
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    plot_interval: int = 2048,
) -> PCBRouterNet:
    """Train single-net routing agent (Milestone 1 target: >95% routing success)."""
    device = torch.device(device_str)
    print(f"🚀 Initializing AI PCB Router Training on device: {device_str.upper()}")

    chk_path = Path(checkpoint_dir)
    chk_path.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate Environment (Single net, 2 pads, 256x256 grid)
    env = PCBRouterEnv(
        grid_size=256,
        num_nets=1,
        num_obstacles=0,
        max_steps_per_net=120,
        snap_radius=6,
    )

    # 2. Instantiate Model & Optimizer
    model = PCBRouterNet(in_channels=10, action_dim=96, d_model=256, num_transformer_layers=2, num_heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    buffer = RolloutBuffer(
        buffer_size=rollout_steps,
        obs_shape=env.observation_space.shape,
        device=device,
    )

    obs_np, info = env.reset()
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

    history = {
        "steps": [],
        "reward": [],
        "completion_rate": [],
        "policy_loss": [],
        "value_loss": [],
    }

    completed_window = []
    episode_reward_window = []
    curr_ep_reward = 0.0
    global_step = 0
    start_time = time.time()

    while global_step < total_timesteps:
        buffer.reset()

        # Collect Rollout
        for step in range(rollout_steps):
            global_step += 1
            with torch.no_grad():
                action_t, log_prob_t, _, value_t = model.get_action_and_value(obs_t)

            action = int(action_t.item())
            next_obs_np, reward, term, trunc, step_info = env.step(action)
            done = term or trunc
            curr_ep_reward += reward

            buffer.add(
                obs=obs_t.squeeze(0),
                action=action_t,
                log_prob=log_prob_t,
                reward=reward,
                done=done,
                value=value_t,
            )

            if done:
                is_comp = 1.0 if step_info.get("completed_nets", 0) > 0 else 0.0
                completed_window.append(is_comp)
                episode_reward_window.append(curr_ep_reward)
                if len(completed_window) > 50:
                    completed_window.pop(0)
                if len(episode_reward_window) > 50:
                    episode_reward_window.pop(0)

                curr_ep_reward = 0.0
                next_obs_np, step_info = env.reset()

            obs_np = next_obs_np
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

        # Compute GAE advantages
        with torch.no_grad():
            _, last_value = model(obs_t)
        buffer.compute_advantages_and_returns(last_value, done)

        # PPO Update
        total_p_loss, total_v_loss = 0.0, 0.0
        num_updates = 0

        for epoch in range(epochs):
            for mb_obs, mb_actions, mb_old_log_probs, mb_advs, mb_returns, mb_values in buffer.get_minibatches(minibatch_size):
                # Normalize advantages
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                _, new_log_prob, entropy, new_value = model.get_action_and_value(mb_obs, mb_actions)

                # Policy Loss
                ratio = torch.exp(new_log_prob - mb_old_log_probs)
                surr1 = -mb_advs * ratio
                surr2 = -mb_advs * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = torch.max(surr1, surr2).mean()

                # Value Loss
                value_loss = 0.5 * ((new_value.squeeze() - mb_returns) ** 2).mean()

                # Total Loss
                loss = policy_loss - ent_coef * entropy.mean() + vf_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                total_p_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                num_updates += 1

        # Logging & Stats
        avg_comp = float(np.mean(completed_window) * 100.0) if completed_window else 0.0
        avg_rew = float(np.mean(episode_reward_window)) if episode_reward_window else 0.0
        fps = int(global_step / max(1e-3, (time.time() - start_time)))

        history["steps"].append(global_step)
        history["reward"].append(avg_rew)
        history["completion_rate"].append(avg_comp)
        history["policy_loss"].append(total_p_loss / max(1, num_updates))
        history["value_loss"].append(total_v_loss / max(1, num_updates))

        print(
            f"Step {global_step:>6d}/{total_timesteps} | "
            f"Completion Rate: {avg_comp:6.1f}% | "
            f"Mean Reward: {avg_rew:7.1f} | "
            f"Loss: [P={history['policy_loss'][-1]:.3f}, V={history['value_loss'][-1]:.3f}] | "
            f"Speed: {fps} steps/s"
        )

        # Plot Curves
        if global_step % plot_interval == 0 or global_step >= total_timesteps:
            plot_learning_curves(history, chk_path / "single_net_training_curves.png")
            # Save Checkpoint
            torch.save(
                {
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "completion_rate": avg_comp,
                },
                chk_path / "single_net_router_latest.pt",
            )

    print(f"\n🎉 Milestone 1 Training Complete! Final Completion Rate: {avg_comp:.1f}%")
    return model


def plot_learning_curves(history: Dict[str, List[float]], save_path: Path):
    if len(history["steps"]) < 2:
        return
    steps = history["steps"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
    fig.patch.set_facecolor("#101216")

    for ax in axes:
        ax.set_facecolor("#181b22")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # 1. Completion Rate %
    axes[0].plot(steps, history["completion_rate"], color="#00ffcc", lw=2.0)
    axes[0].set_title("Single-Net Routing Success Rate (%)", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[0].set_ylim(-5, 105)
    axes[0].set_xlabel("Training Steps", color="#8b949e")
    axes[0].grid(True, alpha=0.15)

    # 2. Mean Reward
    axes[1].plot(steps, history["reward"], color="#ffaa00", lw=2.0)
    axes[1].set_title("Mean Episode Reward", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Training Steps", color="#8b949e")
    axes[1].grid(True, alpha=0.15)

    # 3. Policy & Value Losses
    axes[2].plot(steps, history["policy_loss"], color="#ff0055", lw=1.5, label="Policy Loss")
    axes[2].plot(steps, history["value_loss"], color="#aa00ff", lw=1.5, label="Value Loss")
    axes[2].set_title("Losses (PPO Clip & MSE)", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Training Steps", color="#8b949e")
    axes[2].legend(facecolor="#181b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    axes[2].grid(True, alpha=0.15)

    plt.tight_layout()
    fig.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    train_single_net_policy()
