"""PPO & Actor-Critic Curriculum Trainer for AI PCB Router Platform.

Trains PCBRouterNet across progressive stages:
- Stage 1: Single Net (Basics)
- Stage 2: Single Net + Obstacles
- Stage 3: Multi-Net Dense Routing

Features:
- Real-time live updating dashboard in Colab (IPython.display)
- Dynamic 3-panel learning curves updating every rollout
- GPU / CPU auto-detection with Mixed Precision support
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
from training.reward_scaling import RewardScaler


def train_single_net_policy(
    total_timesteps: int = 40_000,
    rollout_steps: int = 512,
    epochs: int = 4,
    minibatch_size: int = 64,
    lr: float = 3e-4,
    clip_coef: float = 0.2,
    value_clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    ent_coef_final: float = 0.001,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    gamma: float = 0.99,
    num_nets: int = 1,
    num_obstacles: int = 0,
    enable_layer_via: bool = True,
    max_steps_per_net: int = 120,
    max_net_restarts: int = 0,
    max_no_progress_steps: int = 20,
    target_steps_per_net: Optional[float] = None,
    target_success_rate: float = 0.95,
    checkpoint_dir: str = "/content/drive/MyDrive/pcb_ai_router/checkpoints",
    device_str: Optional[str] = None,
    plot_interval: int = 1024,
) -> PCBRouterNet:
    """Train a routing agent (Milestone 1 target: >95% routing success on stage 1).

    `num_nets`/`num_obstacles` select the curriculum stage; the function name
    predates stage 2/3 support and is kept so existing checkpoints/imports
    don't break."""
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"🚀 Initializing AI PCB Router Training on device: {device_str.upper()}")
    if device_str == "cuda":
        print(f"   GPU Model: {torch.cuda.get_device_name(0)}")

    chk_path = Path(checkpoint_dir)
    chk_path.mkdir(parents=True, exist_ok=True)
    plot_path = chk_path / "single_net_training_curves.png"

    # 1. Instantiate Environment
    env = PCBRouterEnv(
        grid_size=256,
        num_nets=num_nets,
        num_obstacles=num_obstacles,
        max_steps_per_net=max_steps_per_net,
        max_net_restarts=max_net_restarts,
        max_no_progress_steps=max_no_progress_steps,
        snap_radius=6,
        enable_layer_via=enable_layer_via,
    )

    # 2. Instantiate Model & Optimizer
    action_dim = 96 if enable_layer_via else 24
    model = PCBRouterNet(in_channels=10, action_dim=action_dim, d_model=256, num_transformer_layers=2, num_heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    buffer = RolloutBuffer(
        buffer_size=rollout_steps,
        obs_shape=env.observation_space.shape,
        device=device,
        gamma=gamma,
    )
    # Scales the reward the LEARNER sees by a running estimate of the
    # discounted return's std -- see reward_scaling.py. Raw reward stays in
    # curr_ep_reward/episode_reward_window for reporting; only what the
    # buffer stores and the value loss trains against goes through this.
    reward_scaler = RewardScaler(gamma=gamma)

    obs_np, info = env.reset()
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

    history = {
        "steps": [],
        "reward": [],
        "completion_rate": [],
        "policy_loss": [],
        "value_loss": [],
        "steps_to_complete": [],
    }

    completed_window: List[float] = []
    episode_reward_window: List[float] = []
    # Steps used by SUCCESSFUL episodes only -- a failed/timed-out episode's
    # step count is the budget, not a measure of efficiency, and would just
    # flatten this into max_steps_per_net regardless of how the policy is
    # actually doing.
    steps_to_complete_window: List[float] = []
    # Whether restart-on-jam is even firing -- select_deterministic_action's
    # retry-avoidance alone may resolve most jams before
    # max_consecutive_collisions triggers a restart, in which case the
    # dead-zone signal (see environment.py's _net_dead_zones) never gets
    # exercised during training regardless of how many steps are spent.
    restarts_window: List[int] = []
    total_restarts_seen = 0
    curr_ep_reward = 0.0
    global_step = 0
    start_time = time.time()
    rollout_count = 0

    # Distance curriculum gated on measured success, not a fixed step count.
    # The old version advanced at hard-coded step thresholds regardless of
    # whether the current distance was mastered -- which is what produced a
    # success rate that LOOKED like decay (42.9% -> 5%) but was actually the
    # task quietly getting harder out from under a policy that hadn't
    # converged on the easy case yet. Matches the >80%-success auto-advance
    # rule this repo already uses for other curricula (AI_ARCHITECTURE.md,
    # RL_PLAN.md) -- this trainer just never applied it.
    DIST_CURRICULUM = [50, 100, None]
    dist_stage = 0
    dist_curriculum_window: List[float] = []
    env.max_pad_dist = DIST_CURRICULUM[dist_stage]

    # In-notebook display helper
    in_colab = False
    try:
        from IPython.display import clear_output, display, Image
        in_colab = True
    except ImportError:
        pass

    while global_step < total_timesteps:
        rollout_count += 1
        buffer.reset()

        # -------------------------------------------------------------
        # Collect Rollout
        # -------------------------------------------------------------
        for step in range(rollout_steps):
            global_step += 1
            with torch.no_grad():
                action_t, log_prob_t, _, value_t = model.get_action_and_value(obs_t)

            action = int(action_t.item())
            next_obs_np, reward, term, trunc, step_info = env.step(action)
            done = term or trunc
            curr_ep_reward += reward  # raw units, for reporting only
            scaled_reward = reward_scaler.scale(reward, done)

            buffer.add(
                obs=obs_t.squeeze(0),
                action=action_t,
                log_prob=log_prob_t,
                reward=scaled_reward,
                done=done,
                value=value_t,
            )

            if done:
                is_comp = 1.0 if step_info.get("completed_nets", 0) > 0 else 0.0
                completed_window.append(is_comp)
                episode_reward_window.append(curr_ep_reward)
                if len(completed_window) > 40:
                    completed_window.pop(0)
                if len(episode_reward_window) > 40:
                    episode_reward_window.pop(0)
                if is_comp:
                    steps_to_complete_window.append(float(step_info.get("total_steps", 0)))
                    if len(steps_to_complete_window) > 40:
                        steps_to_complete_window.pop(0)

                ep_restarts = step_info.get("total_restarts", 0)
                total_restarts_seen += ep_restarts
                restarts_window.append(ep_restarts)
                if len(restarts_window) > 40:
                    restarts_window.pop(0)

                # Progressive Distance Curriculum, gated on measured success
                dist_curriculum_window.append(is_comp)
                if len(dist_curriculum_window) > 40:
                    dist_curriculum_window.pop(0)
                if (
                    dist_stage < len(DIST_CURRICULUM) - 1
                    and len(dist_curriculum_window) >= 20
                    and float(np.mean(dist_curriculum_window)) >= 0.85
                ):
                    dist_stage += 1
                    env.max_pad_dist = DIST_CURRICULUM[dist_stage]
                    dist_curriculum_window = []
                    print(f"  >> distance curriculum advanced: max_pad_dist={env.max_pad_dist}")
                    sys.stdout.flush()

                curr_ep_reward = 0.0
                next_obs_np, step_info = env.reset()

            obs_np = next_obs_np
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

        # -------------------------------------------------------------
        # Compute GAE Advantages
        # -------------------------------------------------------------
        with torch.no_grad():
            _, last_value = model(obs_t)
        buffer.compute_advantages_and_returns(last_value, done)

        # -------------------------------------------------------------
        # PPO Optimization Epochs
        # -------------------------------------------------------------
        # Entropy decays linearly ent_coef -> ent_coef_final over training.
        # Measured directly: a completely UNTRAINED, randomly-initialized
        # model scored 37/50 on the fixed eval boards -- identical to every
        # trained checkpoint so far. A constant entropy bonus keeps the
        # policy perpetually penalized for committing to a confident choice,
        # so even where the advantage signal DOES favor deviating from the
        # init-time default (tight obstacle corners), the deterministic
        # argmax never actually commits to it. Same schedule this repo's
        # other env already uses (docs/RL_PLAN.md: "entropy 0.01 decaying to
        # 0.001").
        progress = min(1.0, global_step / max(1, total_timesteps))
        current_ent_coef = ent_coef + (ent_coef_final - ent_coef) * progress

        total_p_loss, total_v_loss = 0.0, 0.0
        num_updates = 0

        for epoch in range(epochs):
            for mb_obs, mb_actions, mb_old_log_probs, mb_advs, mb_returns, mb_values in buffer.get_minibatches(minibatch_size):
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                _, new_log_prob, entropy, new_value = model.get_action_and_value(mb_obs, mb_actions)

                # Policy Loss
                ratio = torch.exp(new_log_prob - mb_old_log_probs)
                surr1 = -mb_advs * ratio
                surr2 = -mb_advs * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = torch.max(surr1, surr2).mean()

                # Value Loss, clipped (PPO2-style): bounds how far one
                # minibatch can drag the value estimate, the other half of
                # fixing the V-loss-thrashing pattern -- reward scaling fixes
                # the TARGET's scale, this fixes the update's step size.
                new_value = new_value.squeeze()
                v_unclipped = (new_value - mb_returns) ** 2
                v_clipped_pred = mb_values + torch.clamp(
                    new_value - mb_values, -value_clip_coef, value_clip_coef
                )
                v_clipped = (v_clipped_pred - mb_returns) ** 2
                value_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                # Total Loss
                loss = policy_loss - current_ent_coef * entropy.mean() + vf_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                total_p_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                num_updates += 1

        # -------------------------------------------------------------
        # Live Stats & Dashboard
        # -------------------------------------------------------------
        avg_comp = float(np.mean(completed_window) * 100.0) if completed_window else 0.0
        avg_rew = float(np.mean(episode_reward_window)) if episode_reward_window else 0.0
        avg_steps_to_complete = float(np.mean(steps_to_complete_window)) if steps_to_complete_window else 0.0
        elapsed = time.time() - start_time
        fps = int(global_step / max(1e-3, elapsed))
        p_loss = total_p_loss / max(1, num_updates)
        v_loss = total_v_loss / max(1, num_updates)

        history["steps"].append(global_step)
        history["reward"].append(avg_rew)
        history["completion_rate"].append(avg_comp)
        history["policy_loss"].append(p_loss)
        history["value_loss"].append(v_loss)
        history["steps_to_complete"].append(avg_steps_to_complete)

        # Print live stream
        progress_pct = (global_step / total_timesteps) * 100.0
        steps_to_complete_str = f"{avg_steps_to_complete:5.1f}" if steps_to_complete_window else "  n/a"
        restarts_str = ""
        if max_net_restarts > 0:
            avg_restarts = float(np.mean(restarts_window)) if restarts_window else 0.0
            restarts_str = f"Restarts: {avg_restarts:.2f}/ep (total {total_restarts_seen}) | "
        status_line = (
            f"[{progress_pct:5.1f}%] Step {global_step:>6d}/{total_timesteps} | "
            f"Success: {avg_comp:5.1f}% | "
            f"Reward: {avg_rew:6.1f} | "
            f"Steps/net: {steps_to_complete_str} | "
            f"{restarts_str}"
            f"Ent: {current_ent_coef:.4f} | "
            f"Loss: [π={p_loss:6.3f}, V={v_loss:6.3f}] | "
            f"Speed: {fps:>4d} steps/s"
        )
        print(status_line)
        sys.stdout.flush()

        # target_steps_per_net makes total_timesteps a safety cap rather
        # than the actual stopping rule: keep training past whatever budget
        # was passed as long as it's still improving efficiency, and stop as
        # soon as it's both reliable (target_success_rate) AND efficient
        # (avg steps/net at or below target), instead of stopping at a fixed
        # step count regardless of whether the routes it finds are any good.
        efficiency_reached = (
            target_steps_per_net is not None
            and len(steps_to_complete_window) >= 20
            and avg_comp >= target_success_rate * 100.0
            and avg_steps_to_complete <= target_steps_per_net
        )

        # Update curves plot & save checkpoint
        if efficiency_reached or global_step % plot_interval == 0 or global_step >= total_timesteps:
            plot_learning_curves(history, plot_path)
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

        if efficiency_reached:
            print(
                f"  >> target efficiency reached: Steps/net {avg_steps_to_complete:.1f} <= "
                f"{target_steps_per_net} at {avg_comp:.1f}% success -- stopping early"
            )
            sys.stdout.flush()
            break

    print(f"\n================================================================================")
    print(f"🎉 Milestone 1 Training Complete! Final Routing Success: {avg_comp:.1f}%")
    print(f"================================================================================")
    return model


def plot_learning_curves(history: Dict[str, List[float]], save_path: Path):
    if len(history["steps"]) < 2:
        return
    steps = history["steps"]
    fig, axes = plt.subplots(1, 4, figsize=(24, 5), dpi=110)
    fig.patch.set_facecolor("#101216")

    for ax in axes:
        ax.set_facecolor("#181b22")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # 1. Completion Rate %
    axes[0].plot(steps, history["completion_rate"], color="#00ffcc", lw=2.2)
    axes[0].axhline(95.0, color="#ff4444", linestyle="--", alpha=0.7, label="Milestone Target (95%)")
    axes[0].set_title("Single-Net Routing Success Rate (%)", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[0].set_ylim(-5, 105)
    axes[0].set_xlabel("Training Steps", color="#8b949e")
    axes[0].legend(facecolor="#181b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    axes[0].grid(True, alpha=0.15)

    # 2. Mean Reward
    axes[1].plot(steps, history["reward"], color="#ffaa00", lw=2.2)
    axes[1].set_title("Mean Episode Reward", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Training Steps", color="#8b949e")
    axes[1].grid(True, alpha=0.15)

    # 3. Policy & Value Losses
    axes[2].plot(steps, history["policy_loss"], color="#ff0055", lw=1.5, label="Policy Loss (π)")
    axes[2].plot(steps, history["value_loss"], color="#aa00ff", lw=1.5, label="Value Loss (V)")
    axes[2].set_title("Losses (PPO Clip & MSE)", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Training Steps", color="#8b949e")
    axes[2].legend(facecolor="#181b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    axes[2].grid(True, alpha=0.15)

    # 4. Steps to complete (successful episodes only -- a failed episode's
    # step count is just the budget, not an efficiency signal)
    axes[3].plot(steps, history["steps_to_complete"], color="#00bbff", lw=2.2)
    axes[3].set_title("Steps to Complete (successful nets)", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Training Steps", color="#8b949e")
    axes[3].set_ylim(bottom=0)
    axes[3].grid(True, alpha=0.15)

    plt.tight_layout()
    fig.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    train_single_net_policy()
