"""PPO Trainer for LineGeometryPolicy with continuous 1-D heading action.

Extends the training infrastructure for Gaussian policy on line-segment observations.
Includes observation normalization, reward scaling, and Drive checkpointing.
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

from pcbworld.env.line_obs import NUM_GLOBAL, LineObsConfig
from pcbworld.env.line_route_env import LineRouteEnv, RewardWeights
from models.line_geometry_policy import LineGeometryPolicy
from training.replay_buffer import RolloutBuffer
from training.reward_scaling import RewardScaler


def train_line_policy(
    board_path: str,
    total_timesteps: int = 200_000,
    rollout_steps: int = 512,
    epochs: int = 4,
    minibatch_size: int = 64,
    lr: float = 3e-4,
    clip_coef: float = 0.2,
    value_clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    max_steps_per_episode: int = 200,
    track_width_nm: int = 250_000,
    checkpoint_dir: str = "/content/drive/MyDrive/pcb_line_router/checkpoints",
    device_str: Optional[str] = None,
    plot_interval: int = 4096,
    eval_interval: int = 20_000,
    eval_episodes: int = 10,
) -> LineGeometryPolicy:
    """Train a line-geometry routing policy with PPO.

    Args:
        board_path: Path to .kicad_pcb board file
        total_timesteps: Total environment steps to train
        rollout_steps: Steps per rollout per worker
        epochs: PPO epochs per rollout
        minibatch_size: Minibatch size for PPO updates
        lr: Learning rate
        clip_coef: PPO clip coefficient
        value_clip_coef: Value function clip coefficient
        ent_coef: Entropy coefficient
        vf_coef: Value function loss coefficient
        max_grad_norm: Gradient clipping norm
        gamma: Discount factor
        gae_lambda: GAE lambda
        max_steps_per_episode: Max steps per episode in env
        track_width_nm: Track width in nm
        checkpoint_dir: Directory for checkpoints and plots
        device_str: Device string (cuda/cpu), auto-detected if None
        plot_interval: Steps between plotting learning curves
        eval_interval: Steps between evaluation runs
        eval_episodes: Number of episodes for evaluation

    Returns:
        Trained LineGeometryPolicy
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"🚀 Initializing Line-Geometry PCB Router Training on {device_str.upper()}")
    if device_str == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    chk_path = Path(checkpoint_dir)
    chk_path.mkdir(parents=True, exist_ok=True)
    plot_path = chk_path / "line_policy_training_curves.png"

    # Environment (single instance for now; multiprocessing later).
    # `board_path` may be a directory: the env samples a board per reset, so
    # a pool trains against varied geometry instead of memorising one board.
    obs_config = LineObsConfig(max_steps=max_steps_per_episode)
    env = LineRouteEnv(
        board_path=board_path,
        track_width_nm=track_width_nm,
        max_steps_per_net=max_steps_per_episode,
        obs_config=obs_config,
        gamma=gamma,
    )

    # Model
    model = LineGeometryPolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    # Buffers & scaling. One flat observation per step -- the env emits a
    # single Box (see line_obs.build_observation) and the policy splits it,
    # so the buffer needs no per-key bookkeeping.
    buffer = RolloutBuffer(
        buffer_size=rollout_steps,
        obs_shape=env.observation_space.shape,
        device=device,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    reward_scaler = RewardScaler(gamma=gamma)

    # Observation normalization (running mean/std on global vector only --
    # the segment rows are one-hots and a validity flag, and normalising
    # those would destroy the mask).
    global_dim = NUM_GLOBAL
    global_mean = torch.zeros(global_dim, device=device)
    global_var = torch.ones(global_dim, device=device)
    global_count = torch.tensor(1e-4, device=device)

    def normalize_global(global_vec: torch.Tensor, update: bool = True) -> torch.Tensor:
        """Standardise the global block, optionally advancing the statistics.

        `update` is False during the PPO epochs. Re-fitting the running mean
        on the same rollout four times over would let the buffer's own data
        dominate the statistics by its epoch count rather than its sample
        count, and shift the normalisation between the epochs of a single
        update -- so the ratio in the clip objective would compare log-probs
        taken under two different observation scalings."""
        nonlocal global_mean, global_var, global_count
        if update:
            # Update running stats
            batch_mean = global_vec.mean(dim=0)
            batch_var = global_vec.var(dim=0, unbiased=False)
            batch_count = global_vec.shape[0]

            delta = batch_mean - global_mean
            total_count = global_count + batch_count

            new_mean = global_mean + delta * batch_count / total_count
            m_a = global_var * global_count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta**2 * global_count * batch_count / total_count
            new_var = m2 / total_count

            global_mean = new_mean.detach()
            global_var = new_var.detach()
            global_count = total_count.detach()

        return (global_vec - global_mean) / (torch.sqrt(global_var) + 1e-8)

    def to_tensors(flat_obs):
        """Env observation -> (globals, segment rows, mask), batched to (1, ...).

        The split lives in line_obs.py so the widths cannot drift; doing it
        by hand here is how the geodesic features got dropped last time."""
        flat = torch.as_tensor(flat_obs, dtype=torch.float32, device=device).unsqueeze(0)
        g, seg, mask = model.split(flat)
        return g, seg, mask.bool()

    def episode_completion(step_info) -> float:
        """Fraction of the episode's nets that actually committed.

        `info["completed"]` is a LIST of net names, so `info.get("completed")`
        is truthy the moment ONE net lands -- which reads as 100% on a 24-net
        board where 23 failed."""
        num = step_info.get("num_nets") or 0
        return len(step_info.get("completed", ())) / num if num else 0.0

    # Reset env
    obs, info = env.reset()
    global_vec, segments, segment_mask = to_tensors(obs)
    global_vec_norm = normalize_global(global_vec)

    history = {
        "steps": [],
        "reward": [],
        "completion_rate": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "eval_completion_rate": [],
        "eval_reward": [],
    }

    completed_window: List[float] = []
    episode_reward_window: List[float] = []
    curr_ep_reward = 0.0
    global_step = 0
    start_time = time.time()
    rollout_count = 0

    # Distance curriculum (matches RL_PLAN)
    DIST_CURRICULUM = [50, 100, None]  # max pad distance in mm (None = no limit)
    dist_stage = 0
    dist_curriculum_window: List[float] = []
    # Note: LineRouteEnv doesn't currently support max_pad_dist; would need board gen integration

    # In-notebook display
    in_colab = False
    try:
        from IPython.display import clear_output, display, Image
        in_colab = True
    except ImportError:
        pass

    print(f"Starting training: {total_timesteps:,} steps, rollout={rollout_steps}, epochs={epochs}")

    while global_step < total_timesteps:
        rollout_count += 1
        buffer.reset()

        # -------------------------------------------------------------
        # Collect Rollout
        # -------------------------------------------------------------
        for step in range(rollout_steps):
            global_step += 1

            with torch.no_grad():
                action_t, log_prob_t, _, value_t = model.get_action_and_value(
                    global_vec_norm, segments, segment_mask
                )

            action = action_t.squeeze(0).cpu().numpy()  # shape (1,)
            prev_obs = obs
            next_obs, reward, term, trunc, step_info = env.step(action)
            done = term or trunc
            curr_ep_reward += reward
            scaled_reward = reward_scaler.scale(reward, done)

            # The RAW observation goes in the buffer, not the normalised
            # globals: normalisation statistics keep moving during the
            # rollout, so a stored pre-normalised row would be scaled by
            # different statistics than the update re-scales the rest with.
            buffer.add(
                obs=torch.as_tensor(prev_obs, dtype=torch.float32, device=device),
                action=action_t.squeeze(0),
                log_prob=log_prob_t.squeeze(0),
                reward=scaled_reward,
                done=done,
                value=value_t.squeeze(0),
            )

            if done:
                completed_window.append(episode_completion(step_info))
                episode_reward_window.append(curr_ep_reward)
                if len(completed_window) > 40:
                    completed_window.pop(0)
                if len(episode_reward_window) > 40:
                    episode_reward_window.pop(0)

                curr_ep_reward = 0.0
                next_obs, step_info = env.reset()

            # Next observation
            obs = next_obs
            global_vec, segments, segment_mask = to_tensors(obs)
            global_vec_norm = normalize_global(global_vec)

        # -------------------------------------------------------------
        # Compute GAE Advantages
        # -------------------------------------------------------------
        with torch.no_grad():
            last_value = model.get_value(global_vec_norm, segments, segment_mask)
        buffer.compute_advantages_and_returns(last_value.squeeze(0), done)

        # -------------------------------------------------------------
        # PPO Optimization Epochs
        # -------------------------------------------------------------
        total_p_loss, total_v_loss, total_entropy = 0.0, 0.0, 0.0
        num_updates = 0

        for epoch in range(epochs):
            for (
                mb_obs,
                mb_actions,
                mb_old_log_probs,
                mb_advs,
                mb_returns,
                mb_values,
            ) in buffer.get_minibatches(minibatch_size):
                mb_global, mb_segments, mb_mask = model.split(mb_obs)
                mb_mask = mb_mask.bool()
                mb_global = normalize_global(mb_global, update=False)
                mb_actions = mb_actions.unsqueeze(-1)
                mb_old_log_probs = mb_old_log_probs.unsqueeze(-1)
                mb_advs = mb_advs.unsqueeze(-1)

                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                _, new_log_prob, entropy, new_value = model.get_action_and_value(
                    mb_global, mb_segments, mb_mask, mb_actions
                )

                # Policy Loss
                ratio = torch.exp(new_log_prob - mb_old_log_probs)
                surr1 = -mb_advs * ratio
                surr2 = -mb_advs * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = torch.max(surr1, surr2).mean()

                # Value Loss (clipped)
                new_value = new_value.squeeze(-1)
                v_unclipped = (new_value - mb_returns) ** 2
                v_clipped_pred = mb_values + torch.clamp(
                    new_value - mb_values, -value_clip_coef, value_clip_coef
                )
                v_clipped = (v_clipped_pred - mb_returns) ** 2
                value_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                # Total Loss
                loss = policy_loss - ent_coef * entropy.mean() + vf_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                total_p_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # -------------------------------------------------------------
        # Logging & Checkpointing
        # -------------------------------------------------------------
        avg_comp = float(np.mean(completed_window) * 100.0) if completed_window else 0.0
        avg_rew = float(np.mean(episode_reward_window)) if episode_reward_window else 0.0
        avg_entropy = total_entropy / max(1, num_updates)
        elapsed = time.time() - start_time
        fps = int(global_step / max(1e-3, elapsed))
        p_loss = total_p_loss / max(1, num_updates)
        v_loss = total_v_loss / max(1, num_updates)

        history["steps"].append(global_step)
        history["reward"].append(avg_rew)
        history["completion_rate"].append(avg_comp)
        history["policy_loss"].append(p_loss)
        history["value_loss"].append(v_loss)
        history["entropy"].append(avg_entropy)

        progress_pct = (global_step / total_timesteps) * 100.0
        status_line = (
            f"[{progress_pct:5.1f}%] Step {global_step:>7d}/{total_timesteps} | "
            f"Success: {avg_comp:5.1f}% | "
            f"Reward: {avg_rew:7.1f} | "
            f"Entropy: {avg_entropy:5.3f} | "
            f"Loss: [π={p_loss:6.3f}, V={v_loss:6.3f}] | "
            f"Speed: {fps:>4d} steps/s"
        )
        print(status_line)
        sys.stdout.flush()

        # Evaluation
        if global_step % eval_interval == 0 and global_step > 0:
            eval_comp, eval_rew = evaluate_policy(
                model, board_path, track_width_nm, max_steps_per_episode,
                eval_episodes, device, global_mean, global_var
            )
            history["eval_completion_rate"].append(eval_comp)
            history["eval_reward"].append(eval_rew)
            print(f"  📊 EVAL: Completion={eval_comp:.1f}%, Reward={eval_rew:.1f}")

        # Plot & checkpoint
        if global_step % plot_interval == 0 or global_step >= total_timesteps:
            plot_learning_curves(history, plot_path)
            torch.save(
                {
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_mean": global_mean.cpu(),
                    "global_var": global_var.cpu(),
                    "global_count": global_count.cpu(),
                    "history": history,
                    "completion_rate": avg_comp,
                },
                chk_path / "line_policy_latest.pt",
            )

    print(f"\n{'='*80}")
    print(f"🎉 Training Complete! Final Routing Success: {avg_comp:.1f}%")
    print(f"{'='*80}")
    return model


def evaluate_policy(
    model: LineGeometryPolicy,
    board_path: str,
    track_width_nm: int,
    max_steps: int,
    num_episodes: int,
    device: torch.device,
    global_mean: torch.Tensor,
    global_var: torch.Tensor,
) -> tuple[float, float]:
    """Evaluate policy deterministically (mean action)."""
    model.eval()
    env = LineRouteEnv(
        board_path=board_path,
        track_width_nm=track_width_nm,
        max_steps_per_net=max_steps,
        obs_config=LineObsConfig(max_steps=max_steps),
    )

    completions = []
    rewards = []

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            info: dict = {}

            while not done:
                flat = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                global_vec, segments, segment_mask = model.split(flat)
                segment_mask = segment_mask.bool()

                # Normalize
                global_vec_norm = (global_vec - global_mean.to(device)) / (torch.sqrt(global_var.to(device)) + 1e-8)

                # Deterministic: use mean
                dist, _ = model.forward(global_vec_norm, segments, segment_mask)
                action = dist.mean.squeeze(0).cpu().numpy()

                obs, reward, term, trunc, info = env.step(action)
                ep_reward += reward
                done = term or trunc

            num_nets = info.get("num_nets") or 0
            completions.append(len(info.get("completed", ())) / num_nets if num_nets else 0.0)
            rewards.append(ep_reward)

    model.train()
    return float(np.mean(completions) * 100), float(np.mean(rewards))


def plot_learning_curves(history: Dict[str, List[float]], save_path: Path):
    if len(history["steps"]) < 2:
        return

    steps = history["steps"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=110)
    fig.patch.set_facecolor("#101216")
    axes = axes.flatten()

    for ax in axes:
        ax.set_facecolor("#181b22")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # 1. Completion Rate
    axes[0].plot(steps, history["completion_rate"], color="#00ffcc", lw=2.2)
    axes[0].axhline(80.0, color="#ff4444", linestyle="--", alpha=0.7, label="Curriculum Target (80%)")
    axes[0].set_title("Routing Success Rate (%)", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[0].set_ylim(-5, 105)
    axes[0].set_xlabel("Training Steps", color="#8b949e")
    axes[0].legend(facecolor="#181b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    axes[0].grid(True, alpha=0.15)

    # 2. Mean Reward
    axes[1].plot(steps, history["reward"], color="#ffaa00", lw=2.2)
    axes[1].set_title("Mean Episode Reward", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Training Steps", color="#8b949e")
    axes[1].grid(True, alpha=0.15)

    # 3. Policy Loss
    axes[2].plot(steps, history["policy_loss"], color="#ff0055", lw=1.5)
    axes[2].set_title("Policy Loss (PPO Clip)", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Training Steps", color="#8b949e")
    axes[2].grid(True, alpha=0.15)

    # 4. Value Loss
    axes[3].plot(steps, history["value_loss"], color="#aa00ff", lw=1.5)
    axes[3].set_title("Value Loss (Clipped MSE)", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Training Steps", color="#8b949e")
    axes[3].grid(True, alpha=0.15)

    # 5. Entropy
    axes[4].plot(steps, history["entropy"], color="#00aaff", lw=1.5)
    axes[4].set_title("Policy Entropy", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[4].set_xlabel("Training Steps", color="#8b949e")
    axes[4].grid(True, alpha=0.15)

    # 6. Evaluation
    if history["eval_completion_rate"]:
        eval_steps = [history["steps"][i] for i in range(0, len(history["steps"]), len(history["steps"]) // len(history["eval_completion_rate"]))][:len(history["eval_completion_rate"])]
        axes[5].plot(eval_steps, history["eval_completion_rate"], color="#00ff88", lw=2.2, marker='o', label="Eval Success")
        axes[5].plot(eval_steps, history["eval_reward"], color="#ff8800", lw=1.5, marker='s', label="Eval Reward")
        axes[5].set_title("Evaluation Metrics", color="#e6edf3", fontsize=11, fontweight="bold")
        axes[5].legend(facecolor="#181b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    axes[5].set_xlabel("Training Steps", color="#8b949e")
    axes[5].grid(True, alpha=0.15)

    plt.tight_layout()
    fig.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # Example: train on a generated board
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=str, required=True, help="Path to .kicad_pcb board")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--checkpoint-dir", type=str, default="/content/drive/MyDrive/pcb_line_router/checkpoints")
    args = parser.parse_args()

    train_line_policy(board_path=args.board, total_timesteps=args.steps, checkpoint_dir=args.checkpoint_dir)