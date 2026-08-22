"""PPO baseline trainer for PCBRouteEnv and LineRouteEnv.

Supports:
  - Standard MLP ActorCritic (for flat raster/scalar envs)
  - LineActorCritic (for line-geometry observation with segment pooling)
  - Observation normalization for the 8 global features via RunningMeanStd
  - Checkpoint saving to local / Google Drive path
  - Tracking of net completion rates alongside episode rewards
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from pcbworld.agents.line_policy import LineActorCritic, RunningMeanStd
from pcbworld.env.line_obs import NUM_GLOBAL, NUM_SEGMENT_FEATURES


@dataclasses.dataclass
class PPOConfig:
    total_timesteps: int = 20_000
    rollout_steps: int = 512
    epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    device: str = "cpu"
    checkpoint_interval: int = 5_000
    checkpoint_dir: str | None = None
    init_checkpoint: str | None = None
    normalize_globals: bool = True



class ActorCritic(nn.Module):
    """Standard MLP Actor-Critic fallback for flat vector observations."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.action_mean = nn.Linear(hidden_size, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        mean = self.action_mean(features)
        std = self.action_log_std.exp().expand_as(mean)
        value = self.value_head(features).squeeze(-1)
        return Normal(mean, std), value

    def act(self, obs: torch.Tensor):
        dist, value = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


@dataclasses.dataclass
class RolloutBuffer:
    obs: list
    actions: list
    log_probs: list
    rewards: list
    values: list
    dones: list

    @classmethod
    def empty(cls) -> "RolloutBuffer":
        return cls([], [], [], [], [], [])

    def add(self, obs, action, log_prob, reward, value, done) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def _preprocess_obs(obs: np.ndarray, rms: RunningMeanStd | None) -> np.ndarray:
    if rms is None:
        return obs
    norm_obs = obs.copy()
    norm_obs[:NUM_GLOBAL] = rms.normalize(obs[:NUM_GLOBAL])
    return norm_obs


def collect_rollout(
    env,
    policy: nn.Module,
    obs: np.ndarray,
    n_steps: int,
    device: str,
    rms: RunningMeanStd | None = None,
):
    buffer = RolloutBuffer.empty()
    episode_rewards = []
    current_episode_reward = 0.0
    completed_nets = 0
    failed_nets = 0

    for _ in range(n_steps):
        # Update running mean/std on the raw 8 global features
        if rms is not None:
            rms.update(obs[:NUM_GLOBAL])

        processed_obs = _preprocess_obs(obs, rms)
        obs_t = torch.as_tensor(processed_obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_t, log_prob_t, value_t = policy.act(obs_t)

        action = action_t.squeeze(0).cpu().numpy()
        clipped_action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, reward, terminated, truncated, info = env.step(clipped_action)
        done = terminated or truncated

        buffer.add(
            processed_obs,
            action,
            log_prob_t.item(),
            reward,
            value_t.item(),
            float(done),
        )

        current_episode_reward += reward
        obs = next_obs

        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            if "completed" in info and "failed" in info:
                completed_nets += len(info["completed"])
                failed_nets += len(info["failed"])
            obs, _info = env.reset()

    processed_last_obs = _preprocess_obs(obs, rms)
    with torch.no_grad():
        last_value = policy.act(
            torch.as_tensor(processed_last_obs, dtype=torch.float32, device=device).unsqueeze(0)
        )[2].item()

    rollout_info = {
        "episode_rewards": episode_rewards,
        "completed_nets": completed_nets,
        "failed_nets": failed_nets,
    }
    return buffer, obs, last_value, rollout_info


def ppo_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    last_value: float,
    cfg: PPOConfig,
) -> dict:
    obs = torch.as_tensor(np.asarray(buffer.obs), dtype=torch.float32, device=cfg.device)
    actions = torch.as_tensor(np.asarray(buffer.actions), dtype=torch.float32, device=cfg.device)
    old_log_probs = torch.as_tensor(np.asarray(buffer.log_probs), dtype=torch.float32, device=cfg.device)
    rewards = np.asarray(buffer.rewards, dtype=np.float32)
    values = np.asarray(buffer.values, dtype=np.float32)
    dones = np.asarray(buffer.dones, dtype=np.float32)

    advantages, returns = compute_gae(rewards, values, dones, last_value, cfg.gamma, cfg.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=cfg.device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=cfg.device)

    n = len(buffer.rewards)
    indices = np.arange(n)

    last_stats = {}

    for _epoch in range(cfg.epochs):
        np.random.shuffle(indices)

        for start in range(0, n, cfg.minibatch_size):
            batch_idx = indices[start : start + cfg.minibatch_size]
            if len(batch_idx) == 0:
                continue

            batch_idx_t = torch.as_tensor(batch_idx, dtype=torch.long, device=cfg.device)

            log_probs, entropy, value = policy.evaluate(obs[batch_idx_t], actions[batch_idx_t])
            ratio = torch.exp(log_probs - old_log_probs[batch_idx_t])

            batch_adv = advantages_t[batch_idx_t]
            surrogate1 = ratio * batch_adv
            surrogate2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * batch_adv
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            value_loss = ((value - returns_t[batch_idx_t]) ** 2).mean()
            entropy_loss = -entropy.mean()

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                + cfg.entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            last_stats = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": -entropy_loss.item(),
            }

    return last_stats


def save_checkpoint(
    checkpoint_dir: str | Path,
    steps_done: int,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    rms: RunningMeanStd | None,
    cfg: PPOConfig,
    stats: dict,
) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)

    state = {
        "steps_done": steps_done,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rms_mean": rms.mean if rms is not None else None,
        "rms_var": rms.var if rms is not None else None,
        "rms_count": rms.count if rms is not None else None,
        "config": dataclasses.asdict(cfg),
        "last_stats": stats,
    }

    torch.save(state, path / "policy_latest.pt")
    torch.save(state, path / f"policy_{steps_done}.pt")

    stats_file = path / "training_stats.jsonl"
    with open(stats_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"steps": steps_done, **stats}) + "\n")


def train(env, cfg: PPOConfig | None = None) -> nn.Module:
    cfg = cfg or PPOConfig()

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # Detect if observation matches LineRouteEnv (globals + K * 12 features)
    is_line_obs = (obs_dim >= NUM_GLOBAL) and ((obs_dim - NUM_GLOBAL) % NUM_SEGMENT_FEATURES == 0)

    if is_line_obs:
        policy = LineActorCritic(action_dim=action_dim).to(cfg.device)
        rms = RunningMeanStd(shape=(NUM_GLOBAL,)) if cfg.normalize_globals else None
    else:
        policy = ActorCritic(obs_dim, action_dim, cfg.hidden_size).to(cfg.device)
        rms = None

    if cfg.init_checkpoint and os.path.isfile(cfg.init_checkpoint):
        print(f"Loading initial weights from {cfg.init_checkpoint}...")
        chk = torch.load(cfg.init_checkpoint, map_location=cfg.device, weights_only=False)
        if "policy_state_dict" in chk:
            policy.load_state_dict(chk["policy_state_dict"])
        if rms is not None and chk.get("rms_mean") is not None:
            rms.mean = np.array(chk["rms_mean"], dtype=np.float32)
            rms.var = np.array(chk["rms_var"], dtype=np.float32)
            rms.count = float(chk.get("rms_count", 1.0))

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)


    obs, _info = env.reset()
    steps_done = 0
    last_checkpoint_step = 0

    while steps_done < cfg.total_timesteps:
        buffer, obs, last_value, rollout_info = collect_rollout(
            env, policy, obs, cfg.rollout_steps, cfg.device, rms=rms
        )
        stats = ppo_update(policy, optimizer, buffer, last_value, cfg)
        steps_done += cfg.rollout_steps

        episodes = rollout_info["episode_rewards"]
        mean_reward = float(np.mean(episodes)) if episodes else float("nan")
        comp = rollout_info["completed_nets"]
        failed = rollout_info["failed_nets"]
        comp_rate = (comp / (comp + failed) * 100.0) if (comp + failed) > 0 else float("nan")

        comp_str = f"completion_rate={comp_rate:.1f}% ({comp}/{comp+failed})" if (comp + failed) > 0 else "completion_rate=n/a"

        print(
            f"steps={steps_done:6d} episodes={len(episodes):3d} "
            f"mean_reward={mean_reward:8.2f} {comp_str} "
            f"policy_loss={stats.get('policy_loss', float('nan')):7.4f} "
            f"value_loss={stats.get('value_loss', float('nan')):7.4f} "
            f"entropy={stats.get('entropy', float('nan')):6.4f}"
        )

        if cfg.checkpoint_dir and (steps_done - last_checkpoint_step >= cfg.checkpoint_interval or steps_done >= cfg.total_timesteps):
            save_checkpoint(
                cfg.checkpoint_dir,
                steps_done,
                policy,
                optimizer,
                rms,
                cfg,
                {"mean_reward": mean_reward, "completion_rate": comp_rate, **stats},
            )
            last_checkpoint_step = steps_done

    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board_path", help=".kicad_pcb file to train on")
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Path to initial policy checkpoint to resume/transfer from")
    parser.add_argument("--enable-ripup", action="store_true", help="Enable rip-up and reroute for conflicting traces")
    parser.add_argument("--max-ripups", type=int, default=8, help="Max rip-up operations per episode")
    parser.add_argument("--use-legacy-env", action="store_true", help="Use PCBRouteEnv instead of LineRouteEnv")
    args = parser.parse_args()

    if args.use_legacy_env:
        from pcbworld.env.pcb_route_env import PCBRouteEnv
        env = PCBRouteEnv(args.board_path)
    else:
        from pcbworld.env.line_route_env import LineRouteEnv
        env = LineRouteEnv(
            args.board_path,
            enable_ripup=args.enable_ripup,
            max_ripups_per_episode=args.max_ripups,
        )


    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        init_checkpoint=args.init_checkpoint,
    )

    train(env, cfg)


if __name__ == "__main__":
    main()
