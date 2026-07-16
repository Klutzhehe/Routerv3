"""Minimal single-process PPO baseline for PCBRouteEnv.

Deliberately small and dependency-light (plain PyTorch, no RL framework)
rather than matching the PCBWorld paper's exact network (their Fourier-
feature-encoded Transformer policy) -- this is meant as the "does the
plumbing work at all" sanity check ROADMAP.md's step 5 asks for before any
of the deferred "novel SOTA agent" directions (rip-up-reroute, graph-
transformer policy, net-ordering meta-policy) get attempted on top of it.

Only ever runs against pcbworld.env.pcb_route_env.PCBRouteEnv, which needs
the Colab-built pcbworld_pns_bridge -- this file's control flow (rollout
collection, GAE, the clipped-surrogate update) is exercised locally against
a fake env in tests/test_ppo_baseline.py, but the actual routing reward
signal has never been observed end to end. Treat a first training run's
numbers as "does this crash and produce finite losses", not "is this a
good policy" -- matching against the paper's own numbers is future work
once this at least runs.

Single environment, single process -- see docs/performance.md for why
real training throughput needs multiple OS-process env workers
(gymnasium.vector.AsyncVectorEnv or similar); not implemented here since
this is the "does the loop work" baseline, not the throughput-tuned
trainer.
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


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


class ActorCritic(nn.Module):
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


def collect_rollout(env, policy: ActorCritic, obs, n_steps: int, device: str):
    buffer = RolloutBuffer.empty()
    episode_rewards = []
    current_episode_reward = 0.0

    for _ in range(n_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_t, log_prob_t, value_t = policy.act(obs_t)

        action = action_t.squeeze(0).cpu().numpy()
        clipped_action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, reward, terminated, truncated, _info = env.step(clipped_action)
        done = terminated or truncated

        buffer.add(
            obs,
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
            obs, _info = env.reset()

    with torch.no_grad():
        last_value = policy.act(
            torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        )[2].item()

    return buffer, obs, last_value, episode_rewards


def ppo_update(policy: ActorCritic, optimizer, buffer: RolloutBuffer, last_value: float, cfg: PPOConfig) -> dict:
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


def train(env, cfg: PPOConfig | None = None) -> ActorCritic:
    cfg = cfg or PPOConfig()

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    policy = ActorCritic(obs_dim, action_dim, cfg.hidden_size).to(cfg.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    obs, _info = env.reset()
    steps_done = 0

    while steps_done < cfg.total_timesteps:
        buffer, obs, last_value, episode_rewards = collect_rollout(
            env, policy, obs, cfg.rollout_steps, cfg.device
        )
        stats = ppo_update(policy, optimizer, buffer, last_value, cfg)
        steps_done += cfg.rollout_steps

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else float("nan")
        print(
            f"steps={steps_done} episodes={len(episode_rewards)} "
            f"mean_episode_reward={mean_reward:.3f} "
            f"policy_loss={stats.get('policy_loss', float('nan')):.4f} "
            f"value_loss={stats.get('value_loss', float('nan')):.4f} "
            f"entropy={stats.get('entropy', float('nan')):.4f}"
        )

    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board_path", help=".kicad_pcb file to train on")
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    args = parser.parse_args()

    # Deferred import: pcbworld_pns_bridge only exists after the Colab
    # build in notebooks/00_setup.ipynb.
    from pcbworld.env.pcb_route_env import PCBRouteEnv

    env = PCBRouteEnv(args.board_path)
    cfg = PPOConfig(total_timesteps=args.total_timesteps, rollout_steps=args.rollout_steps)
    train(env, cfg)


if __name__ == "__main__":
    main()
