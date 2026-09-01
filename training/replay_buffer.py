"""Replay Buffer and Generalized Advantage Estimation (GAE) for PCB Router.

Supports both flat observations (legacy) and Dict observations (line geometry).
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Union, Dict, Any


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Union[tuple[int, ...], Dict[str, tuple[int, ...]], None],
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.is_dict_obs = isinstance(obs_shape, dict)

        if self.is_dict_obs:
            # Dict observation: store each key separately
            self.observations = {
                key: torch.zeros((buffer_size, *shape), dtype=torch.float32, device=device)
                for key, shape in obs_shape.items()
            }
            # segment_mask is bool
            if "segment_mask" in self.observations:
                self.observations["segment_mask"] = torch.zeros(
                    (buffer_size, *obs_shape["segment_mask"]), dtype=torch.bool, device=device
                )
        else:
            # Flat observation
            self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)

        # Actions: continuous for line geometry, discrete for legacy
        self.actions = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=device)

        self.advantages = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.returns = torch.zeros((buffer_size,), dtype=torch.float32, device=device)

        self.ptr = 0

    def add(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
    ):
        idx = self.ptr
        if self.is_dict_obs:
            for key, tensor in obs.items():
                self.observations[key][idx].copy_(tensor)
        else:
            self.observations[idx].copy_(obs)
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = float(reward)
        self.dones[idx] = 1.0 if done else 0.0
        self.values[idx] = value.squeeze()
        self.ptr += 1

    def compute_advantages_and_returns(self, last_value: torch.Tensor, done: bool):
        last_val = last_value.squeeze()
        last_gae_lam = 0.0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - (1.0 if done else 0.0)
                next_value = last_val
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size: int):
        indices = np.random.permutation(self.buffer_size)
        for start_idx in range(0, self.buffer_size, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            if self.is_dict_obs:
                batch_obs = {key: tensor[batch_indices] for key, tensor in self.observations.items()}
            else:
                batch_obs = self.observations[batch_indices]
            yield (
                batch_obs,
                self.actions[batch_indices],
                self.log_probs[batch_indices],
                self.advantages[batch_indices],
                self.returns[batch_indices],
                self.values[batch_indices],
            )

    def reset(self):
        self.ptr = 0
