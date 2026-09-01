"""LineGeometryPolicy: Actor-Critic for 1-D heading action on line-segment observations.

Network architecture:
- Per-segment MLP: NUM_SEGMENT_FEATURES -> 64 -> 64 (shared weights)
- Masked max-pool + masked mean-pool -> 128
- Global MLP: NUM_GLOBAL -> 64
- Concat 192 -> 128 -> 128
- Actor: 1 mean + 1 learned log_std (Gaussian over [-1, 1])
- Critic: 1

The two input widths come from pcbworld/env/line_obs.py rather than being
written here, and that is load-bearing rather than tidy. Hard-coded as 8 and
11 they silently described a DIFFERENT observation than the env emits: the
env's global block is 15 wide, and the seven it does not carry are
base_heading_cos/sin, geodesic_dist, clearance_now, clearance_ahead and
geo_dir_cos/sin -- i.e. every feature that exists specifically so the policy
can see an obstacle before it hits one. A network built to the wrong width
does not error, it just reads the first 8 numbers and routes blind.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from pcbworld.env.line_obs import NUM_GLOBAL, NUM_SEGMENT_FEATURES, split_observation


class LineGeometryPolicy(nn.Module):
    def __init__(
        self,
        segment_dim: int = NUM_SEGMENT_FEATURES,
        global_dim: int = NUM_GLOBAL,
        k_segments: int = 32,
        hidden_dim: int = 64,
        pooled_dim: int = 128,
        trunk_dim: int = 128,
        action_dim: int = 1,
        log_std_init: float = -0.5,  # ~0.6 std, reasonable exploration
    ):
        super().__init__()
        self.segment_dim = segment_dim
        self.global_dim = global_dim
        self.k_segments = k_segments
        self.action_dim = action_dim

        # Per-segment encoder (shared weights)
        self.segment_mlp = nn.Sequential(
            nn.Linear(segment_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Pooling: masked max + masked mean → 2 * hidden_dim
        self.pool_proj = nn.Linear(2 * hidden_dim, pooled_dim)

        # Global encoder
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Shared trunk
        trunk_in = pooled_dim + hidden_dim  # 128 + 64 = 192
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, trunk_dim),
            nn.ReLU(inplace=True),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU(inplace=True),
        )

        # Actor head: mean + log_std
        self.actor_mean = nn.Linear(trunk_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

        # Critic head
        self.critic = nn.Linear(trunk_dim, 1)

        # Initialize actor near zero (mean-0 policy = straight at target)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

    def split(self, obs: torch.Tensor, extra_globals: int = 0):
        """Flat env observation -> the three tensors forward() wants.

        One place that knows the layout, shared with the env, so a stage that
        appends features (the diff-pair/tune env's leg-kind one-hot) cannot
        put the policy and the env out of step."""
        return split_observation(obs, extra_globals=extra_globals)

    def forward(
        self,
        global_vec: torch.Tensor,      # (B, NUM_GLOBAL [+ extras])
        segments: torch.Tensor,        # (B, K, NUM_SEGMENT_FEATURES)
        segment_mask: torch.Tensor,    # (B, K) bool
    ) -> Tuple[Normal, torch.Tensor]:
        """
        Returns:
            dist: Normal distribution over action (B, 1)
            value: state value estimate (B, 1)
        """
        B, K, _ = segments.shape

        # Encode segments: (B, K, 11) -> (B, K, 64)
        seg_encoded = self.segment_mlp(segments)  # (B, K, 64)

        # Masked pooling.
        #
        # The -inf fill has to be undone where a row has NO valid segments,
        # or max() returns -inf and every downstream number is NaN. That is
        # not a hypothetical: LineRouteEnv returns an all-zero observation
        # once the last net is done, so the terminal state of EVERY episode
        # has an empty mask, and the NaN lands in the GAE bootstrap value and
        # from there in the gradient.
        any_valid = segment_mask.any(dim=1, keepdim=True)  # (B, 1)
        masked_max = seg_encoded.masked_fill(~segment_mask.unsqueeze(-1), -float('inf')).max(dim=1)[0]  # (B, 64)
        masked_max = torch.where(any_valid, masked_max, torch.zeros_like(masked_max))
        # Mean pool over valid segments
        masked_sum = seg_encoded.masked_fill(~segment_mask.unsqueeze(-1), 0.0).sum(dim=1)  # (B, 64)
        valid_counts = segment_mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # (B, 1)
        masked_mean = masked_sum / valid_counts  # (B, 64)

        # Concat max + mean -> (B, 128)
        pooled = torch.cat([masked_max, masked_mean], dim=-1)
        pooled = self.pool_proj(pooled)  # (B, 128)

        # Encode global vector
        global_encoded = self.global_mlp(global_vec)  # (B, 64)

        # Trunk
        trunk_in = torch.cat([pooled, global_encoded], dim=-1)  # (B, 192)
        features = self.trunk(trunk_in)  # (B, 128)

        # Actor
        mean = self.actor_mean(features)  # (B, 1)
        mean = torch.tanh(mean)  # Bound to [-1, 1]
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std).clamp(min=1e-4)
        dist = Normal(mean, std)

        # Critic
        value = self.critic(features)  # (B, 1)

        return dist, value

    def get_action_and_value(
        self,
        global_vec: torch.Tensor,
        segments: torch.Tensor,
        segment_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate action for PPO step.

        Returns:
            action: (B, 1) sampled or passed action
            log_prob: (B, 1) log probability
            entropy: (B, 1) distribution entropy
            value: (B, 1) state value
        """
        dist, value = self.forward(global_vec, segments, segment_mask)

        if action is None:
            action = dist.sample()
        else:
            action = action.clamp(-1.0, 1.0)  # Ensure valid range

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return action, log_prob, entropy, value

    def get_value(
        self,
        global_vec: torch.Tensor,
        segments: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get value estimate only (for GAE bootstrap)."""
        _, value = self.forward(global_vec, segments, segment_mask)
        return value