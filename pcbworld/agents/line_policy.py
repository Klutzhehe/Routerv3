"""Line-geometry policy and value network for LineRouteEnv.

Spec from docs/RL_PLAN.md:
  per-segment MLP   12 → 64 → 64          (shared weights)
  masked max-pool + masked mean-pool  →  128
  global MLP         8 → 64
  concat 192 → 128 → 128
    actor  → 1 mean + 1 learned log_std
    critic → 1

~47k parameters, small enough to train on CPU beside the env workers.

Crucial property:
  The actor's final linear layer is initialized with near-zero gain so that
  an untrained policy emits a ≈ 0 (within ~1e-3 radians). Because the
  observation frame points +x at the target pad, a = 0 walks straight at
  the pad -- putting the untrained policy AT the greedy straight-line baseline
  (8/24 nets) rather than below it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from pcbworld.env.line_obs import NUM_GLOBAL, NUM_SEGMENT_FEATURES


class RunningMeanStd:
    """Tracks running mean and variance using Welford's algorithm.

    `update` takes a BATCH: shape (N, *shape). A single unbatched sample of
    shape `shape` is accepted and promoted to (1, *shape) -- it has to be,
    because the trainer feeds one step at a time and the un-promoted version
    of this was silently wrong for the whole of the first curriculum run.
    `np.mean(x, axis=0)` on a 1-D (8,) array reduces across the FEATURES and
    returns a scalar, so all eight globals converged on one shared mean and
    one shared std: head_collides (0/1) ended up shifted and scaled by
    statistics dominated by dist_to_target (0-5), which is exactly the signal
    the policy most needed to see. The old unit test passed a (100, 8) array
    and so never exercised the shape the trainer actually used.
    """

    def __init__(self, shape: tuple[int, ...] = (NUM_GLOBAL,), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with batch x (shape: (N, *shape))."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == len(self.mean.shape):
            x = x[None, ...]
        if x.shape[1:] != self.mean.shape:
            raise ValueError(
                f"RunningMeanStd expects trailing shape {self.mean.shape}, got {x.shape}"
            )

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean.astype(np.float32)
        self.var = new_var.astype(np.float32)
        self.count = tot_count

    def load_from_checkpoint(self, chk: dict) -> bool:
        """Restore stats from a checkpoint dict, or refuse if they do not fit.

        NUM_GLOBAL changed from 8 to 10 when base_heading_cos/sin were added,
        so every checkpoint written before that carries 8-element stats. Silently
        assigning them would leave `mean` and `var` a different length from the
        slice they divide, and numpy would either raise somewhere far away or
        broadcast something meaningless. Returns False and leaves the stats at
        their identity values instead."""
        if chk.get("rms_mean") is None:
            return False
        mean = np.asarray(chk["rms_mean"], dtype=np.float32)
        if mean.shape != self.mean.shape:
            print(
                f"  [rms] checkpoint stats are {mean.shape}, this build needs "
                f"{self.mean.shape} -- ignoring them and re-estimating from scratch."
            )
            return False
        self.mean = mean
        self.var = np.asarray(chk["rms_var"], dtype=np.float32)
        self.count = float(chk.get("rms_count", 1.0))
        return True

    def normalize(
        self, x: np.ndarray | torch.Tensor, clip: float = 10.0
    ) -> np.ndarray | torch.Tensor:
        """Normalize x using current running statistics, clipped to +/- `clip`.

        The clip is not decoration. Several globals are constant for a whole
        stage -- head_layer, target_layer and length_slack are identically 0
        until the multi-layer and length-tuning stages -- so their running
        variance decays toward 0 and the epsilon in the denominator is all
        that stands between a float wobble and a six-figure input. 10.0 is the
        usual VecNormalize default.
        """
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
            std = torch.as_tensor(np.sqrt(self.var + 1e-8), device=x.device, dtype=x.dtype)
            return torch.clamp((x - mean) / std, -clip, clip)
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)


class LineActorCritic(nn.Module):
    """Actor-Critic network for the line-geometry observation."""

    def __init__(
        self,
        action_dim: int = 1,
        num_global: int = NUM_GLOBAL,
        num_segment_features: int = NUM_SEGMENT_FEATURES,
        seg_hidden_dim: int = 64,
        global_hidden_dim: int = 64,
        trunk_hidden_dim: int = 128,
        initial_log_std: float = -1.2,
    ):
        super().__init__()

        self.num_global = num_global
        self.num_segment_features = num_segment_features
        self.action_dim = action_dim

        # Per-segment MLP (shared across all K segments)
        self.segment_mlp = nn.Sequential(
            nn.Linear(num_segment_features, seg_hidden_dim),
            nn.Tanh(),
            nn.Linear(seg_hidden_dim, seg_hidden_dim),
            nn.Tanh(),
        )

        # Global features MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(num_global, global_hidden_dim),
            nn.Tanh(),
        )

        # Combined trunk: (global_hidden_dim + 2 * seg_hidden_dim) -> trunk_hidden_dim -> trunk_hidden_dim
        combined_in_dim = global_hidden_dim + 2 * seg_hidden_dim  # 64 + 128 = 192
        self.trunk = nn.Sequential(
            nn.Linear(combined_in_dim, trunk_hidden_dim),
            nn.Tanh(),
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim),
            nn.Tanh(),
        )

        # Actor head
        self.action_mean = nn.Linear(trunk_hidden_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.full((action_dim,), initial_log_std, dtype=torch.float32))

        # Critic head
        self.value_head = nn.Linear(trunk_hidden_dim, 1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        # Standard orthogonal init for feature extraction layers
        for module in (self.segment_mlp, self.global_mlp, self.trunk):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=float(np.sqrt(2)))
                    nn.init.zeros_(layer.bias)

        # Actor mean output layer: near-zero gain so initial mean is approx 0.0
        nn.init.orthogonal_(self.action_mean.weight, gain=1e-3)
        nn.init.zeros_(self.action_mean.bias)

        # Value head
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract combined (B, 128) representation from flat (B, flat_size) observation."""
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        global_feats = obs[:, : self.num_global]
        seg_raw = obs[:, self.num_global :]
        k_nearest = seg_raw.shape[1] // self.num_segment_features
        seg_feats = seg_raw.view(-1, k_nearest, self.num_segment_features)

        # Segment valid mask is the last feature (index 11)
        valid_mask = seg_feats[:, :, 11:12]  # (B, K, 1)

        # Embed all segments: (B, K, seg_hidden_dim)
        seg_emb = self.segment_mlp(seg_feats)

        # Masked mean-pool
        mask_sum = torch.clamp(valid_mask.sum(dim=1), min=1.0)  # (B, 1)
        mean_pool = (seg_emb * valid_mask).sum(dim=1) / mask_sum  # (B, seg_hidden_dim)

        # Masked max-pool
        # Fill masked items with large negative value
        masked_emb = torch.where(valid_mask.bool(), seg_emb, torch.full_like(seg_emb, -1e9))
        max_pool = torch.max(masked_emb, dim=1).values  # (B, seg_hidden_dim)
        # In case all segments in an item are invalid, replace -1e9 with 0
        has_any_valid = (valid_mask.sum(dim=1) > 0.5)  # (B, 1)
        max_pool = torch.where(has_any_valid, max_pool, torch.zeros_like(max_pool))

        pooled_segments = torch.cat([max_pool, mean_pool], dim=-1)  # (B, 2 * seg_hidden_dim) = 128
        global_emb = self.global_mlp(global_feats)  # (B, global_hidden_dim) = 64

        combined = torch.cat([global_emb, pooled_segments], dim=-1)  # (B, 192)
        return self.trunk(combined)  # (B, 128)

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.extract_features(obs)
        mean = self.action_mean(features)
        std = self.action_log_std.exp().expand_as(mean)
        value = self.value_head(features).squeeze(-1)
        return Normal(mean, std), value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value
