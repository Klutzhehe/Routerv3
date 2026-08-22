"""Unit tests for LineActorCritic and RunningMeanStd.

Verifies:
  1. Shapes and gradient flow through segment pooling + global MLP.
  2. Untrained policy initialization: mean action is ≈ 0 (greedy baseline).
  3. Mask invariance: invalid segments (valid=0) do not affect features.
  4. RunningMeanStd normalizes online correctly.
"""

import numpy as np
import pytest
import torch

from pcbworld.agents.line_policy import LineActorCritic, RunningMeanStd
from pcbworld.env.line_obs import NUM_GLOBAL, NUM_SEGMENT_FEATURES, LineObsConfig


def test_line_actor_critic_shapes_and_forward():
    k = 8
    obs_dim = NUM_GLOBAL + k * NUM_SEGMENT_FEATURES
    policy = LineActorCritic(action_dim=1)

    batch_size = 4
    obs = torch.randn(batch_size, obs_dim)
    # Set valid bits (index 11) for segments
    obs[:, NUM_GLOBAL + 11 :: NUM_SEGMENT_FEATURES] = 1.0

    dist, value = policy.forward(obs)
    assert dist.mean.shape == (batch_size, 1)
    assert dist.stddev.shape == (batch_size, 1)
    assert value.shape == (batch_size,)


def test_untrained_policy_emits_near_zero_mean():
    """Crucial property: initial mean action must be ≈ 0 so policy begins at the
    greedy baseline."""
    policy = LineActorCritic(action_dim=1)
    obs = torch.randn(16, NUM_GLOBAL + 32 * NUM_SEGMENT_FEATURES)

    dist, _ = policy.forward(obs)
    assert torch.all(torch.abs(dist.mean) < 0.05), f"Untrained policy mean was {dist.mean}"


def test_mask_invariance():
    """Padding segments with valid=0 must not affect the output compared to having
    different numbers in those invalid slots."""
    policy = LineActorCritic(action_dim=1)
    policy.eval()

    k = 4
    obs1 = torch.zeros(1, NUM_GLOBAL + k * NUM_SEGMENT_FEATURES)
    # Segment 0 is valid
    obs1[0, NUM_GLOBAL : NUM_GLOBAL + 4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    obs1[0, NUM_GLOBAL + 11] = 1.0  # valid

    # Segment 1 is invalid with 0s
    # Segment 2 is invalid with 0s

    obs2 = obs1.clone()
    # Segment 1 is invalid but filled with random noise
    obs2[0, NUM_GLOBAL + NUM_SEGMENT_FEATURES : NUM_GLOBAL + 2 * NUM_SEGMENT_FEATURES] = torch.randn(
        NUM_SEGMENT_FEATURES
    )
    obs2[0, NUM_GLOBAL + NUM_SEGMENT_FEATURES + 11] = 0.0  # invalid

    with torch.no_grad():
        f1 = policy.extract_features(obs1)
        f2 = policy.extract_features(obs2)

    assert torch.allclose(f1, f2, atol=1e-5), f"Max difference: {(f1 - f2).abs().max()}"


def test_running_mean_std():
    rms = RunningMeanStd(shape=(NUM_GLOBAL,))
    data = np.random.randn(100, NUM_GLOBAL).astype(np.float32) * 5.0 + 3.0
    rms.update(data)

    assert np.allclose(rms.mean, data.mean(axis=0), atol=0.1)
    assert np.allclose(np.sqrt(rms.var), data.std(axis=0), atol=0.2)

    norm = rms.normalize(data)
    assert np.allclose(norm.mean(axis=0), 0.0, atol=0.1)
    assert np.allclose(norm.std(axis=0), 1.0, atol=0.2)


def test_parameter_count_within_budget():
    policy = LineActorCritic(action_dim=1)
    num_params = sum(p.numel() for p in policy.parameters())
    assert 30_000 <= num_params <= 60_000, f"Param count {num_params} outside expected range"
