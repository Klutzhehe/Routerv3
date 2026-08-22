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


def test_rms_update_accepts_a_single_unbatched_sample():
    """The trainer feeds one step at a time, as a bare (NUM_GLOBAL,) array.

    Regression for the bug that flattened the first curriculum run: np.mean(x,
    axis=0) on a 1-D array reduces across the FEATURES and returns a scalar, so
    every global converged on one shared mean and one shared std. The original
    test only ever passed a (100, NUM_GLOBAL) batch, which takes the correct
    path, so the trainer's actual call signature went unexercised.
    """
    rng = np.random.default_rng(0)
    # Deliberately mismatched per-feature scales -- feature i ~ N(i, 1).
    batch = np.stack(
        [rng.normal(loc=i, scale=1.0, size=4000) for i in range(NUM_GLOBAL)], axis=1
    ).astype(np.float32)

    rms = RunningMeanStd(shape=(NUM_GLOBAL,))
    for sample in batch:
        rms.update(sample)  # 1-D, exactly as collect_rollout() calls it

    assert np.allclose(rms.mean, batch.mean(axis=0), atol=0.1)
    assert np.allclose(np.sqrt(rms.var), batch.std(axis=0), atol=0.2)
    # The bug's signature: every feature collapsing onto one shared statistic.
    assert not np.allclose(rms.mean, rms.mean[0]), "per-feature means collapsed"


def test_rms_matches_whether_fed_batched_or_one_at_a_time():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(500, NUM_GLOBAL)).astype(np.float32) * [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][:NUM_GLOBAL]

    batched = RunningMeanStd(shape=(NUM_GLOBAL,))
    batched.update(data)

    streamed = RunningMeanStd(shape=(NUM_GLOBAL,))
    for row in data:
        streamed.update(row)

    assert np.allclose(batched.mean, streamed.mean, atol=1e-3)
    assert np.allclose(np.sqrt(batched.var), np.sqrt(streamed.var), atol=1e-2)


def test_rms_rejects_a_genuinely_wrong_shape():
    rms = RunningMeanStd(shape=(NUM_GLOBAL,))
    with pytest.raises(ValueError):
        rms.update(np.zeros((4, NUM_GLOBAL + 3), dtype=np.float32))


def test_rms_normalize_clips_a_constant_feature():
    """head_layer, target_layer and length_slack are identically 0 for a whole
    stage, so their running variance decays toward 0. Without the clip, any
    float wobble divided by that variance is an unbounded network input."""
    rms = RunningMeanStd(shape=(NUM_GLOBAL,))
    for _ in range(5000):
        rms.update(np.zeros(NUM_GLOBAL, dtype=np.float32))

    out = rms.normalize(np.full(NUM_GLOBAL, 1.0, dtype=np.float32))
    assert np.all(np.abs(out) <= 10.0 + 1e-5)

    out_t = rms.normalize(torch.full((NUM_GLOBAL,), 1.0))
    assert torch.all(out_t.abs() <= 10.0 + 1e-5)


def test_rms_refuses_stale_checkpoint_stats():
    """NUM_GLOBAL went 8 -> 10 with the base-heading pair, so every pre-fix
    checkpoint carries stats of the wrong length. They must be dropped, not
    assigned."""
    rms = RunningMeanStd(shape=(NUM_GLOBAL,))
    stale = {
        "rms_mean": np.arange(NUM_GLOBAL - 2, dtype=np.float32),
        "rms_var": np.ones(NUM_GLOBAL - 2, dtype=np.float32),
        "rms_count": 1234.0,
    }
    assert rms.load_from_checkpoint(stale) is False
    assert rms.mean.shape == (NUM_GLOBAL,)
    assert np.allclose(rms.mean, 0.0)

    fresh = {
        "rms_mean": np.arange(NUM_GLOBAL, dtype=np.float32),
        "rms_var": np.ones(NUM_GLOBAL, dtype=np.float32),
        "rms_count": 99.0,
    }
    assert rms.load_from_checkpoint(fresh) is True
    assert np.allclose(rms.mean, np.arange(NUM_GLOBAL))
