"""Unit tests for the new AI PCB Autorouter Platform (10-channel grid + PCBRouterNet).
"""

import pytest
import numpy as np
import torch

from pcbworld.board_generator import generate_random_board
from pcbworld.environment import PCBRouterEnv
from models.pcb_encoder import PCBEncoder
from models.router_policy import PCBRouterNet
from training.replay_buffer import RolloutBuffer
from training.evaluation import evaluate_policy


def test_board_generator():
    board = generate_random_board(grid_size=256, num_nets=3, num_obstacles=2, seed=42)
    assert len(board.nets) == 3
    assert len(board.obstacles) == 2
    assert board.copper_grid.shape == (2, 256, 256)


def test_environment_reset_and_step():
    env = PCBRouterEnv(grid_size=256, num_nets=1, num_obstacles=1, seed=42)
    obs, info = env.reset()

    # Observation shape verification (10, 256, 256)
    assert obs.shape == (10, 256, 256)
    assert obs.dtype == np.float32

    # Step verification (action 0..95)
    action = 14  # Some valid action
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert next_obs.shape == (10, 256, 256)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_pcb_router_net():
    model = PCBRouterNet(in_channels=10, action_dim=96, d_model=128, num_transformer_layers=2, num_heads=4)
    dummy_input = torch.randn(2, 10, 256, 256)

    dist, value = model(dummy_input)
    assert dist.logits.shape == (2, 96)
    assert value.shape == (2, 1)

    action, log_prob, entropy, val = model.get_action_and_value(dummy_input)
    assert action.shape == (2,)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    assert val.shape == (2, 1)


def test_rollout_buffer():
    device = torch.device("cpu")
    buffer = RolloutBuffer(buffer_size=16, obs_shape=(10, 256, 256), device=device)

    for i in range(16):
        obs = torch.randn(10, 256, 256)
        act = torch.tensor(i % 96)
        log_prob = torch.tensor(-1.5)
        reward = 1.0
        done = (i == 15)
        val = torch.tensor([0.5])
        buffer.add(obs, act, log_prob, reward, done, val)

    buffer.compute_advantages_and_returns(last_value=torch.tensor([0.0]), done=True)
    batches = list(buffer.get_minibatches(batch_size=8))
    assert len(batches) == 2
