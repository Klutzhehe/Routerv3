"""Runs the PPO baseline's actual training loop against a fake bridge.

Exercises rollout collection, GAE, the clipped-surrogate update,
LineActorCritic, RunningMeanStd, and checkpointing end to end.
"""

import math
import tempfile
from pathlib import Path

from tests import fake_bridge

fake_bridge.install()

from pcbworld.agents.ppo_baseline import PPOConfig, train  # noqa: E402
from pcbworld.env.line_route_env import LineRouteEnv  # noqa: E402
from pcbworld.env.pcb_route_env import PCBRouteEnv  # noqa: E402


def test_legacy_training_loop_runs_and_produces_finite_losses():
    env = PCBRouteEnv("fake_board.kicad_pcb", max_steps_per_net=5)
    cfg = PPOConfig(
        total_timesteps=256,
        rollout_steps=64,
        epochs=2,
        minibatch_size=16,
        hidden_size=16,
    )

    policy = train(env, cfg)

    for param in policy.parameters():
        assert math.isfinite(param.detach().abs().sum().item()), "NaN/Inf in policy weights after training"


def test_line_route_env_training_loop_with_checkpointing():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LineRouteEnv("fake_board.kicad_pcb", max_steps_per_net=5)
        cfg = PPOConfig(
            total_timesteps=256,
            rollout_steps=64,
            epochs=2,
            minibatch_size=16,
            checkpoint_dir=tmpdir,
            checkpoint_interval=128,
        )

        policy = train(env, cfg)

        for param in policy.parameters():
            assert math.isfinite(param.detach().abs().sum().item()), "NaN/Inf in policy weights after training"

        # Verify checkpoints exist
        chk_latest = Path(tmpdir) / "policy_latest.pt"
        assert chk_latest.exists()

        stats_file = Path(tmpdir) / "training_stats.jsonl"
        assert stats_file.exists()


if __name__ == "__main__":
    test_legacy_training_loop_runs_and_produces_finite_losses()
    test_line_route_env_training_loop_with_checkpointing()
    print("OK")
