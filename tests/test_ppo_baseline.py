"""Runs the PPO baseline's actual training loop against a fake bridge.

Exercises rollout collection, GAE, and the clipped-surrogate update
end to end -- catches crashes/NaNs in the training loop itself. Does not
validate that the policy learns anything meaningful (the fake bridge's
push()/fix() always succeed, so there's no real routing signal to learn
from) -- only Colab, against the real router, can tell us that.
"""

import math

from tests import fake_bridge

fake_bridge.install()

from pcbworld.agents.ppo_baseline import PPOConfig, train  # noqa: E402
from pcbworld.env.pcb_route_env import PCBRouteEnv  # noqa: E402


def test_training_loop_runs_and_produces_finite_losses():
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


if __name__ == "__main__":
    test_training_loop_runs_and_produces_finite_losses()
    print("OK")
