"""Exercises PCBRouteEnv's Python control flow against tests/fake_bridge.py.

Does not validate real PNS::ROUTER behavior -- see fake_bridge.py's
docstring. Catches Python-level bugs only.
"""

from tests import fake_bridge

fake_bridge.install()

from pcbworld.env.pcb_route_env import PCBRouteEnv  # noqa: E402


def test_reset_returns_valid_observation():
    env = PCBRouteEnv("fake_board.kicad_pcb", max_steps_per_net=5)
    obs, info = env.reset()

    assert env.observation_space.contains(obs)
    assert info["net"] == "net_0"


def test_episode_terminates_when_all_nets_finished():
    env = PCBRouteEnv("fake_board.kicad_pcb", max_steps_per_net=5)
    obs, _info = env.reset()

    terminated = False
    for _ in range(10):
        action = env.action_space.sample()
        action[2] = 1.0  # always attempt fix -- fake bridge always succeeds
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        if terminated:
            break

    assert terminated, "episode should terminate once both fake nets finish"


def test_truncates_after_max_steps_without_fix():
    env = PCBRouteEnv("fake_board.kicad_pcb", max_steps_per_net=3)
    env.reset()

    truncated = False
    for _ in range(3):
        action = env.action_space.sample()
        action[2] = -1.0  # never attempt fix
        _obs, _reward, _terminated, truncated, _info = env.step(action)

    assert truncated, "net should be abandoned (truncated) after max_steps_per_net"


if __name__ == "__main__":
    test_reset_returns_valid_observation()
    test_episode_terminates_when_all_nets_finished()
    test_truncates_after_max_steps_without_fix()
    print("OK")
