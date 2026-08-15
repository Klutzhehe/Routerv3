"""Exercises SimpleRouteEnv's Python control flow against tests/fake_bridge.py.

Does not validate real PNS::ROUTER behavior -- see fake_bridge.py's
docstring. Catches Python-level bugs only.
"""

from tests import fake_bridge

fake_bridge.install()

from pcbworld.env.simple_route_env import SimpleRouteEnv  # noqa: E402


def test_reset_returns_valid_observation():
    env = SimpleRouteEnv("fake_board.kicad_pcb", max_steps=5)
    obs, info = env.reset()

    assert env.observation_space.contains(obs)
    assert info["net"] == "net_0"


def test_episode_terminates_once_within_snap_radius():
    # Fake bridge's push()/fix() always succeed, so a single step sized to
    # exactly close the 20mm gap between net_0's pads (fake_bridge.py)
    # should land within SNAP_RADIUS_NM and finish.
    env = SimpleRouteEnv("fake_board.kicad_pcb", step_size_nm=20_000_000, max_steps=5)
    env.reset()

    action = [1.0, 0.0]  # net_0 runs pad-to-pad along +x (see fake_bridge.py)
    _obs, reward, terminated, _truncated, _info = env.step(action)

    assert terminated
    assert isinstance(reward, float)


def test_truncates_after_max_steps_without_reaching_target():
    env = SimpleRouteEnv("fake_board.kicad_pcb", step_size_nm=1, max_steps=3)
    env.reset()

    truncated = False
    for _ in range(3):
        _obs, _reward, _terminated, truncated, _info = env.step([0.0, 0.0])

    assert truncated, "episode should truncate after max_steps without reaching the target"


if __name__ == "__main__":
    test_reset_returns_valid_observation()
    test_episode_terminates_once_within_snap_radius()
    test_truncates_after_max_steps_without_reaching_target()
    print("OK")
