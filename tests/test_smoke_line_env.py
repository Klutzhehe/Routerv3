"""Exercises smoke_line_env.py's episode loop and verdict against the fake.

The verdict is what the Colab side reports back, so it has to be right about
the one thing it exists to judge: whether greedy (a=0) reproduces the
independently measured straight-line baseline. A script that prints HEALTHY
on a broken env would send the next session into PPO against geometry that
cannot teach anything.
"""

from __future__ import annotations

import numpy as np

from tests import fake_bridge

fake_bridge.install()

import pcbworld_pns_bridge as bridge  # noqa: E402

MM = 1_000_000

_NETS = [
    fake_bridge.NetPad("net_0", "J1:1", 0, 0, -1),
    fake_bridge.NetPad("net_0", "J2:1", 5 * MM, 0, -1),
    fake_bridge.NetPad("net_1", "J3:1", 0, 10 * MM, -1),
    fake_bridge.NetPad("net_1", "J4:1", 5 * MM, 10 * MM, -1),
]


def _install_nets(nets=None):
    bridge.PNSBridge = lambda: fake_bridge.FakePNSBridge(nets=nets or _NETS)


def test_run_reports_both_policies_and_finishes(capsys, monkeypatch):
    import scripts.smoke_line_env as smoke

    _install_nets()
    monkeypatch.setattr(smoke, "_load_bridge", lambda d: bridge)
    results = smoke.run("fake.kicad_pcb", num_nets=2, bridge_dir=None, seed=0)

    assert set(results) == {"greedy", "random"}
    # The fake never rejects, so greedy routes everything it attempts.
    assert len(results["greedy"]["completed"]) == 2
    assert results["greedy"]["terminated"]
    assert results["greedy"]["step_times"], "per-step timing was not recorded"


def test_verdict_calls_a_zero_completion_greedy_run_broken(capsys, monkeypatch):
    """a=0 is meant to BE the straight-line router. Routing nothing is
    structural, and the script must say so rather than shrug."""
    import scripts.smoke_line_env as smoke

    _install_nets()
    monkeypatch.setattr(smoke, "_load_bridge", lambda d: bridge)

    real_episode = smoke._run_episode

    def barren(env, policy, rng):
        result = real_episode(env, policy, rng)
        result["completed"] = []
        return result

    monkeypatch.setattr(smoke, "_run_episode", barren)
    smoke.run("fake.kicad_pcb", num_nets=2, bridge_dir=None, seed=0)

    out = capsys.readouterr().out
    assert "BROKEN" in out
    assert "HEALTHY" not in out


def test_verdict_warns_when_random_matches_greedy(capsys, monkeypatch):
    """If the action does not change outcomes there is no gradient to follow,
    which is worth saying before a GPU is booked."""
    import scripts.smoke_line_env as smoke

    _install_nets()
    monkeypatch.setattr(smoke, "_load_bridge", lambda d: bridge)
    smoke.run("fake.kicad_pcb", num_nets=2, bridge_dir=None, seed=0)

    out = capsys.readouterr().out
    # The fake accepts everything, so random ties greedy -- exactly the case
    # the warning exists for.
    assert "WARNING" in out and "no worse than greedy" in out


def test_episode_asserts_on_a_non_finite_observation(monkeypatch):
    import scripts.smoke_line_env as smoke
    from pcbworld.env.line_route_env import LineRouteEnv

    _install_nets()
    env = LineRouteEnv("fake.kicad_pcb", max_nets=1, step_size_nm=1 * MM)
    original = env._observe
    monkeypatch.setattr(
        env, "_observe", lambda: np.full_like(original(), np.nan)
    )
    try:
        smoke._run_episode(env, "greedy", np.random.default_rng(0))
    except AssertionError as exc:
        assert "observation" in str(exc)
    else:
        raise AssertionError("a NaN observation should have been caught")
