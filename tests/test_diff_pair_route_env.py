"""Exercises DiffPairRouteEnv's Python control flow against
tests/fake_bridge.py.

Does not validate real PNS::ROUTER behavior -- see fake_bridge.py's
docstring. Catches Python-level bugs only (leg sequencing, net-name
parsing, tune-leg target/length bookkeeping), not whether real
diff-pair/meander geometry comes out correctly (that's ROADMAP.md item 7's
Colab verification).
"""

from tests import fake_bridge

fake_bridge.install()

import pcbworld_pns_bridge as bridge  # noqa: E402

from pcbworld.env.diff_pair_route_env import DiffPairRouteEnv  # noqa: E402

MM = 1_000_000

# One plain net, one diff pair (P/N, pitch 0.5mm), and one length-matched
# group (reference 15mm, member routed at 20mm then tuned toward 15mm) --
# exercises all three leg kinds (direct, diff_pair, tune) in one board,
# matching generate_board.py's naming convention.
_MIXED_NETS = [
    fake_bridge.NetPad("net_0", "J1:1", 0, 0, -1),
    fake_bridge.NetPad("net_0", "J2:1", 20 * MM, 0, -1),
    fake_bridge.NetPad("diffpair_0_P", "J3:1", 0, 20 * MM, -1),
    fake_bridge.NetPad("diffpair_0_P", "J4:1", 20 * MM, 20 * MM, -1),
    fake_bridge.NetPad("diffpair_0_N", "J5:1", 0, int(20.5 * MM), -1),
    fake_bridge.NetPad("diffpair_0_N", "J6:1", 20 * MM, int(20.5 * MM), -1),
    fake_bridge.NetPad("lengthgrp_0_0", "J7:1", 0, 30 * MM, -1),
    fake_bridge.NetPad("lengthgrp_0_0", "J8:1", 15 * MM, 30 * MM, -1),
    fake_bridge.NetPad("lengthgrp_0_1", "J9:1", 0, 40 * MM, -1),
    fake_bridge.NetPad("lengthgrp_0_1", "J10:1", 20 * MM, 40 * MM, -1),
]


def _make_env(**kwargs) -> DiffPairRouteEnv:
    # PNSBridge() takes no constructor args (real bridge: nets come from
    # load_board()) -- swap the factory itself so this test's board has
    # diff-pair/length-group nets without touching the shared default
    # fixture other envs' tests rely on.
    bridge.PNSBridge = lambda: fake_bridge.FakePNSBridge(nets=_MIXED_NETS)
    return DiffPairRouteEnv("fake_mixed_board.kicad_pcb", **kwargs)


def test_reset_starts_with_the_plain_net_direct_leg():
    env = _make_env(max_steps_per_leg=5)
    obs, info = env.reset()

    assert env.observation_space.contains(obs)
    assert info["net"] == "net_0"
    assert info["kind"] == "direct"


def test_episode_visits_five_legs_in_order():
    # 1 plain net + 1 diff-pair leg + (1 direct + 1 direct + 1 tune) for
    # the 2-member length group = 5 legs total.
    env = _make_env(step_size_nm=20_000_000, max_steps_per_leg=5)
    env.reset()

    seen_kinds = []
    action = [1.0, 0.0, 1.0, -1.0]  # push straight toward target, always fix
    for _ in range(5):
        _obs, _reward, terminated, truncated, info = env.step(action)
        seen_kinds.append((info["net"], info["kind"]))
        if terminated or truncated:
            break

    assert seen_kinds == [
        ("net_0", "direct"),
        ("diffpair_0_P", "diff_pair"),
        ("lengthgrp_0_0", "direct"),
        ("lengthgrp_0_1", "direct"),
        ("lengthgrp_0_1", "tune"),
    ]
    assert terminated, "episode should terminate once every leg finishes"


def test_tune_leg_reports_length_mismatch():
    # lengthgrp_0_1 is routed straight at 20mm, then tuned toward
    # lengthgrp_0_0's 15mm. The fake bridge draws tune legs as a straight
    # line from the tune leg's start (the prior segment's midpoint) to the
    # target too, so it won't land exactly on 15mm -- what matters here is
    # that the env actually computes and reports a mismatch, not that the
    # fake's approximation is geometrically accurate (see fake_bridge.py).
    env = _make_env(step_size_nm=20_000_000, max_steps_per_leg=5)
    env.reset()

    action = [1.0, 0.0, 1.0, -1.0]
    info = {}
    for _ in range(5):
        _obs, _reward, terminated, truncated, info = env.step(action)
        if info.get("kind") == "tune" and "length_mismatch_nm" in info:
            break
        if terminated or truncated:
            break

    assert "length_mismatch_nm" in info, f"expected a tune leg to report length_mismatch_nm, got {info}"


def test_truncates_a_leg_after_max_steps_without_fix():
    env = _make_env(step_size_nm=1, max_steps_per_leg=3)
    env.reset()

    truncated = False
    for _ in range(3):
        _obs, _reward, _terminated, truncated, _info = env.step([0.0, 0.0, -1.0, -1.0])

    assert truncated, "leg should truncate after max_steps_per_leg without a fix"


if __name__ == "__main__":
    test_reset_starts_with_the_plain_net_direct_leg()
    test_episode_visits_five_legs_in_order()
    test_tune_leg_reports_length_mismatch()
    test_truncates_a_leg_after_max_steps_without_fix()
    print("OK")
