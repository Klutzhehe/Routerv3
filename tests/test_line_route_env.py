"""Exercises LineRouteEnv's control flow against tests/fake_bridge.py.

Python-level only -- the fake never rejects a push() or a fix(), so these
prove sequencing, reward bookkeeping and observation wiring, not that the
reward shape is right against real router behaviour. That is a Colab
question.

The reward tests are worth the effort anyway: potential-based shaping is easy
to write in a way that still trains but is quietly farmable (backtracking for
reward), and the a=0 convention silently depends on line_obs.py's frame
agreeing with the env's heading maths. Both are pinned here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests import fake_bridge

fake_bridge.install()

import pcbworld_pns_bridge as bridge  # noqa: E402

from pcbworld.env.line_obs import GLOBAL_INDEX, NUM_GLOBAL, LineObsConfig  # noqa: E402
from pcbworld.env.line_route_env import LineRouteEnv, RewardWeights  # noqa: E402

MM = 1_000_000

# Two nets, one clearly shorter, so shortest-first ordering is observable.
_NETS = [
    fake_bridge.NetPad("net_0", "J1:1", 0, 0, -1),
    fake_bridge.NetPad("net_0", "J2:1", 20 * MM, 0, -1),
    fake_bridge.NetPad("net_1", "J3:1", 0, 10 * MM, -1),
    fake_bridge.NetPad("net_1", "J4:1", 5 * MM, 10 * MM, -1),
]


def _make_env(nets=None, **kwargs) -> LineRouteEnv:
    # Swap the factory rather than the shared default fixture, same pattern
    # tests/test_diff_pair_route_env.py uses.
    bridge.PNSBridge = lambda: fake_bridge.FakePNSBridge(nets=nets or _NETS)
    kwargs.setdefault("obs_config", LineObsConfig(k_nearest=8, max_steps=40))
    kwargs.setdefault("max_steps_per_net", 40)
    board_path = kwargs.pop("board_path", "fake_board.kicad_pcb")
    return LineRouteEnv(board_path, **kwargs)



def _one_net(**kwargs):
    return _make_env(nets=_NETS[:2], **kwargs)


# -- spaces and reset ----------------------------------------------------


def test_action_space_is_one_dimensional():
    """The whole design rests on this: one turn angle, not (dx, dy)."""
    env = _one_net()
    assert env.action_space.shape == (1,)


def test_reset_returns_an_observation_in_the_declared_space():
    env = _one_net()
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert info["net"] == "net_0"
    assert info["num_nets"] == 1


def test_nets_default_to_shortest_first():
    """Net ordering is a deliberate heuristic, not a learned decision."""
    env = _make_env()
    env.reset()
    assert env._nets == ["net_1", "net_0"]  # 5mm span before 20mm


def test_explicit_net_order_overrides_the_heuristic():
    env = _make_env(net_order=["net_0", "net_1"])
    env.reset()
    assert env._nets == ["net_0", "net_1"]


# -- the action convention ----------------------------------------------


def test_zero_action_walks_straight_at_the_target():
    """a = 0 must reproduce the greedy straight-line router -- that is what
    puts an untrained mean-zero policy AT the 9/24 baseline rather than
    below it. If this drifts, training starts from noise."""
    env = _one_net()
    env.reset()
    env.step(np.array([0.0], dtype=np.float32))

    # net_0 runs along +x from the origin, so a straight step must move in x
    # and not in y.
    assert env._pos[0] == pytest.approx(env.step_size_nm, rel=1e-6)
    assert env._pos[1] == pytest.approx(0.0, abs=1.0)


def test_full_deflection_turns_ninety_degrees():
    env = _one_net()
    env.reset()
    env.step(np.array([1.0], dtype=np.float32))
    assert env._pos[0] == pytest.approx(0.0, abs=1.0)
    assert abs(env._pos[1]) == pytest.approx(env.step_size_nm, rel=1e-6)


def test_actions_outside_the_box_are_clipped_not_wrapped():
    """A turn of 3*pi/2 from an unclipped action would point the head
    backwards -- silently, and only for out-of-range samples."""
    env = _one_net()
    env.reset()
    env.step(np.array([5.0], dtype=np.float32))
    assert env._pos[0] == pytest.approx(0.0, abs=1.0)


def test_heading_is_relative_to_the_current_bearing_not_a_fixed_axis():
    """The turn is applied on top of the live bearing to the target, so the
    convention keeps holding after the head has wandered off the start line."""
    env = _one_net()
    env.reset()
    env.step(np.array([1.0], dtype=np.float32))   # off the axis
    before = env._pos
    env.step(np.array([0.0], dtype=np.float32))   # straight at the target again

    bearing = math.atan2(env._target_xy[1] - before[1], env._target_xy[0] - before[0])
    assert env._pos[0] == pytest.approx(before[0] + env.step_size_nm * math.cos(bearing), abs=2.0)
    assert env._pos[1] == pytest.approx(before[1] + env.step_size_nm * math.sin(bearing), abs=2.0)



# -- reward --------------------------------------------------------------


def test_moving_toward_the_target_beats_moving_away():
    env = _one_net()
    env.reset()
    _, toward, *_ = env.step(np.array([0.0], dtype=np.float32))

    env.reset()
    _, away, *_ = env.step(np.array([1.0], dtype=np.float32))
    assert toward > away


def test_shaping_is_potential_based_so_a_round_trip_cannot_be_farmed():
    """Ng et al.: the shaping terms telescope. Going out and coming back must
    net out to (roughly) the step costs alone -- otherwise a policy can spin
    in place for reward and never route anything."""
    env = _one_net(reward_weights=RewardWeights(step=0.0, collision=0.0))
    env.reset()
    total = 0.0
    for action in (1.0, -1.0):  # +90 then -90: out and back
        _, r, *_ = env.step(np.array([action], dtype=np.float32))
        total += r
    assert abs(total) < 0.2, f"round trip netted {total}, should telescope to ~0"


def test_completing_a_net_pays_the_completion_bonus():
    env = _one_net(step_size_nm=20 * MM)  # one hop lands inside the snap radius
    env.reset()
    _, reward, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert terminated
    assert info["completed"] == ["net_0"]
    assert reward > 5.0


def test_running_out_of_steps_fails_the_net_and_charges_for_it():
    env = _one_net(max_steps_per_net=3, step_size_nm=1, reward_weights=RewardWeights(step=0.0))
    env.reset()
    rewards = [env.step(np.array([0.0], dtype=np.float32))[1] for _ in range(3)]
    assert rewards[-1] < -1.0, rewards
    assert env._failed == ["net_0"]


def test_detour_penalty_scales_with_wasted_length():
    """A wandering route that still connects should score below a direct one."""
    direct = _one_net(step_size_nm=20 * MM)
    direct.reset()
    _, straight_reward, *_ = direct.step(np.array([0.0], dtype=np.float32))

    wandering = _one_net(step_size_nm=20 * MM)
    wandering.reset()
    wandering.step(np.array([1.0], dtype=np.float32))       # detour first
    _, _, terminated, _, _ = wandering.step(np.array([0.0], dtype=np.float32))
    total = 0.0
    steps = 0
    while not terminated and steps < 10:
        _, r, terminated, _, _ = wandering.step(np.array([0.0], dtype=np.float32))
        total += r
        steps += 1
    assert wandering._routed_len > direct._routed_len


# -- multi-net sequencing ------------------------------------------------


def test_episode_visits_every_net_then_terminates():
    env = _make_env(step_size_nm=1 * MM, max_steps_per_net=40)
    env.reset()
    terminated, guard = False, 0
    while not terminated and guard < 100:
        _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
        guard += 1

    assert terminated
    # Shortest-first, and both reachable at 1mm per step within 40 steps.
    assert env._completed == ["net_1", "net_0"], (env._completed, env._failed)


def test_a_step_larger_than_the_remaining_distance_overshoots_the_snap_zone():
    """Not a bug, but a real constraint on configuration: the head advances a
    FIXED distance, so if step_size_nm is much larger than snap_radius_nm the
    head can jump straight over the snap zone and orbit the pad forever.
    Keep snap_radius_nm >= step_size_nm / 2 and a straight-at-target step can
    never skip past it. Pinned because the symptom -- 'the agent never
    finishes nets' -- looks like a learning failure, not a config one."""
    # net_1 spans 5mm; a 20mm step lands 15mm beyond the pad.
    env = _make_env(nets=_NETS[2:], step_size_nm=20 * MM, max_steps_per_net=4)
    env.reset()
    terminated, guard = False, 0
    while not terminated and guard < 10:
        _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
        guard += 1
    assert env._failed == ["net_1"] and not env._completed


def test_a_failed_net_does_not_end_the_episode():
    """Congestion is only learnable if later nets still route after an
    earlier one fails."""
    env = _make_env(max_steps_per_net=2, step_size_nm=1)
    env.reset()
    terminated, guard = False, 0
    while not terminated and guard < 20:
        _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
        guard += 1
    assert env._failed == ["net_1", "net_0"], env._failed


# -- the caching that the timing data forced ------------------------------


def test_board_geometry_is_fetched_once_per_net_not_once_per_step():
    """get_board_geometry() is ~0.13ms against a ~0.03ms step. Fetching it
    per step would make the observation, not the router, the bottleneck."""
    env = _make_env(step_size_nm=20 * MM)
    inner = env.bridge
    calls = {"n": 0}
    original = inner.get_board_geometry

    def counted():
        calls["n"] += 1
        return original()

    inner.get_board_geometry = counted

    env.reset()
    after_reset = calls["n"]
    terminated, guard = False, 0
    while not terminated and guard < 20:
        _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
        guard += 1

    # One per net (two nets), plus reset's own. Certainly not one per step.
    assert calls["n"] <= after_reset + len(env._nets), calls["n"]


def test_drc_is_not_run_during_stepping_by_default():
    env = _one_net(step_size_nm=20 * MM)
    calls = {"n": 0}
    original = env.bridge.run_drc
    env.bridge.run_drc = lambda: (calls.__setitem__("n", calls["n"] + 1), original())[1]

    env.reset()
    env.step(np.array([0.0], dtype=np.float32))
    assert calls["n"] == 0


# -- observation wiring ---------------------------------------------------


def test_other_nets_pads_appear_as_obstacles_and_own_pads_do_not():
    env = _make_env(net_order=["net_0", "net_1"])
    env.reset()
    own = {"net_0"}
    kinds = {s.net for s in env._obstacles}
    assert "net_1" in kinds
    assert not (kinds & own), "the routed net's own pads must not be obstacles"


def test_unrouted_nets_appear_as_ghost_segments():
    from pcbworld.env.line_obs import KIND_GHOST

    env = _make_env(net_order=["net_0", "net_1"])
    env.reset()
    ghosts = [s for s in env._obstacles if s.kind == KIND_GHOST]
    assert [g.net for g in ghosts] == ["net_1"]


def test_observation_stays_finite_and_in_space_across_a_whole_episode():
    env = _make_env(step_size_nm=3 * MM)
    obs, _ = env.reset()
    terminated, guard = False, 0
    while not terminated and guard < 60:
        obs, reward, terminated, _, _ = env.step(
            np.array([np.sin(guard)], dtype=np.float32)
        )
        assert env.observation_space.contains(obs), f"step {guard}"
        assert np.isfinite(obs).all() and math.isfinite(reward)
        guard += 1


def test_globals_report_progress_within_the_observation():
    env = _one_net()
    obs, _ = env.reset()
    start_dist = obs[0]
    obs, *_ = env.step(np.array([0.0], dtype=np.float32))
    assert obs[0] < start_dist
    # By name: NUM_GLOBAL - 1 stopped being length_slack when the base-heading
    # pair was appended, and base_heading_sin is also 0.0 here, so the old
    # positional form kept passing while testing nothing.
    assert obs[GLOBAL_INDEX["length_slack"]] == 0.0  # unused until the tuning stage


def test_board_pool_support():
    boards = ["fake_board1.kicad_pcb", "fake_board2.kicad_pcb"]
    env = _make_env(board_path=boards)
    assert env.board_paths == boards
    obs, info = env.reset()
    assert env.observation_space.contains(obs)


def test_ripup_and_reroute_flow():
    # Two nets: net_1 routes first and commits. When net_0 routes, we inject a collision with net_1.
    env = _make_env(enable_ripup=True, max_ripups_per_episode=2, max_steps_per_net=3, step_size_nm=1)
    env.reset()
    # Route net_1 to completion
    env._completed.append("net_1")
    env._net_index = 0
    env._nets = ["net_0", "net_1"]
    env._begin_net()

    # Simulate collision with net_1
    env.bridge.head_collides = lambda: True
    env.bridge.get_head_obstacle = lambda: fake_bridge.HeadObstacle(found=True, net="net_1", kind="segment", x=0, y=0)


    # Take steps until timeout triggers rip-up
    for _ in range(3):
        env.step(np.array([0.0], dtype=np.float32))

    # net_1 should have been ripped up and re-queued
    assert env._ripup_count == 1
    assert "net_1" in env._ripups_performed
    assert "net_1" not in env._completed
    assert env._nets.count("net_1") >= 1


def test_diff_pair_support():
    diff_nets = [
        fake_bridge.NetPad("diffpair_0_P", "J1:1", 0, 0, -1),
        fake_bridge.NetPad("diffpair_0_P", "J2:1", 20 * fake_bridge.MM, 0, -1),
        fake_bridge.NetPad("diffpair_0_N", "J1:2", 0, 1 * fake_bridge.MM, -1),
        fake_bridge.NetPad("diffpair_0_N", "J2:2", 20 * fake_bridge.MM, 1 * fake_bridge.MM, -1),
    ]
    env = _make_env(nets=diff_nets, step_size_nm=20 * fake_bridge.MM)
    obs, info = env.reset()
    assert env._nets == ["diffpair_0_P"]
    assert env.bridge._mode == 2  # MODE_ROUTE_DIFF_PAIR
    # Take a step and finish
    obs, reward, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert "diffpair_0_P" in info["completed"]
    assert "diffpair_0_N" in info["completed"]


def test_length_matched_group_support():
    len_nets = [
        fake_bridge.NetPad("lengthgrp_0_0", "J1:1", 0, 0, -1),
        fake_bridge.NetPad("lengthgrp_0_0", "J2:1", 20 * fake_bridge.MM, 0, -1),
        fake_bridge.NetPad("lengthgrp_0_1", "J3:1", 0, 5 * fake_bridge.MM, -1),
        fake_bridge.NetPad("lengthgrp_0_1", "J4:1", 10 * fake_bridge.MM, 5 * fake_bridge.MM, -1),
    ]
    env = _make_env(nets=len_nets, step_size_nm=20 * fake_bridge.MM)
    obs, info = env.reset()
    assert env._nets == ["lengthgrp_0_0", "lengthgrp_0_1"]
    # Route member 0 (the reference)
    obs, reward, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert "lengthgrp_0_0" in info["completed"]
    assert "0" in env._length_group_refs

    # Member 1 should now observe length slack from the reference
    assert obs[7] > 0.0  # length_slack is reported in the 8th global feature







# -- the colliding regime ------------------------------------------------
#
# Everything below is about the branch step() takes when head_collides() is
# true. It is the branch that decides multi-net boards -- an open board never
# enters it, which is exactly why 2-3 net evaluation looked perfect while
# 4+ net training sat at ~62% -- and until the base-heading pair was added to
# the observation it turned on state the policy could not see.


# One 0.5 mm step expressed in the observation's length_scale (10 mm) units.
_ONE_STEP_IN_SCALE_UNITS = 500_000 / (10.0 * MM)


class _AlwaysCollidingBridge(fake_bridge.FakePNSBridge):
    def head_collides(self) -> bool:
        return bool(self._routing_active)


def _colliding_env(**kwargs) -> LineRouteEnv:
    bridge.PNSBridge = lambda: _AlwaysCollidingBridge(nets=_NETS[:2])
    kwargs.setdefault("obs_config", LineObsConfig(k_nearest=8, max_steps=40))
    kwargs.setdefault("max_steps_per_net", 40)
    return LineRouteEnv("fake_board.kicad_pcb", **kwargs)


def _base_heading(obs):
    return math.atan2(obs[GLOBAL_INDEX["base_heading_sin"]], obs[GLOBAL_INDEX["base_heading_cos"]])


def test_base_heading_is_zero_while_not_colliding():
    """Off the target bearing the env re-aims every step, so the base heading
    IS the bearing and the offset is 0 -- which keeps a=0 walking at the pad."""
    env = _one_net()
    obs, _ = env.reset()
    assert _base_heading(obs) == pytest.approx(0.0, abs=1e-6)
    for a in (0.0, 0.4, -0.7):
        obs, *_ = env.step(np.array([a], dtype=np.float32))
        assert _base_heading(obs) == pytest.approx(0.0, abs=1e-6)


def test_observation_exposes_the_frame_the_next_turn_uses():
    """The load-bearing test.

    While colliding, step() turns from its own previous heading rather than
    from the target bearing. If that heading is not in the observation the
    policy is steering blind, and no amount of network capacity recovers it.
    So: read the offset out of the observation, steer by exactly minus it, and
    the head must end up moving straight at the pad -- strictly better than the
    a=0 action that would be correct in the non-colliding frame.
    """
    from pcbworld.env.line_route_env import MAX_TURN_RAD

    def _probe(action_from_obs):
        env = _colliding_env()
        obs, _ = env.reset()
        obs, *_ = env.step(np.array([0.5], dtype=np.float32))  # veer off, now colliding
        before = obs[0]
        obs, *_ = env.step(np.array([action_from_obs(obs)], dtype=np.float32))
        return before - obs[0]  # distance closed, in length_scale units

    informed = _probe(lambda o: np.clip(-_base_heading(o) / MAX_TURN_RAD, -1.0, 1.0))
    naive = _probe(lambda o: 0.0)

    assert informed > naive, "the observation does not determine the turn that aims at the pad"
    # Aiming correctly closes a full step of distance; a=0 cannot, because it
    # keeps the 45-degree veer it inherited from _prev_heading.
    assert informed == pytest.approx(_ONE_STEP_IN_SCALE_UNITS, rel=1e-3)


class _FrozenCollidingBridge(_AlwaysCollidingBridge):
    """Collides AND refuses to move: the real jam, and the only case the cap
    is meant to cut. push() reports the head back at where it already was,
    which is what RM_MARK_OBSTACLES does when the move is into an obstacle."""

    def push(self, x: int, y: int, item_id: int = -1) -> bool:
        if self._current_pos is None:
            return False
        return super().push(*self._current_pos, item_id)


def _frozen_env(**kwargs) -> LineRouteEnv:
    bridge.PNSBridge = lambda: _FrozenCollidingBridge(nets=_NETS[:2])
    kwargs.setdefault("obs_config", LineObsConfig(k_nearest=8, max_steps=40))
    kwargs.setdefault("max_steps_per_net", 40)
    return LineRouteEnv("fake_board.kicad_pcb", **kwargs)


def test_a_frozen_net_is_abandoned_after_max_collision_steps():
    """Uncapped, a jammed net bled 0.5/step for its whole 120-step budget: -60,
    91% of the failed net's return and 5.8x what a clean success paid."""
    env = _frozen_env(reward_weights=RewardWeights(max_collision_steps=5))
    env.reset()

    total = 0.0
    for i in range(4):
        _, r, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        total += r
        assert info["collision_run"] == i + 1
        assert info["failed"] == [], f"gave up early at step {i + 1}"

    _, r, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    total += r
    assert info["failed"] == ["net_0"]
    assert info["net_index"] == 1, "should have moved on to the next net"
    assert total > -10.0, f"jam cost {total:.2f}; the cap is not bounding it"


def test_contour_following_is_not_treated_as_a_jam():
    """A head sliding along an obstacle collides continuously but keeps moving.

    Counting that as a jam would abandon the net in the middle of the detour --
    the one manoeuvre the cap exists to make affordable -- so the agent would
    never collect a completed detour to learn from. Only colliding AND frozen
    counts.
    """
    env = _colliding_env(reward_weights=RewardWeights(max_collision_steps=3))
    env.reset()
    for i in range(20):
        _, _, terminated, _, info = env.step(np.array([0.3], dtype=np.float32))
        if terminated:
            break
        assert info["collides"], "fixture should be colliding throughout"
        assert info["collision_run"] == 0, f"moving head counted as jammed at step {i + 1}"
        assert info["failed"] == []


def test_the_collision_run_resets_when_the_head_gets_free():
    """Consecutive, not cumulative: breaking free and re-jamming elsewhere is
    normal contour-following, not a jam."""

    class _Intermittent(_FrozenCollidingBridge):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.calls = 0

        def push(self, x: int, y: int, item_id: int = -1) -> bool:
            self.calls += 1
            if self.calls % 3 == 0:          # every third step it gets free
                return fake_bridge.FakePNSBridge.push(self, x, y, item_id)
            return super().push(x, y, item_id)

    bridge.PNSBridge = lambda: _Intermittent(nets=_NETS[:2])
    env = LineRouteEnv(
        "fake_board.kicad_pcb",
        obs_config=LineObsConfig(k_nearest=8, max_steps=40),
        max_steps_per_net=40,
        reward_weights=RewardWeights(max_collision_steps=3),
    )
    env.reset()

    for _ in range(12):
        _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
        if terminated:
            break
        assert info["collision_run"] <= 2
        assert info["failed"] == [], "a run that keeps breaking is not a jam"


def test_a_failed_net_still_costs_less_than_a_success_pays():
    """The reward-shape claim, as arithmetic rather than as a comment.

    `collision` is charged once, on the transition into contact: measured on
    Colab, head_collides() stays true on 86.2% of the steps after the first
    one, so a per-step charge spends nearly all its weight punishing a mistake
    that already happened -- and the net is lost at that first contact anyway,
    because fix() refuses a route that touched anything.
    """
    w = RewardWeights()
    budget = 120  # max_steps_per_net used by the curriculum
    worst = -w.collision - w.step * budget - w.net_failed
    assert abs(worst) < w.net_done, (
        f"the worst a failed net can cost is {worst:.1f} against a success worth "
        f"+{w.net_done:.1f}; failure must not dominate the gradient"
    )


def test_the_collision_charge_is_levied_once_not_per_step():
    env = _colliding_env()
    env.reset()
    charges = []
    for _ in range(6):
        _, r, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert info["collides"]
        charges.append(r)
    # first step pays the contact charge, the rest do not
    assert charges[0] < charges[1] - 1.0, (
        f"first contact {charges[0]:+.3f} should cost far more than the steps "
        f"after it ({charges[1]:+.3f}); the charge is still per-step"
    )


def test_failed_nets_record_why_they_failed():
    """Three different problems wear the same "37% did not route" number.

    Reaching the pad and having fix() refused is not a steering failure;
    running out of steps is. Until they are separated, "the agent cannot get
    around obstacles" is an assumption rather than a measurement -- and three
    training runs were spent on that assumption.
    """
    env = _one_net(max_steps_per_net=3, step_size_nm=1)
    env.reset()
    info = None
    for _ in range(5):
        _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
        if terminated:
            break
    assert info["failed"] == ["net_0"]
    assert info["failure_reasons"] == {"net_0": "out_of_steps"}


def test_a_jammed_net_is_recorded_separately_from_a_starved_one():
    env = _frozen_env(reward_weights=RewardWeights(max_collision_steps=3))
    env.reset()
    info = None
    for _ in range(4):
        _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["failure_reasons"] == {"net_0": "jammed"}


def test_failure_reasons_reset_between_episodes():
    env = _one_net(max_steps_per_net=3, step_size_nm=1)
    env.reset()
    for _ in range(5):
        if env.step(np.array([0.0], dtype=np.float32))[2]:
            break
    _, info = env.reset()
    assert info["failure_reasons"] == {}


class _RefusesToStart(fake_bridge.FakePNSBridge):
    """start_route() refuses, and nothing is live afterwards -- as PNS behaves."""

    def start_route(self, x, y, item_id, layer):
        super().start_route(x, y, item_id, layer)
        self._routing_active = False
        return False

    def push(self, x, y, item_id=-1):
        return super().push(x, y, item_id) if self._routing_active else False

    def fix(self, x, y, item_id=-1, force_finish=False, force_commit=False):
        if not self._routing_active:
            return False
        return super().fix(x, y, item_id, force_finish, force_commit)


def test_a_head_with_no_live_route_does_not_teleport_to_the_origin():
    """Regression for a bug that corrupted every never-started net.

    get_head_geometry() returns a ZEROED struct when no route is live. Reading
    end_x/end_y out of it without checking `active` moved the head to the board
    origin and billed _routed_len for the trip -- so a net whose start_route()
    was refused spent its whole budget reporting a position the router never
    had, with a detour_ratio and shaping potential computed from it. On stage-1
    boards that is a third of all nets, and it is indistinguishable downstream
    from a net that merely failed to navigate.
    """
    bridge.PNSBridge = lambda: _RefusesToStart(nets=_NETS[:2])
    env = LineRouteEnv(
        "fake_board.kicad_pcb",
        obs_config=LineObsConfig(k_nearest=8, max_steps=6),
        max_steps_per_net=6,
    )
    env.reset()
    start = env._pos

    for _ in range(3):
        _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert env._pos == start, f"head moved to {env._pos} with no live route"
        assert info["routed_length_nm"] == 0.0, "billed for travel that never happened"


def test_a_refused_start_is_not_filed_as_a_navigation_failure():
    bridge.PNSBridge = lambda: _RefusesToStart(nets=_NETS[:2])
    env = LineRouteEnv(
        "fake_board.kicad_pcb",
        obs_config=LineObsConfig(k_nearest=8, max_steps=4),
        max_steps_per_net=4,
    )
    env.reset()
    info = None
    for _ in range(6):
        _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
        if terminated:
            break
    assert info["failure_reasons"]["net_0"] == "route_never_started"
    assert info["failure_progress"]["net_0"] == pytest.approx(0.0)


class _RefusesFix(fake_bridge.FakePNSBridge):
    calls = 0

    def fix(self, x, y, item_id=-1, force_finish=False, force_commit=False):
        _RefusesFix.calls += 1
        return False


def test_a_refused_fix_ends_the_net_instead_of_burning_the_budget():
    """Regression for the bug that hid the real failure mode.

    _try_finish() calls stop_routing() when fix() refuses, so nothing can move
    or succeed afterwards -- but it left net_done False. The episode then spent
    the rest of the budget re-calling fix() on a dead route (101 times on a
    120-step net), appending the net to _failed once per step, and finally let
    _abandon() OVERWRITE the recorded reason with "out_of_steps".

    That is why the failure breakdown read 100% "never reached the pad" and 0%
    "fix() refused", while the per-net diagnostic showed those same heads
    closing 99% of the distance to the target. Both described the same nets.
    """
    _RefusesFix.calls = 0
    bridge.PNSBridge = lambda: _RefusesFix(nets=_NETS[:2])
    env = LineRouteEnv(
        "fake_board.kicad_pcb",
        obs_config=LineObsConfig(k_nearest=8, max_steps=120),
        max_steps_per_net=120,
    )
    env.reset()

    info, steps = None, 0
    for _ in range(200):
        _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
        steps += 1
        if terminated:
            break

    assert info["failure_reasons"]["net_0"] == "fix_rejected", "reason was overwritten"
    assert info["failed"] == ["net_0"], f"net recorded {len(info['failed'])} times"
    assert _RefusesFix.calls == 1, f"fix() retried on a dead route {_RefusesFix.calls} times"
    assert steps < 60, f"burned {steps} steps on a net that was already resolved"


def test_a_refused_fix_records_the_state_it_was_refused_in():
    """WHY fix() refuses is the open question, so the refusal carries context."""
    _RefusesFix.calls = 0
    bridge.PNSBridge = lambda: _RefusesFix(nets=_NETS[:2])
    env = LineRouteEnv(
        "fake_board.kicad_pcb",
        obs_config=LineObsConfig(k_nearest=8, max_steps=120),
        max_steps_per_net=120,
    )
    env.reset()
    info = None
    for _ in range(200):
        _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
        if terminated:
            break

    detail = info["fix_refusals"]["net_0"]
    assert detail["dist_nm"] <= env.snap_radius_nm
    assert detail["steps"] > 0
    assert "colliding_at_fix" in detail and "collision_steps" in detail
    assert detail["detour_ratio"] >= 1.0


def test_a_refusal_records_what_the_head_was_touching():
    """"Colliding at fix()" is not actionable on its own.

    Colliding with the net's OWN target pad means the router is treating this
    net's endpoint as an obstacle -- a bridge-integration bug with real
    headroom. Colliding with another net's copper means the route genuinely
    crosses something, which the oracle says is worth about one net in 124.
    The completion number cannot tell those apart; this can.
    """

    class _RefusesAndReportsOwnPad(fake_bridge.FakePNSBridge):
        def head_collides(self):
            return bool(self._routing_active)

        def get_head_obstacle(self):
            # the net's own target pad, which is where net_0's B pad sits
            return fake_bridge.HeadObstacle(
                found=True, net="net_0", kind="pad", x=20 * MM, y=0
            )

        def fix(self, x, y, item_id=-1, force_finish=False, force_commit=False):
            return False

    bridge.PNSBridge = lambda: _RefusesAndReportsOwnPad(nets=_NETS[:2])
    env = LineRouteEnv(
        "fake_board.kicad_pcb",
        obs_config=LineObsConfig(k_nearest=8, max_steps=120),
        max_steps_per_net=120,
    )
    env.reset()
    info = None
    for _ in range(200):
        _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
        if terminated:
            break

    detail = info["fix_refusals"]["net_0"]
    for key in ("obstacle_on_approach", "obstacle_at_fix"):
        o = detail[key]
        assert o["probe"] is True and o["found"] is True
        assert o["is_own_net"] is True, "own target pad must not read as another net"
        assert o["dist_to_target_nm"] == pytest.approx(0.0, abs=1.0)
    assert detail["collides_after_snap"] is True


def test_the_obstacle_probe_never_takes_the_episode_down():
    class _AngryProbe(fake_bridge.FakePNSBridge):
        def head_collides(self):
            return bool(self._routing_active)

        def get_head_obstacle(self):
            raise RuntimeError("bridge exploded")

        def fix(self, x, y, item_id=-1, force_finish=False, force_commit=False):
            return False

    bridge.PNSBridge = lambda: _AngryProbe(nets=_NETS[:2])
    env = LineRouteEnv(
        "fake_board.kicad_pcb",
        obs_config=LineObsConfig(k_nearest=8, max_steps=120),
        max_steps_per_net=120,
    )
    env.reset()
    info = None
    for _ in range(200):
        _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
        if terminated:
            break
    assert info["fix_refusals"]["net_0"]["obstacle_at_fix"] == {"probe": False}
