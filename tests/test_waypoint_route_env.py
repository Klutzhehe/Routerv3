import pytest
import numpy as np

from tests.fake_bridge import (
    install,
    FakePNSBridge,
    NetPad,
    MM,
)

install()

from pcbworld.env.waypoint_route_env import (
    WaypointRouteEnv,
    read_head_state,
    HeadState,
    ACT_WAYPOINT,
    ACT_FIX,
    ACT_TUNE,
    ACT_RIPUP,
    DEFAULT_TOLERANCE_NM,
)


def _make_mixed_fixture():
    return [
        NetPad("net_0", "J1:1", 0, 0, -1),
        NetPad("net_0", "J2:1", 10 * MM, 0, -1),
        NetPad("diffpair_0_P", "J3:1", 0, 5 * MM, -1),
        NetPad("diffpair_0_P", "J4:1", 10 * MM, 5 * MM, -1),
        NetPad("diffpair_0_N", "J5:1", 0, 6 * MM, -1),
        NetPad("diffpair_0_N", "J6:1", 10 * MM, 6 * MM, -1),
        NetPad("lengthgrp_0_0", "J7:1", 0, 15 * MM, -1),
        NetPad("lengthgrp_0_0", "J8:1", 10 * MM, 15 * MM, -1),
        NetPad("lengthgrp_0_1", "J9:1", 0, 20 * MM, -1),
        NetPad("lengthgrp_0_1", "J10:1", 10 * MM, 20 * MM, -1),
    ]


def test_fake_bridge_new_calls(monkeypatch):
    install()
    import pcbworld_pns_bridge as bridge

    bridge_inst = bridge.PNSBridge(nets=_make_mixed_fixture())
    bridge_inst.load_board("fake.kicad_pcb")

    # 1. get_design_rules
    rules = bridge_inst.get_design_rules()
    assert rules.track_width > 0
    assert rules.via_diameter > 0
    assert rules.clearance > 0

    # 2. get_head_geometry & head_collides when idle
    idle_head = bridge_inst.get_head_geometry()
    assert not idle_head.active
    assert not bridge_inst.head_collides()

    # 3. start_route and push
    pads = bridge_inst.net_pads()
    p0 = pads[0]
    cands = bridge_inst.query_hover_items(p0.x, p0.y, layer=0)
    assert bridge_inst.start_route(p0.x, p0.y, cands[0].id, 0)

    active_head = bridge_inst.get_head_geometry()
    assert active_head.active
    assert active_head.end_x == p0.x
    assert active_head.end_y == p0.y

    # Push a waypoint
    bridge_inst.push(5 * MM, 2 * MM)
    pushed_head = bridge_inst.get_head_geometry()
    assert pushed_head.active
    assert pushed_head.end_x == 5 * MM
    assert pushed_head.end_y == 2 * MM
    assert pushed_head.length > 0
    assert len(pushed_head.segments) == 1

    # 4. fix and commit
    bridge_inst.fix(10 * MM, 0, force_finish=True, force_commit=True)
    bridge_inst.commit_routing()

    geom = bridge_inst.get_board_geometry()
    assert len(geom.tracks) >= 1

    # 5. rip_up
    removed = bridge_inst.rip_up("net_0")
    assert removed >= 1
    geom_after = bridge_inst.get_board_geometry()
    assert len([t for t in geom_after.tracks if t.net == "net_0"]) == 0


def test_head_state_adapter(monkeypatch):
    install()
    import pcbworld_pns_bridge as bridge

    bridge_inst = bridge.PNSBridge()
    bridge_inst.load_board("fake.kicad_pcb")

    state = read_head_state(bridge_inst)
    assert isinstance(state, HeadState)
    assert not state.active
    assert state.end_x == 0
    assert not state.collides


def test_waypoint_route_env_lifecycle(monkeypatch):
    install()
    import pcbworld_pns_bridge as bridge

    monkeypatch.setattr(
        bridge, "PNSBridge", lambda: FakePNSBridge(nets=_make_mixed_fixture())
    )

    env = WaypointRouteEnv("fake.kicad_pcb", grid_resolution=32, max_waypoints_per_leg=10)
    obs, info = env.reset()

    assert obs.shape == (12,)
    assert info["kind"] == "direct"
    assert info["net"] == "net_0"

    # Check action masks
    masks = env.action_masks()
    assert masks["action_type"][ACT_WAYPOINT]
    assert masks["action_type"][ACT_FIX]
    assert masks["action_type"][ACT_RIPUP]
    assert not masks["action_type"][ACT_TUNE]  # Tune is only valid on tune legs

    # Step a waypoint
    gx, gy = env.nm_to_grid(5 * MM, 0)
    obs, reward, terminated, truncated, step_info = env.step([ACT_WAYPOINT, gx, gy, 0])
    assert obs.shape == (12,)
    assert not terminated
    assert "deviation_nm" in step_info

    # Layer switch waypoint
    obs, reward, terminated, truncated, step_info = env.step([ACT_WAYPOINT, gx, gy, 1])
    assert not terminated

    # Rip up current leg
    obs, reward, terminated, truncated, step_info = env.step([ACT_RIPUP, 0, 0, 0])
    assert "ripped_up_count" in step_info

    # Finish net_0
    obs, reward, terminated, truncated, step_info = env.step([ACT_FIX, 0, 0, 0])
    assert not terminated  # More legs remain
    assert env._current_leg.kind == "diff_pair"

    env.close()


def test_waypoint_route_env_length_tune_tolerance_band(monkeypatch):
    install()
    import pcbworld_pns_bridge as bridge

    monkeypatch.setattr(
        bridge, "PNSBridge", lambda: FakePNSBridge(nets=_make_mixed_fixture())
    )

    # Test with tolerance band covering the mismatch
    env = WaypointRouteEnv(
        "fake.kicad_pcb",
        grid_resolution=32,
        tolerance_nm=10 * MM,  # Wide tolerance band
    )
    env.reset()

    # Fast forward through net_0, diffpair_0_P, lengthgrp_0_0 (reference), lengthgrp_0_1 (baseline)
    env.step([ACT_FIX, 0, 0, 0])  # net_0
    env.step([ACT_FIX, 0, 0, 0])  # diffpair_0_P
    env.step([ACT_FIX, 0, 0, 0])  # lengthgrp_0_0
    env.step([ACT_FIX, 0, 0, 0])  # lengthgrp_0_1 baseline

    # Now at Leg 5: lengthgrp_0_1 (tune leg)
    assert env._current_leg.kind == "tune"
    masks = env.action_masks()
    assert masks["action_type"][ACT_TUNE]
    assert not masks["action_type"][ACT_WAYPOINT]

    # Execute 1-step tune action (anchor=0, amp_idx=2, spacing_idx=1)
    obs, reward, terminated, truncated, info = env.step([ACT_TUNE, 0, 2, 1])
    assert terminated  # All legs finished
    assert "within_tolerance" in info
    assert info["within_tolerance"]
    assert "length_mismatch_nm" in info
    assert reward > 0  # Earned net finish bonus since within tolerance band
