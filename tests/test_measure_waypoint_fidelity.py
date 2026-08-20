"""Exercises measure_waypoint_fidelity.py's retry/detour control flow.

tests/fake_bridge.py's FakePNSBridge never rejects a push(), so running the
script against it (as the module docstring's own smoke test does) only
proves "direct" ever fires -- the polyline and detour retry branches, which
are the actual interesting logic in this script, are untouched. This file
adds a bridge double that rejects on purpose so those branches run.

Like tests/fake_bridge.py, this is Python control-flow coverage only, not a
router model: it has no notion of real geometry or clearance, it just
decides accept/reject from a scripted rule. Whether push() *in reality*
rejects that often, and whether a perpendicular detour *in reality* rescues
it, can only be answered by the real bridge in Colab -- that's what this
script is for.
"""

from __future__ import annotations

import sys
import types
from collections import namedtuple

import pytest

MM = 1_000_000

Candidate = namedtuple("Candidate", ["id", "x", "y", "kind", "net"])
NetPad = namedtuple("NetPad", ["net", "pad_name", "x", "y", "layer"])
DRCViolation = namedtuple("DRCViolation", ["error_code", "message", "severity", "x", "y"])
TrackSegment = namedtuple(
    "TrackSegment", ["x1", "y1", "x2", "y2", "width", "layer", "net", "is_arc"]
)
BoardGeometry = namedtuple(
    "BoardGeometry", ["tracks", "vias", "pads", "zones", "courtyards", "board_edge"]
)


class ScriptedBridge:
    """A push() is rejected iff `reject(x, y)` says so. Otherwise behaves
    like tests/fake_bridge.py's FakePNSBridge: trivially accepts, and
    commit_routing() records a single straight start->last_pushed segment
    (good enough here -- this test checks *which strategy the script picks
    and how far it gets*, not committed-track shape)."""

    def __init__(self, nets, reject, fix_reject=None) -> None:
        self._nets = nets
        self._reject = reject
        self._fix_reject = fix_reject or (lambda x, y: False)
        self._committed: list[TrackSegment] = []
        self._active_net: str | None = None
        self._start_xy = None
        self._last_xy = None
        self._pending = False
        self.fix_calls: list[tuple] = []  # (x, y, item_id, force_finish, force_commit)

    def load_board(self, path):
        return True

    def reset(self):
        self._committed = []

    def set_mode(self, mode):
        pass

    def set_collision_mode(self, mode):
        pass

    def set_track_width(self, w):
        pass

    def net_pads(self):
        return list(self._nets)

    def query_hover_items(self, x, y, layer=-1, slop_radius=100000):
        for p in self._nets:
            if abs(p.x - x) <= slop_radius and abs(p.y - y) <= slop_radius:
                return [Candidate(0, x, y, "pad", p.net)]
        return [Candidate(0, x, y, "pad", "")]

    def start_route(self, x, y, item_id, layer):
        self._active_net = next((p.net for p in self._nets if p.x == x and p.y == y), None)
        self._start_xy = (x, y)
        self._last_xy = (x, y)
        self._pending = False
        return True

    def push(self, x, y, item_id=-1):
        if self._reject(x, y):
            return False
        self._last_xy = (x, y)
        return True

    def fix(self, x, y, item_id=-1, force_finish=False, force_commit=False):
        self.fix_calls.append((x, y, item_id, force_finish, force_commit))
        if self._fix_reject(x, y):
            return False
        self._last_xy = (x, y)
        self._pending = True
        return True

    def commit_routing(self):
        if self._pending and self._active_net and self._start_xy:
            x1, y1 = self._start_xy
            x2, y2 = self._last_xy
            self._committed = [t for t in self._committed if t.net != self._active_net]
            self._committed.append(
                TrackSegment(x1, y1, x2, y2, 250_000, 0, self._active_net, False)
            )
        self._pending = False

    def stop_routing(self):
        pass

    def run_drc(self):
        return []

    def get_board_geometry(self):
        return BoardGeometry(list(self._committed), [], [], [], [], [])


def _install(nets, reject, fix_reject=None) -> ScriptedBridge:
    """Installs a fake pcbworld_pns_bridge module and returns the single
    ScriptedBridge instance it will hand back from PNSBridge() -- run()
    only ever constructs one, so tests that need to inspect call history
    (e.g. fix_calls) can hold onto this."""
    bridge = ScriptedBridge(nets, reject, fix_reject)
    module = types.ModuleType("pcbworld_pns_bridge")
    module.PNSBridge = lambda: bridge
    module.MODE_ROUTE_SINGLE = 1
    module.RM_MARK_OBSTACLES = 0
    sys.modules["pcbworld_pns_bridge"] = module
    return bridge


@pytest.fixture(autouse=True)
def _clean_bridge_module():
    sys.modules.pop("pcbworld_pns_bridge", None)
    yield
    sys.modules.pop("pcbworld_pns_bridge", None)


def _two_pad_net(name: str, start=(0, 0), target=(10 * MM, 0)):
    return [
        NetPad(name, "J1:1", start[0], start[1], -1),
        NetPad(name, "J2:1", target[0], target[1], -1),
    ]


class _FakeCandidate:
    def __init__(self, kind, id):
        self.kind, self.id = kind, id


def test_pick_pad_candidate_prefers_pad_kind_regardless_of_order():
    """Regression: a Colab run against a 24-net board showed 16/24 nets
    failing at fix() despite push() reaching the target every time. Real
    query_hover_items() doesn't sort its hits, so on a board with many
    already-committed tracks, candidates[0] can be an unrelated track
    instead of the pad -- fix()ing against its id then correctly fails for
    a reason that has nothing to do with the net actually being routed."""
    from scripts.measure_waypoint_fidelity import _pick_pad_candidate

    warnings: list[str] = []
    candidates = [_FakeCandidate("segment", 5), _FakeCandidate("pad", 7), _FakeCandidate("segment", 9)]
    picked = _pick_pad_candidate(candidates, "net_x/target", warnings)
    assert picked.id == 7
    assert warnings == []


def test_pick_pad_candidate_falls_back_and_warns_when_no_pad_present():
    from scripts.measure_waypoint_fidelity import _pick_pad_candidate

    warnings: list[str] = []
    candidates = [_FakeCandidate("segment", 3), _FakeCandidate("via", 4)]
    picked = _pick_pad_candidate(candidates, "net_x/target", warnings)
    assert picked.id == 3  # no pad available -- falls back to candidates[0]
    assert len(warnings) == 1
    assert "net_x/target" in warnings[0]
    assert "segment" in warnings[0] and "via" in warnings[0]


def test_retries_polyline_when_direct_push_succeeds_but_direct_fix_fails():
    """Regression for the structural gap found from real Colab data: three
    Colab runs showed push() accepting 72/72 attempts while fix() rejected
    ~67% of them, even with force_finish/force_commit=True. An earlier
    version of this retry loop only escalated past 'direct' when push()
    itself rejected -- since push() apparently never rejects a single big
    hop, 'reached=True' was set right after push() succeeded and fix()'s
    rejection was never retried, so polyline/detour had literally never run
    against the real bridge across all three runs. push() always succeeds
    here; fix() rejects on its first call only, forcing the script to prove
    it now retries a fix() rejection by moving to 'polyline', not just a
    push() rejection."""
    from scripts.measure_waypoint_fidelity import run

    calls = {"n": 0}

    def fix_reject_first_only(x, y):
        calls["n"] += 1
        return calls["n"] == 1

    _install(_two_pad_net("net_0"), reject=lambda x, y: False, fix_reject=fix_reject_first_only)
    results = run("board.kicad_pcb", num_nets=1, bridge_dir=None)
    assert results[0].strategy == "polyline", results[0].strategy
    assert results[0].reached_target
    assert calls["n"] == 2, "expected exactly 2 fix() calls: direct (rejected), polyline (accepted)"


def test_fix_is_called_with_force_finish_and_force_commit():
    """Regression: an earlier version of the script called fix(x, y,
    item_id, False, False), matching simple_route_env.py's convention. A
    Colab run against the real bridge showed that every net where push()
    reached the target still failed at this exact call -- 17/24 nets, every
    one with accepted == requested waypoints, meaning push() succeeded and
    only fix() rejected. pcb_route_env.py and diff_pair_route_env.py had
    already fixed this (commit 7f746b6, 'use force_finish=True,
    force_commit=True in fix() to snap to target pad'); this script had
    just copied the older, unfixed convention."""
    from scripts.measure_waypoint_fidelity import run

    bridge = _install(_two_pad_net("net_0"), reject=lambda x, y: False)
    run("board.kicad_pcb", num_nets=1, bridge_dir=None)
    assert bridge.fix_calls, "fix() was never called"
    for x, y, item_id, force_finish, force_commit in bridge.fix_calls:
        assert force_finish is True, f"fix() called with force_finish={force_finish}"
        assert force_commit is True, f"fix() called with force_commit={force_commit}"


def test_direct_push_succeeds_when_nothing_rejects():
    from scripts.measure_waypoint_fidelity import run

    _install(_two_pad_net("net_0"), reject=lambda x, y: False)
    results = run("board.kicad_pcb", num_nets=1, bridge_dir=None)
    assert results[0].strategy == "direct"
    assert results[0].reached_target


def test_polyline_rescues_a_rejected_direct_push():
    """Reject exactly the first push() call (the direct one-shot to the
    target) and accept every push after -- forces the
    direct-fails -> polyline-succeeds path, and nothing else."""
    from scripts.measure_waypoint_fidelity import run

    calls = {"n": 0}

    def reject_first_only(x, y):
        calls["n"] += 1
        return calls["n"] == 1

    _install(_two_pad_net("net_0"), reject=reject_first_only)
    results = run("board.kicad_pcb", num_nets=1, bridge_dir=None)
    assert results[0].strategy == "polyline"
    assert results[0].reached_target
    assert results[0].waypoints_accepted == results[0].waypoints_requested


def test_detour_rescues_a_net_the_polyline_cannot_reach():
    """Reject the direct attempt's push and the polyline attempt's first
    push, accept everything after -- forces the script through
    direct -> polyline -> detour and proves the detour branch is reachable
    at all, not just present in the code.

    Only 2 calls need rejecting, not every polyline waypoint:
    try_push_sequence (see the module under test) stops at the FIRST
    rejection rather than attempting the whole list, so the polyline
    attempt only ever issues one push() call before falling through to
    detour. This test's threshold is coupled to that short-circuit
    behavior on purpose -- the assertions below would catch it drifting.
    """
    from scripts.measure_waypoint_fidelity import run

    calls = {"n": 0}
    REJECT_THROUGH_CALL = 2  # call 1: direct's push(target). call 2: polyline's first push.

    def reject_first_few(x, y):
        calls["n"] += 1
        return calls["n"] <= REJECT_THROUGH_CALL

    _install(_two_pad_net("net_0"), reject=reject_first_few)
    results = run("board.kicad_pcb", num_nets=1, bridge_dir=None)
    assert results[0].strategy.startswith("detour"), results[0].strategy
    assert results[0].reached_target
    assert calls["n"] > REJECT_THROUGH_CALL, (
        "the detour branch never issued a push() past the rejection threshold -- "
        "either it didn't run, or try_push_sequence's short-circuit behavior changed"
    )


def test_unreachable_net_is_reported_as_failed_not_crashed():
    from scripts.measure_waypoint_fidelity import run

    _install(_two_pad_net("net_0"), reject=lambda x, y: True)
    results = run("board.kicad_pcb", num_nets=1, bridge_dir=None)
    assert results[0].strategy == "failed"
    assert not results[0].reached_target
    assert results[0].max_deviation_nm is None


def test_only_plain_nets_are_selected_and_num_nets_caps_the_count():
    from scripts.measure_waypoint_fidelity import run

    nets = _two_pad_net("net_0") + _two_pad_net("net_1", start=(0, 20 * MM), target=(10 * MM, 20 * MM))
    nets += [NetPad("diffpair_0_P", "J5:1", 0, 40 * MM, -1),
             NetPad("diffpair_0_P", "J6:1", 10 * MM, 40 * MM, -1)]
    _install(nets, reject=lambda x, y: False)
    results = run("board.kicad_pcb", num_nets=1, bridge_dir=None)
    assert len(results) == 1
    assert results[0].net == "net_0"  # diffpair_* excluded, net_1 capped out


def test_save_image_writes_a_nonempty_png(tmp_path):
    """--save-image is optional and reads the SAME bridge instance's live
    get_board_geometry()/net_pads() -- ScriptedBridge's fixture (via
    fake_bridge-style commit_routing()) is enough to prove the wiring
    works; pcbworld.viz.render_board's own rendering correctness is
    tests/test_render_board.py's job, not this file's."""
    from scripts.measure_waypoint_fidelity import run

    _install(_two_pad_net("net_0"), reject=lambda x, y: False)
    out = tmp_path / "board.png"
    run("board.kicad_pcb", num_nets=1, bridge_dir=None, save_image=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_image_omitted_by_default_does_not_touch_matplotlib():
    """matplotlib is an optional dependency (see requirements.txt) -- a
    caller who never passes --save-image should never pay for the import,
    and this project has run entirely without matplotlib installed for
    most of its history."""
    import sys

    from scripts.measure_waypoint_fidelity import run

    had_matplotlib = "matplotlib" in sys.modules
    _install(_two_pad_net("net_0"), reject=lambda x, y: False)
    run("board.kicad_pcb", num_nets=1, bridge_dir=None)
    if not had_matplotlib:
        assert "matplotlib" not in sys.modules
