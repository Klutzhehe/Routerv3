"""Exercises diagnose_via_hop.py's protocol sweep and verdict engine.

The verdict is what the Colab side reports back, so a confidently wrong
conclusion is worse than none -- the previous round's verdict overclaimed
("this is the layer-change primitive") on evidence that only showed a via
being committed, never a route continuing on the far layer. These tests pin
the distinction that overclaim blurred: head state looking right is NOT the
same as copper existing on two layers.

Control-flow tests only. Whether the real PNS placer survives a
non-finishing fix() is exactly what the script exists to find out.
"""

from __future__ import annotations

import sys
import types
from collections import namedtuple

import pytest

from tests import fake_bridge

MM = 1_000_000

Candidate = namedtuple("Candidate", ["id", "x", "y", "kind", "net"])
NetPad = namedtuple("NetPad", ["net", "pad_name", "x", "y", "layer"])
TrackSegment = namedtuple(
    "TrackSegment", ["x1", "y1", "x2", "y2", "width", "layer", "net", "is_arc"]
)
ViaGeom = namedtuple(
    "ViaGeom", ["x", "y", "diameter", "drill", "layer_top", "layer_bottom", "net"]
)
BoardGeometry = namedtuple(
    "BoardGeometry", ["tracks", "vias", "pads", "zones", "courtyards", "board_edge"]
)
HeadGeometry = namedtuple(
    "HeadGeometry", ["active", "segments", "vias", "end_x", "end_y", "layer", "length"]
)

BACK_LAYER = 2


class HopBridge:
    """Bridge double parameterised by how fix() behaves mid-route.

    rules:
      "hop_on_nonfinishing"  a fix() with force_finish=False keeps the placer
                             alive and flips the layer if a via is pending --
                             the script's hypothesis
      "never_hops"           fix() always ends the route (the behaviour the
                             previous round actually observed)
      "head_lies"            the head reports a flipped layer and the route
                             stays active, but NO copper is ever committed on
                             the far layer -- the false positive these tests
                             exist to catch
    """

    def __init__(self, rule: str) -> None:
        self.rule = rule
        self.layer = 0
        self.active = False
        self.via_pending = False
        self.pos = (0, 0)
        self.tracks: list[TrackSegment] = []
        self.vias: list[ViaGeom] = []
        self._pending_tracks: list[TrackSegment] = []
        self._pending_vias: list[ViaGeom] = []

    def load_board(self, path):
        self.layer, self.active, self.via_pending = 0, False, False
        self.tracks, self.vias = [], []
        self._pending_tracks, self._pending_vias = [], []
        return True

    def set_mode(self, m):
        pass

    def set_collision_mode(self, m):
        pass

    def set_track_width(self, w):
        pass

    def set_via_diameter(self, d):
        pass

    def set_via_drill(self, d):
        pass

    def net_pads(self):
        return [
            NetPad("net_0", "J1:1", 0, 0, -1),
            NetPad("net_0", "J2:1", 20 * MM, 0, -1),
        ]

    def query_hover_items(self, x, y, layer=-1, slop_radius=100000):
        return [Candidate(1, x, y, "pad", "net_0")]

    def start_route(self, x, y, item_id, layer):
        self.active, self.layer, self.pos = True, layer, (x, y)
        self.via_pending = False
        return True

    def push(self, x, y, item_id=-1):
        if not self.active:
            return False
        self._pending_tracks.append(
            TrackSegment(self.pos[0], self.pos[1], x, y, 250_000, self.layer, "net_0", False)
        )
        self.pos = (x, y)
        return True

    def toggle_via_placement(self):
        # Matches the observed real behaviour: nothing appears in the head,
        # the via is only latent until a fix() materialises it.
        self.via_pending = True

    def fix(self, x, y, item_id=-1, force_finish=False, force_commit=False):
        if not self.active:
            return False
        self._pending_tracks.append(
            TrackSegment(self.pos[0], self.pos[1], x, y, 250_000, self.layer, "net_0", False)
        )
        self.pos = (x, y)

        keeps_placer = self.rule in ("hop_on_nonfinishing", "head_lies") and not force_finish
        if self.via_pending and keeps_placer:
            self._pending_vias.append(ViaGeom(x, y, 600_000, 300_000, 0, BACK_LAYER, "net_0"))
            self.layer = BACK_LAYER if self.layer == 0 else 0
            self.via_pending = False
        if not keeps_placer:
            self.active = False
        return True

    def commit_routing(self):
        if self.rule == "head_lies":
            # Everything lands on the front layer regardless of what the head
            # claimed -- head state right, copper wrong.
            self.tracks.extend(
                t._replace(layer=0) for t in self._pending_tracks
            )
        else:
            self.tracks.extend(self._pending_tracks)
            self.vias.extend(self._pending_vias)
        self._pending_tracks, self._pending_vias = [], []
        self.active = False

    def stop_routing(self):
        self.active = False
        self._pending_tracks, self._pending_vias = [], []

    def get_head_geometry(self):
        return HeadGeometry(self.active, [], [], self.pos[0], self.pos[1], self.layer, 0.0)

    def get_board_geometry(self):
        return BoardGeometry(list(self.tracks), list(self.vias), [], [], [], [])


def _install(rule: str) -> HopBridge:
    bridge = HopBridge(rule)
    module = types.ModuleType("pcbworld_pns_bridge")
    module.PNSBridge = lambda: bridge
    module.MODE_ROUTE_SINGLE = 1
    module.RM_MARK_OBSTACLES = 0
    sys.modules["pcbworld_pns_bridge"] = module
    return bridge


@pytest.fixture(autouse=True)
def _clean_bridge_module():
    # Restore the original module OBJECT -- see the same fixture in
    # tests/test_diagnose_layer_switch.py for why identity matters here.
    saved = sys.modules.get("pcbworld_pns_bridge")
    sys.modules.pop("pcbworld_pns_bridge", None)
    yield
    if saved is not None:
        sys.modules["pcbworld_pns_bridge"] = saved
    else:
        fake_bridge.install()


def _named(results):
    return {r.name: r for r in results}


def test_a_nonfinishing_fix_that_keeps_the_placer_alive_is_detected_as_a_hop():
    from scripts.diagnose_via_hop import run

    _install("hop_on_nonfinishing")
    results = _named(run("smd1.kicad_pcb", bridge_dir=None))

    hop = results["fix_ff0_fc0_toggle,push"]
    assert hop.hopped, hop.notes
    assert hop.layer_before == 0 and hop.layer_after == BACK_LAYER


def test_the_force_finish_control_reproduces_the_previous_rounds_dead_end():
    """force_finish=True is the call the previous round made; it must still
    show the placer dying, or the double is not modelling what was seen."""
    from scripts.diagnose_via_hop import run

    _install("hop_on_nonfinishing")
    results = _named(run("smd1.kicad_pcb", bridge_dir=None))

    control = results["fix_ff1_fc1_toggle,push"]
    assert control.active_after is False
    assert not control.hopped


def test_full_two_hop_commits_vias_and_tracks_on_both_layers():
    from scripts.diagnose_via_hop import run, _verdict

    _install("hop_on_nonfinishing")
    results = run("smd1.kicad_pcb", bridge_dir=None)
    full = _named(results)["full_two_hop"]

    assert full.committed_vias == 2, full.notes
    assert sorted(full.tracks_per_layer) == [0, BACK_LAYER], full.tracks_per_layer
    assert any("LAYER HOPPING CONFIRMED ON DISK" in line for line in _verdict(results))


def test_a_lying_head_does_not_produce_a_confirmed_verdict():
    """The exact overclaim the previous round made: head state looks like a
    hop, but committed copper is single-layer. Must NOT confirm."""
    from scripts.diagnose_via_hop import run, _verdict

    _install("head_lies")
    results = run("smd1.kicad_pcb", bridge_dir=None)
    lines = _verdict(results)

    assert not any(line.startswith("***") for line in lines), lines
    assert any("PARTIAL" in line for line in lines), lines
    assert any("stay single-layer" in line for line in lines), lines


def test_no_hop_anywhere_recommends_staying_single_layer():
    from scripts.diagnose_via_hop import run, _verdict

    _install("never_hops")
    results = run("smd1.kicad_pcb", bridge_dir=None)
    lines = _verdict(results)

    assert any("NO MID-ROUTE HOP" in line for line in lines), lines
    assert any("stay single-layer" in line for line in lines), lines


def test_a_trial_that_raises_is_recorded_not_fatal():
    from scripts.diagnose_via_hop import run

    bridge = _install("hop_on_nonfinishing")
    original = bridge.toggle_via_placement
    calls = {"n": 0}

    def exploding():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return original()

    bridge.toggle_via_placement = exploding
    results = _named(run("smd1.kicad_pcb", bridge_dir=None))

    assert any(r.error and "boom" in r.error for r in results.values())
    assert "full_two_hop" in results, "run stopped early after the exception"
