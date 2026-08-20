"""Control-flow coverage for measure_layer_hop_rescue.py.

Like tests/test_measure_waypoint_fidelity.py, this is Python control-flow
coverage only -- the double below has no notion of real PNS::ROUTER
geometry, layer stack-up, or via clearance. It exists to prove the script's
two-phase wiring (route everything same-layer, then try a layer hop on
whatever failed) does what it's supposed to, not to predict what the real
bridge will do with switch_layer()/toggle_via_placement() -- see that
script's module docstring on why those two calls are unverified API
surface and a null Colab result from them is inconclusive rather than
negative.
"""

from __future__ import annotations

import sys
import types
from collections import namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

MM = 1_000_000

Candidate = namedtuple("Candidate", ["id", "x", "y", "kind", "net"])
NetPad = namedtuple("NetPad", ["net", "pad_name", "x", "y", "layer"])
TrackSegment = namedtuple(
    "TrackSegment", ["x1", "y1", "x2", "y2", "width", "layer", "net", "is_arc"]
)
BoardGeometry = namedtuple(
    "BoardGeometry", ["tracks", "vias", "pads", "zones", "courtyards", "board_edge"]
)


class LayerHopBridge:
    """push() always succeeds, matching what three real Colab runs actually
    observed (72/72 push() calls accepted, only fix() ever rejected) --
    modeling push() as rejectable here would let a test "rescue" a net by
    accident at the wrong step and not notice.

    fix_reject(x, y) gates fix() while on layer 0. fix() also succeeds
    unconditionally once switched to exactly `rescue_layer` --
    toggle_via_placement() deliberately does nothing here, so a test can
    tell which of the two insertion methods the script under test actually
    used to rescue a net."""

    def __init__(self, nets, fix_reject, rescue_layer) -> None:
        self._nets = nets
        self._fix_reject = fix_reject
        self._rescue_layer = rescue_layer
        self._layer = 0
        self._committed: list[TrackSegment] = []
        self._active_net: str | None = None
        self._start_xy = None
        self._last_xy = None
        self._pending = False

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
        self._layer = 0  # every fresh route starts on the front layer
        self._pending = False
        return True

    def push(self, x, y, item_id=-1):
        self._last_xy = (x, y)
        return True

    def switch_layer(self, layer):
        self._layer = layer
        return True

    def toggle_via_placement(self):
        pass  # deliberately a no-op -- see class docstring

    def fix(self, x, y, item_id=-1, force_finish=False, force_commit=False):
        if self._layer == 0:
            if self._fix_reject(x, y):
                return False
        elif self._layer != self._rescue_layer:
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
                TrackSegment(x1, y1, x2, y2, 250_000, self._layer, self._active_net, False)
            )
        self._pending = False

    def stop_routing(self):
        pass

    def run_drc(self):
        return []

    def get_board_geometry(self):
        return BoardGeometry(list(self._committed), [], [], [], [], [])


def _install(nets, fix_reject, rescue_layer=31) -> LayerHopBridge:
    bridge = LayerHopBridge(nets, fix_reject, rescue_layer)
    module = types.ModuleType("pcbworld_pns_bridge")
    module.PNSBridge = lambda: bridge
    module.MODE_ROUTE_SINGLE = 1
    module.RM_MARK_OBSTACLES = 0
    sys.modules["pcbworld_pns_bridge"] = module
    return bridge


def _two_pad_net(name: str, start=(0, 0), target=(10 * MM, 0)):
    return [
        NetPad(name, "J1:1", start[0], start[1], -1),
        NetPad(name, "J2:1", target[0], target[1], -1),
    ]


def _clean():
    sys.modules.pop("pcbworld_pns_bridge", None)
    sys.modules.pop("measure_layer_hop_rescue", None)
    sys.modules.pop("measure_waypoint_fidelity", None)


def test_layer_hop_rescues_a_net_the_front_layer_cannot():
    """push() always succeeds (see LayerHopBridge's docstring on why); fix()
    rejects on layer 0 for every point, forcing phase 1 to fail this net via
    every one of its 6 same-layer attempts. Phase 2's switch_layer(31) then
    puts fix() on the one layer this double treats as reachable."""
    _clean()
    _install(_two_pad_net("net_0"), fix_reject=lambda x, y: True, rescue_layer=31)
    from measure_layer_hop_rescue import run

    rescued = run("board.kicad_pcb", num_nets=1, back_layer=31, bridge_dir=None)
    assert rescued == {"net_0": "switch_layer"}
    _clean()


def test_reports_inconclusive_not_a_crash_when_nothing_rescues():
    """Regression against a genuinely plausible outcome: the real API is
    unverified, so a real run could easily rescue nothing. The script must
    say so cleanly, not throw. Here: the script is told back_layer=99, but
    the double only ever accepts fix() on layer 31 -- so even after a
    successful switch_layer(99) call, fix() still rejects on the new layer,
    same as it would if switch_layer() were a no-op against the real router."""
    _clean()
    _install(_two_pad_net("net_0"), fix_reject=lambda x, y: True, rescue_layer=31)
    from measure_layer_hop_rescue import run

    rescued = run("board.kicad_pcb", num_nets=1, back_layer=99, bridge_dir=None)
    assert rescued == {}
    _clean()


def test_nets_already_succeeding_same_layer_are_left_alone():
    _clean()
    _install(_two_pad_net("net_0"), fix_reject=lambda x, y: False, rescue_layer=31)
    from measure_layer_hop_rescue import run

    rescued = run("board.kicad_pcb", num_nets=1, back_layer=31, bridge_dir=None)
    assert rescued == {}  # nothing needed rescuing -- phase 2 never had a candidate
    _clean()
