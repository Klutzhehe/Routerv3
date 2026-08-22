"""Exercises diagnose_layer_switch.py's trial harness and verdict engine.

The verdict engine is the part that actually has to be right. Per AGENTS.md
the Colab side reports output rather than diagnosing, so a wrong conclusion
printed confidently is worse than no conclusion at all -- it would send the
next session after the wrong hypothesis, which is exactly the failure mode
this script exists to end (two Colab rounds, one hypothesis each).

These are Python control-flow and decision-logic tests only. Whether the
REAL router accepts switch_layer() from a THT pad is precisely what the
script exists to find out and cannot be answered here.
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
ViaGeom = namedtuple(
    "ViaGeom", ["x", "y", "diameter", "drill", "layer_top", "layer_bottom", "net"]
)
BoardGeometry = namedtuple(
    "BoardGeometry", ["tracks", "vias", "pads", "zones", "courtyards", "board_edge"]
)
HeadGeometry = namedtuple(
    "HeadGeometry", ["active", "segments", "vias", "end_x", "end_y", "layer", "length"]
)
HeadVia = namedtuple("HeadVia", ["x", "y", "layer_top", "layer_bottom"])


class DiagnosticBridge:
    """Bridge double whose layer-change behavior is chosen by `rule`.

    Board pad type is inferred from the loaded path containing "tht" -- the
    real bridge has no such notion, but the script's H2 trial is defined by
    which BOARD FILE it loads, so the double needs some way to tell them
    apart.

    rules:
      "reject_all"      every switch_layer() returns False (the historical
                        0-for-32 behavior)
      "h2"              accepted iff the board is THT, or the target layer
                        is the one the head is already on (an SMD pad
                        trivially spans its own layer)
      "only_layer_2"    accepted iff target layer == 2 (models a wrong
                        PCB_LAYER_ID constant having caused the rejections)
      "h4"              switch_layer() always False, but
                        toggle_via_placement() flips the layer and its via
                        survives commit_routing()
      "lies"            returns False while actually changing the layer
    """

    def __init__(self, rule: str) -> None:
        self.rule = rule
        self.pad_type = "smd"
        self.layer = 0
        self.active = False
        self.head_vias: list[HeadVia] = []
        self.committed_vias: list[ViaGeom] = []
        self.via_diameter: int | None = None
        self.via_drill: int | None = None
        self.switch_calls: list[int] = []
        self.pos = (0, 0)
        # Ordered log of the calls whose SEQUENCE is load-bearing: the
        # unset-via sweep is only honest if nothing set a via size first,
        # and reset() cannot undo that (see the script's module docstring).
        self.call_log: list[str] = []

    # -- setup ----------------------------------------------------------
    def load_board(self, path):
        self.call_log.append("load_board")
        self.pad_type = "tht" if "tht" in path else "smd"
        # Mirrors the real LoadBoard(): a brand-new PNS::ROUTER, so via
        # sizes revert. Only this call clears them -- reset() does not.
        self.via_diameter = self.via_drill = None
        self.layer, self.active, self.head_vias = 0, False, []
        return True

    def reset(self):
        # Mirrors PNS_BRIDGE::Reset(): strips committed copper and drops any
        # in-progress route, but deliberately does NOT touch via sizes.
        self.call_log.append("reset")
        self.active, self.head_vias = False, []
        self.committed_vias = []

    def set_mode(self, mode):
        pass

    def set_collision_mode(self, mode):
        pass

    def set_track_width(self, w):
        pass

    def set_via_diameter(self, d):
        self.call_log.append("set_via_diameter")
        self.via_diameter = d

    def set_via_drill(self, d):
        self.via_drill = d

    # -- routing --------------------------------------------------------
    def net_pads(self):
        return [
            NetPad("net_0", "J1:1", 0, 0, -1),
            NetPad("net_0", "J2:1", 20 * MM, 0, -1),
        ]

    def query_hover_items(self, x, y, layer=-1, slop_radius=100000):
        return [Candidate(1, x, y, "pad", "net_0")]

    def start_route(self, x, y, item_id, layer):
        self.active, self.layer, self.pos = True, layer, (x, y)
        self.head_vias = []
        return True

    def push(self, x, y, item_id=-1):
        self.pos = (x, y)
        return True

    def switch_layer(self, layer):
        self.call_log.append(f"switch_layer({layer})")
        self.switch_calls.append(layer)
        if self.rule == "only_layer_2":
            accepted = layer == 2
        elif self.rule == "h2":
            accepted = self.pad_type == "tht" or layer == self.layer
        elif self.rule == "lies":
            self.layer = layer
            return False
        else:  # reject_all, h4
            accepted = False
        if accepted:
            self.layer = layer
        return accepted

    def toggle_via_placement(self):
        if self.rule == "h4" and self.active:
            self.head_vias.append(HeadVia(self.pos[0], self.pos[1], 0, 2))
            self.layer = 2 if self.layer == 0 else 0

    def fix(self, x, y, item_id=-1, force_finish=False, force_commit=False):
        return self.rule == "h4"

    def commit_routing(self):
        for hv in self.head_vias:
            self.committed_vias.append(
                ViaGeom(hv.x, hv.y, 600_000, 300_000, hv.layer_top, hv.layer_bottom, "net_0")
            )
        self.active, self.head_vias = False, []

    def stop_routing(self):
        self.active, self.head_vias = False, []

    def get_head_geometry(self):
        return HeadGeometry(
            active=self.active,
            segments=[],
            vias=list(self.head_vias),
            end_x=self.pos[0],
            end_y=self.pos[1],
            layer=self.layer,
            length=0.0,
        )

    def get_board_geometry(self):
        return BoardGeometry([], list(self.committed_vias), [], [], [], [])


def _install(rule: str) -> DiagnosticBridge:
    bridge = DiagnosticBridge(rule)
    module = types.ModuleType("pcbworld_pns_bridge")
    module.PNSBridge = lambda: bridge
    module.MODE_ROUTE_SINGLE = 1
    module.RM_MARK_OBSTACLES = 0
    sys.modules["pcbworld_pns_bridge"] = module
    return bridge


@pytest.fixture(autouse=True)
def _clean_bridge_module():
    # Restore the ORIGINAL module OBJECT, not merely a working module.
    # tests/test_diff_pair_route_env.py binds this module to a global at
    # import time (`import pcbworld_pns_bridge as bridge`) and later mutates
    # that object's PNSBridge to install its mixed diff-pair/length-group
    # board. Leaving a DIFFERENT object in sys.modules orphans that
    # reference: its env's deferred import resolves to the replacement and
    # silently gets fake_bridge's default 2-plain-net fixture instead, so a
    # tune leg never appears and its test fails somewhere unrelated to this
    # file. This file sorts alphabetically before that one, so it is the
    # first to be able to cause it -- found exactly that way, not in theory.
    saved = sys.modules.get("pcbworld_pns_bridge")
    sys.modules.pop("pcbworld_pns_bridge", None)
    yield
    if saved is not None:
        sys.modules["pcbworld_pns_bridge"] = saved
    else:
        fake_bridge.install()


def _named(results):
    return {r.name: r for r in results}


# -- the trial harness ---------------------------------------------------


def test_layer_id_sweep_finds_the_real_constant_and_later_trials_adopt_it():
    """A wrong PCB_LAYER_ID is indistinguishable from a structural refusal --
    both are just False. KiCad 9 renumbered PCB_LAYER_ID and this process
    cannot import pcbnew to ask (hard constraint 1), so the sweep has to
    close that gap itself rather than trusting a default."""
    from scripts.diagnose_layer_switch import run

    _install("only_layer_2")
    results = _named(run("smd.kicad_pcb", tht_board=None, bridge_dir=None))

    assert results["sweep_set_layer_2"].accepted
    assert not results["sweep_set_layer_1"].accepted
    # Post-sweep trials must have adopted 2, which is also LAYER_ID_SWEEP[0]
    # here -- vias_040_020 accepting proves the adopted value reached them.
    assert results["vias_040_020"].accepted, "later trials did not adopt the accepted layer id"


def test_the_unset_via_sweep_runs_before_anything_sets_a_via_size():
    """The H1 test only means something if 'via sizes were never configured'
    is literally true when that sweep runs. reset() does not undo a via-size
    setting -- only a fresh load_board() does -- so this is guaranteed by
    trial ORDER alone, which makes it worth pinning down."""
    from scripts.diagnose_layer_switch import LAYER_ID_SWEEP, run

    bridge = _install("reject_all")
    run("smd.kicad_pcb", tht_board=None, bridge_dir=None)

    first_set = bridge.call_log.index("set_via_diameter")
    unset_sweep_switches = [
        i
        for i, call in enumerate(bridge.call_log)
        if call.startswith("switch_layer") and i < first_set
    ]
    assert len(unset_sweep_switches) == len(LAYER_ID_SWEEP), (
        "the whole unset sweep must complete before any set_via_diameter() call; "
        f"got {len(unset_sweep_switches)} of {len(LAYER_ID_SWEEP)}"
    )


def test_only_one_load_board_per_distinct_board():
    """load_board() twice on one PNS_BRIDGE segfaulted a whole Colab run --
    a use-after-free in LoadBoard's teardown order, since fixed in
    pns_bridge.cpp. Trials are separated by reset() instead, so the fixed
    path is exercised once per board rather than once per trial."""
    from scripts.diagnose_layer_switch import run

    bridge = _install("reject_all")
    run("smd.kicad_pcb", tht_board="tht.kicad_pcb", bridge_dir=None)

    assert bridge.call_log.count("load_board") == 2, bridge.call_log.count("load_board")
    assert bridge.call_log.count("reset") > 2, "trials should be separated by reset()"


def test_the_tht_trial_runs_last_so_the_second_load_board_risks_least():
    from scripts.diagnose_layer_switch import run

    _install("reject_all")
    results = run("smd.kicad_pcb", tht_board="tht.kicad_pcb", bridge_dir=None)
    assert results[-1].name == "tht_start"


def test_a_trial_that_raises_is_recorded_not_fatal():
    """A crashed trial must not cost the rest of the Colab session."""
    from scripts.diagnose_layer_switch import run

    bridge = _install("reject_all")
    original = bridge.switch_layer
    calls = {"n": 0}

    def exploding(layer):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return original(layer)

    bridge.switch_layer = exploding
    results = _named(run("smd.kicad_pcb", tht_board=None, bridge_dir=None))

    assert any(r.error and "boom" in r.error for r in results.values())
    assert results["toggle_via_placement"] is not None, "run stopped early after the exception"


def test_readback_disagreement_is_visible_when_the_return_value_lies():
    """A False return with a changed layer is itself a finding -- a bare
    bool cannot show it, which is why every trial snapshots the head."""
    from scripts.diagnose_layer_switch import run

    _install("lies")
    results = _named(run("smd.kicad_pcb", tht_board=None, bridge_dir=None))

    trial = results["sweep_set_layer_2"]
    assert trial.accepted is False
    assert trial.layer_changed, "layer readback should disagree with the return value"


# -- the verdict engine --------------------------------------------------


def test_verdict_confirms_h2_when_tht_succeeds_and_smd_fails():
    from scripts.diagnose_layer_switch import run, _verdict

    _install("h2")
    results = _named(run("smd.kicad_pcb", tht_board="tht.kicad_pcb", bridge_dir=None))
    lines = _verdict(results, back_layer=2, tht_tested=True)

    # Match on "H2 ... CONFIRMED" specifically: the same-layer control line
    # also mentions H2 (by name, as a hypothesis it leaves alive), so a bare
    # "first line containing H2" match would test the wrong line.
    confirmed = [line for line in lines if "H2" in line and "CONFIRMED" in line]
    assert confirmed, lines
    assert "--pad-type tht" in confirmed[0] or "toggle_via_placement" in confirmed[0]


def test_verdict_kills_h1_when_the_two_sweeps_are_identical():
    """The H1 test is the pairwise diff between the unset-via sweep and the
    configured-via sweep, at every candidate layer id -- not one guessed id."""
    from scripts.diagnose_layer_switch import run, _verdict

    _install("reject_all")
    results = _named(run("smd.kicad_pcb", tht_board=None, bridge_dir=None))
    lines = _verdict(results, back_layer=2, tht_tested=False)

    assert any("H1 (via size) DEAD" in line for line in lines), lines


def test_verdict_reports_structural_refusal_when_the_same_layer_noop_is_rejected():
    """The control that knocks out H1 and H2 together: switching to the layer
    the head is already on needs no via and trivially satisfies any
    layer-span requirement, so a rejection there is refusal before geometry."""
    from scripts.diagnose_layer_switch import run, _verdict

    _install("reject_all")
    results = _named(run("smd.kicad_pcb", tht_board=None, bridge_dir=None))
    lines = _verdict(results, back_layer=2, tht_tested=False)

    control = [line for line in lines if line.startswith("CONTROL")]
    assert control and "REJECTED" in control[0]
    assert "knocks out H1 and H2" in control[0]


def test_verdict_confirms_h4_when_toggle_via_commits_real_copper():
    from scripts.diagnose_layer_switch import run, _verdict

    _install("h4")
    results = _named(run("smd.kicad_pcb", tht_board=None, bridge_dir=None))
    lines = _verdict(results, back_layer=2, tht_tested=False)

    h4 = [line for line in lines if "H4 CONFIRMED" in line]
    assert h4, lines
    assert results["toggle_via_placement"].committed_vias == 1


def test_verdict_says_h2_untested_and_recommends_single_layer_when_nothing_works():
    """The run has to say plainly that its leading hypothesis went untested,
    rather than reporting a confident dead end built on eight other trials."""
    from scripts.diagnose_layer_switch import run, _verdict

    _install("reject_all")
    results = _named(run("smd.kicad_pcb", tht_board=None, bridge_dir=None))
    lines = _verdict(results, back_layer=2, tht_tested=False)

    assert any("H2 UNTESTED" in line for line in lines), lines
    assert any("NO PRIMITIVE WORKED" in line for line in lines), lines


def test_verdict_does_not_declare_a_dead_end_when_something_did_work():
    from scripts.diagnose_layer_switch import run, _verdict

    _install("h4")
    results = _named(run("smd.kicad_pcb", tht_board=None, bridge_dir=None))
    lines = _verdict(results, back_layer=2, tht_tested=False)

    assert not any("NO PRIMITIVE WORKED" in line for line in lines), lines
