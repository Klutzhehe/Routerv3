"""A fake pcbworld_pns_bridge module for testing pcbworld.env/agents locally.

pcbworld_pns_bridge only exists after the Colab build (see ROADMAP.md) --
nothing that imports it can run on a normal dev machine. This fake
implements just enough of the bound API (bindings.cpp) to exercise
PCBRouteEnv's and ppo_baseline's Python control flow (rollout collection,
GAE, the PPO update, env step/reset bookkeeping) locally. It does *not*
model real push-and-shove routing -- every push()/fix() call trivially
succeeds and DRC always reports one fixed "violation", so passing tests
here only mean "the Python glue doesn't crash and produces finite
numbers", not "the routing/reward logic is correct against a real board".
That can only be checked in Colab.

Call install() before importing anything that does
`import pcbworld_pns_bridge`.
"""

import sys
import types
from collections import namedtuple

Candidate = namedtuple("Candidate", ["id", "x", "y", "kind", "net"])
NetPad = namedtuple("NetPad", ["net", "pad_name", "x", "y", "layer"])
DRCViolation = namedtuple("DRCViolation", ["error_code", "message", "severity", "x", "y"])

MM = 1_000_000


class FakePNSBridge:
    def __init__(self, nets=None):
        # Default fixture: two 2-terminal nets, matching
        # pcbworld/data/generate_board.py's 2-pad-per-net design.
        self._nets = nets or [
            NetPad("net_0", "J1:1", 0, 0, -1),
            NetPad("net_0", "J2:1", 20 * MM, 0, -1),
            NetPad("net_1", "J3:1", 0, 10 * MM, -1),
            NetPad("net_1", "J4:1", 20 * MM, 10 * MM, -1),
        ]
        self.loaded = False

    def load_board(self, path):
        self.loaded = True
        return True

    def reset(self):
        pass

    def set_mode(self, mode):
        pass

    def set_track_width(self, w):
        pass

    def set_via_diameter(self, d):
        pass

    def set_via_drill(self, d):
        pass

    def net_pads(self):
        return list(self._nets)

    def query_hover_items(self, x, y, layer=-1, slop_radius=100000):
        return [Candidate(0, x, y, "pad", "")]

    def start_route(self, x, y, item_id, layer):
        return True

    def push(self, x, y, item_id=-1):
        return True

    def fix(self, x, y, item_id=-1, force_finish=False, force_commit=False):
        return True

    def commit_routing(self):
        pass

    def stop_routing(self):
        pass

    def toggle_via_placement(self):
        pass

    def run_drc(self):
        return [DRCViolation(1, "clearance too small", "error", 0, 0)]


def install() -> None:
    module = types.ModuleType("pcbworld_pns_bridge")
    module.PNSBridge = FakePNSBridge
    module.MODE_ROUTE_SINGLE = 1
    module.MODE_ROUTE_DIFF_PAIR = 2
    module.MODE_TUNE_SINGLE = 3
    module.MODE_TUNE_DIFF_PAIR = 4
    module.MODE_TUNE_DIFF_PAIR_SKEW = 5
    sys.modules["pcbworld_pns_bridge"] = module
