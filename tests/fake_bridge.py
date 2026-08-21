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
TrackSegment = namedtuple(
    "TrackSegment", ["x1", "y1", "x2", "y2", "width", "layer", "net", "is_arc"]
)
BoardGeometry = namedtuple(
    "BoardGeometry", ["tracks", "vias", "pads", "zones", "courtyards", "board_edge"]
)

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

        # Minimal routing-state tracking -- enough for get_board_geometry()
        # to report something after commit_routing(), which
        # DiffPairRouteEnv's length-tuning legs read back to find a
        # reference net's length and a to-be-tuned net's existing straight
        # segment. Still not real geometry: commit_routing() always records
        # a single straight start->target segment, regardless of what mode
        # was set or how many push() calls happened in between.
        self._committed_tracks: list[TrackSegment] = []
        self._candidate_net: dict[int, str] = {}
        self._next_candidate_id = 0
        self._active_net: str | None = None
        self._route_start_xy: tuple[int, int] | None = None
        self._route_target_xy: tuple[int, int] | None = None
        self._pending_commit = False

    def load_board(self, path):
        self.loaded = True
        return True

    def reset(self):
        self._committed_tracks = []

    def set_mode(self, mode):
        pass

    def set_collision_mode(self, mode):
        pass

    def set_track_width(self, w):
        pass

    def set_via_diameter(self, d):
        pass

    def set_via_drill(self, d):
        pass

    def set_diff_pair_gap(self, gap):
        pass

    def set_diff_pair_via_gap(self, gap):
        pass

    def set_diff_pair_width(self, width):
        pass

    def set_target_length(self, length):
        pass

    def set_meander_max_amplitude(self, max_amp):
        pass

    def set_meander_spacing(self, spacing):
        pass

    def net_pads(self):
        return list(self._nets)

    def query_hover_items(self, x, y, layer=-1, slop_radius=100000):
        # Pads first (exact fixture positions, same as before), then
        # already-committed fake tracks by proximity to their midpoint --
        # the latter is what lets a tune leg's start_route() find the
        # segment its direct leg just committed (real bridge: a
        # query_hover_items at a track's midpoint resolves to that
        # segment; see ROADMAP.md item 7's Colab-verified tune-mode test).
        for p in self._nets:
            if abs(p.x - x) <= slop_radius and abs(p.y - y) <= slop_radius:
                cid = self._next_candidate_id
                self._next_candidate_id += 1
                self._candidate_net[cid] = p.net
                return [Candidate(cid, x, y, "pad", p.net)]
        for t in self._committed_tracks:
            mid_x, mid_y = (t.x1 + t.x2) // 2, (t.y1 + t.y2) // 2
            if abs(mid_x - x) <= slop_radius and abs(mid_y - y) <= slop_radius:
                cid = self._next_candidate_id
                self._next_candidate_id += 1
                self._candidate_net[cid] = t.net
                return [Candidate(cid, x, y, "segment", t.net)]
        return [Candidate(0, x, y, "pad", "")]

    def start_route(self, x, y, item_id, layer):
        self._active_net = self._candidate_net.get(item_id)
        self._route_start_xy = (x, y)
        self._route_target_xy = None
        self._pending_commit = False
        return True

    def push(self, x, y, item_id=-1):
        return True

    def fix(self, x, y, item_id=-1, force_finish=False, force_commit=False):
        self._route_target_xy = (x, y)
        self._pending_commit = True
        return True

    def commit_routing(self):
        if self._pending_commit and self._active_net and self._route_start_xy:
            x1, y1 = self._route_start_xy
            x2, y2 = self._route_target_xy or (x1, y1)
            # Re-tuning a net replaces its previous fake segment rather
            # than appending a second one, matching the real router's "one
            # net, one committed route" invariant closely enough for this
            # fake's purposes.
            self._committed_tracks = [
                t for t in self._committed_tracks if t.net != self._active_net
            ]
            self._committed_tracks.append(
                TrackSegment(x1, y1, x2, y2, 250_000, 0, self._active_net, False)
            )
        self._pending_commit = False

    def stop_routing(self):
        pass

    def toggle_via_placement(self):
        pass

    def run_drc(self):
        return [DRCViolation(1, "clearance too small", "error", 0, 0)]

    def get_board_geometry(self):
        return BoardGeometry(
            tracks=list(self._committed_tracks),
            vias=[],
            pads=[],
            zones=[],
            courtyards=[],
            board_edge=[],
        )


def install() -> None:
    # Reuse the existing module object across repeated install() calls
    # (every test file that imports this one calls install() again at its
    # own collection time). If each call swapped in a brand-new
    # types.ModuleType, a test file that does `import pcbworld_pns_bridge
    # as bridge` and later monkeypatches `bridge.PNSBridge` (e.g.
    # test_diff_pair_route_env.py) would be mutating a module object that
    # a *different* file's later install() call had already replaced in
    # sys.modules -- the env under test would silently pick up the
    # untouched default fixture instead of the monkeypatched one.
    module = sys.modules.get("pcbworld_pns_bridge")
    if module is None:
        module = types.ModuleType("pcbworld_pns_bridge")
        sys.modules["pcbworld_pns_bridge"] = module
    module.PNSBridge = FakePNSBridge
    module.MODE_ROUTE_SINGLE = 1
    module.MODE_ROUTE_DIFF_PAIR = 2
    module.MODE_TUNE_SINGLE = 3
    module.MODE_TUNE_DIFF_PAIR = 4
    module.MODE_TUNE_DIFF_PAIR_SKEW = 5
    module.RM_MARK_OBSTACLES = 0
    module.RM_SHOVE = 1
    module.RM_WALKAROUND = 2
