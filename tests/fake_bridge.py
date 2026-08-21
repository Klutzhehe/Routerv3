"""A fake pcbworld_pns_bridge module for testing pcbworld.env/agents locally.

pcbworld_pns_bridge only exists after the Colab build (see ROADMAP.md) --
nothing that imports it can run on a normal dev machine. This fake
implements just enough of the bound API (bindings.cpp) to exercise
PCBRouteEnv's, DiffPairRouteEnv's, pcbworld/agent's RouterTools/loop, and
ppo_baseline's Python control flow (rollout collection, GAE, the PPO
update, env step/reset bookkeeping, head state tracking, rip-up) locally.
It does
*not* model real push-and-shove routing -- every push()/fix() call
trivially succeeds and DRC always reports one fixed "violation", so passing
tests here only mean "the Python glue doesn't crash and produces finite
numbers", not "the routing/reward logic is correct against a real board".
That can only be checked in Colab.

Call install() before importing anything that does
`import pcbworld_pns_bridge`.
"""

from __future__ import annotations

import sys
import types
from collections import namedtuple

Candidate = namedtuple("Candidate", ["id", "x", "y", "kind", "net"])
NetPad = namedtuple("NetPad", ["net", "pad_name", "x", "y", "layer"])
DRCViolation = namedtuple("DRCViolation", ["error_code", "message", "severity", "x", "y"])
TrackSegment = namedtuple(
    "TrackSegment", ["x1", "y1", "x2", "y2", "width", "layer", "net", "is_arc"]
)
ViaGeom = namedtuple(
    "ViaGeom", ["x", "y", "diameter", "drill", "layer_top", "layer_bottom", "net"]
)
PadGeom = namedtuple(
    "PadGeom", ["x", "y", "size_x", "size_y", "layer_top", "layer_bottom", "net", "pad_name"]
)
ZoneGeom = namedtuple("ZoneGeom", ["outline", "layer", "is_keepout", "net"])
FootprintBBox = namedtuple("FootprintBBox", ["x1", "y1", "x2", "y2", "reference"])
EdgeShape = namedtuple("EdgeShape", ["shape_type", "x1", "y1", "x2", "y2", "width"])
BoardGeometry = namedtuple(
    "BoardGeometry", ["tracks", "vias", "pads", "zones", "courtyards", "board_edge"]
)

HeadSegment = namedtuple("HeadSegment", ["x1", "y1", "x2", "y2", "width", "layer"])
HeadVia = namedtuple("HeadVia", ["x", "y", "layer_top", "layer_bottom"])
HeadGeometry = namedtuple(
    "HeadGeometry", ["active", "segments", "vias", "end_x", "end_y", "layer", "length"]
)

DesignRules = namedtuple(
    "DesignRules",
    [
        "track_width",
        "via_diameter",
        "via_drill",
        "clearance",
        "min_track_width",
        "min_via_diameter",
        "min_via_drill",
        "min_hole_to_hole",
    ],
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

        self._committed_tracks: list[TrackSegment] = []
        self._committed_vias: list[ViaGeom] = []
        self._candidate_net: dict[int, str] = {}
        self._next_candidate_id = 0
        self._active_net: str | None = None
        self._route_start_xy: tuple[int, int] | None = None
        self._route_target_xy: tuple[int, int] | None = None
        self._current_pos: tuple[int, int] | None = None
        self._current_layer = 0
        self._pending_commit = False
        self._routing_active = False

        # In-progress head tracking
        self._head_segments: list[HeadSegment] = []
        self._head_vias: list[HeadVia] = []
        self._head_length: float = 0.0

        # Settings
        self._mode = 1
        self._collision_mode = 0
        self._track_width = 250_000
        self._via_diameter = 600_000
        self._via_drill = 300_000
        self._diff_pair_gap = 150_000
        self._diff_pair_via_gap = 150_000
        self._diff_pair_width = 200_000
        self._target_length = 0
        self._meander_max_amplitude = 2_500_000
        self._meander_spacing = 1_000_000

        # Design rules fixture
        self._design_rules = DesignRules(
            track_width=250_000,
            via_diameter=600_000,
            via_drill=300_000,
            clearance=200_000,
            min_track_width=200_000,
            min_via_diameter=500_000,
            min_via_drill=250_000,
            min_hole_to_hole=250_000,
        )

    def load_board(self, path: str) -> bool:
        self.loaded = True
        return True

    def save_board(self, path: str) -> bool:
        return True

    def net_names(self) -> list[str]:
        return sorted({p.net for p in self._nets if p.net})

    def reset(self) -> None:
        self._committed_tracks = []
        self._committed_vias = []
        self._stop_head()

    def set_mode(self, mode: int) -> None:
        self._mode = mode

    def set_collision_mode(self, mode: int) -> None:
        self._collision_mode = mode

    def set_track_width(self, w: int) -> None:
        self._track_width = w

    def set_via_diameter(self, d: int) -> None:
        self._via_diameter = d

    def set_via_drill(self, d: int) -> None:
        self._via_drill = d

    def set_diff_pair_gap(self, gap: int) -> None:
        self._diff_pair_gap = gap

    def set_diff_pair_via_gap(self, gap: int) -> None:
        self._diff_pair_via_gap = gap

    def set_diff_pair_width(self, width: int) -> None:
        self._diff_pair_width = width

    def set_target_length(self, length: int) -> None:
        self._target_length = length

    def set_meander_max_amplitude(self, max_amp: int) -> None:
        self._meander_max_amplitude = max_amp

    def set_meander_spacing(self, spacing: int) -> None:
        self._meander_spacing = spacing

    def net_pads(self) -> list[NetPad]:
        return list(self._nets)

    def query_hover_items(self, x: int, y: int, layer: int = -1, slop_radius: int = 100000) -> list[Candidate]:
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

    def start_route(self, x: int, y: int, item_id: int, layer: int) -> bool:
        self._active_net = self._candidate_net.get(item_id)
        self._route_start_xy = (x, y)
        self._current_pos = (x, y)
        self._current_layer = layer if layer >= 0 else 0
        self._route_target_xy = None
        self._pending_commit = False
        self._routing_active = True
        self._head_segments = []
        self._head_vias = []
        self._head_length = 0.0
        return True

    def push(self, x: int, y: int, item_id: int = -1) -> bool:
        if not self._routing_active or self._current_pos is None:
            return False
        x1, y1 = self._current_pos
        dx = x - x1
        dy = y - y1
        seg_len = (dx * dx + dy * dy) ** 0.5
        self._head_segments.append(
            HeadSegment(x1, y1, x, y, self._track_width, self._current_layer)
        )
        self._head_length += seg_len
        self._current_pos = (x, y)
        return True

    def fix(self, x: int, y: int, item_id: int = -1, force_finish: bool = False, force_commit: bool = False) -> bool:
        if not self._routing_active:
            return False
        self._route_target_xy = (x, y)
        self._pending_commit = True
        return True

    def commit_routing(self) -> None:
        if self._pending_commit and self._active_net and self._route_start_xy:
            x1, y1 = self._route_start_xy
            x2, y2 = self._route_target_xy or self._current_pos or (x1, y1)
            # Re-tuning a net replaces its previous fake segment
            self._committed_tracks = [
                t for t in self._committed_tracks if t.net != self._active_net
            ]
            if self._head_segments:
                for seg in self._head_segments:
                    self._committed_tracks.append(
                        TrackSegment(seg.x1, seg.y1, seg.x2, seg.y2, seg.width, seg.layer, self._active_net, False)
                    )
            else:
                self._committed_tracks.append(
                    TrackSegment(x1, y1, x2, y2, self._track_width, self._current_layer, self._active_net, False)
                )
            for hv in self._head_vias:
                self._committed_vias.append(
                    ViaGeom(hv.x, hv.y, self._via_diameter, self._via_drill, hv.layer_top, hv.layer_bottom, self._active_net)
                )
        self._pending_commit = False
        self._stop_head()

    def stop_routing(self) -> None:
        self._stop_head()

    def _stop_head(self) -> None:
        self._routing_active = False
        self._head_segments = []
        self._head_vias = []
        self._head_length = 0.0
        self._current_pos = None
        self._active_net = None
        self._pending_commit = False

    def toggle_via_placement(self) -> None:
        if self._routing_active and self._current_pos is not None:
            top_l = min(0, self._current_layer)
            bot_l = max(1, self._current_layer)
            self._head_vias.append(HeadVia(self._current_pos[0], self._current_pos[1], top_l, bot_l))
            self._current_layer = 1 if self._current_layer == 0 else 0

    def switch_layer(self, layer: int) -> bool:
        if self._routing_active:
            self._current_layer = layer
            if self._current_pos is not None:
                self._head_vias.append(HeadVia(self._current_pos[0], self._current_pos[1], 0, 1))
        return True

    def run_drc(self) -> list[DRCViolation]:
        return [DRCViolation(1, "clearance too small", "error", 0, 0)]

    def get_board_geometry(self) -> BoardGeometry:
        pads_geom = [
            PadGeom(p.x, p.y, 500_000, 500_000, 0, 1, p.net, p.pad_name)
            for p in self._nets
        ]
        return BoardGeometry(
            tracks=list(self._committed_tracks),
            vias=list(self._committed_vias),
            pads=pads_geom,
            zones=[],
            courtyards=[],
            board_edge=[EdgeShape("segment", 0, 0, 50 * MM, 50 * MM, 100_000)],
        )

    def get_head_geometry(self) -> HeadGeometry:
        if not self._routing_active or self._current_pos is None:
            return HeadGeometry(
                active=False,
                segments=[],
                vias=[],
                end_x=0,
                end_y=0,
                layer=0,
                length=0.0,
            )
        return HeadGeometry(
            active=True,
            segments=list(self._head_segments),
            vias=list(self._head_vias),
            end_x=self._current_pos[0],
            end_y=self._current_pos[1],
            layer=self._current_layer,
            length=self._head_length,
        )

    def head_collides(self) -> bool:
        if not self._routing_active:
            return False
        return False

    def get_design_rules(self) -> DesignRules:
        return self._design_rules

    def rip_up(self, net: str) -> int:
        before_count = len(self._committed_tracks) + len(self._committed_vias)
        self._committed_tracks = [t for t in self._committed_tracks if t.net != net]
        self._committed_vias = [v for v in self._committed_vias if v.net != net]
        after_count = len(self._committed_tracks) + len(self._committed_vias)
        return before_count - after_count


def install() -> None:
    module = sys.modules.get("pcbworld_pns_bridge")
    if module is None:
        module = types.ModuleType("pcbworld_pns_bridge")
        sys.modules["pcbworld_pns_bridge"] = module
    module.PNSBridge = FakePNSBridge
    module.Candidate = Candidate
    module.NetPad = NetPad
    module.DRCViolation = DRCViolation
    module.TrackSegment = TrackSegment
    module.ViaGeom = ViaGeom
    module.PadGeom = PadGeom
    module.ZoneGeom = ZoneGeom
    module.FootprintBBox = FootprintBBox
    module.EdgeShape = EdgeShape
    module.BoardGeometry = BoardGeometry
    module.HeadSegment = HeadSegment
    module.HeadVia = HeadVia
    module.HeadGeometry = HeadGeometry
    module.DesignRules = DesignRules
    module.MODE_ROUTE_SINGLE = 1
    module.MODE_ROUTE_DIFF_PAIR = 2
    module.MODE_TUNE_SINGLE = 3
    module.MODE_TUNE_DIFF_PAIR = 4
    module.MODE_TUNE_DIFF_PAIR_SKEW = 5
    module.RM_MARK_OBSTACLES = 0
    module.RM_SHOVE = 1
    module.RM_WALKAROUND = 2
