"""LineDiffPairTuneEnv: Multi-net routing with diff-pair and length-tuning primitives.

Extends the LineRouteEnv design to handle:
- Differential pairs: one MODE_ROUTE_DIFF_PAIR leg driven by P net
- Length-matched groups: reference net direct, then members: direct + MODE_TUNE_SINGLE

Uses the same line-segment observation and 1-D heading action as LineRouteEnv,
with 3 extra leg-kind features appended to the flat vector.

## Why this file imports geodesic.py

The first version of this env shaped nothing at all: per step it charged a
step cost and a collision penalty, and paid out only at the leg's end. With
no progress term the only gradient available was "stop colliding", whose
cheapest solution is to not move -- and with a straight-line term instead,
the gradient points INTO whatever is in the way (measured: +0.0545 to step at
the obstacle against -0.0445 to round it; see pcbworld/env/geodesic.py).

So the potential here is the same obstacle-free cost-to-go LineRouteEnv uses,
built from the same segment list, at the same measured inflation. A diff-pair
leg has more copper to get around than a single-ended one, not less.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pcbworld.env.geodesic import GeodesicConfig, GeodesicField
from pcbworld.env.line_obs import (
    KIND_PAD,
    MM,
    LineObsConfig,
    board_segments,
    build_observation,
    ghost_segment,
    nearest_obstacle_gap,
    pad_to_segment,
)
from pcbworld.env.line_route_env import MAX_TURN_RAD, RewardWeights

# Leg kind one-hot order
_KIND_INDEX = {"direct": 0, "diff_pair": 1, "tune": 2}
NUM_LEG_KINDS = len(_KIND_INDEX)


@dataclasses.dataclass
class _Leg:
    kind: str  # "direct" | "diff_pair" | "tune"
    net: str
    mode: int  # PNS::ROUTER_MODE constant
    reference_net: Optional[str] = None  # for tune legs


def _parse_board_nets(pads) -> Tuple[List[str], Dict[str, Dict[str, str]], Dict[str, Dict[int, str]]]:
    """Groups net_pads() output by generate_board.py's naming convention."""
    plain: set[str] = set()
    diff_pairs: Dict[str, Dict[str, str]] = {}
    length_groups: Dict[str, Dict[int, str]] = {}

    for pad in pads:
        net = pad.net
        if not net:
            continue
        if net.startswith("diffpair_"):
            base, leg = net.rsplit("_", 1)
            diff_pairs.setdefault(base, {})[leg] = net
        elif net.startswith("lengthgrp_"):
            _, group_idx, member_idx = net.split("_")
            length_groups.setdefault(group_idx, {})[int(member_idx)] = net
        else:
            plain.add(net)

    return sorted(plain), diff_pairs, length_groups


def _build_legs(pads, bridge_module) -> List[_Leg]:
    """Build ordered leg sequence from board nets."""
    plain, diff_pairs, length_groups = _parse_board_nets(pads)

    legs: List[_Leg] = [_Leg("direct", net, bridge_module.MODE_ROUTE_SINGLE) for net in plain]

    for base in sorted(diff_pairs):
        pair = diff_pairs[base]
        assert "P" in pair and "N" in pair, f"diff pair {base!r} missing P or N leg"
        legs.append(_Leg("diff_pair", pair["P"], bridge_module.MODE_ROUTE_DIFF_PAIR))

    for group_idx in sorted(length_groups):
        members = length_groups[group_idx]
        ordered_nets = [members[i] for i in sorted(members)]
        assert len(ordered_nets) >= 2, f"length group {group_idx!r} needs >= 2 members"
        reference_net = ordered_nets[0]
        legs.append(_Leg("direct", reference_net, bridge_module.MODE_ROUTE_SINGLE))
        for member_net in ordered_nets[1:]:
            legs.append(_Leg("direct", member_net, bridge_module.MODE_ROUTE_SINGLE))
            legs.append(_Leg("tune", member_net, bridge_module.MODE_TUNE_SINGLE, reference_net=reference_net))

    return legs


def _net_length_nm(geometry, net: str) -> float:
    """Sum committed segment lengths for a net."""
    total = 0.0
    for seg in geometry.tracks:
        if seg.net == net:
            total += math.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1)
    return total


def _net_midpoint(geometry, net: str) -> Optional[Tuple[int, int]]:
    """First committed segment's midpoint for tune leg start."""
    for seg in geometry.tracks:
        if seg.net == net:
            return (seg.x1 + seg.x2) // 2, (seg.y1 + seg.y2) // 2
    return None


class LineDiffPairTuneEnv(gym.Env):
    """Routes one leg at a time; an episode sequences every leg on the board.

    Observation: flat Box, identical layout to LineRouteEnv's plus three
    trailing leg-kind features (see LineObsConfig.extra_globals).

    Action: Box(-1, 1, shape=(1,)) -- turn relative to the target bearing.

    Reward: geodesic potential shaping + step cost + first-contact collision
    charge + leg terminals + length-mismatch penalty on tune legs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        board_path: str,
        *,
        track_width_nm: int = 250_000,
        via_diameter_nm: int = 600_000,
        via_drill_nm: int = 300_000,
        diff_pair_gap_nm: int = 150_000,
        diff_pair_width_nm: int = 200_000,
        meander_max_amplitude_nm: int = 2_500_000,
        meander_spacing_nm: int = 1_000_000,
        step_size_nm: int = 500_000,
        snap_radius_nm: int = 400_000,
        max_steps_per_leg: int = 120,
        gamma: float = 0.99,
        obs_config: Optional[LineObsConfig] = None,
        geodesic_config: Optional[GeodesicConfig] = None,
        reward_weights: Optional[RewardWeights] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        import pcbworld_pns_bridge as bridge

        self._bridge_module = bridge
        self.bridge = bridge.PNSBridge()
        self.board_path = board_path
        self.track_width_nm = track_width_nm
        self.via_diameter_nm = via_diameter_nm
        self.via_drill_nm = via_drill_nm
        self.diff_pair_gap_nm = diff_pair_gap_nm
        self.diff_pair_width_nm = diff_pair_width_nm
        self.meander_max_amplitude_nm = meander_max_amplitude_nm
        self.meander_spacing_nm = meander_spacing_nm
        self.step_size_nm = step_size_nm
        self.snap_radius_nm = snap_radius_nm
        self.max_steps_per_leg = max_steps_per_leg
        self.gamma = gamma
        self.weights = reward_weights or RewardWeights()

        base = obs_config or LineObsConfig(max_steps=max_steps_per_leg)
        # The three leg-kind features are this env's own; everything else is
        # the shared contract.
        self.obs_config = dataclasses.replace(base, extra_globals=NUM_LEG_KINDS)
        self._geodesic_config = geodesic_config or GeodesicConfig(cell_nm=float(step_size_nm))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_config.flat_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._board_loaded = False
        self._pending_legs: List[_Leg] = []
        self._current_leg: Optional[_Leg] = None
        self._pads: list = []
        self._target_xy: Tuple[float, float] = (0.0, 0.0)
        self._target_item_id: int = -1
        self._start_xy: Tuple[float, float] = (0.0, 0.0)
        self._pos: Tuple[float, float] = (0.0, 0.0)
        self._target_layer: int = 0
        self._steps_this_leg = 0
        self._routed_length_nm: float = 0.0
        self._straight_line_dist_nm: float = 1.0
        self._collides = False
        self._prev_heading = 0.0
        self._completed: List[str] = []
        self._failed: List[str] = []

        # Per-leg caches. Committed copper only changes when a leg finishes,
        # so both the obstacle list and the field it feeds live exactly that
        # long -- the same lifetime LineRouteEnv gives them.
        self._static_segments: list = []
        self._pad_geoms: list = []
        self._obstacles: list = []
        self._field: Optional[GeodesicField] = None

        if seed is not None:
            self.reset(seed=seed)

    # -- gym API --------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if not self._board_loaded:
            assert self.bridge.load_board(self.board_path), f"load_board failed: {self.board_path}"
            self._board_loaded = True
        else:
            self.bridge.reset()

        self.bridge.set_track_width(self.track_width_nm)
        self.bridge.set_via_diameter(self.via_diameter_nm)
        self.bridge.set_via_drill(self.via_drill_nm)
        self.bridge.set_diff_pair_gap(self.diff_pair_gap_nm)
        self.bridge.set_diff_pair_width(self.diff_pair_width_nm)
        self.bridge.set_meander_max_amplitude(self.meander_max_amplitude_nm)
        self.bridge.set_meander_spacing(self.meander_spacing_nm)
        # push() must be a pure validator, not a router that shoves other
        # traces aside to accommodate the agent.
        self.bridge.set_collision_mode(self._bridge_module.RM_MARK_OBSTACLES)

        self._pads = list(self.bridge.net_pads())
        self._pending_legs = _build_legs(self._pads, self._bridge_module)
        assert self._pending_legs, f"no routable nets on board: {self.board_path}"

        self._completed, self._failed = [], []
        self._refresh_static_segments()
        self._start_next_leg()
        return self._observe(), self._info()

    def step(self, action):
        assert self._current_leg is not None, "step() before reset()"

        turn = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0)) * MAX_TURN_RAD
        weights = self.weights

        dx = self._target_xy[0] - self._pos[0]
        dy = self._target_xy[1] - self._pos[1]
        bearing = math.atan2(dy, dx) if (dx or dy) else 0.0
        heading = bearing + turn
        self._prev_heading = heading

        prev_pos = self._pos
        was_colliding = self._collides

        goal = (
            self._pos[0] + self.step_size_nm * math.cos(heading),
            self._pos[1] + self.step_size_nm * math.sin(heading),
        )
        self.bridge.push(int(goal[0]), int(goal[1]), -1)

        # Read the head back rather than trusting the requested point --
        # push() is ROUTER::Move() and need not land where it was told.
        head = self.bridge.get_head_geometry()
        if head.active:
            self._routed_length_nm += math.hypot(
                head.end_x - self._pos[0], head.end_y - self._pos[1]
            )
            self._pos = (float(head.end_x), float(head.end_y))
        self._collides = bool(self.bridge.head_collides())
        self._steps_this_leg += 1

        reward = self._shaping(prev_pos, self._pos) - weights.step
        # Charged on the TRANSITION into contact: the leg is lost at the first
        # touch regardless, so one marker at the deciding step beats thirty
        # diffuse ones afterwards.
        if self._collides and not was_colliding:
            reward -= weights.collision

        info: dict[str, Any] = {"net": self._current_leg.net, "kind": self._current_leg.kind}
        dist = math.hypot(
            self._target_xy[0] - self._pos[0], self._target_xy[1] - self._pos[1]
        )

        leg_finished = False
        truncated = False

        if dist <= self.snap_radius_nm:
            leg_finished = True
            if self.bridge.fix(
                int(self._target_xy[0]), int(self._target_xy[1]), self._target_item_id, True, True
            ):
                self.bridge.commit_routing()
                self._completed.append(self._current_leg.net)
                reward += weights.net_done
                info["completed"] = True
                detour = self._routed_length_nm / self._straight_line_dist_nm
                info["detour_ratio"] = detour
                reward -= weights.detour * max(0.0, detour - 1.0)
            else:
                # fix() refused: the leg is over either way -- stop_routing()
                # kills the head, so re-calling fix() every remaining step
                # would only burn the budget and overwrite the reason.
                self.bridge.stop_routing()
                self._failed.append(self._current_leg.net)
                reward -= weights.net_failed
                info["fix_rejected"] = True

        if not leg_finished and self._steps_this_leg >= self.max_steps_per_leg:
            self.bridge.stop_routing()
            self._failed.append(self._current_leg.net)
            reward -= weights.net_failed
            leg_finished = True
            truncated = True
            info["timeout"] = True

        terminated = False
        if leg_finished:
            if self._current_leg.kind == "tune":
                geometry = self.bridge.get_board_geometry()
                target_length = _net_length_nm(geometry, self._current_leg.reference_net)
                actual_length = _net_length_nm(geometry, self._current_leg.net)
                mismatch = abs(actual_length - target_length)
                info["length_mismatch_nm"] = mismatch
                reward -= weights.length_mismatch * mismatch / self.obs_config.length_scale

            if self._pending_legs:
                # Copper changed, so the obstacle cache and its field are
                # stale exactly here -- once per leg, which is the point.
                self._refresh_static_segments()
                self._start_next_leg()
            else:
                terminated = not truncated
                self._current_leg = None

        return self._observe(), float(reward), terminated, truncated, {**info, **self._info()}

    def close(self) -> None:
        pass

    # -- internals ------------------------------------------------------

    def _start_next_leg(self) -> None:
        leg = self._pending_legs.pop(0)
        self._current_leg = leg
        self._steps_this_leg = 0
        self._routed_length_nm = 0.0
        self._collides = False

        pads = [p for p in self._pads if p.net == leg.net]
        assert len(pads) >= 2, f"net {leg.net!r} has {len(pads)} pads, need >= 2"
        start_pad, target_pad = pads[0], pads[1]

        self.bridge.set_mode(leg.mode)

        if leg.kind == "tune":
            geometry = self.bridge.get_board_geometry()
            target_length = _net_length_nm(geometry, leg.reference_net)
            assert target_length > 0, f"reference net {leg.reference_net!r} not routed yet"
            self.bridge.set_target_length(int(target_length))

            start_xy = _net_midpoint(geometry, leg.net)
            assert start_xy is not None, f"tune leg needs {leg.net!r} already routed"
        else:
            start_xy = (start_pad.x, start_pad.y)

        self._start_xy = (float(start_xy[0]), float(start_xy[1]))
        self._target_xy = (float(target_pad.x), float(target_pad.y))
        self._target_item_id = self._pad_candidate(int(target_pad.x), int(target_pad.y))
        self._target_layer = target_pad.layer
        self._pos = self._start_xy
        self._prev_heading = math.atan2(
            self._target_xy[1] - self._start_xy[1], self._target_xy[0] - self._start_xy[0]
        )
        self._straight_line_dist_nm = max(
            1.0,
            math.hypot(
                self._target_xy[0] - self._start_xy[0], self._target_xy[1] - self._start_xy[1]
            ),
        )

        start_id = self._pad_candidate(int(start_xy[0]), int(start_xy[1]))
        assert self.bridge.start_route(int(start_xy[0]), int(start_xy[1]), start_id, 0), (
            f"start_route failed for {leg.net!r}"
        )
        self._rebuild_obstacles()

    def _pad_candidate(self, x: int, y: int) -> int:
        """Prefer a pad hit over whatever the hit-test happened to return
        first: query_hover_items() is not sorted by kind or distance, and on a
        board with committed copper an unrelated track passing within the snap
        radius hands fix() the wrong item id."""
        candidates = self.bridge.query_hover_items(x, y, layer=0, slop_radius=self.snap_radius_nm)
        if not candidates:
            return -1
        pads = [c for c in candidates if c.kind == "pad"]
        return (pads[0] if pads else candidates[0]).id

    def _refresh_static_segments(self) -> None:
        geometry = self.bridge.get_board_geometry()
        self._static_segments = board_segments(geometry)
        self._pad_geoms = list(geometry.pads)

    def _rebuild_obstacles(self) -> None:
        self._obstacles = self._build_obstacles()
        self._field = GeodesicField.build(
            self._obstacles,
            head=self._pos,
            target=self._target_xy,
            config=self._geodesic_config,
        )
        if not self._field.reachable:
            # No obstacle-free route exists. Fall back to the straight line
            # for the whole leg rather than per query: a potential that
            # changes definition mid-leg puts a reward spike on the switch.
            self._field = None

    def _build_obstacles(self) -> list:
        leg = self._current_leg
        net = leg.net if leg else ""
        segments = list(self._static_segments)

        # A pad is an obstacle unless it belongs to the leg being routed. For
        # a diff pair both P and N pads are endpoints, not obstacles.
        own_nets = {net}
        if net.startswith("diffpair_") and net.endswith("_P"):
            own_nets.add(net[:-2] + "_N")

        for pad in self._pad_geoms:
            if pad.net not in own_nets:
                segments.append(pad_to_segment(pad, kind=KIND_PAD))

        # Legs still to come enter as the straight lines they will have to
        # span, so the policy can see where future copper needs room. Ghosts
        # are excluded from the FIELD (see geodesic._blocked_mask) but not
        # from the observation.
        seen = set(own_nets)
        for future in self._pending_legs:
            if future.net in seen or future.net in self._completed:
                continue
            seen.add(future.net)
            future_pads = [p for p in self._pads if p.net == future.net]
            if len(future_pads) >= 2:
                a, b = future_pads[0], future_pads[1]
                segments.append(ghost_segment((a.x, a.y), (b.x, b.y), future.net))

        return segments

    def _geodesic_dist(self, pos: Tuple[float, float]) -> float:
        if self._field is not None:
            c = self._field.cost_to_go(pos[0], pos[1])
            if math.isfinite(c):
                return c
        return math.hypot(self._target_xy[0] - pos[0], self._target_xy[1] - pos[1])

    def _potential(self, pos: Tuple[float, float]) -> float:
        return -self.weights.progress * self._geodesic_dist(pos) / self.obs_config.length_scale

    def _shaping(self, prev_pos, pos) -> float:
        return self.gamma * self._potential(pos) - self._potential(prev_pos)

    def _clearance_at(self, pos) -> float:
        return nearest_obstacle_gap(pos[0], pos[1], self._obstacles)

    def _step_ahead(self) -> Tuple[float, float]:
        bearing = math.atan2(
            self._target_xy[1] - self._pos[1], self._target_xy[0] - self._pos[0]
        )
        return (
            self._pos[0] + self.step_size_nm * math.cos(bearing),
            self._pos[1] + self.step_size_nm * math.sin(bearing),
        )

    def _geodesic_direction(self):
        if self._field is None:
            return None
        return self._field.descent_direction(self._pos[0], self._pos[1], self.step_size_nm)

    def _observe(self) -> np.ndarray:
        if self._current_leg is None:
            return np.zeros(self.obs_config.flat_size, dtype=np.float32)

        kind_onehot = np.zeros(NUM_LEG_KINDS, dtype=np.float32)
        kind_onehot[_KIND_INDEX[self._current_leg.kind]] = 1.0

        length_slack = 0.0
        if self._current_leg.kind == "tune":
            geometry = self.bridge.get_board_geometry()
            reference = _net_length_nm(geometry, self._current_leg.reference_net)
            length_slack = max(0.0, reference - self._routed_length_nm)

        return build_observation(
            self._obstacles,
            head=self._pos,
            target=self._target_xy,
            head_layer=0,
            target_layer=self._target_layer,
            own_net=self._current_leg.net,
            steps_taken=self._steps_this_leg,
            routed_length=self._routed_length_nm,
            straight_line_length=self._straight_line_dist_nm,
            head_collides=self._collides,
            config=self.obs_config,
            length_slack=length_slack,
            geodesic_dist=self._geodesic_dist(self._pos),
            clearance_now=self._clearance_at(self._pos),
            clearance_ahead=self._clearance_at(self._step_ahead()),
            geodesic_direction=self._geodesic_direction(),
            extra=kind_onehot,
        )

    def _info(self) -> dict:
        return {
            "leg": self._current_leg.net if self._current_leg else None,
            "leg_kind": self._current_leg.kind if self._current_leg else None,
            "legs_remaining": len(self._pending_legs),
            "completed": list(self._completed),
            "failed": list(self._failed),
            "steps": self._steps_this_leg,
            "collides": self._collides,
            "routed_length_nm": self._routed_length_nm,
        }
