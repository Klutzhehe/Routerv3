"""WaypointRouteEnv: Gymnasium environment for discrete waypoint-level PCB routing.

Key Architecture Decisions (Phase B):
1. No A* Planner: The policy directly emits coarse waypoints (one step = one waypoint),
   and pcbworld_pns_bridge's push()/fix() walks and validates them in RM_MARK_OBSTACLES mode.
2. Length Tuning is NOT a Waypoint Problem: A tune leg is a ONE-STEP macro-action
   (anchor, amplitude, spacing) scored against a tolerance band (~0.25 mm residual
   is meander granularity, not an error).
3. Single Head-State Adapter: All reads of get_head_geometry() and head_collides()
   are encapsulated in `read_head_state(bridge) -> HeadState` for a single point of truth.
4. Masked MultiDiscrete Action Space & Dense Deviation Rewards:
   The agent is rewarded for geometric progress and penalized for deviations between
   the requested waypoint and actual router head placement.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MM = 1_000_000
SNAP_RADIUS_NM = int(0.5 * MM)
DEFAULT_TOLERANCE_NM = int(0.25 * MM)  # 0.25 mm meander granularity tolerance


@dataclasses.dataclass
class HeadState:
    """Normalized snapshot of the active router head state."""

    active: bool
    end_x: int
    end_y: int
    layer: int
    length_nm: float
    collides: bool
    num_segments: int
    num_vias: int


def read_head_state(bridge: Any) -> HeadState:
    """Single adapter function wrapping all head-state reads from the bridge.

    If the C++ bridge in Colab changes return types or semantics, only this
    function needs to be modified.
    """
    geom = bridge.get_head_geometry()
    collides = bridge.head_collides() if hasattr(bridge, "head_collides") else False
    return HeadState(
        active=bool(geom.active),
        end_x=int(geom.end_x),
        end_y=int(geom.end_y),
        layer=int(geom.layer),
        length_nm=float(geom.length),
        collides=bool(collides),
        num_segments=len(geom.segments) if hasattr(geom, "segments") else 0,
        num_vias=len(geom.vias) if hasattr(geom, "vias") else 0,
    )


@dataclasses.dataclass
class WaypointRewardWeights:
    step_penalty: float = 0.05
    deviation_penalty: float = 1.0e-6  # per nm of |requested - actual| head position
    collision_penalty: float = 2.0
    via_penalty: float = 5.0
    drc_penalty: float = 10.0
    progress_weight: float = 10.0  # potential shaping toward target
    net_finished_bonus: float = 20.0
    length_mismatch_penalty: float = 2.0e-6  # per nm of excess |actual - target| > tolerance
    invalid_action_penalty: float = 1.0


@dataclasses.dataclass
class _Leg:
    kind: str  # "direct" | "diff_pair" | "tune"
    net: str
    mode: int
    reference_net: str | None = None


# Action Type Codes
ACT_WAYPOINT = 0  # Push a waypoint at (grid_x, grid_y)
ACT_FIX = 1       # Attempt fix/finish to target pad
ACT_TUNE = 2      # Single-step length tune macro-action
ACT_RIPUP = 3     # Rip up current net and re-attempt

_KIND_INDEX = {"direct": 0, "diff_pair": 1, "tune": 2}


class WaypointRouteEnv(gym.Env):
    """Discrete waypoint routing environment with masked MultiDiscrete action space."""

    metadata = {"render_modes": []}

    # Amplitude presets for tune legs (in nm)
    AMPLITUDE_PRESETS_NM = [
        int(1.0 * MM),
        int(1.5 * MM),
        int(2.0 * MM),
        int(2.5 * MM),
        int(3.0 * MM),
    ]

    # Spacing presets for tune legs (in nm)
    SPACING_PRESETS_NM = [
        int(0.5 * MM),
        int(0.8 * MM),
        int(1.0 * MM),
        int(1.2 * MM),
    ]

    def __init__(
        self,
        board_path: str,
        grid_resolution: int = 32,
        board_size_mm: float = 50.0,
        max_waypoints_per_leg: int = 20,
        tolerance_nm: int = DEFAULT_TOLERANCE_NM,
        reward_weights: WaypointRewardWeights | None = None,
    ) -> None:
        super().__init__()

        import pcbworld_pns_bridge as bridge

        self._bridge_module = bridge
        self.bridge = bridge.PNSBridge()
        self.board_path = board_path
        self.grid_resolution = grid_resolution
        self.board_size_nm = int(board_size_mm * MM)
        self.max_waypoints_per_leg = max_waypoints_per_leg
        self.tolerance_nm = tolerance_nm
        self.reward_weights = reward_weights or WaypointRewardWeights()

        # MultiDiscrete Action Layout:
        # [0]: Action Type (0=ACT_WAYPOINT, 1=ACT_FIX, 2=ACT_TUNE, 3=ACT_RIPUP)
        # [1]: Grid X [0, grid_resolution-1] (or anchor index for tune)
        # [2]: Grid Y [0, grid_resolution-1] (or amplitude index for tune)
        # [3]: Layer / Spacing flag (0=keep layer / spacing 0, 1=switch layer / spacing 1, etc.)
        max_param = max(len(self.SPACING_PRESETS_NM), 2)
        self.action_space = spaces.MultiDiscrete(
            [4, self.grid_resolution, self.grid_resolution, max_param]
        )

        # Observation Layout (12 floats):
        # [0..1]: Current head (x, y) normalized to [0, 1]
        # [2..3]: Target pad (x, y) normalized to [0, 1]
        # [4..5]: Delta to target (dx, dy) in mm
        # [6]: Distance to target normalized by board diagonal
        # [7]: Current routing layer (0.0 or 1.0)
        # [8]: Step fraction remaining
        # [9..11]: Leg kind one-hot [direct, diff_pair, tune]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        self._board_loaded = False
        self._pending_legs: list[_Leg] = []
        self._current_leg: _Leg | None = None
        self._target_xy: tuple[int, int] | None = None
        self._target_item_id: int = -1
        self._start_xy: tuple[int, int] | None = None
        self._last_head_pos: tuple[int, int] | None = None
        self._steps_this_leg = 0
        self._via_count = 0
        self._violations_last_check = 0
        self._last_potential = 0.0

    # -- Grid mapping helpers ----------------------------------------------

    def grid_to_nm(self, gx: int, gy: int) -> tuple[int, int]:
        """Maps discrete grid coordinates to board nanometer coordinates."""
        nx = int((gx + 0.5) / self.grid_resolution * self.board_size_nm)
        ny = int((gy + 0.5) / self.grid_resolution * self.board_size_nm)
        return nx, ny

    def nm_to_grid(self, x: int, y: int) -> tuple[int, int]:
        """Maps board nanometer coordinates to nearest discrete grid cell."""
        gx = int(np.clip(x / self.board_size_nm * self.grid_resolution, 0, self.grid_resolution - 1))
        gy = int(np.clip(y / self.board_size_nm * self.grid_resolution, 0, self.grid_resolution - 1))
        return gx, gy

    # -- Action Masking ----------------------------------------------------

    def action_masks(self) -> dict[str, np.ndarray]:
        """Returns boolean validity masks for each dimension of MultiDiscrete."""
        mask_action_type = np.zeros(4, dtype=bool)
        mask_gx = np.ones(self.grid_resolution, dtype=bool)
        mask_gy = np.ones(self.grid_resolution, dtype=bool)
        mask_param = np.zeros(self.action_space.nvec[3], dtype=bool)

        if self._current_leg is None:
            return {
                "action_type": mask_action_type,
                "grid_x": mask_gx,
                "grid_y": mask_gy,
                "param": mask_param,
            }

        if self._current_leg.kind == "tune":
            # Tune leg: only ACT_TUNE is valid (1-step macro action)
            mask_action_type[ACT_TUNE] = True
            mask_gx = np.zeros(self.grid_resolution, dtype=bool)
            mask_gx[0] = True  # Anchor index
            mask_gy = np.zeros(self.grid_resolution, dtype=bool)
            mask_gy[: len(self.AMPLITUDE_PRESETS_NM)] = True
            mask_param[: len(self.SPACING_PRESETS_NM)] = True
        else:
            # Route legs (direct / diff_pair)
            mask_action_type[ACT_WAYPOINT] = True
            mask_action_type[ACT_FIX] = True
            mask_action_type[ACT_RIPUP] = True
            mask_param[:2] = True  # 0 = keep layer, 1 = switch layer

        return {
            "action_type": mask_action_type,
            "grid_x": mask_gx,
            "grid_y": mask_gy,
            "param": mask_param,
        }

    # -- gymnasium.Env Interface ------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        if not self._board_loaded:
            assert self.bridge.load_board(self.board_path), f"load_board failed: {self.board_path}"
            self._board_loaded = True
        else:
            self.bridge.reset()

        # Apply default design rules from the board
        rules = self.bridge.get_design_rules()
        self.bridge.set_track_width(rules.track_width)
        self.bridge.set_via_diameter(rules.via_diameter)
        self.bridge.set_via_drill(rules.via_drill)
        self.bridge.set_mode(self._bridge_module.MODE_ROUTE_SINGLE)
        self.bridge.set_collision_mode(self._bridge_module.RM_MARK_OBSTACLES)

        pads = self.bridge.net_pads()
        self._pending_legs = self._build_legs(pads)
        assert self._pending_legs, f"No routable nets found on {self.board_path}"

        self._via_count = 0
        self._violations_last_check = 0

        self._start_next_leg()
        return self._observe(), {"net": self._current_leg.net, "kind": self._current_leg.kind}

    def step(self, action: np.ndarray | list[int]):
        assert self._current_leg is not None, "step() called with no active leg; call reset() first"

        action = np.asarray(action, dtype=int)
        act_type, gx, gy, param = action[0], action[1], action[2], action[3]
        weights = self.reward_weights
        reward = -weights.step_penalty
        terminated = False
        truncated = False
        info: dict[str, Any] = {"net": self._current_leg.net, "kind": self._current_leg.kind}

        self._steps_this_leg += 1
        leg_finished = False

        # --- 1. TUNE LEG MACRO-ACTION (1-step execution) ---
        if self._current_leg.kind == "tune":
            amp_idx = min(gy, len(self.AMPLITUDE_PRESETS_NM) - 1)
            spacing_idx = min(param, len(self.SPACING_PRESETS_NM) - 1)
            amplitude_nm = self.AMPLITUDE_PRESETS_NM[amp_idx]
            spacing_nm = self.SPACING_PRESETS_NM[spacing_idx]

            self.bridge.set_meander_max_amplitude(amplitude_nm)
            self.bridge.set_meander_spacing(spacing_nm)

            # Fix and commit tuned meander
            fixed = self.bridge.fix(
                self._target_xy[0], self._target_xy[1], self._target_item_id, True, True
            )
            self.bridge.commit_routing()
            leg_finished = True

            # Evaluate against tolerance band (e.g. 0.25 mm)
            geometry = self.bridge.get_board_geometry()
            target_length = self._get_net_length_nm(geometry, self._current_leg.reference_net)
            actual_length = self._get_net_length_nm(geometry, self._current_leg.net)
            mismatch_nm = abs(actual_length - target_length)
            excess_mismatch = max(0.0, mismatch_nm - self.tolerance_nm)

            reward -= weights.length_mismatch_penalty * excess_mismatch
            if excess_mismatch == 0.0:
                reward += weights.net_finished_bonus
            info["actual_length_nm"] = actual_length
            info["target_length_nm"] = target_length
            info["length_mismatch_nm"] = mismatch_nm
            info["within_tolerance"] = excess_mismatch == 0.0

        # --- 2. ROUTE LEGS (WAYPOINT / FIX / RIPUP) ---
        elif act_type == ACT_WAYPOINT:
            # Layer switch if requested
            if param > 0:
                head_state = read_head_state(self.bridge)
                new_layer = 1 if head_state.layer == 0 else 0
                self.bridge.switch_layer(new_layer)
                self._via_count += 1
                reward -= weights.via_penalty

            # Push waypoint
            wx, wy = self.grid_to_nm(gx, gy)
            self.bridge.push(wx, wy, -1)

            # Read back resulting head state via single adapter function
            head_state = read_head_state(self.bridge)

            # Dense deviation penalty: distance between requested waypoint and router head
            deviation_nm = float(np.hypot(head_state.end_x - wx, head_state.end_y - wy))
            reward -= weights.deviation_penalty * deviation_nm
            info["deviation_nm"] = deviation_nm

            # Collision penalty
            if head_state.collides:
                reward -= weights.collision_penalty
                info["head_collision"] = True

            # Potential progress shaping toward target pad
            current_dist = float(
                np.hypot(
                    (self._target_xy[0] - head_state.end_x) / MM,
                    (self._target_xy[1] - head_state.end_y) / MM,
                )
            )
            potential = -current_dist
            reward += weights.progress_weight * (potential - self._last_potential)
            self._last_potential = potential
            self._last_head_pos = (head_state.end_x, head_state.end_y)

        elif act_type == ACT_FIX:
            fixed = self.bridge.fix(
                self._target_xy[0], self._target_xy[1], self._target_item_id, True, True
            )
            if fixed:
                self.bridge.commit_routing()
                leg_finished = True
                reward += weights.net_finished_bonus
            else:
                reward -= weights.invalid_action_penalty

        elif act_type == ACT_RIPUP:
            removed = self.bridge.rip_up(self._current_leg.net)
            self.bridge.stop_routing()
            reward -= weights.invalid_action_penalty * 0.5
            # Restart route for this net
            self._re_start_current_leg()
            info["ripped_up_count"] = removed

        # Check step budget
        if not leg_finished and self._steps_this_leg >= self.max_waypoints_per_leg:
            self.bridge.stop_routing()
            truncated = True
            leg_finished = True

        # Process leg completion / DRC
        if leg_finished:
            violations = self.bridge.run_drc()
            errors = sum(1 for v in violations if v.severity == "error")
            reward -= weights.drc_penalty * max(0, errors - self._violations_last_check)
            self._violations_last_check = errors
            info["drc_errors"] = errors

            if self._pending_legs:
                self._start_next_leg()
            else:
                if not truncated:
                    terminated = True
                self._current_leg = None

        return self._observe(), reward, terminated, truncated, info

    def close(self) -> None:
        if self._board_loaded:
            self.bridge.stop_routing()

    # -- Internal Leg Management & Helpers ---------------------------------

    def _start_next_leg(self) -> None:
        leg = self._pending_legs.pop(0)
        self._current_leg = leg
        self._steps_this_leg = 0

        pads = [p for p in self.bridge.net_pads() if p.net == leg.net]
        assert len(pads) >= 2, f"Net {leg.net!r} has {len(pads)} pad(s), need >= 2"
        start_pad, target_pad = pads[0], pads[1]

        self.bridge.set_mode(leg.mode)

        if leg.kind == "tune":
            geometry = self.bridge.get_board_geometry()
            target_length = self._get_net_length_nm(geometry, leg.reference_net)
            assert target_length > 0, f"Tune leg reference {leg.reference_net!r} unrouted"
            self.bridge.set_target_length(int(target_length))

            start_xy = self._get_net_midpoint(geometry, leg.net)
            assert start_xy is not None, f"Tune leg {leg.net!r} has no baseline track"
        else:
            start_xy = (start_pad.x, start_pad.y)

        candidates = self.bridge.query_hover_items(
            start_xy[0], start_xy[1], layer=0, slop_radius=SNAP_RADIUS_NM
        )
        assert candidates, f"No start candidate for {leg.net!r}"
        start_id = [c for c in candidates if c.kind in ("pad", "segment")][0].id

        assert self.bridge.start_route(start_xy[0], start_xy[1], start_id, 0), (
            f"start_route failed for {leg.net!r}"
        )

        target_candidates = self.bridge.query_hover_items(
            target_pad.x, target_pad.y, layer=0, slop_radius=SNAP_RADIUS_NM
        )
        target_id = [c for c in target_candidates if c.kind in ("pad", "segment")][0].id

        self._start_xy = start_xy
        self._target_xy = (target_pad.x, target_pad.y)
        self._target_item_id = target_id
        self._last_head_pos = start_xy
        self._last_potential = -float(
            np.hypot(
                (self._target_xy[0] - start_xy[0]) / MM,
                (self._target_xy[1] - start_xy[1]) / MM,
            )
        )

    def _re_start_current_leg(self) -> None:
        """Re-initializes the active leg after rip-up."""
        if self._current_leg is None:
            return
        leg = self._current_leg
        pads = [p for p in self.bridge.net_pads() if p.net == leg.net]
        start_pad = pads[0]
        candidates = self.bridge.query_hover_items(
            start_pad.x, start_pad.y, layer=0, slop_radius=SNAP_RADIUS_NM
        )
        start_id = [c for c in candidates if c.kind == "pad"][0].id
        self.bridge.set_mode(leg.mode)
        self.bridge.start_route(start_pad.x, start_pad.y, start_id, 0)
        self._last_head_pos = (start_pad.x, start_pad.y)

    def _observe(self) -> np.ndarray:
        if self._current_leg is None or self._target_xy is None:
            return np.zeros(12, dtype=np.float32)

        head_state = read_head_state(self.bridge)
        hx = (head_state.end_x if head_state.active else self._last_head_pos[0]) / self.board_size_nm
        hy = (head_state.end_y if head_state.active else self._last_head_pos[1]) / self.board_size_nm
        tx = self._target_xy[0] / self.board_size_nm
        ty = self._target_xy[1] / self.board_size_nm

        dx_mm = (self._target_xy[0] - (head_state.end_x if head_state.active else self._last_head_pos[0])) / MM
        dy_mm = (self._target_xy[1] - (head_state.end_y if head_state.active else self._last_head_pos[1])) / MM
        diag_mm = float(np.hypot(self.board_size_nm / MM, self.board_size_nm / MM))
        dist_norm = float(np.hypot(dx_mm, dy_mm)) / (diag_mm or 1.0)

        layer_float = float(head_state.layer)
        budget_frac = 1.0 - (self._steps_this_leg / max(1, self.max_waypoints_per_leg))

        kind_onehot = np.zeros(3, dtype=np.float32)
        kind_onehot[_KIND_INDEX[self._current_leg.kind]] = 1.0

        return np.array(
            [
                hx,
                hy,
                tx,
                ty,
                dx_mm,
                dy_mm,
                dist_norm,
                layer_float,
                budget_frac,
                kind_onehot[0],
                kind_onehot[1],
                kind_onehot[2],
            ],
            dtype=np.float32,
        )

    def _build_legs(self, pads: list) -> list[_Leg]:
        plain: set[str] = set()
        diff_pairs: dict[str, dict[str, str]] = {}
        length_groups: dict[str, dict[int, str]] = {}

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

        bridge = self._bridge_module
        legs: list[_Leg] = [_Leg("direct", net, bridge.MODE_ROUTE_SINGLE) for net in sorted(plain)]

        for base in sorted(diff_pairs):
            pair = diff_pairs[base]
            legs.append(_Leg("diff_pair", pair["P"], bridge.MODE_ROUTE_DIFF_PAIR))

        for group_idx in sorted(length_groups):
            members = length_groups[group_idx]
            ordered_nets = [members[i] for i in sorted(members)]
            ref_net = ordered_nets[0]
            legs.append(_Leg("direct", ref_net, bridge.MODE_ROUTE_SINGLE))
            for member_net in ordered_nets[1:]:
                legs.append(_Leg("direct", member_net, bridge.MODE_ROUTE_SINGLE))
                legs.append(_Leg("tune", member_net, bridge.MODE_TUNE_SINGLE, reference_net=ref_net))

        return legs

    def _get_net_length_nm(self, geometry: Any, net: str | None) -> float:
        if not net:
            return 0.0
        return sum(
            float(np.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1))
            for seg in geometry.tracks
            if seg.net == net
        )

    def _get_net_midpoint(self, geometry: Any, net: str) -> tuple[int, int] | None:
        for seg in geometry.tracks:
            if seg.net == net:
                return (seg.x1 + seg.x2) // 2, (seg.y1 + seg.y2) // 2
        return None
