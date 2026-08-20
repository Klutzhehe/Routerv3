"""Gym env wrapping pcbworld_pns_bridge's diff-pair and length-tuning
primitives (PNS_BRIDGE::SetMode's MODE_ROUTE_DIFF_PAIR/MODE_TUNE_SINGLE --
see ROADMAP.md item 7, Colab-verified against real geometry).

Untested against a real bridge -- pcbworld_pns_bridge only builds inside
the Colab flow (see ROADMAP.md); this has never been run end to end, same
caveat as pcb_route_env.py/simple_route_env.py. Locally verified only
against tests/fake_bridge.py (Python control flow, not real router
behavior) via tests/test_diff_pair_route_env.py.

Design (ROADMAP.md's "novel SOTA agent" section): the agent does NOT learn
diff-pair coupling or meander geometry from raw pushes -- PNS's own
placers already do that. This env instead exposes those placers as
higher-level routing *primitives* the agent invokes per net/net-group,
while still using the same low-level (dx, dy) push action as
SimpleRouteEnv/PCBRouteEnv to steer the *driving* net toward its target --
the router's own diff-pair coupling / meander insertion handles the rest.

Consumes pcbworld/data/generate_board.py's net-name convention directly:
  - "net_<i>"                 -> a direct MODE_ROUTE_SINGLE leg.
  - "diffpair_<i>_P"/"_N"     -> one MODE_ROUTE_DIFF_PAIR leg, driven by
                                  the P leg; PNS finds N via net-name
                                  matching (confirmed in Colab -- see
                                  ROADMAP.md item 7).
  - "lengthgrp_<g>_<member>"  -> the group's first member (lowest index)
                                  routes as a direct leg and becomes the
                                  length reference; every other member
                                  gets a direct leg (establishes a
                                  straight baseline track) immediately
                                  followed by a MODE_TUNE_SINGLE leg that
                                  re-opens that same track and tunes it
                                  toward the reference's actual routed
                                  length (read back via
                                  get_board_geometry(), not assumed).

Net/leg sequencing within each of the three groups is fixed (sorted by
index) rather than caller-supplied -- unlike PCBRouteEnv's net_order,
ROADMAP.md's net-ordering meta-policy question doesn't have an answer yet
for how diff-pair/tune legs should interleave with plain nets, so this
deliberately doesn't pretend to support arbitrary ordering yet.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MM = 1_000_000  # KiCad internal units are nm; 1mm = 1e6 nm
SNAP_RADIUS_NM = int(0.5 * MM)

# Leg kind one-hot order for the observation vector's last 3 dims.
_KIND_INDEX = {"direct": 0, "diff_pair": 1, "tune": 2}


@dataclasses.dataclass
class RewardWeights:
    wirelength: float = 1.0e-6
    via: float = 5.0
    drc_violation: float = 10.0
    net_finished: float = 20.0
    step_penalty: float = 0.05
    invalid_move: float = 0.1
    # Only applied at a tune leg's finish: penalizes the gap between the
    # tuned net's actual length and its group reference's length.
    length_mismatch: float = 1.0e-6  # per nm of |actual - target|


@dataclasses.dataclass
class _Leg:
    kind: str  # "direct" | "diff_pair" | "tune"
    net: str  # the net whose pads define this leg's start/target
    mode: int  # PNS::ROUTER_MODE constant
    reference_net: str | None = None  # "tune" legs only


def _parse_board_nets(pads) -> tuple[list[str], dict[str, dict[str, str]], dict[str, dict[int, str]]]:
    """Groups net_pads() output by generate_board.py's naming convention.

    Returns (plain_nets, diff_pairs, length_groups) where diff_pairs maps
    "diffpair_<i>" -> {"P": netname, "N": netname} and length_groups maps
    "<g>" -> {member_index: netname}.
    """
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

    return sorted(plain), diff_pairs, length_groups


def _build_legs(pads) -> list[_Leg]:
    plain, diff_pairs, length_groups = _parse_board_nets(pads)

    # Deferred import -- see module docstring on why this can't be a
    # top-level import.
    import pcbworld_pns_bridge as bridge

    legs: list[_Leg] = [_Leg("direct", net, bridge.MODE_ROUTE_SINGLE) for net in plain]

    for base in sorted(diff_pairs):
        pair = diff_pairs[base]
        assert "P" in pair and "N" in pair, f"diff pair {base!r} missing a P or N leg: {pair}"
        legs.append(_Leg("diff_pair", pair["P"], bridge.MODE_ROUTE_DIFF_PAIR))

    for group_idx in sorted(length_groups):
        members = length_groups[group_idx]
        ordered_nets = [members[i] for i in sorted(members)]
        assert len(ordered_nets) >= 2, (
            f"length-matched group {group_idx!r} has {len(ordered_nets)} member(s), "
            f"need at least 2 for length matching to mean anything"
        )
        reference_net = ordered_nets[0]
        legs.append(_Leg("direct", reference_net, bridge.MODE_ROUTE_SINGLE))
        for member_net in ordered_nets[1:]:
            legs.append(_Leg("direct", member_net, bridge.MODE_ROUTE_SINGLE))
            legs.append(
                _Leg("tune", member_net, bridge.MODE_TUNE_SINGLE, reference_net=reference_net)
            )

    return legs


def _segment_length_nm(seg) -> float:
    return float(np.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1))


def _net_length_nm(geometry, net: str) -> float:
    """Sums every committed segment's length for `net`. Straight-line
    approximation for arcs (matches TrackSegment's own documented
    simplification, pns_bridge.h)."""
    return sum(_segment_length_nm(seg) for seg in geometry.tracks if seg.net == net)


def _net_midpoint(geometry, net: str) -> tuple[int, int] | None:
    """First committed segment's midpoint for `net`, or None if it hasn't
    been routed yet. Used to re-acquire an already-routed straight net as
    a PNS item for a tune leg's start_route() -- mirrors ROADMAP.md item
    7's Colab-verified tune-mode test, which grabbed the segment via a
    query_hover_items() at its midpoint rather than at a pad."""
    for seg in geometry.tracks:
        if seg.net == net:
            return (seg.x1 + seg.x2) // 2, (seg.y1 + seg.y2) // 2
    return None


class DiffPairRouteEnv(gym.Env):
    """Routes one leg per step-batch; a full episode sequences every leg
    generate_board.py's diff-pair/length-matched-group net naming implies
    (see module docstring).

    Action: Box(shape=(4,), low=-1, high=1) -- identical layout to
      PCBRouteEnv: [0:2] push delta (x, y) on the leg's driving net,
      [2] fix threshold, [3] via threshold.

    Observation: Box(shape=(8,)) -- PCBRouteEnv's 5-vector
      ([dx_to_target_mm, dy_to_target_mm, progress_fraction, via_count,
      drc_errors_last_check]) plus a 3-dim one-hot for the current leg's
      kind (direct/diff_pair/tune), so a shared policy trunk can condition
      on which primitive is active without needing the full two-stream
      graph+raster encoder ROADMAP.md describes (that's a separate,
      not-yet-built piece -- this one-hot is the minimum needed to make
      the existing PPO baseline's MLP primitive-aware).

    Reward: PCBRouteEnv's shaping (wirelength, via, DRC) plus, for tune
      legs only, a penalty on the gap between the tuned net's actual
      length (read back via get_board_geometry() after commit) and its
      group reference's length -- not assumed to be zero just because
      set_target_length() was called with that value.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        board_path: str,
        track_width_nm: int = 250_000,
        via_diameter_nm: int = 600_000,
        via_drill_nm: int = 300_000,
        diff_pair_gap_nm: int = 150_000,
        diff_pair_width_nm: int = 200_000,
        meander_max_amplitude_nm: int = 2_500_000,
        meander_spacing_nm: int = 1_000_000,
        step_size_nm: int = 200_000,
        max_steps_per_leg: int = 60,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        super().__init__()

        # Deferred import -- see module docstring.
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
        self.max_steps_per_leg = max_steps_per_leg
        self.reward_weights = reward_weights or RewardWeights()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self._board_loaded = False
        self._pending_legs: list[_Leg] = []
        self._current_leg: _Leg | None = None
        self._target_xy: tuple[int, int] | None = None
        self._target_item_id: int = -1
        self._start_xy: tuple[int, int] | None = None
        self._pos_xy: tuple[int, int] | None = None
        self._steps_this_leg = 0
        self._via_count = 0
        self._violations_last_check = 0

    # -- gymnasium.Env interface -----------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        if not self._board_loaded:
            assert self.bridge.load_board(self.board_path), (
                f"load_board failed: {self.board_path}"
            )
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

        pads = self.bridge.net_pads()
        self._pending_legs = _build_legs(pads)
        assert self._pending_legs, f"no routable nets found on board: {self.board_path}"

        self._via_count = 0
        self._violations_last_check = 0

        self._start_next_leg()
        return self._observe(), {"net": self._current_leg.net, "kind": self._current_leg.kind}

    def step(self, action: np.ndarray):
        assert self._current_leg is not None, (
            "step() called with no leg in progress -- call reset() first"
        )

        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        weights = self.reward_weights
        reward = -weights.step_penalty
        terminated = False
        truncated = False
        info: dict[str, Any] = {"net": self._current_leg.net, "kind": self._current_leg.kind}

        if action[3] > 0:
            self.bridge.toggle_via_placement()
            self._via_count += 1
            reward -= weights.via

        new_x = int(self._pos_xy[0] + action[0] * self.step_size_nm)
        new_y = int(self._pos_xy[1] + action[1] * self.step_size_nm)

        if self.bridge.push(new_x, new_y, -1):
            step_len = float(np.hypot(new_x - self._pos_xy[0], new_y - self._pos_xy[1]))
            reward -= weights.wirelength * step_len
            self._pos_xy = (new_x, new_y)
        else:
            reward -= weights.invalid_move

        self._steps_this_leg += 1
        leg_finished = False

        if action[2] > 0:
            if self.bridge.fix(
                self._target_xy[0], self._target_xy[1], self._target_item_id, True, True
            ):
                leg_finished = True
                reward += weights.net_finished

        if not leg_finished and self._steps_this_leg >= self.max_steps_per_leg:
            self.bridge.stop_routing()
            truncated = True
            leg_finished = True

        if leg_finished:
            self.bridge.commit_routing()
            violations = self.bridge.run_drc()
            errors = sum(1 for v in violations if v.severity == "error")
            reward -= weights.drc_violation * max(0, errors - self._violations_last_check)
            self._violations_last_check = errors
            info["drc_errors"] = errors

            if self._current_leg.kind == "tune":
                geometry = self.bridge.get_board_geometry()
                target_length = _net_length_nm(geometry, self._current_leg.reference_net)
                actual_length = _net_length_nm(geometry, self._current_leg.net)
                mismatch = abs(actual_length - target_length)
                reward -= weights.length_mismatch * mismatch
                info["length_mismatch_nm"] = mismatch

            if self._pending_legs:
                self._start_next_leg()
            else:
                if not truncated:
                    terminated = True
                self._current_leg = None

        return self._observe(), reward, terminated, truncated, info

    def close(self) -> None:
        pass

    # -- internals ---------------------------------------------------------

    def _start_next_leg(self) -> None:
        leg = self._pending_legs.pop(0)
        self._current_leg = leg
        self._steps_this_leg = 0

        pads = [p for p in self.bridge.net_pads() if p.net == leg.net]
        assert len(pads) >= 2, (
            f"net {leg.net!r} has {len(pads)} pad(s) -- two-terminal routing "
            f"needs at least 2 to route between"
        )
        start_pad, target_pad = pads[0], pads[1]

        self.bridge.set_mode(leg.mode)

        if leg.kind == "tune":
            geometry = self.bridge.get_board_geometry()
            target_length = _net_length_nm(geometry, leg.reference_net)
            assert target_length > 0, (
                f"tune leg for {leg.net!r} needs its reference net "
                f"{leg.reference_net!r} already routed with nonzero length"
            )
            self.bridge.set_target_length(int(target_length))

            start_xy = _net_midpoint(geometry, leg.net)
            assert start_xy is not None, (
                f"tune leg for {leg.net!r} needs it already routed (a prior "
                f"direct leg for the same net) so there's a segment to re-open"
            )
        else:
            start_xy = (start_pad.x, start_pad.y)

        candidates = self.bridge.query_hover_items(
            start_xy[0], start_xy[1], layer=0, slop_radius=SNAP_RADIUS_NM
        )
        assert candidates, f"no candidate at start point for leg {leg.net!r} ({start_xy})"

        assert self.bridge.start_route(start_xy[0], start_xy[1], candidates[0].id, 0), (
            f"start_route failed for leg {leg.net!r} (kind={leg.kind})"
        )

        target_candidates = self.bridge.query_hover_items(
            target_pad.x, target_pad.y, layer=0, slop_radius=SNAP_RADIUS_NM
        )
        assert target_candidates, f"no candidate at target pad for leg {leg.net!r}"

        self._start_xy = start_xy
        self._target_xy = (target_pad.x, target_pad.y)
        self._target_item_id = target_candidates[0].id
        self._pos_xy = start_xy

    def _observe(self) -> np.ndarray:
        kind_onehot = np.zeros(3, dtype=np.float32)

        if self._pos_xy is None or self._target_xy is None or self._current_leg is None:
            return np.concatenate([np.zeros(5, dtype=np.float32), kind_onehot])

        kind_onehot[_KIND_INDEX[self._current_leg.kind]] = 1.0

        dx = (self._target_xy[0] - self._pos_xy[0]) / MM
        dy = (self._target_xy[1] - self._pos_xy[1]) / MM
        start_dist = (
            float(
                np.hypot(
                    (self._target_xy[0] - self._start_xy[0]) / MM,
                    (self._target_xy[1] - self._start_xy[1]) / MM,
                )
            )
            or 1.0
        )
        progress = 1.0 - min(1.0, float(np.hypot(dx, dy)) / start_dist)

        base = np.array(
            [dx, dy, progress, float(self._via_count), float(self._violations_last_check)],
            dtype=np.float32,
        )
        return np.concatenate([base, kind_onehot])
