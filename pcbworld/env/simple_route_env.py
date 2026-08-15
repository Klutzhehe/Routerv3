"""Minimal single-net Gym env wrapping pcbworld_pns_bridge -- v1 of the RL
routing environment, deliberately smaller than pcb_route_env.py.

Kept as a separate module (rather than simplifying pcb_route_env.py in
place) so the existing multi-net env stays available as a reference/
fallback. Scope is intentionally small to minimize debugging surface: one
net per episode, one copper layer, a pure Cartesian (dx, dy) action -- no
via/layer/diff-pair logic, no multi-net sequencing. Obstacles are simply
every other net's pads; pcbworld/data/generate_board.py's existing
num_nets parameter already gives "no obstacles" at num_nets=1 and growing
point-obstacle density as num_nets increases, so no new board-generation
work was needed for this.

Untested against a real bridge -- pcbworld_pns_bridge only builds inside
the Colab flow (see ROADMAP.md); this has never been run end to end. The
next verification step is running an episode in Colab and fixing whatever
the real router's behavior disagrees with here, same as pcb_route_env.py.

Depends on PNS_BRIDGE::SetCollisionMode() (pcbworld/engine/cpp/
pns_bridge.{h,cpp}), added alongside this file and equally unverified --
see that method's doc comment for why RM_MARK_OBSTACLES (not the
ROUTING_SETTINGS default of Shove) is required for push() to be a pure
validator rather than a router that quietly repairs collisions itself.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MM = 1_000_000  # KiCad internal units are nm; 1mm = 1e6 nm
K_NEAREST = 8
SNAP_RADIUS_NM = int(0.5 * MM)


@dataclasses.dataclass
class RewardWeights:
    progress: float = 1.0e-6  # per nm of straight-line distance closed toward target
    collision: float = 0.5    # push() rejected by the router
    net_finished: float = 20.0
    step_penalty: float = 0.05
    timeout: float = 5.0      # distinct from just missing net_finished -- see below


class SimpleRouteEnv(gym.Env):
    """Routes exactly one net per episode with a 2-D Cartesian action.

    Action: Box(shape=(2,), low=-1, high=1) -- push delta (x, y), scaled by
      step_size_nm and applied to the router's current point via
      bridge.push(). Deliberately Cartesian, not (distance, angle): a polar
      action has an angle-wraparound discontinuity and a degenerate
      direction when distance is near zero, neither of which Cartesian
      deltas have.

    A route finishes automatically once the head is within SNAP_RADIUS_NM
    of the target *and* bridge.fix() confirms real connectivity -- there is
    no separate "attempt fix" action dimension to learn; the agent's only
    job is choosing where to move.

    Observation: Box(shape=(3 + 2*K_NEAREST,)) -- [dx_to_target_mm,
      dy_to_target_mm, progress_fraction] followed by the K_NEAREST nearest
      other-net pads' (rel_x_mm, rel_y_mm), sorted by distance and
      zero-padded if fewer than K_NEAREST exist. No board-wide graph, no
      route history beyond the current position -- those are later-stage
      additions once this smallest version is verified against the real
      bridge (see ROADMAP.md).

    Reward: true potential-based shaping on distance-to-target (Ng et al.
      -- provably policy-invariant and automatically revisit-safe), not a
      penalty on raw distance traveled -- the latter would penalize a
      legal detour around an obstacle more than the collision it's
      avoiding.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        board_path: str,
        net_name: str | None = None,
        track_width_nm: int = 250_000,
        step_size_nm: int = 200_000,
        max_steps: int = 80,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        super().__init__()

        # Deferred import -- see module docstring; importing this module
        # somewhere the bridge isn't built (e.g. a local editor's type
        # checker) shouldn't hard-fail.
        import pcbworld_pns_bridge as bridge

        self._bridge_module = bridge
        self.bridge = bridge.PNSBridge()
        self.board_path = board_path
        self.net_name = net_name  # None => first net discovered
        self.track_width_nm = track_width_nm
        self.step_size_nm = step_size_nm
        self.max_steps = max_steps
        self.reward_weights = reward_weights or RewardWeights()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 + 2 * K_NEAREST,), dtype=np.float32
        )

        self._board_loaded = False
        self._current_net: str | None = None
        self._target_xy: tuple[int, int] | None = None
        self._target_item_id: int = -1
        self._start_xy: tuple[int, int] | None = None
        self._pos_xy: tuple[int, int] | None = None
        self._obstacle_xy: list[tuple[int, int]] = []
        self._steps = 0
        self._prev_dist = 0.0

    # -- gymnasium.Env interface -----------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        if not self._board_loaded:
            assert self.bridge.load_board(self.board_path), (
                f"load_board failed: {self.board_path}"
            )
            self._board_loaded = True
        else:
            self.bridge.reset()  # strip routing, keep footprint placement

        self.bridge.set_mode(self._bridge_module.MODE_ROUTE_SINGLE)
        # LoadBoard() already defaults collision mode to RM_MARK_OBSTACLES
        # (see pns_bridge.cpp), but set it explicitly here too so a caller
        # that flips it to RM_SHOVE/RM_WALKAROUND for a classical-baseline
        # comparison run doesn't leave that setting stuck across a reset()
        # that reuses this same bridge instance.
        self.bridge.set_collision_mode(self._bridge_module.RM_MARK_OBSTACLES)
        self.bridge.set_track_width(self.track_width_nm)

        pads = self.bridge.net_pads()
        self._current_net = self.net_name or next(p.net for p in pads if p.net)
        net_pads = [p for p in pads if p.net == self._current_net]
        assert len(net_pads) >= 2, (
            f"net {self._current_net!r} has {len(net_pads)} pad(s) -- need "
            f"at least 2 to route between"
        )
        start_pad, target_pad = net_pads[0], net_pads[1]

        self._obstacle_xy = [(p.x, p.y) for p in pads if p.net != self._current_net]

        # layer=0 (F_Cu) matches pcb_route_env.py's and
        # notebooks/00_setup.ipynb's toy-board assumption -- true for
        # generate_board.py's output, not a general board.
        candidates = self.bridge.query_hover_items(
            start_pad.x, start_pad.y, layer=0, slop_radius=SNAP_RADIUS_NM
        )
        assert candidates, f"no candidate at start pad for net {self._current_net!r}"
        assert self.bridge.start_route(start_pad.x, start_pad.y, candidates[0].id, 0), (
            f"start_route failed for net {self._current_net!r}"
        )

        target_candidates = self.bridge.query_hover_items(
            target_pad.x, target_pad.y, layer=0, slop_radius=SNAP_RADIUS_NM
        )
        assert target_candidates, f"no candidate at target pad for net {self._current_net!r}"

        self._start_xy = (start_pad.x, start_pad.y)
        self._target_xy = (target_pad.x, target_pad.y)
        self._target_item_id = target_candidates[0].id
        self._pos_xy = (start_pad.x, start_pad.y)
        self._steps = 0
        self._prev_dist = self._dist_to_target()

        return self._observe(), {"net": self._current_net}

    def step(self, action: np.ndarray):
        assert self._pos_xy is not None, "step() called before reset()"

        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        weights = self.reward_weights
        reward = -weights.step_penalty
        terminated = False
        truncated = False
        info: dict[str, Any] = {"net": self._current_net}

        new_x = int(self._pos_xy[0] + action[0] * self.step_size_nm)
        new_y = int(self._pos_xy[1] + action[1] * self.step_size_nm)

        if self.bridge.push(new_x, new_y, -1):
            self._pos_xy = (new_x, new_y)
            new_dist = self._dist_to_target()
            # Potential-based shaping: reward progress toward the target,
            # not raw distance moved -- see class docstring.
            reward += weights.progress * (self._prev_dist - new_dist)
            self._prev_dist = new_dist
        else:
            # Router refused the segment (collision/clearance). Position
            # doesn't move; environment neither auto-corrects nor picks an
            # alternative -- it only reports the rejection.
            reward -= weights.collision
            info["collision"] = True

        self._steps += 1

        if self._dist_to_target() <= SNAP_RADIUS_NM and self.bridge.fix(
            self._target_xy[0], self._target_xy[1], self._target_item_id, False, False
        ):
            reward += weights.net_finished
            terminated = True
            self.bridge.commit_routing()
        elif self._steps >= self.max_steps:
            # Distinct penalty from simply missing net_finished, so a
            # policy can't learn "sit still and let the clock run out" as
            # a way to avoid collision penalties from genuine attempts.
            self.bridge.stop_routing()
            reward -= weights.timeout
            truncated = True

        return self._observe(), reward, terminated, truncated, info

    def close(self) -> None:
        pass

    # -- internals ---------------------------------------------------------

    def _dist_to_target(self) -> float:
        return float(
            np.hypot(
                self._target_xy[0] - self._pos_xy[0], self._target_xy[1] - self._pos_xy[1]
            )
        )

    def _observe(self) -> np.ndarray:
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

        nearest = sorted(
            self._obstacle_xy,
            key=lambda o: (o[0] - self._pos_xy[0]) ** 2 + (o[1] - self._pos_xy[1]) ** 2,
        )[:K_NEAREST]
        obs_feats = np.zeros(2 * K_NEAREST, dtype=np.float32)
        for i, (ox, oy) in enumerate(nearest):
            obs_feats[2 * i] = (ox - self._pos_xy[0]) / MM
            obs_feats[2 * i + 1] = (oy - self._pos_xy[1]) / MM

        return np.concatenate([np.array([dx, dy, progress], dtype=np.float32), obs_feats])
