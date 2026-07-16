"""Gym-style env wrapping pcbworld_pns_bridge for RL routing.

Untested against a real bridge -- pcbworld_pns_bridge only builds inside
the Colab flow in notebooks/00_setup.ipynb (see ROADMAP.md), so this has
never been run end to end. Written directly against the bound API in
pcbworld/engine/cpp/bindings.cpp (query_hover_items/start_route/push/fix/
commit_routing/reset/net_pads/run_drc), same as the notebook's routing/DRC
cells; the next verification step is running an episode in Colab and
fixing whatever the real router's behavior disagrees with here.

Constraints this respects (see docs/performance.md, ROADMAP.md):
 - One PNSBridge instance per OS process. This env only ever imports
   pcbworld_pns_bridge, never system pcbnew -- vectorize across envs with
   multiprocessing (e.g. gymnasium.vector.AsyncVectorEnv), not threads.
 - The bridge import is deferred into __init__ so importing this module
   somewhere the bridge isn't built (e.g. a local editor's type checker)
   doesn't hard-fail.

Net sequencing (`net_order`) is left as a caller-supplied parameter rather
than decided here -- ROADMAP.md flags net ordering as "itself a design
decision for the env", i.e. a place a meta-policy could plug in later
(see ROADMAP.md's "net-ordering meta-policy" candidate direction).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MM = 1_000_000  # KiCad internal units are nm; 1mm = 1e6 nm


@dataclasses.dataclass
class RewardWeights:
    wirelength: float = 1.0e-6   # per nm of push distance actually applied
    via: float = 5.0
    drc_violation: float = 10.0
    net_finished: float = 20.0
    step_penalty: float = 0.05   # per step, encourages finishing quickly
    invalid_move: float = 0.1    # extra penalty when the router refuses a push


class PCBRouteEnv(gym.Env):
    """Routes one net per "leg"; a full episode sequences every net on the board.

    Action: Box(shape=(4,), low=-1, high=1) --
      [0:2] push delta (x, y), scaled by `step_size_nm` and applied to the
            router's current point via bridge.push().
      [2]   fix threshold: > 0 attempts bridge.fix() at the target this step.
      [3]   via threshold: > 0 calls bridge.toggle_via_placement() before
            the push.

    Observation: Box(shape=(5,)) -- [dx_to_target_mm, dy_to_target_mm,
      progress_fraction, via_count_this_episode, drc_errors_last_check].
      Deliberately a small-MLP-friendly vector, not the PCBWorld paper's
      Fourier-feature geometry encoding -- the graph/transformer encoder
      question is explicitly deferred (see ROADMAP.md's "novel SOTA agent"
      candidate directions).

    Reward: potential-based shaping on wirelength, via count, and DRC
      violation count. RunDRC() only runs once per net-finish (not every
      step) -- DRC_ENGINE is a full-board check, too expensive to call on
      every push/move.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        board_path: str,
        net_order: list[str] | None = None,
        track_width_nm: int = 250_000,
        via_diameter_nm: int = 600_000,
        via_drill_nm: int = 300_000,
        step_size_nm: int = 200_000,
        max_steps_per_net: int = 60,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        super().__init__()

        # Deferred import -- see module docstring.
        import pcbworld_pns_bridge as bridge

        self._bridge_module = bridge
        self.bridge = bridge.PNSBridge()
        self.board_path = board_path
        self.net_order = net_order  # None => discovery order from net_pads()
        self.track_width_nm = track_width_nm
        self.via_diameter_nm = via_diameter_nm
        self.via_drill_nm = via_drill_nm
        self.step_size_nm = step_size_nm
        self.max_steps_per_net = max_steps_per_net
        self.reward_weights = reward_weights or RewardWeights()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self._board_loaded = False
        self._pending_nets: list[str] = []
        self._current_net: str | None = None
        self._target_xy: tuple[int, int] | None = None
        self._target_item_id: int = -1
        self._start_xy: tuple[int, int] | None = None
        self._pos_xy: tuple[int, int] | None = None
        self._steps_this_net = 0
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
            self.bridge.reset()  # strip routing, keep footprint placement

        self.bridge.set_mode(self._bridge_module.MODE_ROUTE_SINGLE)
        self.bridge.set_track_width(self.track_width_nm)
        self.bridge.set_via_diameter(self.via_diameter_nm)
        self.bridge.set_via_drill(self.via_drill_nm)

        self._pending_nets = (
            list(self.net_order) if self.net_order else self._discover_net_order()
        )
        assert self._pending_nets, f"no nets found on board: {self.board_path}"

        self._via_count = 0
        self._violations_last_check = 0

        self._start_next_net()
        return self._observe(), {"net": self._current_net}

    def step(self, action: np.ndarray):
        assert self._current_net is not None, (
            "step() called with no net in progress -- call reset() first"
        )

        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        weights = self.reward_weights
        reward = -weights.step_penalty
        terminated = False
        truncated = False
        info: dict[str, Any] = {"net": self._current_net}

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
            # Router refused the push (geometry conflict) -- position
            # doesn't move; small extra penalty discourages requesting
            # moves the router can't satisfy.
            reward -= weights.invalid_move

        self._steps_this_net += 1
        net_finished = False

        if action[2] > 0:
            if self.bridge.fix(
                self._target_xy[0], self._target_xy[1], self._target_item_id, False, False
            ):
                net_finished = True
                reward += weights.net_finished

        if not net_finished and self._steps_this_net >= self.max_steps_per_net:
            self.bridge.stop_routing()
            truncated = True
            net_finished = True  # abandon this net and move on regardless

        if net_finished:
            self.bridge.commit_routing()
            violations = self.bridge.run_drc()
            errors = sum(1 for v in violations if v.severity == "error")
            reward -= weights.drc_violation * max(0, errors - self._violations_last_check)
            self._violations_last_check = errors
            info["drc_errors"] = errors

            if self._pending_nets:
                self._start_next_net()
            else:
                # Don't set both flags on the last net's timeout --
                # `truncated` (time-limit abandonment) already says the
                # episode ended; a caller checking `terminated` alone to
                # decide whether to bootstrap from a value estimate should
                # see a consistent signal, not both simultaneously.
                if not truncated:
                    terminated = True
                self._current_net = None

        return self._observe(), reward, terminated, truncated, info

    def close(self) -> None:
        pass

    # -- internals ---------------------------------------------------------

    def _discover_net_order(self) -> list[str]:
        seen: dict[str, None] = {}
        for pad in self.bridge.net_pads():
            if pad.net:
                seen.setdefault(pad.net, None)
        return list(seen.keys())

    def _start_next_net(self) -> None:
        self._current_net = self._pending_nets.pop(0)
        self._steps_this_net = 0

        pads = [p for p in self.bridge.net_pads() if p.net == self._current_net]
        assert len(pads) >= 2, (
            f"net {self._current_net!r} has {len(pads)} pad(s) -- two-terminal "
            f"routing needs at least 2 to route between"
        )
        start_pad, target_pad = pads[0], pads[1]

        # layer=0 (F_Cu) matches notebooks/00_setup.ipynb's toy-board
        # routing cell -- an assumption that every net's start pad sits on
        # the top copper layer, true for pcbworld/data/generate_board.py's
        # output but not a general board.
        candidates = self.bridge.query_hover_items(
            start_pad.x, start_pad.y, layer=0, slop_radius=int(0.5 * MM)
        )
        assert candidates, (
            f"no candidate found at start pad for net {self._current_net!r} "
            f"({start_pad.x}, {start_pad.y})"
        )

        assert self.bridge.start_route(start_pad.x, start_pad.y, candidates[0].id, 0), (
            f"start_route failed for net {self._current_net!r}"
        )

        # Resolve the target pad's own candidate id up front and pass it to
        # fix() (matches notebooks/00_setup.ipynb's verified routing cell:
        # `b.fix(pad_b[0].x, pad_b[0].y, pad_b[0].id, ...)`) rather than -1
        # -- fix() needs the actual target item to snap/connect to, not
        # just a coincident point.
        target_candidates = self.bridge.query_hover_items(
            target_pad.x, target_pad.y, layer=0, slop_radius=int(0.5 * MM)
        )
        assert target_candidates, (
            f"no candidate found at target pad for net {self._current_net!r} "
            f"({target_pad.x}, {target_pad.y})"
        )

        self._start_xy = (start_pad.x, start_pad.y)
        self._target_xy = (target_pad.x, target_pad.y)
        self._target_item_id = target_candidates[0].id
        self._pos_xy = (start_pad.x, start_pad.y)

    def _observe(self) -> np.ndarray:
        if self._pos_xy is None or self._target_xy is None:
            return np.zeros(5, dtype=np.float32)

        dx = (self._target_xy[0] - self._pos_xy[0]) / MM
        dy = (self._target_xy[1] - self._pos_xy[1]) / MM
        start_dist = float(
            np.hypot(
                (self._target_xy[0] - self._start_xy[0]) / MM,
                (self._target_xy[1] - self._start_xy[1]) / MM,
            )
        ) or 1.0
        progress = 1.0 - min(1.0, float(np.hypot(dx, dy)) / start_dist)

        return np.array(
            [dx, dy, progress, float(self._via_count), float(self._violations_last_check)],
            dtype=np.float32,
        )
