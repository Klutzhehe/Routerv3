"""Gymnasium-Compatible PCB Routing Grid Environment (PCBRouterEnv).

Implements round-robin multi-net growth on a 10-channel 256x256 spatial grid:
- 10-channel spatial observation space (Box(0.0, 1.0, (10, 256, 256), float32))
- 96 discrete actions per net growth step (8 directions x 3 distances x 2 layers x 2 vias)
- Fast line-raster collision & clearance checks
- Decoupled from external push/shove heuristics
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pcbworld.board_generator import BoardState, generate_random_board, NetSpec, Pad, Obstacle
from pcbworld.congestion import compute_net_demand_heatmap, compute_geodesic_distance_field, compute_clearance_field
from pcbworld.reward import RewardCalculator


DIST_STEPS = [2, 4, 8]  # Step distances for fast grid traversal


@dataclass
class _NetState:
    """One net's own routing state.

    Round-robin gives every unfinished net one step, then rotates to the
    next, so several nets are mid-route at once -- each one's head position,
    local retry/stall history, and geodesic field have to survive the OTHER
    nets' turns in between, instead of living in singular env-level
    attributes the way they could when exactly one net was ever in progress.
    """
    head_x: int = 0
    head_y: int = 0
    head_layer: int = 0
    head_prev_dir: Optional[int] = None
    steps_taken: int = 0
    collision_run: int = 0
    # Where the last rejected move tried to land, or None if the last step
    # on THIS net was clean -- see Channel 9.
    last_rejected_pos: Optional[Tuple[int, int]] = None
    visited_cells: set = field(default_factory=set)
    restart_count: int = 0
    dead_zones: set = field(default_factory=set)
    best_dist_this_attempt: float = float("inf")
    no_progress_run: int = 0
    recent_positions: deque = field(default_factory=lambda: deque(maxlen=8))
    # Ordered (x, y, layer) waypoints for this net's current attempt, for
    # rendering/export (see PCBRouterEnv.simplify_net_path). Not read by
    # _build_observation or compute_step_reward.
    waypoints: List[Tuple[int, int, int]] = field(default_factory=list)
    # Obstacle-aware cost-to-go to THIS net's target. Depends only on the
    # target position and the (static, shared) obstacle grid -- never on
    # other nets' copper or which net is currently acting -- so it is
    # computed once when the net first starts (or restarts) and simply
    # reused every time round-robin rotates back to it, not recomputed
    # per step. See compute_geodesic_distance_field.
    geodesic_cache: Optional[np.ndarray] = None


class PCBRouterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        grid_size: int = 256,
        num_nets: int = 1,
        num_obstacles: int = 0,
        num_layers: int = 2,
        max_steps_per_net: int = 150,
        snap_radius: int = 8,
        min_pad_dist: int = 20,
        max_pad_dist: Optional[int] = None,
        seed: Optional[int] = None,
        reward_calculator: Optional[RewardCalculator] = None,
        enable_layer_via: bool = True,
        max_consecutive_collisions: Optional[int] = None,
        max_net_restarts: int = 0,
        max_no_progress_steps: int = 20,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_nets = num_nets
        self.num_obstacles = num_obstacles
        self.num_layers = num_layers
        self.max_steps_per_net = max_steps_per_net
        self.snap_radius = snap_radius
        self.min_pad_dist = min_pad_dist
        self.max_pad_dist = max_pad_dist
        # Give up on a net after this many CONSECUTIVE rejected moves, not
        # after the first one -- see the "jammed" check in step(). Defaults
        # to the full action space (24, or 96 with via/layer on) so a
        # deterministic run (select_deterministic_action, which tries a
        # DIFFERENT action each retry) actually exhausts every local
        # alternative before giving up, instead of stopping a third of the
        # way through. Measured directly: the old default of 8 left 5/5
        # tested stuck boards unsolved; a search that tries every candidate
        # before committing solves all 5 of the same boards. No backtracking
        # needed -- just not giving up early.
        self.max_consecutive_collisions = (
            max_consecutive_collisions
            if max_consecutive_collisions is not None
            else (96 if enable_layer_via else 24)
        )
        # On jam (max_consecutive_collisions exhausted), wipe this net's
        # progress and retry it from its source pad instead of abandoning it
        # outright -- see _restart_net(). Retrying single steps from the
        # SAME stuck position (above) only searches alternatives at that one
        # point; if every local option from there is bad, exhausting them
        # cannot help, because they are all evaluated from inside the same
        # trap. Restarting is a bounded, well-defined form of "back up and
        # try a different path" without needing per-step undo history.
        # Default 0 (disabled) -- opt in explicitly; existing behavior stays
        # unchanged unless requested.
        self.max_net_restarts = max_net_restarts
        # "jammed" above only fires on REJECTED moves -- a head oscillating
        # between two valid, non-colliding cells never collides, so
        # collision_run stays at 0 forever and the net just burns the whole
        # max_steps_per_net budget looping. Observed directly: a rendered
        # failed board showed the head alternating between two adjacent
        # spots near an obstacle corner, both legal, neither leading
        # anywhere. Tracked separately from collisions -- see
        # no_progress_run in step().
        self.max_no_progress_steps = max_no_progress_steps
        # board_generator.py sets tgt_layer = src_layer for these stages, and
        # the head starts on the source pad's layer -- so a policy that never
        # touches via/layer is already correctly aligned, and every toggle
        # moves it OFF the one layer where head_layer == target.layer can
        # ever be true again. Measured: with this on and undertrained, a
        # deterministic eval picked via/layer on ~75% of steps (1743 vias /
        # 50 nets) against a correct answer of zero -- matches
        # docs/RL_PLAN.md's Gate A finding that layer/via actions are not
        # viable this early and should be introduced later, not from stage 1.
        self.enable_layer_via = enable_layer_via

        self.reward_calc = reward_calculator or RewardCalculator()

        # Observation Space: (10, H, W) float32 tensor
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10, grid_size, grid_size),
            dtype=np.float32,
        )

        # Action Space: 8 directions * 3 step distances, times 4 if
        # layer/via is enabled (2 layer changes * 2 via flags), else 1.
        self.action_space = spaces.Discrete(96 if enable_layer_via else 24)

        self.board: Optional[BoardState] = None
        # Which board-index net acts on the NEXT step() call, or None once
        # every net has finished (completed or failed out of restarts).
        self.current_net_idx: Optional[int] = None
        # One _NetState per net, keyed by board index -- see _NetState.
        self.net_states: Dict[int, _NetState] = {}
        # Round-robin queue of not-yet-finished net indices. The net at the
        # front acts next; after it acts it's popped, and re-appended at the
        # back unless it just finished -- see step().
        self._active_order: deque = deque()

        self.total_steps: int = 0
        self.total_restarts_this_episode: int = 0
        self.completed_nets: List[int] = []
        self.failed_nets: List[int] = []

        self.wirelength_per_net: Dict[int, float] = {}
        self.vias_per_net: Dict[int, int] = {}
        # Frozen waypoint copy for each net that's finished -- purely for
        # rendering/export (see simplify_net_path).
        self.completed_net_paths: Dict[int, List[Tuple[int, int, int]]] = {}

        self._obs_cache: Optional[np.ndarray] = None
        self._congestion_cache: Optional[np.ndarray] = None
        self._clearance_cache: Optional[np.ndarray] = None
        self._obs_grid: Optional[np.ndarray] = None

        if seed is not None:
            self.reset(seed=seed)

    @property
    def head_x(self) -> int:
        """Convenience read for whichever net is about to act next. For
        per-net bookkeeping across step() calls (e.g. deterministic-eval
        retry avoidance), use info["acted_net_id"] / info["acted_head_pos"]
        instead -- under round-robin a DIFFERENT net is "current" on every
        call, so this property alone can't tell you what a specific net's
        head just did."""
        state = self.net_states.get(self.current_net_idx) if self.current_net_idx is not None else None
        return state.head_x if state is not None else 0

    @property
    def head_y(self) -> int:
        state = self.net_states.get(self.current_net_idx) if self.current_net_idx is not None else None
        return state.head_y if state is not None else 0

    @property
    def head_layer(self) -> int:
        state = self.net_states.get(self.current_net_idx) if self.current_net_idx is not None else None
        return state.head_layer if state is not None else 0

    def decode_action(self, action: int) -> Tuple[int, int, int, int]:
        """Decode a discrete action into (dir_idx, dist_idx, layer_change, via_flag).

        With `enable_layer_via=False` the env only ever hands out an index in
        [0, 24) and layer_change/via_flag are fixed at 0 -- there is no
        "toggle" action to accidentally learn.
        """
        if not self.enable_layer_via:
            dist_idx = action % 3
            dir_idx = action // 3
            return dir_idx, dist_idx, 0, 0
        # action = dir * 12 + dist * 4 + layer * 2 + via
        via_flag = action % 2
        rem = action // 2
        layer_change = rem % 2
        rem = rem // 2
        dist_idx = rem % 3
        dir_idx = rem // 3
        return dir_idx, dist_idx, layer_change, via_flag

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        board_seed = seed if seed is not None else np.random.randint(0, 1_000_000)
        self.board = generate_random_board(
            grid_size=self.grid_size,
            num_nets=self.num_nets,
            num_obstacles=self.num_obstacles,
            num_layers=self.num_layers,
            min_pad_dist=self.min_pad_dist,
            max_pad_dist=self.max_pad_dist,
            seed=board_seed,
        )

        self.completed_nets = []
        self.failed_nets = []
        self.wirelength_per_net = {net.net_id: 0.0 for net in self.board.nets}
        self.vias_per_net = {net.net_id: 0 for net in self.board.nets}
        self.completed_net_paths = {}
        self.total_steps = 0
        # Episode-lifetime count, unlike a net's own restart_count which is
        # per net -- lets a caller (train.py) see whether restarts are
        # firing at all, since select_deterministic_action's retry-avoidance
        # alone may resolve most jams before max_consecutive_collisions
        # triggers one.
        self.total_restarts_this_episode = 0

        # Obstacle grid must exist before any net's geodesic field can be
        # built, so this runs before _init_net_state.
        self._precompute_static_caches()

        self.net_states = {}
        for idx in range(len(self.board.nets)):
            self._init_net_state(idx)
        self._active_order = deque(range(len(self.board.nets)))
        self.current_net_idx = self._active_order[0] if self._active_order else None

        self._update_congestion_cache()

        obs = self._build_observation()
        info = self._get_info()
        return obs, info

    def _init_net_state(self, idx: int):
        """Build a fresh _NetState for the net at board index idx -- head at
        its source pad, empty retry/stall history, and this net's own
        geodesic field computed once up front (see _NetState.geodesic_cache)."""
        net = self.board.nets[idx]
        state = _NetState()
        state.head_x = net.source_pad.x
        state.head_y = net.source_pad.y
        state.head_layer = net.source_pad.layer
        state.visited_cells = {(state.head_x, state.head_y)}
        state.waypoints = [(state.head_x, state.head_y, state.head_layer)]
        state.geodesic_cache = compute_geodesic_distance_field(
            self.grid_size, net.target_pad.x, net.target_pad.y, self._obs_grid
        )
        state.best_dist_this_attempt = self._geo_dist_at(state.geodesic_cache, state.head_x, state.head_y)
        self.net_states[idx] = state

    def _restart_net(self, idx: int):
        """Wipe THIS net's progress and try it again from its source pad.

        Deliberately does NOT touch steps_taken or restart_count -- those
        are what bound how many restarts a net gets before max_steps_per_net
        or max_net_restarts ends it for real, same overall budget, just a
        fresh attempt within it.

        Also deliberately does NOT reset dead_zones -- a restart with an
        otherwise identical starting observation and an empty local
        retry-history would make a DETERMINISTIC policy retrace the exact
        same path into the exact same jam, every time, for zero benefit.
        Recording the trap location HERE, before it's wiped below, is what
        makes a restart actually different from the attempt that just
        failed: the observation the restarted attempt sees, once it
        re-approaches this spot, is not identical to what the failed
        attempt saw there.
        """
        net = self.board.nets[idx]
        state = self.net_states[idx]
        self.total_restarts_this_episode += 1
        state.dead_zones.update(state.recent_positions)
        state.dead_zones.add((state.head_x, state.head_y))
        if state.last_rejected_pos is not None:
            state.dead_zones.add(state.last_rejected_pos)
        state.recent_positions = deque(maxlen=8)
        for layer_grid in self.board.copper_grid:
            layer_grid[layer_grid == net.net_id] = 0
        self.wirelength_per_net[net.net_id] = 0.0
        self.vias_per_net[net.net_id] = 0
        state.head_x = net.source_pad.x
        state.head_y = net.source_pad.y
        state.head_layer = net.source_pad.layer
        state.head_prev_dir = None
        state.collision_run = 0
        state.last_rejected_pos = None
        state.visited_cells = {(state.head_x, state.head_y)}
        state.waypoints = [(state.head_x, state.head_y, state.head_layer)]
        state.best_dist_this_attempt = self._geo_dist_at(state.geodesic_cache, state.head_x, state.head_y)
        state.no_progress_run = 0

    def _precompute_static_caches(self):
        """Precompute obstacle masks and clearance field. Congestion is NOT
        computed here -- it depends on current_net_idx (to exclude whichever
        net is about to act), which isn't set yet at this point in reset();
        the caller computes it explicitly once net state exists."""
        obs_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for obs in self.board.obstacles:
            x1 = max(0, obs.x1)
            x2 = min(self.grid_size, obs.x2)
            y1 = max(0, obs.y1)
            y2 = min(self.grid_size, obs.y2)
            obs_grid[y1:y2, x1:x2] = 1.0

        self._clearance_cache = compute_clearance_field(self.grid_size, obs_grid)
        self._obs_grid = obs_grid

    def _geo_dist_at(self, geodesic_cache: Optional[np.ndarray], x: float, y: float) -> float:
        if geodesic_cache is None:
            return 0.0
        gx = int(np.clip(round(x), 0, self.grid_size - 1))
        gy = int(np.clip(round(y), 0, self.grid_size - 1))
        return float(geodesic_cache[gy, gx])

    def _geo_descent_dir(self, geodesic_cache: Optional[np.ndarray], x: float, y: float) -> Tuple[float, float]:
        """Which way the cost-to-go field decreases fastest -- the direction
        that walks around whatever is in the way instead of into it."""
        if geodesic_cache is None:
            return (0.0, 0.0)
        gx = int(np.clip(round(x), 1, self.grid_size - 2))
        gy = int(np.clip(round(y), 1, self.grid_size - 2))
        ddx = geodesic_cache[gy, gx - 1] - geodesic_cache[gy, gx + 1]
        ddy = geodesic_cache[gy - 1, gx] - geodesic_cache[gy + 1, gx]
        norm = math.hypot(ddx, ddy)
        if norm < 1e-6:
            return (0.0, 0.0)
        return (ddx / norm, ddy / norm)

    def _relative_direction_vector(
        self, geodesic_cache: Optional[np.ndarray], dir_idx: int, x: float, y: float, target_x: float, target_y: float
    ) -> Tuple[float, float]:
        """dir_idx=0 means 'toward the target' (or around whatever the
        geodesic field says is in the way, when one exists) -- the same free
        baseline `pcbworld/env/line_route_env.py`'s bearing-relative frame
        gives the vector agent: a policy that has not learned anything yet
        still walks toward the target by construction, instead of having to
        first infer, from raw pixels, which of 8 FIXED compass directions
        happens to point at wherever this episode's target landed. The old
        DIR_VECTORS table was exactly that -- board-pose-dependent, so the
        "correct" action for dir_idx=0 changed every episode and had to be
        relearned from image content each time. This makes it stationary.

        dir_idx counts 45-degree steps around from there, covering the same
        full circle DIR_VECTORS did.
        """
        gdx, gdy = self._geo_descent_dir(geodesic_cache, x, y) if geodesic_cache is not None else (0.0, 0.0)
        if gdx == 0.0 and gdy == 0.0:
            gdx, gdy = target_x - x, target_y - y
            norm = math.hypot(gdx, gdy)
            gdx, gdy = (gdx / norm, gdy / norm) if norm > 1e-6 else (1.0, 0.0)
        bearing = math.atan2(gdy, gdx)
        angle = bearing + dir_idx * (math.pi / 4.0)
        return math.cos(angle), math.sin(angle)

    def _update_congestion_cache(self):
        """Demand heatmap for space OTHER nets will need -- not this one.

        Measured: with the active net included in `unrouted`, its own
        heatmap peaks at ~0.99-1.0 directly on its OWN straight-line
        corridor (it demands the space it is about to walk through, same as
        every other unrouted net would) and reads lower just off to the
        side. congestion_penalty_scale then charges MORE for staying on the
        direct path to the target than for weaving off it -- a real,
        measured incentive to zigzag, not a training artifact. Matches
        docs/AI_ARCHITECTURE.md's reserve-plane rule: this signal is for
        space LATER nets will need, and must not tax the net currently
        being routed for occupying its own path.

        Recomputed every step under round-robin (cheap -- downsampled,
        vectorized, see compute_net_demand_heatmap) rather than only at net
        transitions, since both "which nets are unrouted" and "which net is
        excluded as active" can change on every single step now.
        """
        active_id = (
            self.board.nets[self.current_net_idx].net_id
            if self.current_net_idx is not None
            else None
        )
        unrouted = [
            net for net in self.board.nets
            if net.net_id not in self.completed_nets and net.net_id != active_id
        ]
        obs_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for obs in self.board.obstacles:
            obs_mask[obs.y1:obs.y2, obs.x1:obs.x2] = 1.0
        self._congestion_cache = compute_net_demand_heatmap(self.grid_size, unrouted, obs_mask)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        idx = self.current_net_idx
        if idx is None:
            raise RuntimeError("step() called with no active net -- check terminated/truncated before calling again")
        state = self.net_states[idx]
        active_net = self.board.nets[idx]
        target = active_net.target_pad

        dir_idx, dist_idx, layer_change, via_flag = self.decode_action(action)
        step_dist = DIST_STEPS[dist_idx]

        prev_x, prev_y, prev_layer = state.head_x, state.head_y, state.head_layer
        dir_x, dir_y = self._relative_direction_vector(state.geodesic_cache, dir_idx, prev_x, prev_y, target.x, target.y)
        # Obstacle-aware cost-to-go, not straight-line -- a straight-line
        # potential pays off walking INTO whatever is in the way right up
        # until the collision fires. See compute_geodesic_distance_field.
        prev_dist = self._geo_dist_at(state.geodesic_cache, prev_x, prev_y)

        # 1. Execute via / layer change
        is_via = False
        if via_flag == 1 or layer_change == 1:
            new_layer = 1 - prev_layer  # Switch between Layer 0 and Layer 1
            is_via = True
            state.head_layer = new_layer
            self.vias_per_net[active_net.net_id] += 1

        # 2. Compute new head coordinate. Rounded to int: _rasterize_line's
        # Bresenham walk increments by whole cells and terminates on
        # x == x1 and y == y1 -- a float target it can step past without
        # ever hitting exactly would loop forever.
        new_x = int(round(prev_x + dir_x * step_dist))
        new_y = int(round(prev_y + dir_y * step_dist))

        step_len = math.hypot(new_x - prev_x, new_y - prev_y)
        state.steps_taken += 1
        self.total_steps += 1

        # 3. Check for boundary collision
        out_of_bounds = (
            new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size
        )

        # 4. Check obstacle and copper collision along raster line
        is_collided = out_of_bounds
        if not is_collided:
            is_collided = self._check_line_collision(
                prev_x, prev_y, new_x, new_y, state.head_layer, active_net.net_id
            )

        # 5. Check if target pad reached -- physical proximity, so Euclidean
        # is correct here even though the reward below uses the geodesic
        # distance (near the pad the two coincide anyway).
        curr_dist = math.hypot(new_x - target.x, new_y - target.y)
        curr_dist_geo = self._geo_dist_at(state.geodesic_cache, new_x, new_y)
        is_connected = False

        is_revisit = False
        if not is_collided:
            # Draw line segment into copper grid
            self._rasterize_line(
                prev_x, prev_y, new_x, new_y, state.head_layer, active_net.net_id
            )
            # Revisit check: did this step cross any cell THIS net has
            # already been through (excluding prev_x,prev_y itself, which is
            # trivially "already visited" by definition). A curving detour
            # around an obstacle does not re-cross its own earlier path, so
            # this only fires on genuine backtracking/looping.
            new_points = self._get_line_points(prev_x, prev_y, new_x, new_y)[1:]
            is_revisit = any(p in state.visited_cells for p in new_points)
            state.visited_cells.update(new_points)
            state.head_x = new_x
            state.head_y = new_y
            state.recent_positions.append((new_x, new_y))
            state.waypoints.append((new_x, new_y, state.head_layer))
            self.wirelength_per_net[active_net.net_id] += step_len

            # Progress tracking -- separate from collisions entirely.
            # Oscillating between two valid, non-colliding cells never
            # collides, so it would otherwise be invisible to "jammed"
            # below and just burn the whole step budget looping. 1.0-cell
            # tolerance for float/geodesic noise, not a real threshold.
            if curr_dist_geo < state.best_dist_this_attempt - 1.0:
                state.best_dist_this_attempt = curr_dist_geo
                state.no_progress_run = 0
            else:
                state.no_progress_run += 1

            if curr_dist <= self.snap_radius and (state.head_layer == target.layer):
                is_connected = True
                # Connect directly to target pad center
                self._rasterize_line(
                    state.head_x, state.head_y, target.x, target.y, target.layer, active_net.net_id
                )
                state.waypoints.append((target.x, target.y, target.layer))
                self.completed_net_paths[active_net.net_id] = list(state.waypoints)
                self.completed_nets.append(active_net.net_id)

        # Check bend
        is_bend = (state.head_prev_dir is not None) and (state.head_prev_dir != dir_idx)
        state.head_prev_dir = dir_idx

        # Heading alignment against the field's descent direction, not the
        # straight bearing to the pad -- this is what actually rewards
        # circling an obstacle instead of pushing into it. Falls back to the
        # straight bearing only where the field has no local gradient (e.g.
        # against the board edge).
        act_norm = math.hypot(dir_x, dir_y)
        gdx, gdy = self._geo_descent_dir(state.geodesic_cache, prev_x, prev_y)
        if act_norm > 1e-4 and (abs(gdx) > 1e-6 or abs(gdy) > 1e-6):
            heading_alignment = float((gdx * dir_x + gdy * dir_y) / act_norm)
        else:
            dx_tgt = target.x - prev_x
            dy_tgt = target.y - prev_y
            tgt_norm = math.hypot(dx_tgt, dy_tgt)
            heading_alignment = float((dx_tgt * dir_x + dy_tgt * dir_y) / (tgt_norm * act_norm)) if (tgt_norm > 1e-4 and act_norm > 1e-4) else 0.0

        # Congestion overlap penalty
        cong_overlap = 0.0
        if self._congestion_cache is not None and not out_of_bounds:
            cong_overlap = float(self._congestion_cache[new_y, new_x])

        # Compute Step Reward
        reward, _breakdown = self.reward_calc.compute_step_reward(
            prev_dist=prev_dist,
            curr_dist=curr_dist_geo,
            step_len=step_len,
            heading_alignment=heading_alignment,
            is_connected=is_connected,
            is_collided=is_collided,
            is_bend=is_bend,
            is_via=is_via,
            congestion_overlap=cong_overlap,
            is_revisit=is_revisit,
        )

        # Collision is a rejected move and a penalty, not an instant kill.
        # Measured directly: a scripted policy that always follows the
        # geodesic field's descent direction still hit obstacle corners
        # ~20-25% of the time (coarse 8-direction/3-distance action space
        # meeting real geometry), and EVERY one of those failures ended the
        # net in under 20 steps, 37-125 cells from the target -- one bad
        # step, no chance to try a different angle from the same spot. Same
        # failure line_route_env.py already measured and fixed for the
        # vector env ("a colliding head is not jammed, it is contouring").
        # Only give up once actually stuck: several consecutive rejected
        # moves in a row, not one.
        state.collision_run = state.collision_run + 1 if is_collided else 0
        stalled = state.no_progress_run >= self.max_no_progress_steps
        jammed = (state.collision_run >= self.max_consecutive_collisions) or stalled
        # WHERE the rejection landed, for Channel 9 -- cleared the instant a
        # move succeeds, so this is strictly "what just happened", not a
        # lingering mark.
        state.last_rejected_pos = (new_x, new_y) if is_collided else None

        # Every local option from the stuck position has now been tried and
        # failed (see max_consecutive_collisions). Restarting from the
        # source pad is a bounded way to "back up and try a different path"
        # instead of only ever searching alternatives from inside the same
        # trap -- see _restart_net(). Still bounded by max_steps_per_net
        # below, so this cannot run forever.
        if jammed and state.restart_count < self.max_net_restarts:
            state.restart_count += 1
            self._restart_net(idx)
            jammed = False

        # Transition logic
        net_timeout = state.steps_taken >= self.max_steps_per_net
        net_done = is_connected or jammed or net_timeout

        if net_done and not is_connected:
            self.failed_nets.append(active_net.net_id)

        acted_net_id = active_net.net_id
        acted_head_pos = (state.head_x, state.head_y, state.head_layer)

        # Round-robin: this net's turn is over regardless of net_done -- if
        # it isn't finished, it goes to the BACK of the queue instead of
        # getting another immediate turn, so every active net grows
        # interleaved rather than one being fully resolved before the next
        # even starts.
        self._active_order.popleft()
        if not net_done:
            self._active_order.append(idx)

        terminated = len(self._active_order) == 0
        truncated = False

        if not terminated:
            self.current_net_idx = self._active_order[0]
            self._update_congestion_cache()
        else:
            self.current_net_idx = None

        obs = self._build_observation()
        info = self._get_info()
        # What THIS step actually did, independent of whichever net is now
        # "current" for the next call -- a caller doing per-net bookkeeping
        # across steps (e.g. deterministic-eval retry avoidance) needs this,
        # not head_x/head_y, since a different net is current every call.
        info["acted_net_id"] = acted_net_id
        info["acted_head_pos"] = acted_head_pos
        return obs, reward, terminated, truncated, info

    def _check_line_collision(
        self, x0: int, y0: int, x1: int, y1: int, layer: int, net_id: int
    ) -> bool:
        """Bresenham line raster check for obstacle & foreign copper collisions."""
        points = self._get_line_points(x0, y0, x1, y1)
        for px, py in points:
            if px < 0 or px >= self.grid_size or py < 0 or py >= self.grid_size:
                return True
            # Check obstacles
            for obs in self.board.obstacles:
                if (obs.layer == -1 or obs.layer == layer) and (
                    obs.x1 <= px <= obs.x2 and obs.y1 <= py <= obs.y2
                ):
                    return True
            # Check foreign copper
            existing_net = self.board.copper_grid[layer, py, px]
            if existing_net != 0 and existing_net != net_id:
                return True
            # Check foreign pads
            for net in self.board.nets:
                if net.net_id != net_id:
                    for pad in [net.source_pad, net.target_pad]:
                        if pad.layer == layer and math.hypot(px - pad.x, py - pad.y) <= pad.radius:
                            return True
        return False

    def _rasterize_line(self, x0: int, y0: int, x1: int, y1: int, layer: int, net_id: int):
        points = self._get_line_points(x0, y0, x1, y1)
        for px, py in points:
            if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                self.board.copper_grid[layer, py, px] = net_id

    def _get_line_points(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points

    @staticmethod
    def _canonical_corner(x0: int, y0: int, x1: int, y1: int) -> Optional[Tuple[int, int]]:
        """The single corner point connecting (x0,y0) to (x1,y1) using only
        horizontal/vertical/45-degree segments -- diagonal first for
        whichever axis has less distance to cover, then straight for the
        remainder on the other axis. None if the direct segment is already
        canonical (dx==0, dy==0, or |dx|==|dy|) and needs no corner at all.

        Real PCB traces are conventionally built from exactly these angles,
        never an arbitrary-angle cut -- and since raw routed step deltas are
        essentially never already canonical (see simplify_net_path), a
        shortcut has to go through a corner like this one to look like a
        trace instead of a diagonal ruler-line.
        """
        dx, dy = x1 - x0, y1 - y0
        if dx == 0 or dy == 0 or abs(dx) == abs(dy):
            return None
        diag = min(abs(dx), abs(dy))
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        return (x0 + diag * sx, y0 + diag * sy)

    def simplify_net_path(self, net_id: int) -> List[Tuple[int, int, int]]:
        """Collapse a completed net's raw stepped waypoints into a minimal
        set of straight, canonical-angle segments, for rendering/export only.

        The raw path in completed_net_paths is exactly what the policy
        walked -- correct, but its "toward target" direction is re-derived
        every step from a coarse, downsampled geodesic field gradient (see
        _geo_descent_dir), which wobbles even along what should be a
        straight run, AND dir_idx is an offset from that continuously
        varying bearing (not one of a fixed set of compass directions -- see
        _relative_direction_vector), so raw step deltas are essentially
        never exactly axis/diagonal-aligned. This does not touch the copper
        grid, reward, or any training-facing state -- it's a pure post-hoc
        geometry cleanup, the same separation real grid autorouters draw
        between "found a valid path" (the RL policy's job) and "looks like
        a manufacturable trace" (this).

        Greedy farthest line-of-sight shortcut: from each kept waypoint,
        try the farthest later waypoint first and work backward, connecting
        through _canonical_corner's single corner point (or directly, if
        already canonical) -- accept the first candidate where every
        resulting segment is collision-free (obstacles, foreign copper,
        foreign pads -- via _check_line_collision, an exact Bresenham raster
        check). Falls back to the immediate next waypoint (always safe --
        it's literally the step the policy already took) if no farther
        shortcut clears. Every accepted segment is therefore both
        provably valid and drawn at one of the router's canonical angles.
        """
        raw = self.completed_net_paths.get(net_id)
        if not raw or len(raw) < 3:
            return list(raw) if raw else []

        simplified = [raw[0]]
        i = 0
        n = len(raw)
        while i < n - 1:
            x0, y0, l0 = raw[i]
            chosen = i + 1
            corner_point: Optional[Tuple[int, int, int]] = None
            for j in range(n - 1, i, -1):
                x1, y1, l1 = raw[j]
                if l1 != l0:
                    continue
                corner = self._canonical_corner(x0, y0, x1, y1)
                if corner is None:
                    if self._check_line_collision(x0, y0, x1, y1, l0, net_id):
                        continue
                    chosen, corner_point = j, None
                    break
                cx, cy = corner
                if self._check_line_collision(x0, y0, cx, cy, l0, net_id):
                    continue
                if self._check_line_collision(cx, cy, x1, y1, l0, net_id):
                    continue
                chosen, corner_point = j, (cx, cy, l0)
                break
            if corner_point is not None:
                simplified.append(corner_point)
            simplified.append(raw[chosen])
            i = chosen
        return simplified

    def _build_observation(self) -> np.ndarray:
        """Construct the (10, 256, 256) spatial observation tensor, for
        whichever net is current_net_idx (about to act next)."""
        obs = np.zeros((10, self.grid_size, self.grid_size), dtype=np.float32)

        idx = self.current_net_idx
        active_net = self.board.nets[idx] if idx is not None else None
        state = self.net_states.get(idx) if idx is not None else None

        head_x = state.head_x if state is not None else 0
        head_y = state.head_y if state is not None else 0
        head_layer = state.head_layer if state is not None else 0
        geodesic_cache = state.geodesic_cache if state is not None else None
        last_rejected_pos = state.last_rejected_pos if state is not None else None
        collision_run = state.collision_run if state is not None else 0
        dead_zones = state.dead_zones if state is not None else set()

        # Channel 0: Existing copper (binary/normalized)
        obs[0] = (self.board.copper_grid[head_layer] > 0).astype(np.float32)

        # Channel 1: Obstacles
        for obs_rect in self.board.obstacles:
            if obs_rect.layer == -1 or obs_rect.layer == head_layer:
                obs[1, obs_rect.y1:obs_rect.y2, obs_rect.x1:obs_rect.x2] = 1.0

        # Channel 2: Pads (Source & Target)
        for net in self.board.nets:
            for pad in [net.source_pad, net.target_pad]:
                if pad.layer == head_layer:
                    y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
                    mask = (x_coords - pad.x) ** 2 + (y_coords - pad.y) ** 2 <= pad.radius ** 2
                    obs[2, mask] = 1.0

        # Channel 3: Current Routing Head (Gaussian spot)
        if 0 <= head_x < self.grid_size and 0 <= head_y < self.grid_size:
            y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
            dist_sq = (x_coords - head_x) ** 2 + (y_coords - head_y) ** 2
            obs[3] = np.exp(-0.5 * dist_sq / 16.0).astype(np.float32)

        # Channel 4: Unrouted net demand heatmap
        if self._congestion_cache is not None:
            obs[4] = self._congestion_cache

        # Channel 5: Critical net importance
        if active_net is not None:
            obs[5].fill(float(active_net.importance))

        # Channel 6: Clearance cost field
        if self._clearance_cache is not None:
            obs[6] = self._clearance_cache

        # Channel 7: Obstacle-aware distance-to-go to the target pad (NOT
        # Euclidean -- see compute_geodesic_distance_field). Normalized by
        # the same constant a Euclidean field would use, so the channel's
        # scale stays stationary across nets even though a detour's true
        # cost-to-go can exceed straight-line distance.
        if geodesic_cache is not None:
            max_dist = math.hypot(self.grid_size, self.grid_size)
            obs[7] = np.clip(geodesic_cache / max_dist, 0.0, 1.0)

        # Channel 8: Layer occupancy (1.0 on top layer, 0.0 on bottom)
        obs[8].fill(1.0 if head_layer == 0 else 0.0)

        # Channel 9: Rejection feedback -- WHERE the last move was rejected
        # (Gaussian marker, same shape as the Channel 3 head spot) and HOW
        # STUCK the net currently is (peak intensity = collision_run /
        # max_consecutive_collisions, so a single bump reads faint and
        # approaching "jammed" reads strong). This used to duplicate Channel
        # 4 verbatim -- dead weight, and the one thing actually missing was
        # a signal the policy could react to WITHIN the same net, the direct
        # analogue of an LLM seeing a DRC error before its next move rather
        # than only being reachable through the reward gradient over many
        # future episodes.
        y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
        if last_rejected_pos is not None:
            lx, ly = last_rejected_pos
            intensity = min(1.0, collision_run / max(1, self.max_consecutive_collisions))
            dist_sq = (x_coords - lx) ** 2 + (y_coords - ly) ** 2
            obs[9] = (intensity * np.exp(-0.5 * dist_sq / 16.0)).astype(np.float32)

        # Same channel also carries permanent dead-zone markers -- spots a
        # PREVIOUS restart attempt of this net got jammed at, at full
        # intensity (1.0, more severe than a fresh in-progress rejection,
        # since these are confirmed dead ends rather than a single bad try).
        # Without this, a restart's starting observation is bit-identical to
        # the failed attempt's, and a deterministic policy would just
        # retrace the exact same path into the exact same jam -- see
        # _restart_net().
        for dx, dy in dead_zones:
            dist_sq = (x_coords - dx) ** 2 + (y_coords - dy) ** 2
            obs[9] = np.maximum(obs[9], np.exp(-0.5 * dist_sq / 16.0).astype(np.float32))

        return obs

    def _get_info(self) -> Dict[str, Any]:
        idx = self.current_net_idx
        state = self.net_states.get(idx) if idx is not None else None
        head_pos = (state.head_x, state.head_y, state.head_layer) if state is not None else (0, 0, 0)
        return {
            "completed_nets": len(self.completed_nets),
            "failed_nets": len(self.failed_nets),
            "total_nets": len(self.board.nets) if self.board else 0,
            "completion_rate": (len(self.completed_nets) / len(self.board.nets)) if self.board and len(self.board.nets) > 0 else 0.0,
            "head_pos": head_pos,
            "vias": sum(self.vias_per_net.values()),
            "total_wirelength": sum(self.wirelength_per_net.values()),
            "total_steps": self.total_steps,
            "total_restarts": self.total_restarts_this_episode,
        }
