"""Gymnasium-Compatible PCB Routing Grid Environment (PCBRouterEnv).

Implements sequential multi-net growth on a 10-channel 256x256 spatial grid:
- 10-channel spatial observation space (Box(0.0, 1.0, (10, 256, 256), float32))
- 96 discrete actions per net growth step (8 directions x 3 distances x 2 layers x 2 vias)
- Fast line-raster collision & clearance checks
- Decoupled from external push/shove heuristics
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pcbworld.board_generator import BoardState, generate_random_board, NetSpec, Pad, Obstacle
from pcbworld.congestion import compute_net_demand_heatmap, compute_geodesic_distance_field, compute_clearance_field
from pcbworld.reward import RewardCalculator


DIST_STEPS = [2, 4, 8]  # Step distances for fast grid traversal


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
        max_consecutive_collisions: int = 8,
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
        # after the first one -- see the "jammed" check in step().
        self.max_consecutive_collisions = max_consecutive_collisions
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
        self.current_net_idx: int = 0
        self.head_x: int = 0
        self.head_y: int = 0
        self.head_layer: int = 0
        self.head_prev_dir: Optional[int] = None

        self.steps_taken_current_net: int = 0
        self.collision_run: int = 0
        # Where the last rejected move tried to land, or None if the last
        # step was clean. The ONLY place the policy can perceive its own
        # last mistake -- there is no recurrence/frame history here, so
        # anything not in THIS observation is invisible to it, no matter how
        # informative the reward gradient eventually is. See Channel 9.
        self._last_rejected_pos: Optional[Tuple[int, int]] = None
        self.total_steps: int = 0
        self.completed_nets: List[int] = []
        self.failed_nets: List[int] = []

        self.wirelength_per_net: Dict[int, float] = {}
        self.vias_per_net: Dict[int, int] = {}

        self._obs_cache: Optional[np.ndarray] = None
        self._congestion_cache: Optional[np.ndarray] = None
        self._clearance_cache: Optional[np.ndarray] = None
        self._obs_grid: Optional[np.ndarray] = None
        # Obstacle-aware cost-to-go for the ACTIVE net's target, rebuilt once
        # per net (not per step -- same amortization as _refresh_static_segments
        # in line_route_env.py). This is what lets a detour pay off before the
        # collision fires, instead of only after -- see compute_geodesic_distance_field.
        self._geodesic_cache: Optional[np.ndarray] = None

        if seed is not None:
            self.reset(seed=seed)

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

        self.current_net_idx = 0
        self.completed_nets = []
        self.failed_nets = []
        self.wirelength_per_net = {net.net_id: 0.0 for net in self.board.nets}
        self.vias_per_net = {net.net_id: 0 for net in self.board.nets}
        self.total_steps = 0

        self._init_current_net()
        self._precompute_static_caches()
        self._update_geodesic_cache()

        obs = self._build_observation()
        info = self._get_info()
        return obs, info

    def _init_current_net(self):
        """Initialize head for the active unrouted net."""
        if self.current_net_idx < len(self.board.nets):
            active_net = self.board.nets[self.current_net_idx]
            self.head_x = active_net.source_pad.x
            self.head_y = active_net.source_pad.y
            self.head_layer = active_net.source_pad.layer
            self.head_prev_dir = None
            self.steps_taken_current_net = 0
            self.collision_run = 0
            self._last_rejected_pos = None

    def _precompute_static_caches(self):
        """Precompute obstacle masks, clearance field, and initial congestion."""
        # 1. Obstacle grid
        obs_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for obs in self.board.obstacles:
            x1 = max(0, obs.x1)
            x2 = min(self.grid_size, obs.x2)
            y1 = max(0, obs.y1)
            y2 = min(self.grid_size, obs.y2)
            obs_grid[y1:y2, x1:x2] = 1.0

        self._clearance_cache = compute_clearance_field(self.grid_size, obs_grid)
        self._obs_grid = obs_grid
        self._update_congestion_cache()

    def _update_geodesic_cache(self):
        """Rebuild the cost-to-go field for whichever net is active now.

        Only the target moves between nets; the obstacle grid does not, so
        `self._obs_grid` from `_precompute_static_caches` is reused rather
        than rebuilt.
        """
        if self.current_net_idx >= len(self.board.nets):
            self._geodesic_cache = None
            return
        target = self.board.nets[self.current_net_idx].target_pad
        self._geodesic_cache = compute_geodesic_distance_field(
            self.grid_size, target.x, target.y, self._obs_grid
        )

    def _geo_dist_at(self, x: float, y: float) -> float:
        if self._geodesic_cache is None:
            return 0.0
        gx = int(np.clip(round(x), 0, self.grid_size - 1))
        gy = int(np.clip(round(y), 0, self.grid_size - 1))
        return float(self._geodesic_cache[gy, gx])

    def _geo_descent_dir(self, x: float, y: float) -> Tuple[float, float]:
        """Which way the cost-to-go field decreases fastest -- the direction
        that walks around whatever is in the way instead of into it."""
        if self._geodesic_cache is None:
            return (0.0, 0.0)
        gx = int(np.clip(round(x), 1, self.grid_size - 2))
        gy = int(np.clip(round(y), 1, self.grid_size - 2))
        ddx = self._geodesic_cache[gy, gx - 1] - self._geodesic_cache[gy, gx + 1]
        ddy = self._geodesic_cache[gy - 1, gx] - self._geodesic_cache[gy + 1, gx]
        norm = math.hypot(ddx, ddy)
        if norm < 1e-6:
            return (0.0, 0.0)
        return (ddx / norm, ddy / norm)

    def _relative_direction_vector(self, dir_idx: int, x: float, y: float, target_x: float, target_y: float) -> Tuple[float, float]:
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
        gdx, gdy = self._geo_descent_dir(x, y) if self._geodesic_cache is not None else (0.0, 0.0)
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
        """
        active_id = (
            self.board.nets[self.current_net_idx].net_id
            if self.current_net_idx < len(self.board.nets)
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
        dir_idx, dist_idx, layer_change, via_flag = self.decode_action(action)
        step_dist = DIST_STEPS[dist_idx]

        active_net = self.board.nets[self.current_net_idx]
        target = active_net.target_pad

        prev_x, prev_y, prev_layer = self.head_x, self.head_y, self.head_layer
        dir_x, dir_y = self._relative_direction_vector(dir_idx, prev_x, prev_y, target.x, target.y)
        # Obstacle-aware cost-to-go, not straight-line -- a straight-line
        # potential pays off walking INTO whatever is in the way right up
        # until the collision fires. See compute_geodesic_distance_field.
        prev_dist = self._geo_dist_at(prev_x, prev_y)

        # 1. Execute via / layer change
        is_via = False
        if via_flag == 1 or layer_change == 1:
            new_layer = 1 - prev_layer  # Switch between Layer 0 and Layer 1
            is_via = True
            self.head_layer = new_layer
            self.vias_per_net[active_net.net_id] += 1

        # 2. Compute new head coordinate. Rounded to int: _rasterize_line's
        # Bresenham walk increments by whole cells and terminates on
        # x == x1 and y == y1 -- a float target it can step past without
        # ever hitting exactly would loop forever.
        new_x = int(round(prev_x + dir_x * step_dist))
        new_y = int(round(prev_y + dir_y * step_dist))

        step_len = math.hypot(new_x - prev_x, new_y - prev_y)
        self.steps_taken_current_net += 1
        self.total_steps += 1

        # 3. Check for boundary collision
        out_of_bounds = (
            new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size
        )

        # 4. Check obstacle and copper collision along raster line
        is_collided = out_of_bounds
        if not is_collided:
            is_collided = self._check_line_collision(
                prev_x, prev_y, new_x, new_y, self.head_layer, active_net.net_id
            )

        # 5. Check if target pad reached -- physical proximity, so Euclidean
        # is correct here even though the reward below uses the geodesic
        # distance (near the pad the two coincide anyway).
        curr_dist = math.hypot(new_x - target.x, new_y - target.y)
        curr_dist_geo = self._geo_dist_at(new_x, new_y)
        is_connected = False

        if not is_collided:
            # Draw line segment into copper grid
            self._rasterize_line(
                prev_x, prev_y, new_x, new_y, self.head_layer, active_net.net_id
            )
            self.head_x = new_x
            self.head_y = new_y
            self.wirelength_per_net[active_net.net_id] += step_len

            if curr_dist <= self.snap_radius and (self.head_layer == target.layer):
                is_connected = True
                # Connect directly to target pad center
                self._rasterize_line(
                    self.head_x, self.head_y, target.x, target.y, target.layer, active_net.net_id
                )
                self.completed_nets.append(active_net.net_id)

        # Check bend
        is_bend = (self.head_prev_dir is not None) and (self.head_prev_dir != dir_idx)
        self.head_prev_dir = dir_idx

        # Heading alignment against the field's descent direction, not the
        # straight bearing to the pad -- this is what actually rewards
        # circling an obstacle instead of pushing into it. Falls back to the
        # straight bearing only where the field has no local gradient (e.g.
        # against the board edge).
        act_norm = math.hypot(dir_x, dir_y)
        gdx, gdy = self._geo_descent_dir(prev_x, prev_y)
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
        self.collision_run = self.collision_run + 1 if is_collided else 0
        jammed = self.collision_run >= self.max_consecutive_collisions
        # WHERE the rejection landed, for Channel 9 -- cleared the instant a
        # move succeeds, so this is strictly "what just happened", not a
        # lingering mark.
        self._last_rejected_pos = (new_x, new_y) if is_collided else None

        # Transition / Termination logic
        net_timeout = self.steps_taken_current_net >= self.max_steps_per_net
        net_done = is_connected or jammed or net_timeout

        if net_done and not is_connected:
            self.failed_nets.append(active_net.net_id)

        terminated = False
        truncated = False

        if net_done:
            self.current_net_idx += 1
            if self.current_net_idx < len(self.board.nets):
                self._init_current_net()
                self._update_congestion_cache()
                self._update_geodesic_cache()
            else:
                terminated = True

        obs = self._build_observation()
        info = self._get_info()
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

    def _build_observation(self) -> np.ndarray:
        """Construct the (10, 256, 256) spatial observation tensor."""
        obs = np.zeros((10, self.grid_size, self.grid_size), dtype=np.float32)

        active_net = None
        if self.current_net_idx < len(self.board.nets):
            active_net = self.board.nets[self.current_net_idx]

        # Channel 0: Existing copper (binary/normalized)
        obs[0] = (self.board.copper_grid[self.head_layer] > 0).astype(np.float32)

        # Channel 1: Obstacles
        for obs_rect in self.board.obstacles:
            if obs_rect.layer == -1 or obs_rect.layer == self.head_layer:
                obs[1, obs_rect.y1:obs_rect.y2, obs_rect.x1:obs_rect.x2] = 1.0

        # Channel 2: Pads (Source & Target)
        for net in self.board.nets:
            for pad in [net.source_pad, net.target_pad]:
                if pad.layer == self.head_layer:
                    y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
                    mask = (x_coords - pad.x) ** 2 + (y_coords - pad.y) ** 2 <= pad.radius ** 2
                    obs[2, mask] = 1.0

        # Channel 3: Current Routing Head (Gaussian spot)
        if 0 <= self.head_x < self.grid_size and 0 <= self.head_y < self.grid_size:
            y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
            dist_sq = (x_coords - self.head_x) ** 2 + (y_coords - self.head_y) ** 2
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
        if self._geodesic_cache is not None:
            max_dist = math.hypot(self.grid_size, self.grid_size)
            obs[7] = np.clip(self._geodesic_cache / max_dist, 0.0, 1.0)

        # Channel 8: Layer occupancy (1.0 on top layer, 0.0 on bottom)
        obs[8].fill(1.0 if self.head_layer == 0 else 0.0)

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
        if self._last_rejected_pos is not None:
            lx, ly = self._last_rejected_pos
            intensity = min(1.0, self.collision_run / max(1, self.max_consecutive_collisions))
            y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
            dist_sq = (x_coords - lx) ** 2 + (y_coords - ly) ** 2
            obs[9] = (intensity * np.exp(-0.5 * dist_sq / 16.0)).astype(np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "completed_nets": len(self.completed_nets),
            "failed_nets": len(self.failed_nets),
            "total_nets": len(self.board.nets) if self.board else 0,
            "completion_rate": (len(self.completed_nets) / len(self.board.nets)) if self.board and len(self.board.nets) > 0 else 0.0,
            "head_pos": (self.head_x, self.head_y, self.head_layer),
            "vias": sum(self.vias_per_net.values()),
            "total_wirelength": sum(self.wirelength_per_net.values()),
            "total_steps": self.total_steps,
        }
