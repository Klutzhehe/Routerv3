"""LineRouteEnv -- the RL environment docs/RL_PLAN.md specifies.

One RL step is one 1mm advance in a chosen direction. One episode routes a
sequence of nets on one board. Everything here is downstream of two measured
Colab results rather than of taste:

  1. **The per-step collision signal is real.** head_collides() fired on 100%
     of routing attempts whose fix() was later rejected (n=90) and 0% of
     those accepted (n=9), a mean 1.9 pushes before the fix() call, and a
     control run without the probe reproduced the identical result. So the
     collision penalty below is signal, not noise, and a 1-D action with a
     dense shaped reward is a viable design rather than a hopeful one.

  2. **get_board_geometry() costs ~0.13ms and run_drc() costs 267ms.**
     Committed copper only changes when a net FINISHES, so the board is
     fetched once per net and cached; per step only the head-relative view is
     rebuilt, in numpy. DRC runs at most once per episode. Doing either per
     step would make the observation, not the router, the bottleneck.

## The action, and why it is one number

`a` in [-1, 1] maps to a turn of +/-90 degrees relative to the direction of
the target, then the head advances a fixed step. Because line_obs.py's frame
already points +x at the target, **a = 0 walks straight at the pad** -- so an
untrained mean-zero policy reproduces the greedy straight-line router, which
is the 9/24-net baseline this is trying to beat. Training starts at that
baseline instead of below it, and every parameter update is spent learning
when to deviate rather than learning to aim.

Deliberately not (dx, dy): two dimensions where one will do, and the
straight-at-target prior would then depend on the policy learning a
particular vector rather than falling out of the coordinate system. Step
length becomes a second dimension later (docs/RL_PLAN.md's upgrade path),
once this learns.

There is no "finish" action. A net completes automatically when the head is
within `snap_radius_nm` and fix() confirms real connectivity -- the agent's
only job is choosing where to go, and a learned stop action would be one more
thing to get wrong for no benefit.

**Keep `snap_radius_nm >= step_size_nm / 2`.** The head advances a FIXED
distance, so with a step much larger than the snap radius it can jump clean
over the snap zone and orbit the target forever. The symptom of getting this
wrong -- "the agent never finishes nets" -- reads as a learning failure and
is really a configuration one, which is why it is stated here and pinned by a
test rather than left to be rediscovered.

## Termination

Per net: success (fix() accepted), or the step budget runs out. Per episode:
every net in `net_order` has been attempted. A failed net does not end the
episode -- later nets still route, which is what makes the multi-net stages
teach anything about congestion.

Untested against a real bridge: like every module here that imports
pcbworld_pns_bridge, this can only run inside the Colab flow. The Python
control flow is covered by tests/fake_bridge.py; whether the reward shape is
right against real router behaviour is a Colab question.
"""

from __future__ import annotations

import dataclasses
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pcbworld.env.line_obs import (
    KIND_PAD,
    MM,
    LineObsConfig,
    board_segments,
    build_observation,
    ghost_segment,
    pad_to_segment,
)

MAX_TURN_RAD = math.pi / 2


@dataclasses.dataclass
class RewardWeights:
    """Potential-based progress plus three penalties and two terminals.

    `progress` scales a true potential (Ng et al.): the per-step term is
    gamma*phi(s') - phi(s) with phi = -distance/length_scale, which is
    policy-invariant and automatically revisit-safe -- backtracking cannot be
    farmed for reward. A raw distance-travelled penalty would instead punish a
    legal detour around an obstacle more than the collision it avoids.
    """

    progress: float = 1.0
    step: float = 0.02
    collision: float = 0.5      # the Gate-B-validated per-step failure signal
    net_done: float = 10.0
    net_failed: float = 5.0
    detour: float = 2.0         # per unit of (routed / straight-line - 1)
    drc: float = 0.0            # per violation, once per episode; 0 = off


class LineRouteEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        board_path: str,
        *,
        net_order: list[str] | None = None,
        max_nets: int | None = None,
        obs_config: LineObsConfig | None = None,
        reward_weights: RewardWeights | None = None,
        step_size_nm: int = 1 * MM,
        track_width_nm: int = 250_000,
        snap_radius_nm: int = 500_000,
        max_steps_per_net: int = 80,
        gamma: float = 0.99,
        run_drc_at_episode_end: bool = False,
    ) -> None:
        super().__init__()

        # Deferred: the bridge only exists after the Colab build, and a type
        # checker or a test collecting this file must not hard-fail.
        import pcbworld_pns_bridge as bridge

        self._module = bridge
        self.bridge = bridge.PNSBridge()
        self.board_path = board_path
        self.net_order = net_order
        self.max_nets = max_nets
        self.obs_config = obs_config or LineObsConfig(max_steps=max_steps_per_net)
        self.weights = reward_weights or RewardWeights()
        self.step_size_nm = step_size_nm
        self.track_width_nm = track_width_nm
        self.snap_radius_nm = snap_radius_nm
        self.max_steps_per_net = max_steps_per_net
        self.gamma = gamma
        self.run_drc_at_episode_end = run_drc_at_episode_end

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_config.flat_size,), dtype=np.float32
        )

        self._board_loaded = False
        self._pads: list = []
        self._nets: list[str] = []
        self._net_index = 0
        self._static_segments: list = []   # refreshed once per net, not per step
        self._pad_geoms: list = []         # PadGeom, refreshed alongside
        self._obstacles: list = []         # per-net obstacle set, not per-step
        self._route_active = False
        self._steps = 0
        self._pos = (0.0, 0.0)
        self._start_xy = (0.0, 0.0)
        self._target_xy = (0.0, 0.0)
        self._target_id = -1
        self._straight_len = 1.0
        self._routed_len = 0.0
        self._collides = False
        self._completed: list[str] = []
        self._failed: list[str] = []

    # -- setup ----------------------------------------------------------

    def _pad_candidate(self, x: int, y: int) -> int:
        """query_hover_items() returns hits in the router's own hit-test
        order, NOT sorted by kind or distance. On a board with committed
        copper an unrelated track can pass within the snap radius of a pad,
        and taking candidates[0] then hands fix() the wrong item id -- which
        fails for a reason unrelated to the net being routed and is
        indistinguishable from a real collision in aggregate stats. That bug
        cost a Colab round in measure_waypoint_fidelity.py; the older envs
        still carry the candidates[0] pattern, this one does not."""
        candidates = self.bridge.query_hover_items(
            x, y, layer=0, slop_radius=self.snap_radius_nm
        )
        if not candidates:
            return -1
        pads = [c for c in candidates if c.kind == "pad"]
        return (pads[0] if pads else candidates[0]).id

    def _discover_nets(self) -> list[str]:
        two_pad = {}
        for pad in self._pads:
            if pad.net:
                two_pad.setdefault(pad.net, []).append(pad)
        usable = {n: p for n, p in two_pad.items() if len(p) == 2 and n.startswith("net_")}

        if self.net_order is not None:
            ordered = [n for n in self.net_order if n in usable]
        else:
            # Shortest-first. Net ordering is deliberately a heuristic, not a
            # learned decision (docs/RL_PLAN.md): it removes a combinatorial
            # dimension the policy would otherwise have to explore, and is
            # revisited only if a stage plateaus.
            def span(name: str) -> float:
                a, b = usable[name]
                return math.hypot(a.x - b.x, a.y - b.y)

            ordered = sorted(usable, key=span)

        return ordered[: self.max_nets] if self.max_nets else ordered

    def _refresh_static_segments(self) -> None:
        """Committed copper, board outline, and pad geometry. Called once per
        net, never per step -- get_board_geometry() is ~0.13ms against a
        ~0.03ms step.

        Pads come from here rather than from net_pads(), which the env
        already holds: the two carry different things. net_pads() gives
        NetPad (net, pad name, position, layer) -- enough to find a route's
        endpoints, which is what it is used for. get_board_geometry() gives
        PadGeom, which also carries size_x/size_y, and an obstacle's SIZE is
        exactly what matters for avoiding it. Reaching for the wrong one is a
        plausible mistake, so both are fetched deliberately and used for
        their own job.
        """
        geometry = self.bridge.get_board_geometry()
        self._static_segments = board_segments(geometry)
        self._pad_geoms = list(geometry.pads)

    # -- gym API --------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if not self._board_loaded:
            assert self.bridge.load_board(self.board_path), f"load_board failed: {self.board_path}"
            self._board_loaded = True
            self._pads = list(self.bridge.net_pads())

        # reset() rather than reloading: it strips tracks/vias and keeps
        # footprints, and is the Colab-verified way to get a bare board.
        # Reloading would also work now that LoadBoard's teardown is fixed,
        # but it is far more expensive and buys nothing here.
        self.bridge.reset()
        self.bridge.set_mode(self._module.MODE_ROUTE_SINGLE)
        # push() must be a pure validator, not a router that quietly shoves
        # other traces aside to accommodate the agent.
        self.bridge.set_collision_mode(self._module.RM_MARK_OBSTACLES)
        self.bridge.set_track_width(self.track_width_nm)

        self._nets = self._discover_nets()
        assert self._nets, "no routable two-pad 'net_*' nets on this board"
        self._net_index = 0
        self._completed, self._failed = [], []

        self._refresh_static_segments()
        self._begin_net()
        return self._observe(), self._info()

    def _begin_net(self) -> None:
        net = self._nets[self._net_index]
        a, b = [p for p in self._pads if p.net == net]
        self._start_xy = (float(a.x), float(a.y))
        self._target_xy = (float(b.x), float(b.y))
        self._target_id = self._pad_candidate(int(b.x), int(b.y))
        self._straight_len = max(
            1.0, math.hypot(self._target_xy[0] - self._start_xy[0], self._target_xy[1] - self._start_xy[1])
        )
        self._pos = self._start_xy
        self._routed_len = 0.0
        self._steps = 0
        self._collides = False

        start_id = self._pad_candidate(int(a.x), int(a.y))
        self._route_active = bool(
            self.bridge.start_route(int(a.x), int(a.y), start_id, 0)
        )
        self._rebuild_obstacles()

    def step(self, action):
        turn = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0)) * MAX_TURN_RAD

        # Bearing to the target, then the policy's turn on top of it. This is
        # what makes a=0 mean "straight at the pad" -- the same convention
        # line_obs.py's frame uses, and the two must not drift apart.
        dx = self._target_xy[0] - self._pos[0]
        dy = self._target_xy[1] - self._pos[1]
        bearing = math.atan2(dy, dx) if (dx or dy) else 0.0
        heading = bearing + turn

        goal = (
            self._pos[0] + self.step_size_nm * math.cos(heading),
            self._pos[1] + self.step_size_nm * math.sin(heading),
        )

        prev_dist = math.hypot(dx, dy)
        self.bridge.push(int(goal[0]), int(goal[1]), -1)

        # Read the head back rather than trusting the requested point: push()
        # is ROUTER::Move(), which may not land exactly where it was told, and
        # a route built on the requested position instead of the real one
        # accumulates error silently.
        head = self.bridge.get_head_geometry()
        moved = math.hypot(head.end_x - self._pos[0], head.end_y - self._pos[1])
        self._pos = (float(head.end_x), float(head.end_y))
        self._routed_len += moved
        self._collides = bool(self.bridge.head_collides())
        self._steps += 1

        dist = math.hypot(self._target_xy[0] - self._pos[0], self._target_xy[1] - self._pos[1])
        reward = self._shaping(prev_dist, dist) - self.weights.step
        if self._collides:
            reward -= self.weights.collision

        net_done = False
        if dist <= self.snap_radius_nm:
            net_done = self._try_finish()
            reward += self.weights.net_done if net_done else 0.0
        if not net_done and self._steps >= self.max_steps_per_net:
            self._abandon()
            reward -= self.weights.net_failed
            net_done = True

        if net_done:
            reward -= self.weights.detour * max(0.0, self._routed_len / self._straight_len - 1.0)

        terminated = False
        if net_done:
            self._net_index += 1
            if self._net_index >= len(self._nets):
                terminated = True
                reward += self._episode_end_penalty()
            else:
                # Copper changed, so the cache is stale exactly here -- once
                # per net, which is the whole point of caching it.
                self._refresh_static_segments()
                self._begin_net()

        return self._observe(), float(reward), terminated, False, self._info()

    # -- internals ------------------------------------------------------

    def _shaping(self, prev_dist: float, dist: float) -> float:
        scale = self.obs_config.length_scale
        phi_prev = -self.weights.progress * prev_dist / scale
        phi_next = -self.weights.progress * dist / scale
        return self.gamma * phi_next - phi_prev

    def _try_finish(self) -> bool:
        """force_finish/force_commit=True is the Colab-verified convention
        (commit 7f746b6) and is what snaps the route to the target pad."""
        net = self._nets[self._net_index]
        ok = bool(
            self.bridge.fix(
                int(self._target_xy[0]), int(self._target_xy[1]), self._target_id, True, True
            )
        )
        if ok:
            self.bridge.commit_routing()
            self._completed.append(net)
        else:
            self.bridge.stop_routing()
            self._failed.append(net)
        self._route_active = False
        return ok

    def _abandon(self) -> None:
        self.bridge.stop_routing()
        self._failed.append(self._nets[self._net_index])
        self._route_active = False

    def _episode_end_penalty(self) -> float:
        if not self.run_drc_at_episode_end or self.weights.drc <= 0:
            return 0.0
        # 267ms -- affordable once per episode, never per step.
        return -self.weights.drc * len(self.bridge.run_drc())

    def _rebuild_obstacles(self) -> None:
        """Cache the obstacle list for the CURRENT net.

        Nothing in it changes while a net is being routed: committed copper
        only grows when a net finishes, pads never move, and the agent's own
        in-progress head is not in get_board_geometry() until commit. So this
        is per-net work that was being redone every step -- measured at
        0.745ms/step against a ~0.035ms budget on a 24-net board, where the
        per-step rebuild was allocating a Segment dataclass for every pad and
        every pending net, ~200 objects, before any numpy ran.
        """
        self._obstacles = self._build_obstacles()

    def _build_obstacles(self) -> list:
        net = self._nets[self._net_index] if self._net_index < len(self._nets) else ""
        segments = list(self._static_segments)

        # A pad is an obstacle unless it belongs to the net being routed, in
        # which case it is an endpoint. board_segments() cannot make that call
        # itself -- it has no idea which net is active -- so it is made here.
        for pad in self._pad_geoms:
            if pad.net != net:
                segments.append(pad_to_segment(pad, kind=KIND_PAD))

        # Nets not yet routed enter as the straight lines they will have to
        # span, so the policy can see where future copper still needs room.
        routed = set(self._completed)
        for future in self._nets[self._net_index + 1 :]:
            if future in routed:
                continue
            a, b = [p for p in self._pads if p.net == future]
            segments.append(ghost_segment((a.x, a.y), (b.x, b.y), future))

        return segments

    def _observe(self) -> np.ndarray:
        if self._net_index >= len(self._nets):
            return np.zeros(self.obs_config.flat_size, dtype=np.float32)
        return build_observation(
            self._obstacles,
            head=self._pos,
            target=self._target_xy,
            head_layer=0,
            target_layer=0,
            own_net=self._nets[self._net_index],
            steps_taken=self._steps,
            routed_length=self._routed_len,
            straight_line_length=self._straight_len,
            head_collides=self._collides,
            config=self.obs_config,
        )

    def _info(self) -> dict:
        return {
            "net": self._nets[self._net_index] if self._net_index < len(self._nets) else None,
            "net_index": self._net_index,
            "num_nets": len(self._nets),
            "completed": list(self._completed),
            "failed": list(self._failed),
            "steps": self._steps,
            "collides": self._collides,
            "routed_length_nm": self._routed_len,
        }

    def close(self):
        if self._route_active:
            self.bridge.stop_routing()
            self._route_active = False
