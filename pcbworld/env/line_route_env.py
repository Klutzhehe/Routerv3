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

from pcbworld.env.geodesic import GeodesicConfig, GeodesicField
from pcbworld.env.line_obs import (
    KIND_PAD,
    nearest_obstacle_gap,
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

    ## Why `collision` is capped in TIME, not just in size

    `collision` is charged per step and the old budget was 120 steps, so a net
    that jammed against an obstacle bled 0.5 x 120 = -60. Measured against a
    clean success at +11.4, that made failure 5.8x the weight of success and
    left 91% of a failed net's return sitting in the collision term. The
    dominant gradient was "do not touch anything", and the cheapest policy
    satisfying it is to wander in open copper (-6.2) rather than to route.

    The original diagnosis assumed the router refused to move a colliding
    head, so the penalty accrued against a frozen state. MEASURED ON COLAB,
    that is false: over 2000 steps, 275 collided and the head still advanced
    501244 nm on each of them -- a full 0.5 mm step. A colliding head is not
    jammed, it is contouring, and the whole 120-step budget can legitimately
    be spent in contact.

    So the time cap is not what bounds this; the per-step weight has to. At
    0.5 a fully-contacting net cost -66 against a +26.4 success (2.51x), which
    is the same failure-dominates-success shape the cap was meant to remove.
    At 0.15 the worst case is -24.2 (0.92x) and the measured 13.75%-contact
    case is -8.7, so contact stays a real signal without outweighing the
    reason to route at all.

    `max_collision_steps` stays as a safety net for a genuinely frozen head.
    On the measured behaviour it never fires, which is the correct outcome.

    Two qualifiers, both load-bearing:

    Consecutive rather than cumulative -- the target is the jammed state
    specifically, so escaping and re-contacting elsewhere resets the count.

    STUCK rather than merely colliding. Contour-following holds contact for as
    long as the obstacle lasts, so a plain collision count would abandon the
    net partway through the exact manoeuvre this is meant to make affordable,
    and the agent would never collect a completed detour to learn from. A head
    that is colliding and still advancing is working; one that is colliding and
    not moving is the case worth cutting.
    """

    progress: float = 1.0
    step: float = 0.01          # gentle step cost so detours are viable
    collision: float = 0.5      # charged ONCE, on first contact -- see below
    max_collision_steps: int = 16   # give up after this many CONSECUTIVE collision
                                    # steps; see the note below
    net_done: float = 25.0
    net_failed: float = 5.0
    detour: float = 0.5         # relaxed so going around obstacles is not penalized
    drc: float = 0.0            # per violation, once per episode; 0 = off
    ripup: float = 1.0          # penalty charged per ripped-up net
    length_mismatch: float = 1.0 # penalty per unit length discrepancy on length-matched groups





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
        step_size_nm: int = 500_000,
        track_width_nm: int = 250_000,
        via_diameter_nm: int = 600_000,
        via_drill_nm: int = 300_000,
        snap_radius_nm: int = 400_000,
        max_steps_per_net: int = 100,
        gamma: float = 0.99,

        geodesic_config: GeodesicConfig | None = None,
        run_drc_at_episode_end: bool = False,
        enable_ripup: bool = False,
        max_ripups_per_episode: int = 8,
    ) -> None:
        super().__init__()

        # Deferred: the bridge only exists after the Colab build, and a type
        # checker or a test collecting this file must not hard-fail.
        import pcbworld_pns_bridge as bridge
        import os

        self._module = bridge
        self.bridge = bridge.PNSBridge()
        if isinstance(board_path, (list, tuple)):
            self.board_paths = list(board_path)
        elif os.path.isdir(board_path):
            self.board_paths = sorted([os.path.join(board_path, f) for f in os.listdir(board_path) if f.endswith(".kicad_pcb")])
            assert self.board_paths, f"No .kicad_pcb boards found in directory {board_path}"
        else:
            self.board_paths = [board_path]

        self.board_path = self.board_paths[0]
        self.net_order = net_order
        self.max_nets = max_nets
        self.obs_config = obs_config or LineObsConfig(max_steps=max_steps_per_net)
        self.weights = reward_weights or RewardWeights()
        self.step_size_nm = step_size_nm
        self.track_width_nm = track_width_nm
        self.via_diameter_nm = via_diameter_nm
        self.via_drill_nm = via_drill_nm
        self.snap_radius_nm = snap_radius_nm
        self.max_steps_per_net = max_steps_per_net
        self.gamma = gamma
        self.run_drc_at_episode_end = run_drc_at_episode_end
        self.enable_ripup = enable_ripup
        self.max_ripups_per_episode = max_ripups_per_episode
        self._geodesic_config = geodesic_config or GeodesicConfig(cell_nm=float(step_size_nm))


        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_config.flat_size,), dtype=np.float32
        )

        self._current_board_path: str | None = None
        self._pads: list = []
        self._nets: list[str] = []
        self._net_index = 0
        self._ripup_count = 0
        self._blocking_nets: set[str] = set()
        self._ripups_performed: list[str] = []
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
        self._collision_run = 0
        self._field: GeodesicField | None = None
        self._completed: list[str] = []
        self._failed: list[str] = []
        # Why each failed net failed. Three very different problems wear the
        # same "37% did not route" number, and until they are separated the
        # obvious reading -- "it cannot steer around obstacles" -- is an
        # assumption rather than a measurement.
        self._failure_reasons: dict[str, str] = {}
        self._failure_progress: dict[str, float] = {}
        self._failure_travel: dict[str, float] = {}
        self._fix_refusals: dict[str, dict] = {}
        self._collisions_this_net = 0
        self._min_dist_nm = float("inf")
        self._start_ok = False
        self._start_pad_id = -1
        self._target_pad_id_ok = False
        # Closest the head ever got to the target, per net. With 100% of
        # failures reported as "never reached the pad" on a step budget 3.3x
        # the typical net, this is the number that says whether the head
        # stalls short of the pad or never travels at all.
        self._closest_approach_nm: dict[str, float] = {}
        self._min_dist_nm = float("inf")

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
        usable = {
            n: p
            for n, p in two_pad.items()
            if len(p) == 2 and (
                n.startswith("net_")
                or (n.startswith("diffpair_") and n.endswith("_P"))
                or n.startswith("lengthgrp_")
            )
        }

        if self.net_order is not None:
            ordered = [n for n in self.net_order if n in usable]
        else:
            # Group ordering: ensure lengthgrp_<g>_0 routes before member 1, 2...
            # Within plain/diffpair nets, sort shortest-first.
            def sort_key(name: str) -> tuple[int, float]:
                if name.startswith("lengthgrp_"):
                    parts = name.split("_")
                    g, m = int(parts[1]), int(parts[2])
                    return (1000 + g, float(m))
                a, b = usable[name]
                return (0, math.hypot(a.x - b.x, a.y - b.y))

            ordered = sorted(usable, key=sort_key)

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

        chosen_board = np.random.choice(self.board_paths) if len(self.board_paths) > 1 else self.board_paths[0]
        if self._current_board_path != chosen_board:
            assert self.bridge.load_board(chosen_board), f"load_board failed: {chosen_board}"
            self._current_board_path = chosen_board
            self.board_path = chosen_board
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
        # Inert until the action space can request a layer change, but wrong to
        # leave unset: switch_layer() places a real via internally, and with no
        # via size configured the real bridge rejected it 15/15 times at open
        # board space nowhere near a pad (see scripts/measure_layer_hop_rescue.py,
        # which cost a Colab round to that exact omission). pcb_route_env and
        # diff_pair_route_env, both Colab-verified, always set these before
        # routing; this env was the one that did not.
        self.bridge.set_via_diameter(self.via_diameter_nm)
        self.bridge.set_via_drill(self.via_drill_nm)

        self._nets = self._discover_nets()
        assert self._nets, "no routable two-pad 'net_*' nets on this board"
        self._net_index = 0
        self._completed, self._failed = [], []
        self._failure_reasons = {}
        self._failure_progress = {}
        self._failure_travel = {}
        self._fix_refusals = {}
        self._closest_approach_nm = {}
        self._ripup_count = 0
        self._blocking_nets = set()
        self._ripups_performed = []
        self._length_group_refs: dict[str, float] = {}
        self._target_ref_length = 0.0

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
        self._collision_run = 0
        self._min_dist_nm = math.hypot(
            self._target_xy[0] - self._start_xy[0], self._target_xy[1] - self._start_xy[1]
        )
        self._blocking_nets = set()
        self._target_ref_length = 0.0
        self._prev_heading = math.atan2(self._target_xy[1] - self._start_xy[1], self._target_xy[0] - self._start_xy[0])

        if net.startswith("diffpair_"):
            self.bridge.set_mode(self._module.MODE_ROUTE_DIFF_PAIR)
            self.bridge.set_diff_pair_gap(150_000)
            self.bridge.set_diff_pair_width(200_000)
        elif net.startswith("lengthgrp_"):
            self.bridge.set_mode(self._module.MODE_ROUTE_SINGLE)
            parts = net.split("_")
            group_idx, member_idx = parts[1], int(parts[2])
            if member_idx > 0:
                self._target_ref_length = self._length_group_refs.get(group_idx, 0.0)
        else:
            self.bridge.set_mode(self._module.MODE_ROUTE_SINGLE)


        start_id = self._pad_candidate(int(a.x), int(a.y))
        self._route_active = bool(
            self.bridge.start_route(int(a.x), int(a.y), start_id, 0)
        )
        # A head the router never took charge of cannot be steered anywhere,
        # by any policy. Recorded here so a net that never started is not
        # filed under the same "ran out of steps" heading as one that was
        # genuinely navigating and did not arrive -- they are opposite
        # problems and the first is invisible to everything downstream.
        self._start_ok = self._route_active
        self._start_pad_id = start_id
        self._target_pad_id_ok = self._target_id >= 0
        self._rebuild_obstacles()


    def step(self, action):
        turn = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0)) * MAX_TURN_RAD

        dx = self._target_xy[0] - self._pos[0]
        dy = self._target_xy[1] - self._pos[1]
        bearing = math.atan2(dy, dx) if (dx or dy) else 0.0

        # When colliding against an obstacle, maintain heading momentum so turns
        # contour along the obstacle perimeter rather than getting pulled straight back into it.
        base_heading = self._prev_heading if self._collides else bearing
        heading = base_heading + turn
        self._prev_heading = heading

        prev_pos = self._pos
        was_colliding = self._collides
        goal = (
            self._pos[0] + self.step_size_nm * math.cos(heading),
            self._pos[1] + self.step_size_nm * math.sin(heading),
        )

        self.bridge.push(int(goal[0]), int(goal[1]), -1)

        # Read the head back rather than trusting the requested point: push()
        # is ROUTER::Move(), which may not land exactly where it was told, and
        # a route built on the requested position instead of the real one
        # accumulates error silently.
        head = self.bridge.get_head_geometry()
        # `active` is not decoration. With no live route the bridge returns a
        # zeroed HeadGeometry, and reading end_x/end_y out of it unconditionally
        # teleports the head to the board ORIGIN -- then charges the distance it
        # "travelled" to get there to _routed_len. A net whose start_route()
        # was refused therefore spends its whole budget reporting a head at
        # (0, 0), with a detour_ratio and a shaping potential computed from a
        # position the router never had. Every downstream signal on that net is
        # fiction, and it looks exactly like a net that simply failed to
        # navigate.
        if head.active:
            moved = math.hypot(head.end_x - self._pos[0], head.end_y - self._pos[1])
            self._pos = (float(head.end_x), float(head.end_y))
        else:
            moved = 0.0
        self._routed_len += moved
        self._collides = bool(self.bridge.head_collides())
        # Only a STUCK collision counts toward the jam. Sliding along an
        # obstacle keeps head_collides() true for as long as the contact
        # lasts, so counting every colliding step would abandon the net in
        # the middle of a successful contour -- killing the one manoeuvre
        # this is supposed to make affordable. A frozen head is the thing
        # worth giving up on, and `moved` is what distinguishes the two.
        stuck = self._collides and moved < 0.25 * self.step_size_nm
        self._collision_run = self._collision_run + 1 if stuck else 0
        self._collisions_this_net += int(self._collides)
        self._steps += 1

        if self._collides:
            # Through the guarded helper: this runs on every colliding step, so
            # an exception out of the probe would take a whole training run
            # down for a diagnostic nobody asked to be load-bearing.
            obs_item = self._describe_head_obstacle()
            if obs_item.get("found") and obs_item.get("net") in self._completed:
                self._blocking_nets.add(obs_item["net"])

        dist = math.hypot(self._target_xy[0] - self._pos[0], self._target_xy[1] - self._pos[1])
        self._min_dist_nm = min(self._min_dist_nm, dist)
        reward = self._shaping(prev_pos, self._pos) - self.weights.step
        # Charged on the TRANSITION into contact, not on every step after it.
        # Measured: once a route clips copper, head_collides() stays true on
        # 86.2% of the remaining steps, so a per-step charge spends almost all
        # of its weight punishing a mistake that already happened -- and the
        # net is lost at that first contact regardless, because fix() refuses
        # a route that touched anything. One clear marker at the moment that
        # decides the net beats thirty diffuse ones afterwards.
        if self._collides and not was_colliding:
            reward -= self.weights.collision

        net_done = False
        if dist <= self.snap_radius_nm:
            finished = self._try_finish()
            # The net is OVER either way. _try_finish() calls stop_routing() on
            # a refusal, so nothing can move or succeed afterwards -- but this
            # used to leave net_done False, and the episode then spent the rest
            # of the budget re-calling fix() on a dead route (101 times on a
            # 120-step net), appending the net to _failed once per step, and
            # finally letting _abandon() OVERWRITE the recorded reason with
            # "out_of_steps".
            #
            # That is why the first failure breakdown read 100% "never reached
            # the pad" and 0% "fix() refused" while the per-net diagnostic
            # showed those same heads closing 99% of the distance. Both were
            # measuring the same nets; the reason was being overwritten before
            # anyone could see it.
            net_done = True
            reward += self.weights.net_done if finished else -self.weights.net_failed
        jammed = self._collision_run >= self.weights.max_collision_steps
        if not net_done and (self._steps >= self.max_steps_per_net or jammed):
            # Check if rip-up can clear a blocking trace
            if self.enable_ripup and (self._ripup_count < self.max_ripups_per_episode) and self._blocking_nets:
                victim_net = sorted(self._blocking_nets)[-1]
                self._blocking_nets.remove(victim_net)
                self.bridge.stop_routing()
                self.bridge.rip_up(victim_net)
                if victim_net in self._completed:
                    self._completed.remove(victim_net)
                # Re-queue victim net to be routed later
                self._nets.append(victim_net)
                self._ripup_count += 1
                self._ripups_performed.append(victim_net)
                reward -= self.weights.ripup
                # Retry current net on cleared board
                self._refresh_static_segments()
                self._begin_net()
                net_done = False
            else:
                self._abandon("jammed" if jammed else "out_of_steps")
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

    def _potential(self, pos: tuple[float, float]) -> float:
        """-(distance still to travel) / length_scale.

        Obstacle-free distance when a field exists, straight-line when it does
        not. Never a mix within one net: `_rebuild_obstacles` drops the field
        for the whole net if the target is unreachable, so the definition is
        fixed for the duration and the telescoping stays honest."""
        d = self._geodesic_dist(pos)
        return -self.weights.progress * d / self.obs_config.length_scale

    def _clearance_at(self, pos) -> float:
        """Room before the head clips copper at `pos`."""
        return nearest_obstacle_gap(pos[0], pos[1], self._obstacles)

    def _step_ahead(self) -> tuple[float, float]:
        """Where the head lands next if the policy does nothing (a = 0).

        Deliberately the a=0 landing point rather than a per-action lookahead:
        it is the same base heading step() will use, so the feature answers
        "will going straight clip something", which is the question a=0 keeps
        getting wrong."""
        bearing = math.atan2(
            self._target_xy[1] - self._pos[1], self._target_xy[0] - self._pos[0]
        )
        base = self._prev_heading if self._collides else bearing
        return (
            self._pos[0] + self.step_size_nm * math.cos(base),
            self._pos[1] + self.step_size_nm * math.sin(base),
        )

    def _geodesic_direction(self):
        """Which way the obstacle-free route leaves the head, or None."""
        if self._field is None:
            return None
        return self._field.descent_direction(
            self._pos[0], self._pos[1], self.step_size_nm
        )

    def _geodesic_dist(self, pos: tuple[float, float]) -> float:
        if self._field is not None:
            c = self._field.cost_to_go(pos[0], pos[1])
            if math.isfinite(c):
                return c
        return math.hypot(self._target_xy[0] - pos[0], self._target_xy[1] - pos[1])

    def _shaping(self, prev_pos, pos) -> float:
        return self.gamma * self._potential(pos) - self._potential(prev_pos)

    def _describe_head_obstacle(self) -> dict:
        """WHAT the head is touching, not just that it is touching something.

        The two candidates behind a refused fix() need completely different
        fixes and the completion number cannot tell them apart. If the head is
        colliding with its OWN target pad, the router is treating this net's
        endpoint as an obstacle and the bug is in how the route is started or
        which pad id it was handed -- a bridge-integration problem with real
        headroom behind it. If it is another net's copper, the route genuinely
        crosses something and only better avoidance helps -- which the oracle
        says is worth about one net.
        """
        if not hasattr(self.bridge, "get_head_obstacle"):
            return {"probe": False}
        try:
            o = self.bridge.get_head_obstacle()
        except Exception:  # a probe must never take the episode down
            return {"probe": False}
        if not getattr(o, "found", False):
            return {"probe": True, "found": False}

        own = self._nets[self._net_index] if self._net_index < len(self._nets) else ""
        own_nets = {own}
        if own.startswith("diffpair_") and own.endswith("_P"):
            own_nets.add(own[:-2] + "_N")

        ox, oy = float(getattr(o, "x", 0.0)), float(getattr(o, "y", 0.0))
        return {
            "probe": True,
            "found": True,
            "net": str(getattr(o, "net", "")),
            "kind": str(getattr(o, "kind", "")),
            "is_own_net": str(getattr(o, "net", "")) in own_nets,
            "dist_to_target_nm": math.hypot(
                self._target_xy[0] - ox, self._target_xy[1] - oy
            ),
            "dist_to_start_nm": math.hypot(
                self._start_xy[0] - ox, self._start_xy[1] - oy
            ),
        }

    def _try_finish(self) -> bool:
        """force_finish/force_commit=True is the Colab-verified convention
        (commit 7f746b6) and is what snaps the route to the target pad."""
        net = self._nets[self._net_index]
        # What the approach was touching, before the finish push moves the head
        # onto the pad centre and possibly changes it.
        obstacle_on_approach = self._describe_head_obstacle()
        # Push router head directly into the target pad candidate ID to snap both legs
        self.bridge.push(int(self._target_xy[0]), int(self._target_xy[1]), self._target_id)
        collides_after_snap = bool(self.bridge.head_collides())
        obstacle_at_fix = self._describe_head_obstacle()
        ok = bool(
            self.bridge.fix(
                int(self._target_xy[0]), int(self._target_xy[1]), self._target_id, True, True
            )
        )

        if ok:
            self.bridge.commit_routing()
            self._completed.append(net)
            if net.startswith("diffpair_") and net.endswith("_P"):
                self._completed.append(net[:-2] + "_N")
            elif net.startswith("lengthgrp_"):
                parts = net.split("_")
                group_idx, member_idx = parts[1], int(parts[2])
                if member_idx == 0:
                    self._length_group_refs[group_idx] = self._routed_len
        else:
            self.bridge.stop_routing()
            self._failed.append(net)
            self._failure_reasons[net] = "fix_rejected"
            self._failure_progress[net] = 1.0 - min(
                1.0, self._min_dist_nm / max(1.0, self._straight_len)
            )
            self._failure_travel[net] = self._routed_len
            # WHY fix() refused is now the whole question, so capture the state
            # it refused in rather than only the fact of it.
            self._fix_refusals[net] = {
                "dist_nm": math.hypot(
                    self._target_xy[0] - self._pos[0], self._target_xy[1] - self._pos[1]
                ),
                "colliding_at_fix": bool(self._collides),
                "collision_steps": self._collisions_this_net,
                "steps": self._steps,
                "detour_ratio": self._routed_len / max(1.0, self._straight_len),
                "collides_after_snap": collides_after_snap,
                "obstacle_on_approach": obstacle_on_approach,
                "obstacle_at_fix": obstacle_at_fix,
            }
            if net.startswith("diffpair_") and net.endswith("_P"):
                self._failed.append(net[:-2] + "_N")
        self._route_active = False
        return ok

    def _abandon(self, reason: str = "out_of_steps") -> None:
        net = self._nets[self._net_index]
        if reason == "out_of_steps" and not self._start_ok:
            reason = "route_never_started"
        self.bridge.stop_routing()
        self._failed.append(net)
        self._failure_reasons[net] = reason
        # How far along it got before giving up: 1.0 means it reached the pad,
        # 0.0 means it never left the start. A budget problem and a head that
        # never moved both read as "out of steps" without this.
        dist = math.hypot(
            self._target_xy[0] - self._pos[0], self._target_xy[1] - self._pos[1]
        )
        self._failure_progress[net] = 1.0 - min(1.0, dist / max(1.0, self._straight_len))
        self._failure_travel[net] = self._routed_len
        self._closest_approach_nm[net] = self._min_dist_nm
        if net.startswith("diffpair_") and net.endswith("_P"):
            self._failed.append(net[:-2] + "_N")
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

        # Same lifetime as the obstacle cache it is built from: committed
        # copper only changes when a net finishes, so the cost-to-go field is
        # valid for exactly as long as this list is. ~20ms per net against a
        # ~1.4s rollout.
        self._field = GeodesicField.build(
            self._obstacles,
            head=self._pos,
            target=self._target_xy,
            config=self._geodesic_config,
        )
        if not self._field.reachable:
            # No obstacle-free route exists at all. Fall back to the straight
            # line for the whole net rather than per query -- a potential that
            # changes definition mid-net puts a reward spike on the switch.
            self._field = None

    def _build_obstacles(self) -> list:
        net = self._nets[self._net_index] if self._net_index < len(self._nets) else ""
        segments = list(self._static_segments)

        # A pad is an obstacle unless it belongs to the net being routed, in
        # which case it is an endpoint. For diff pairs, both P and N pads are endpoints.
        own_nets = {net}
        if net.startswith("diffpair_") and net.endswith("_P"):
            own_nets.add(net[:-2] + "_N")

        for pad in self._pad_geoms:
            if pad.net not in own_nets:
                segments.append(pad_to_segment(pad, kind=KIND_PAD))

        # Nets not yet routed enter as the straight lines they will have to
        # span, so the policy can see where future copper still needs room.
        routed = set(self._completed)
        for future in self._nets[self._net_index + 1 :]:
            if future in routed:
                continue
            a, b = [p for p in self._pads if p.net == future]
            segments.append(ghost_segment((a.x, a.y), (b.x, b.y), future))
            if future.startswith("diffpair_") and future.endswith("_P"):
                n_leg = future[:-2] + "_N"
                n_pads = [p for p in self._pads if p.net == n_leg]
                if len(n_pads) == 2:
                    segments.append(ghost_segment((n_pads[0].x, n_pads[0].y), (n_pads[1].x, n_pads[1].y), n_leg))

        return segments


    def _observe(self) -> np.ndarray:
        if self._net_index >= len(self._nets):
            return np.zeros(self.obs_config.flat_size, dtype=np.float32)

        length_slack = 0.0
        if self._target_ref_length > 0:
            length_slack = max(0.0, self._target_ref_length - self._routed_len)

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
            length_slack=length_slack,
            # Mirror step()'s own choice exactly: it turns from _prev_heading
            # while colliding and from the target bearing otherwise. None means
            # "the target bearing", which the obs frame already encodes as +x.
            base_heading=self._prev_heading if self._collides else None,
            geodesic_dist=self._geodesic_dist(self._pos),
            clearance_now=self._clearance_at(self._pos),
            clearance_ahead=self._clearance_at(self._step_ahead()),
            geodesic_direction=self._geodesic_direction(),
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
            "collision_run": self._collision_run,
            "routed_length_nm": self._routed_len,
            "ripup_count": self._ripup_count,
            "ripups_performed": list(self._ripups_performed),
            "failure_reasons": dict(self._failure_reasons),
            "failure_progress": dict(self._failure_progress),
            "failure_travel_nm": dict(self._failure_travel),
            "fix_refusals": dict(self._fix_refusals),
            "route_started": self._start_ok,
            "start_pad_id": self._start_pad_id,
            "target_pad_id": self._target_id,
            "closest_approach_nm": dict(self._closest_approach_nm),
            "snap_radius_nm": self.snap_radius_nm,
        }


    def close(self):
        if self._route_active:
            self.bridge.stop_routing()
            self._route_active = False
