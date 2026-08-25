"""`BatchedRouterWorld` -- the routing environment, as tensors.

`B` independent boards advance in lockstep, each with `K` simultaneously
active routing heads. One `step()` therefore resolves ``B * K`` routing
decisions with a fixed number of gathers and scatters and **no Python loop over
nets, cells, or boards**.

That is the difference between this and every previous thread in the repo. The
raster env (`pcbworld/environment.py`) and the line env
(`pcbworld/env/line_route_env.py`) are both one-board-per-process Python loops;
with `nproc`=2 on Colab that caps the whole project at ~2 environment steps in
flight. Here `B=64, K=8` is 512. Nothing else about the plan changes that
number, and sample starvation is the shared root cause of every stalled
direction in `docs/HANDOVER.md`'s open-questions list.

Two invariants the rest of the system relies on:

1. **Occupancy is the single source of truth for legality.** Pads, keepouts,
   pours, committed copper and vias all live in `occ` with the same encoding
   (`0` free, `net+1` owned, `-1` keepout). Nothing checks legality any other
   way, so there is exactly one place for a legality bug to live.
2. **A move either happens completely or not at all.** A rejected move leaves
   `occ` and the head untouched and reports `rejected=True`. This is what makes
   the per-step collision signal usable as a dense reward -- the same property
   `docs/RL_PLAN.md`'s Gate B measured on PNS (collision fired on 100% of
   rejected fixes and 0% of accepted ones, n=99).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from neuroroute.world import geometry as geo
from neuroroute.world.generator import GeneratedBoard
from neuroroute.world.spec import (
    KIND_DIFF_PAIR,
    NUM_DIRECTIONS,
    REFINE_OFFSETS,
    NUM_STEPS,
    OCC_FREE,
    PHASE_CONNECT,
    PHASE_DONE,
    PHASE_REFINE,
    STEP_LENGTHS,
    BoardSpec,
)

STATUS_PENDING = 0
STATUS_ACTIVE = 1
STATUS_DONE = 2
STATUS_FAILED = 3

#: Length a via adds to a net, in cell-equivalents. Vias are not free -- they
#: cost board area, fabrication yield and signal integrity -- and a policy with
#: a free via action will drill its way out of every problem instead of
#: learning to route.
VIA_LENGTH_COST = 4.0


@dataclass
class WorldConfig:
    batch_size: int = 32
    #: Simultaneously active routing heads per board. The direct multiplier on
    #: sample throughput; capped by geodesic-cache memory (see `head_geo`).
    max_heads: int = 8
    max_nets: int = 256
    #: Polyline vertices stored per leg. Caps route complexity; a route that
    #: needs more than this is almost certainly pathological anyway.
    max_vertices: int = 64
    max_steps_per_net: int = 96
    #: Cells within which a head may snap directly onto its target pad.
    #: MUST be >= max(STEP_LENGTHS) / 2, or a long step jumps clean over the
    #: snap zone and the head orbits its target forever -- a config bug that
    #: reads exactly like a learning failure (docs/HANDOVER.md).
    snap_radius: int = 4
    geodesic_downsample: int = 4
    geodesic_iterations: int = 96
    #: Recompute a head's cached geodesic field every N of that head's steps.
    #: 0 disables refresh (compute once when the head is assigned). The field
    #: ignores other nets' in-progress copper between refreshes, which is the
    #: same approximation `pcbworld/environment.py` makes deliberately.
    geodesic_refresh: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.snap_radius * 2 < max(STEP_LENGTHS):
            raise ValueError(
                f"snap_radius={self.snap_radius} < max step {max(STEP_LENGTHS)}/2; "
                "a head would step over its own snap zone"
            )


@dataclass
class _LegPlan:
    """What one leg of every head slot would do, before anything is written."""

    live: torch.Tensor
    want_via: torch.Tensor
    want_move: torch.Tensor
    legal: torch.Tensor
    pos: torch.Tensor
    newpos: torch.Tensor
    seg_len: torch.Tensor
    #: Flat `occ` indices this leg would write, and which of them are real.
    claim_flat: torch.Tensor
    claim_valid: torch.Tensor


@dataclass
class StepResult:
    """Per-head outcome of one `step()`. All tensors are (B, K)."""

    rejected: torch.Tensor
    moved: torch.Tensor
    via_placed: torch.Tensor
    arrived: torch.Tensor
    progress: torch.Tensor
    exhausted: torch.Tensor
    active: torch.Tensor
    #: (B,) -- nets finished / failed on this step, for board-level reward.
    nets_done: torch.Tensor
    nets_failed: torch.Tensor


class BatchedRouterWorld:
    """The lattice routing world for a batch of boards."""

    def __init__(self, spec: BoardSpec, cfg: WorldConfig):
        self.spec = spec
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tables = geo.build_tables(spec.rules, self.device)

        B, K, N = cfg.batch_size, cfg.max_heads, cfg.max_nets
        L, H, W = spec.num_layers, spec.height_cells, spec.width_cells
        ds = cfg.geodesic_downsample
        h, w = max(1, H // ds), max(1, W // ds)
        dev = self.device

        z = lambda *s, dt=torch.int64: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731

        self.occ = z(B, L, H, W, dt=torch.int16)
        self.static = z(B, L, H, W, dt=torch.int16)
        self.pour = torch.zeros(B, L, H, W, dtype=torch.bool, device=dev)

        self.net_src = z(B, N, 2, 3)
        self.net_dst = z(B, N, 2, 3)
        self.net_kind = z(B, N)
        self.net_width = z(B, N)
        self.net_group = torch.full((B, N), -1, dtype=torch.int64, device=dev)
        self.net_status = torch.full((B, N), STATUS_DONE, dtype=torch.int64, device=dev)
        self.net_valid = torch.zeros(B, N, dtype=torch.bool, device=dev)
        self.net_len = torch.zeros(B, N, 2, dtype=torch.float32, device=dev)
        self.net_target_len = torch.full((B, N), -1.0, dtype=torch.float32, device=dev)
        self.net_vias = z(B, N)
        self.net_split = torch.zeros(B, N, dtype=torch.float32, device=dev)
        self.net_coupled_steps = torch.zeros(B, N, dtype=torch.float32, device=dev)

        self.head_net = torch.full((B, K), -1, dtype=torch.int64, device=dev)
        self.head_pos = z(B, K, 2, 3)
        self.head_steps = z(B, K)
        self.head_phase = torch.full((B, K), PHASE_CONNECT, dtype=torch.int64, device=dev)
        self.head_done = torch.zeros(B, K, 2, dtype=torch.bool, device=dev)
        self.head_geo = torch.full(
            (B, K, 2, L, h, w), float("inf"), dtype=torch.float32, device=dev
        )
        self.head_prev_dist = torch.zeros(B, K, 2, dtype=torch.float32, device=dev)

        self.route_v = z(B, N, 2, cfg.max_vertices, 3, dt=torch.int16)
        self.route_n = z(B, N, 2)

        self.step_count = 0
        self._geo_shape = (h, w)
        # Scratch for `resolve_claims`. Allocated once: it is the size of
        # the occupancy grid and is refilled, not reallocated, each step.
        self._claim_buf = torch.empty(B * L * H * W, dtype=torch.int32, device=dev)

    # -- properties ---------------------------------------------------------

    @property
    def num_layers(self) -> int:
        return self.spec.num_layers

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.spec.num_layers, self.spec.height_cells, self.spec.width_cells

    # -- loading ------------------------------------------------------------

    def load(self, boards: Sequence[GeneratedBoard]) -> None:
        """Install a fresh batch of boards and reset all routing state.

        `boards` must be exactly `batch_size` long. Boards with fewer nets than
        `max_nets` leave the tail marked invalid; nothing downstream reads it,
        because every consumer masks on `net_valid`.
        """
        cfg = self.cfg
        if len(boards) != cfg.batch_size:
            raise ValueError(f"expected {cfg.batch_size} boards, got {len(boards)}")

        self.net_valid.zero_()
        self.net_status.fill_(STATUS_DONE)
        self.net_len.zero_()
        self.net_target_len.fill_(-1.0)
        self.net_vias.zero_()
        self.net_split.zero_()
        self.net_coupled_steps.zero_()
        self.net_group.fill_(-1)
        self.route_n.zero_()
        self.head_net.fill_(-1)
        self.head_steps.zero_()
        self.head_phase.fill_(PHASE_CONNECT)
        self.head_done.zero_()
        self.head_geo.fill_(float("inf"))
        self.step_count = 0

        static = np.stack([b.static for b in boards], axis=0)
        pour = np.stack([b.pour_mask for b in boards], axis=0)
        self.static.copy_(torch.from_numpy(static).to(self.device))
        self.pour.copy_(torch.from_numpy(pour).to(self.device))
        self.occ.copy_(self.static)

        src = np.zeros((cfg.batch_size, cfg.max_nets, 2, 3), dtype=np.int64)
        dst = np.zeros_like(src)
        kind = np.zeros((cfg.batch_size, cfg.max_nets), dtype=np.int64)
        width = np.zeros_like(kind)
        group = np.full_like(kind, -1)
        valid = np.zeros((cfg.batch_size, cfg.max_nets), dtype=bool)

        for bi, board in enumerate(boards):
            for ni, net in enumerate(board.netlist.nets[: cfg.max_nets]):
                legs = net.endpoints()
                for li, (s, d) in enumerate(legs):
                    src[bi, ni, li] = s
                    dst[bi, ni, li] = d
                if len(legs) == 1:
                    src[bi, ni, 1] = src[bi, ni, 0]
                    dst[bi, ni, 1] = dst[bi, ni, 0]
                kind[bi, ni] = net.kind
                width[bi, ni] = net.width_class
                group[bi, ni] = net.group_id
                valid[bi, ni] = True

        t = lambda a: torch.from_numpy(a).to(self.device)  # noqa: E731
        self.net_src.copy_(t(src))
        self.net_dst.copy_(t(dst))
        self.net_kind.copy_(t(kind))
        self.net_width.copy_(t(width))
        self.net_group.copy_(t(group))
        self.net_valid.copy_(t(valid))
        self.net_status = torch.where(
            self.net_valid,
            torch.full_like(self.net_status, STATUS_PENDING),
            torch.full_like(self.net_status, STATUS_DONE),
        )

    # -- scheduling ---------------------------------------------------------

    def idle_slots(self) -> torch.Tensor:
        """(B, K) bool -- head slots available for assignment."""
        return self.head_net < 0

    def pending_mask(self) -> torch.Tensor:
        """(B, N) bool -- nets not yet routed and not currently active."""
        return self.net_valid & (self.net_status == STATUS_PENDING)

    def assign(self, net_idx: torch.Tensor) -> torch.Tensor:
        """Bind nets to idle head slots.

        Parameters
        ----------
        net_idx : (B, K) int64
            Net to place in each slot; `-1` leaves the slot idle. Entries
            targeting an already-active or finished net, or a non-idle slot,
            are silently dropped -- the scheduler head is a *policy*, and a
            policy must not be able to corrupt world state with a bad output.

        Returns
        -------
        (B, K) bool -- which assignments were actually taken.
        """
        B, K = self.head_net.shape
        idle = self.idle_slots()
        bb = torch.arange(B, device=self.device).view(B, 1).expand(B, K)

        want = net_idx.clamp(0, self.cfg.max_nets - 1)
        legal = (
            (net_idx >= 0)
            & idle
            & self.net_valid[bb, want]
            & (self.net_status[bb, want] == STATUS_PENDING)
        )
        # Two slots on one board must not claim the same net in one call.
        first = torch.zeros_like(legal)
        seen = torch.zeros(B, self.cfg.max_nets, dtype=torch.bool, device=self.device)
        for k in range(K):
            take = legal[:, k] & ~seen[torch.arange(B, device=self.device), want[:, k]]
            first[:, k] = take
            seen[torch.arange(B, device=self.device), want[:, k]] |= take
        legal = first

        self.head_net = torch.where(legal, net_idx, self.head_net)
        self.head_steps = torch.where(legal, torch.zeros_like(self.head_steps), self.head_steps)
        self.head_phase = torch.where(
            legal, torch.full_like(self.head_phase, PHASE_CONNECT), self.head_phase
        )
        self.head_done = torch.where(legal.unsqueeze(-1), torch.zeros_like(self.head_done), self.head_done)

        # Start each leg at its source pad.
        sel = self.net_src[bb, want]  # (B, K, 2, 3)
        self.head_pos = torch.where(legal.view(B, K, 1, 1), sel, self.head_pos)

        # A single net's leg 1 is a mirror of leg 0 and must never be routed.
        is_pair = self.net_kind[bb, want] == KIND_DIFF_PAIR
        self.head_done[..., 1] |= legal & ~is_pair

        self.net_status[bb[legal], net_idx[legal]] = STATUS_ACTIVE

        if legal.any():
            self._refresh_geodesic(legal)
            self._seed_routes(legal)
        return legal

    def _seed_routes(self, mask: torch.Tensor) -> None:
        """Record each newly-assigned leg's source pad as vertex 0."""
        B, K = mask.shape
        bb = torch.arange(B, device=self.device).view(B, 1).expand(B, K)
        net = self.head_net.clamp_min(0)
        for leg in range(2):
            sel = mask & ~self.head_done[..., leg]
            if not sel.any():
                continue
            b_i = bb[sel]
            n_i = net[sel]
            self.route_v[b_i, n_i, leg, 0] = self.head_pos[sel][:, leg].to(torch.int16)
            self.route_n[b_i, n_i, leg] = 1

    def _refresh_geodesic(self, mask: torch.Tensor) -> None:
        """Recompute cached cost-to-go fields for the masked head slots.

        Cost is why this is cached rather than recomputed per step: a field is
        `iterations` relaxations over ``L x h x w``, and the field depends only
        on the target and the obstacle set, not on which step the head is on
        (the same reasoning behind `pcbworld/environment.py`'s
        `geodesic_cache`).
        """
        if not bool(mask.any()):
            return
        B, K = mask.shape
        L, H, W = self.shape
        ds = self.cfg.geodesic_downsample
        bb = torch.arange(B, device=self.device).view(B, 1).expand(B, K)

        for leg in range(2):
            sel = mask & ~self.head_done[..., leg]
            if not bool(sel.any()):
                continue
            b_i = bb[sel]
            n_i = self.head_net[sel].clamp_min(0)
            tgt = self.net_dst[b_i, n_i, leg]  # (m, 3)

            own = (n_i + 1).view(-1, 1, 1, 1).to(self.occ.dtype)
            sub = self.occ[b_i]
            blocked = (sub != OCC_FREE) & (sub != own)

            fld = geo.geodesic_field(
                blocked,
                tgt[:, 0],
                tgt[:, 1],
                tgt[:, 2],
                iterations=self.cfg.geodesic_iterations,
                downsample=ds,
                upsample=False,
            )
            self.head_geo[sel, leg] = fld
            self.head_prev_dist[sel, leg] = self._geo_at(fld, self.head_pos[sel][:, leg])

    def _geo_at(self, coarse: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Sample a coarse geodesic field at fine-lattice positions, in fine
        cell units. (m,)

        Bilinear, not nearest-neighbour: at `downsample=4` a nearest lookup is
        constant across each 4x4 block, so the potential-based shaping term
        would be exactly zero for three out of four steps and then jump. The
        policy would see a sparse, jagged reward where a smooth one exists.
        """
        ds = self.cfg.geodesic_downsample
        val = geo.sample_coarse(coarse, pos[:, 0], pos[:, 1], pos[:, 2], ds)
        return torch.nan_to_num(val * ds, posinf=1e6, nan=1e6)

    # -- rip-up -------------------------------------------------------------

    def ripup(self, net_idx: torch.Tensor) -> torch.Tensor:
        """Remove one net's copper per board and return it to the pending pool.

        `docs/RL_PLAN.md` identifies rip-up-and-reroute as the density lever
        once layer changes closed. Here layers work, and rip-up is *also*
        available -- the two are complementary, not alternatives: rip-up fixes
        a bad ordering decision, a via fixes a bad geometric one.

        Returns (B,) bool -- which rip-ups were performed.
        """
        B = net_idx.shape[0]
        ok = (net_idx >= 0) & self.net_valid.gather(1, net_idx.clamp_min(0).view(B, 1)).squeeze(1)
        ok &= self.net_status.gather(1, net_idx.clamp_min(0).view(B, 1)).squeeze(1) == STATUS_DONE
        if not bool(ok.any()):
            return ok

        b_i = torch.arange(B, device=self.device)[ok]
        n_i = net_idx[ok]
        own = (n_i + 1).view(-1, 1, 1, 1).to(self.occ.dtype)
        sub = self.occ[b_i]
        stat = self.static[b_i]
        # Clear this net's copper but restore its pads from the static layer --
        # a ripped-up net still has to start somewhere next time.
        cleared = torch.where(sub == own, torch.zeros_like(sub), sub)
        self.occ[b_i] = torch.where(stat == own, stat, cleared)

        self.net_status[b_i, n_i] = STATUS_PENDING
        self.net_len[b_i, n_i] = 0.0
        self.net_vias[b_i, n_i] = 0
        self.net_split[b_i, n_i] = 0.0
        self.net_coupled_steps[b_i, n_i] = 0.0
        self.route_n[b_i, n_i] = 0
        return ok

    # -- the step -----------------------------------------------------------

    def step(
        self,
        direction: torch.Tensor,
        step_class: torch.Tensor,
        layer_action: torch.Tensor,
        via_class: torch.Tensor,
        width_class: torch.Tensor,
        couple: torch.Tensor,
        bearing: torch.Tensor | None = None,
    ) -> StepResult:
        """Advance every active head by one decision.

        All arguments are (B, K) int64.

        * `direction` is **egocentric**: 0 means "down this leg's own geodesic
          gradient", so the same index means the same intent on every board and
          at every pose. This is why a near-zero-init policy starts at the
          greedy baseline rather than below it.
        * `layer_action` 0 = stay; `j > 0` = place a via and move to layer
          `j - 1`. A layer change *is* a via -- there is no separate action,
          because there is no such thing as changing layer without one.
        * `couple` only means anything for a differential pair.
        * `bearing` (B, K) optionally supplies leg 0's reference bearing. The
          observation builder already computes it, and recomputing it here
          would be a second raycast per head per step for no new information.

        Resolution is **plan -> arbitrate -> commit**, never check-and-write
        per head. With `K` heads acting at once, per-head check-and-write lets
        two heads both pass a legality test against the pre-step occupancy and
        then both write the same cell; one silently loses, its head advances
        anyway, and its route is left with a gap that nothing detects until a
        flood fill much later. That bug was real, and this structure is what
        makes it impossible rather than unlikely.
        """
        B, K = self.head_net.shape
        L, H, W = self.shape
        dev = self.device
        M = B * K
        bb = torch.arange(B, device=dev).view(B, 1).expand(B, K)

        active = self.head_net >= 0
        net = self.head_net.clamp_min(0)
        is_pair = active & (self.net_kind[bb, net] == KIND_DIFF_PAIR)
        # A net's own required width is a floor, not a suggestion: the policy
        # may widen a trace but never narrow one below its electrical spec.
        wclass = torch.maximum(width_class, self.net_width[bb, net])
        coupled = is_pair & (couple > 0)

        self.net_coupled_steps[bb[coupled], net[coupled]] += 1.0
        split = is_pair & ~coupled
        self.net_split[bb[split], net[split]] += 1.0

        ds = self.cfg.geodesic_downsample
        b_flat = bb.reshape(M)
        n_flat = net.reshape(M)
        w_flat = wclass.reshape(M)
        coupled_f = coupled.reshape(M)

        # A coupled pair applies leg 0's bearing to both legs so they travel
        # parallel; an uncoupled pair lets each leg follow its own gradient,
        # which is what lets the policy fan the legs around an obstacle and
        # re-converge afterwards.
        bearings = []
        for leg in range(2):
            if leg == 0 and bearing is not None:
                bearings.append(bearing)
                continue
            fld = self.head_geo[:, :, leg].reshape(M, L, *self._geo_shape)
            pos = self.head_pos[:, :, leg].reshape(M, 3)
            free = geo.raycast(
                self.occ, b_flat, pos[:, 0], pos[:, 1], pos[:, 2], n_flat, self.tables, w_flat
            )
            bearings.append(
                geo.bearing_from_field(
                    fld, pos[:, 0], pos[:, 1], pos[:, 2], self.tables, ds, free_units=free
                ).view(B, K)
            )

        # --- plan: legality only, nothing written ---------------------------
        plans = []
        for leg in range(2):
            live = active & ~self.head_done[..., leg]
            if leg == 1:
                live = live & is_pair
            bear = torch.where(coupled, bearings[0], bearings[leg])
            abs_dir = (bear + direction) % NUM_DIRECTIONS
            plans.append(
                self._plan_leg(
                    leg, live, net, abs_dir, step_class, layer_action, via_class, wclass
                )
            )

        # --- coupled pairs move atomically ----------------------------------
        joint = plans[0].legal & plans[1].legal
        go = []
        for leg in range(2):
            g = plans[leg].legal & plans[leg].live
            go.append(torch.where(coupled_f, joint & plans[leg].live, g))

        # --- arbitrate between simultaneous heads ---------------------------
        ok = geo.resolve_claims(
            self.occ,
            self._claim_buf,
            [plans[0].claim_flat, plans[1].claim_flat],
            [plans[0].claim_valid & go[0].view(M, 1), plans[1].claim_valid & go[1].view(M, 1)],
            [n_flat, n_flat],
        )
        joint_ok = ok[0] & ok[1]
        for leg in range(2):
            go[leg] = go[leg] & torch.where(coupled_f, joint_ok, ok[leg])

        # --- commit ----------------------------------------------------------
        rejected = torch.zeros(M, dtype=torch.bool, device=dev)
        moved = torch.zeros_like(rejected)
        via_placed = torch.zeros_like(rejected)
        arrived_leg = torch.zeros(B, K, 2, dtype=torch.bool, device=dev)
        progress = torch.zeros(M, dtype=torch.float32, device=dev)

        for leg in range(2):
            plan = plans[leg]
            prog, mv, vp, arr = self._commit_leg(leg, plan, go[leg], b_flat, n_flat, w_flat)
            arrived_leg[..., leg] = arr.view(B, K)
            moved |= mv
            via_placed |= vp
            progress += prog
            rejected |= plan.live & ~go[leg]

        rejected = rejected.view(B, K)
        moved = moved.view(B, K)
        via_placed = via_placed.view(B, K)
        progress = progress.view(B, K)

        self.head_steps = torch.where(active, self.head_steps + 1, self.head_steps)

        # A leg that reached its pad stops; a net is done when every live leg
        # has. `head_done` starts True for leg 1 of a single net, so the
        # all-legs-done test needs no is_pair special case.
        self.head_done |= arrived_leg
        net_complete = active & self.head_done.all(dim=-1)
        exhausted = active & ~net_complete & (self.head_steps >= self.cfg.max_steps_per_net)

        nets_done = self._retire(net_complete, STATUS_DONE)
        nets_failed = self._retire(exhausted, STATUS_FAILED)

        if self.cfg.geodesic_refresh > 0:
            due = active & ~net_complete & ~exhausted
            due &= (self.head_steps % self.cfg.geodesic_refresh) == 0
            if bool(due.any()):
                self._refresh_geodesic(due)

        self.step_count += 1
        return StepResult(
            rejected=rejected,
            moved=moved,
            via_placed=via_placed,
            arrived=net_complete,
            progress=progress,
            exhausted=exhausted,
            active=active,
            nets_done=nets_done,
            nets_failed=nets_failed,
        )

    def _plan_leg(
        self,
        leg: int,
        live: torch.Tensor,
        net: torch.Tensor,
        abs_dir: torch.Tensor,
        step_class: torch.Tensor,
        layer_action: torch.Tensor,
        via_class: torch.Tensor,
        width_class: torch.Tensor,
    ) -> "_LegPlan":
        """Work out what one leg of every head slot *would* do. Writes nothing."""
        B, K = live.shape
        L, H, W = self.shape
        dev = self.device
        M = B * K
        flat = lambda t: t.reshape(M, *t.shape[2:])  # noqa: E731

        pos = flat(self.head_pos[:, :, leg])
        b_i = torch.arange(B, device=dev).view(B, 1).expand(B, K).reshape(M)
        n_i = flat(net)
        d_i = flat(abs_dir)
        s_i = flat(step_class)
        la_i = flat(layer_action)
        vc_i = flat(via_class)
        wc_i = flat(width_class)
        live_f = flat(live)

        want_via = live_f & (la_i > 0) & ((la_i - 1) != pos[:, 0])
        want_move = live_f & ~want_via

        tgt_layer = (la_i - 1).clamp(0, L - 1)
        via_lo, via_hi = self._via_span(pos[:, 0], tgt_layer)
        via_flat, via_valid = geo.via_claims(
            self.occ, b_i, via_lo, via_hi, pos[:, 1], pos[:, 2], vc_i, self.tables
        )
        via_valid = via_valid & want_via.view(M, 1)
        via_legal = geo.claims_passable(self.occ, via_flat, via_valid, n_i)

        move_flat, move_valid = geo.move_claims(
            self.occ, b_i, pos[:, 0], pos[:, 1], pos[:, 2], d_i, s_i, wc_i, self.tables
        )
        move_valid = move_valid & want_move.view(M, 1)
        # `claims_passable` only sees in-bounds cells, so a move that runs off
        # the board would trivially pass. `check_moves` counts clear travel
        # units with out-of-bounds treated as blocked, which catches it.
        _, free_units = geo.check_moves(
            self.occ, b_i, pos[:, 0], pos[:, 1], pos[:, 2], d_i, s_i, wc_i, n_i, self.tables
        )
        move_legal = geo.claims_passable(self.occ, move_flat, move_valid, n_i)
        move_legal = move_legal & (free_units >= self.tables.step_len[s_i])

        unit = self.tables.path[d_i, 0, 0]
        n_travel = self.tables.step_len[s_i]
        step_vec = unit * n_travel.view(M, 1)
        move_pos = torch.stack(
            [pos[:, 0], pos[:, 1] + step_vec[:, 0], pos[:, 2] + step_vec[:, 1]], dim=-1
        )
        via_pos = torch.stack([tgt_layer, pos[:, 1], pos[:, 2]], dim=-1)
        newpos = torch.where(want_via.view(M, 1), via_pos, move_pos)

        diag = (unit[:, 0] != 0) & (unit[:, 1] != 0)
        move_len = n_travel.float() * torch.where(
            diag, torch.full((M,), math.sqrt(2.0), device=dev), torch.ones(M, device=dev)
        )
        seg_len = torch.where(want_via, torch.full_like(move_len, VIA_LENGTH_COST), move_len)

        legal = ((want_via & via_legal) | (want_move & move_legal)) & live_f

        return _LegPlan(
            live=live_f,
            want_via=want_via,
            want_move=want_move,
            legal=legal,
            pos=pos,
            newpos=newpos,
            seg_len=seg_len,
            claim_flat=torch.cat([via_flat, move_flat], dim=1),
            claim_valid=torch.cat([via_valid, move_valid], dim=1),
        )

    def _commit_leg(
        self,
        leg: int,
        plan: "_LegPlan",
        go: torch.Tensor,
        b_i: torch.Tensor,
        n_i: torch.Tensor,
        w_i: torch.Tensor,
    ):
        """Write one leg's accepted actions, then try to snap onto the pad."""
        B, K = self.head_net.shape
        L, H, W = self.shape
        dev = self.device
        M = B * K

        geo.write_claims(self.occ, plan.claim_flat, plan.claim_valid, n_i, go)

        pos = torch.where(go.view(M, 1), plan.newpos, plan.pos)
        did_via = go & plan.want_via
        did_move = go & plan.want_move
        self.net_vias[b_i[did_via], n_i[did_via]] += 1
        self.net_len[b_i[go], n_i[go], leg] += plan.seg_len[go]
        self._append_vertex(b_i, n_i, leg, pos, go)

        # --- snap onto the target pad ---------------------------------------
        tgt = self.net_dst[b_i, n_i, leg]
        near = (
            plan.live
            & (pos[:, 0] == tgt[:, 0])
            & ((pos[:, 1] - tgt[:, 1]).abs() <= self.cfg.snap_radius)
            & ((pos[:, 2] - tgt[:, 2]).abs() <= self.cfg.snap_radius)
        )
        arrived = torch.zeros(M, dtype=torch.bool, device=dev)
        if bool(near.any()):
            sflat, svalid = geo.segment_claims(
                self.occ, b_i, pos[:, 0], pos[:, 1], pos[:, 2], tgt[:, 1], tgt[:, 2], w_i, self.tables
            )
            svalid = svalid & near.view(M, 1)
            clear = geo.claims_passable(self.occ, sflat, svalid, n_i)
            cand = near & clear
            (snap_ok,) = geo.resolve_claims(
                self.occ, self._claim_buf, [sflat], [svalid & cand.view(M, 1)], [n_i]
            )
            snap = cand & snap_ok
            geo.write_claims(self.occ, sflat, svalid, n_i, snap)
            dy = (tgt[:, 1] - pos[:, 1]).float()
            dx = (tgt[:, 2] - pos[:, 2]).float()
            self.net_len[b_i[snap], n_i[snap], leg] += torch.sqrt(dy[snap] ** 2 + dx[snap] ** 2)
            pos = torch.where(snap.view(M, 1), tgt, pos)
            self._append_vertex(b_i, n_i, leg, pos, snap)
            arrived = snap

        self.head_pos[:, :, leg] = pos.view(B, K, 3)

        # --- potential-based progress ---------------------------------------
        fld = self.head_geo[:, :, leg].reshape(M, L, *self._geo_shape)
        new_dist = self._geo_at(fld, pos)
        prev = self.head_prev_dist[:, :, leg].reshape(M)
        prog = torch.where(plan.live, prev - new_dist, torch.zeros_like(prev))
        self.head_prev_dist[:, :, leg] = new_dist.view(B, K)

        return prog, did_move, did_via, arrived

    def _via_span(self, cur: torch.Tensor, tgt: torch.Tensor):
        if self.spec.layers.through_only:
            lo = torch.zeros_like(cur)
            hi = torch.full_like(cur, self.num_layers - 1)
            return lo, hi
        return torch.minimum(cur, tgt), torch.maximum(cur, tgt)

    def _append_vertex(
        self,
        b_i: torch.Tensor,
        n_i: torch.Tensor,
        leg: int,
        pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Record an accepted head position as a polyline vertex.

        The polyline is what the KiCad exporter emits and what the refine phase
        edits, so it has to be kept in step with `occ` rather than
        reconstructed from it -- reconstructing a route from an occupancy grid
        is ambiguous once two nets touch.
        """
        if not bool(mask.any()):
            return
        b = b_i[mask]
        n = n_i[mask]
        cnt = self.route_n[b, n, leg]
        room = cnt < self.cfg.max_vertices
        b, n, cnt = b[room], n[room], cnt[room]
        self.route_v[b, n, leg, cnt] = pos[mask][room].to(torch.int16)
        self.route_n[b, n, leg] = cnt + 1

    def _retire(self, mask: torch.Tensor, status: int) -> torch.Tensor:
        """Free head slots whose nets finished (or failed). Returns (B,) counts."""
        if not bool(mask.any()):
            return torch.zeros(self.head_net.shape[0], dtype=torch.int64, device=self.device)
        B, K = mask.shape
        bb = torch.arange(B, device=self.device).view(B, 1).expand(B, K)
        self.net_status[bb[mask], self.head_net[mask]] = status
        self.head_net = torch.where(mask, torch.full_like(self.head_net, -1), self.head_net)
        self.head_phase = torch.where(
            mask,
            torch.full_like(self.head_phase, PHASE_REFINE if status == STATUS_DONE else PHASE_DONE),
            self.head_phase,
        )
        return mask.sum(dim=1)

    # -- readout ------------------------------------------------------------

    def completion(self) -> torch.Tensor:
        """(B,) fraction of valid nets routed. **This, not reward, is the
        number to track** -- `docs/RL_PLAN.md` measured a random policy scoring
        -330 reward against greedy's -177 while completing *more* nets."""
        done = ((self.net_status == STATUS_DONE) & self.net_valid).sum(dim=1).float()
        total = self.net_valid.sum(dim=1).float()
        # A board with no nets is vacuously complete. Scoring it 0% instead
        # lets a degenerate board masquerade as a routing failure, which is
        # exactly how a generator bug hides inside a training curve.
        return torch.where(total > 0, done / total.clamp_min(1.0), torch.ones_like(done))

    def board_stats(self) -> dict[str, torch.Tensor]:
        valid = self.net_valid
        done = valid & (self.net_status == STATUS_DONE)
        pair = valid & (self.net_kind == KIND_DIFF_PAIR)
        steps = self.net_coupled_steps + self.net_split
        return {
            "completion": self.completion(),
            "routed": done.sum(dim=1),
            "failed": (valid & (self.net_status == STATUS_FAILED)).sum(dim=1),
            "pending": (valid & (self.net_status == STATUS_PENDING)).sum(dim=1),
            "vias": self.net_vias.sum(dim=1),
            "wirelength": (self.net_len.sum(dim=-1) * done.float()).sum(dim=1),
            "split_fraction": (
                (self.net_split * pair.float()).sum(dim=1)
                / steps.mul(pair.float()).sum(dim=1).clamp_min(1.0)
            ),
        }

    def pair_gap_error(self) -> torch.Tensor:
        """(B, N) mean |gap - nominal| / nominal over each pair's routed length.

        Measured from the polylines rather than assumed from the action, so a
        policy that decouples, drifts, and re-couples is scored on the copper
        it actually produced.
        """
        B, N, _, V, _ = self.route_v.shape
        v0 = self.route_v[:, :, 0].float()
        v1 = self.route_v[:, :, 1].float()
        n0 = self.route_n[:, :, 0]
        n1 = self.route_n[:, :, 1]
        k = torch.minimum(n0, n1).clamp_min(1)
        idx = torch.arange(V, device=self.device).view(1, 1, V)
        m = (idx < k.unsqueeze(-1)).float()

        d = torch.linalg.vector_norm(v0[..., 1:] - v1[..., 1:], dim=-1)  # (B, N, V)
        nominal = torch.ones_like(d)
        err = ((d - nominal).abs() / nominal.clamp_min(1e-6) * m).sum(-1) / m.sum(-1).clamp_min(1.0)
        is_pair = self.net_kind == KIND_DIFF_PAIR
        return torch.where(is_pair, err, torch.zeros_like(err))

    def pair_skew(self) -> torch.Tensor:
        """(B, N) |len_P - len_N| in cells, 0 for non-pairs."""
        skew = (self.net_len[..., 0] - self.net_len[..., 1]).abs()
        return torch.where(self.net_kind == KIND_DIFF_PAIR, skew, torch.zeros_like(skew))

    def group_length_error(self) -> torch.Tensor:
        """(B, N) |routed length - group target| in cells.

        The target is the group's **longest routed member**, resolved here at
        runtime rather than fixed in the netlist -- which is what length
        matching actually means, and matches how `docs/ROUTER_CAPABILITIES.md`
        describes reading a tune target off the reference net's real routed
        length.
        """
        B, N = self.net_group.shape
        length = self.net_len.sum(dim=-1)
        gid = self.net_group
        has = gid >= 0
        out = torch.zeros_like(length)
        ngroups = int(gid.max().item()) + 1 if bool(has.any()) else 0
        for g in range(ngroups):
            m = has & (gid == g)
            if not bool(m.any()):
                continue
            longest = (length * m.float()).amax(dim=1, keepdim=True)
            self.net_target_len = torch.where(m, longest.expand_as(length), self.net_target_len)
            out = torch.where(m, (length - longest).abs(), out)
        return out

    # -- refine phase: length tuning, diff-pair repair, wirelength ----------
    #
    # DESIGN.md section 4. Once a net is topologically connected it enters a
    # SECOND MDP over the same board: drag a polyline vertex sideways. That is
    # literally what a person does in the KiCad editor, and it is the smallest
    # action that changes a route's LENGTH without re-solving its connectivity.
    #
    # A meander is not a primitive here. A meander is what an optimal policy of
    # this MDP looks like -- alternating drags of adjacent vertices, with
    # amplitude and spacing chosen by the policy. That is the strongest form of
    # "the policy learned it" available, and unlike asking length matching to
    # emerge from the connect phase alone it is dense-reward and short-horizon,
    # so it is actually trainable.
    #
    # Deliberately NOT `MODE_TUNE_SINGLE`: KiCad's own meander solver would
    # make length matching "work" without the policy ever learning it.

    def refine(
        self,
        net_idx: torch.Tensor,
        leg_idx: torch.Tensor,
        vertex_idx: torch.Tensor,
        offset_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Drag one interior polyline vertex sideways, one net per board.

        All arguments are (B,). `net_idx < 0` is a no-op for that board.
        `vertex_idx` selects an interior vertex; `offset_idx` indexes
        `REFINE_OFFSETS` and is applied perpendicular to the local run of the
        route.

        Returns (B,) bool -- which drags were accepted.

        Erase-then-check-then-restore is safe here because refine acts on at
        most one net per board: `stamp_segments(erase=True)` only clears cells
        this net owns, and the restore path re-stamps the identical cells,
        which are guaranteed still free because nothing else wrote in between.
        """
        B = self.head_net.shape[0]
        dev = self.device
        idx = torch.arange(B, device=dev)

        n = net_idx.clamp_min(0)
        leg = leg_idx.clamp(0, 1)
        count = self.route_n[idx, n, leg]

        # Only interior vertices can be dragged: the endpoints are pads.
        v = vertex_idx.clamp_min(1)
        ok = (
            (net_idx >= 0)
            & self.net_valid[idx, n]
            & (self.net_status[idx, n] == STATUS_DONE)
            & (count >= 3)
            & (v <= (count - 2).clamp_min(1))
        )
        if not bool(ok.any()):
            return torch.zeros(B, dtype=torch.bool, device=dev)

        v = torch.minimum(v, (count - 2).clamp_min(1))
        prev = self.route_v[idx, n, leg, (v - 1).clamp_min(0)].long()
        cur = self.route_v[idx, n, leg, v].long()
        nxt = self.route_v[idx, n, leg, (v + 1).clamp(0, self.cfg.max_vertices - 1)].long()

        # A vertex where the layer changes is a via, not a corner -- dragging
        # it would move the via and orphan the copper on the far layer.
        ok = ok & (prev[:, 0] == cur[:, 0]) & (cur[:, 0] == nxt[:, 0])

        # Perpendicular to the local run, snapped to the lattice's 8 directions.
        dy = (nxt[:, 1] - prev[:, 1]).float()
        dx = (nxt[:, 2] - prev[:, 2]).float()
        ang = torch.atan2(dy, dx) + math.pi / 2.0
        step = 2.0 * math.pi / NUM_DIRECTIONS
        perp_dir = torch.round(ang / step).long() % NUM_DIRECTIONS
        unit = self.tables.path[perp_dir, 0, 0]                      # (B, 2)

        offsets = torch.as_tensor(REFINE_OFFSETS, device=dev)
        amount = offsets[offset_idx.clamp(0, len(REFINE_OFFSETS) - 1)]
        new_y = cur[:, 1] + unit[:, 0] * amount
        new_x = cur[:, 2] + unit[:, 1] * amount
        ok = ok & (amount != 0)

        H, W = self.spec.height_cells, self.spec.width_cells
        ok = ok & (new_y >= 0) & (new_y < H) & (new_x >= 0) & (new_x < W)
        if not bool(ok.any()):
            return torch.zeros(B, dtype=torch.bool, device=dev)

        wc = self.net_width[idx, n]
        lay = cur[:, 0]

        def seg(a, b, active):
            return (idx, lay, a[:, 1], a[:, 2], b[:, 1], b[:, 2], wc, n, active)

        # 1. Clear the two segments that meet at this vertex.
        for a, b in ((prev, cur), (cur, nxt)):
            geo.stamp_segments(
                self.occ, idx, lay, a[:, 1], a[:, 2], b[:, 1], b[:, 2], wc, n,
                self.tables, active=ok, erase=True,
            )

        # 2. Would the dragged route be legal?
        legal = ok.clone()
        for a_y, a_x, b_y, b_x in (
            (prev[:, 1], prev[:, 2], new_y, new_x),
            (new_y, new_x, nxt[:, 1], nxt[:, 2]),
        ):
            legal = legal & geo.check_segments(
                self.occ, idx, lay, a_y, a_x, b_y, b_x, wc, n, self.tables
            )

        # 3. Commit the drag, or put the original copper back exactly.
        moved = torch.stack([lay, new_y, new_x], dim=-1)
        for a, b, active in (
            (prev, moved, legal), (moved, nxt, legal),
            (prev, cur, ok & ~legal), (cur, nxt, ok & ~legal),
        ):
            geo.stamp_segments(
                self.occ, idx, lay, a[:, 1], a[:, 2], b[:, 1], b[:, 2], wc, n,
                self.tables, active=active,
            )

        def dist(a_y, a_x, b_y, b_x):
            return torch.sqrt(
                (b_y - a_y).float() ** 2 + (b_x - a_x).float() ** 2
            )

        old_len = dist(prev[:, 1], prev[:, 2], cur[:, 1], cur[:, 2]) + dist(
            cur[:, 1], cur[:, 2], nxt[:, 1], nxt[:, 2]
        )
        new_len = dist(prev[:, 1], prev[:, 2], new_y, new_x) + dist(
            new_y, new_x, nxt[:, 1], nxt[:, 2]
        )
        delta = torch.where(legal, new_len - old_len, torch.zeros_like(old_len))
        self.net_len[idx, n, leg] += delta

        sel = legal
        if bool(sel.any()):
            self.route_v[idx[sel], n[sel], leg[sel], v[sel]] = moved[sel].to(torch.int16)
        return legal

    def refinable(self) -> torch.Tensor:
        """(B, N) bool -- nets that are routed and have a draggable interior."""
        long_enough = self.route_n.amax(dim=-1) >= 3
        return self.net_valid & (self.net_status == STATUS_DONE) & long_enough

    def length_error(self) -> torch.Tensor:
        """(B, N) |routed length - target| in cells, for length-matched nets.

        This is what the refine phase is optimising. `group_length_error`
        resolves the target as the group's longest routed member, which is what
        length matching actually means -- the same way
        `docs/ROUTER_CAPABILITIES.md` describes reading a tune target off the
        reference net's real routed length rather than a number fixed up front.
        """
        return self.group_length_error()
