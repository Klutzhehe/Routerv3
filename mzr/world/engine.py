"""The batched simultaneous-frontier routing world.

`mzr/DESIGN.md` sections 1-3. This is the piece that differs from NeuroRoute;
the geometry kernels underneath it are ported unchanged because they were
verified against KiCad's own DRC engine.

**What is different, and why.** NeuroRoute kept `K=8` head slots and a learned
scheduler that decided which nets occupied them. That scheduler received zero
gradient on every run in the project's history, so in practice it was a greedy
sequential router: early nets committed copper, later nets took what was left,
no net ever yielded. Completion plateaued at 60-65%.

Here there are no slots. **Every net is live from step 0 and every net's
frontiers advance on every macro-step.** Ordering is not scheduled; it emerges
from the congestion price (`price.py`), which makes contested cells expensive
so nets negotiate for room rather than race for it.

Two consequences worth stating because they are the reason the design works:

* **The horizon collapses.** A sequential router's episode is the sum of all
  net lengths (~10,000 steps). Here every frontier moves each macro-step and
  every net grows from *both* pads, so an episode is
  ``max_net_length / (2 * mean_step)`` -- about 48 macro-steps, **independent
  of net count**. That is what puts this in range of a learned latent model.
* **Arbitration is load-bearing, not a detail.** With every frontier acting at
  once, per-frontier check-and-write would let two frontiers both pass a
  legality test against the pre-step occupancy and both write the same cell.
  One silently loses, its frontier advances anyway, and its route is left with
  a hole that nothing detects until a flood fill much later. That bug was real
  in NeuroRoute. `step()` is **plan -> arbitrate -> commit**, which makes it
  impossible rather than unlikely.

Frontier indexing: ``f = (net * 2 + leg) * 2 + end``, so all frontier state is
``(B, F, ...)`` with ``F = max_nets * 2 * 2``. Leg 1 exists only for
differential pairs; end 0 grows from `src`, end 1 from `dst`, and each targets
the *other* end's pad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

import mzr.world.geometry as geo
from mzr.world.generator import GeneratedBoard
from mzr.world.price import CongestionPrice
from mzr.world.spec import (
    END_DST,
    END_SRC,
    KIND_DIFF_PAIR,
    NUM_DIRECTIONS,
    NUM_ENDS,
    OCC_FREE,
    STEP_LENGTHS,
    BoardSpec,
    PriceRules,
    RipupRules,
)

#: Net lifecycle. Unlike NeuroRoute there is no PENDING-waiting-for-a-slot
#: state: a valid net is ROUTING from load() until it finishes, fails, or is
#: ripped up (which returns it to ROUTING, not to a queue).
STATUS_ROUTING = 0
STATUS_DONE = 1
STATUS_FAILED = 2

#: Length charged for a via, in cell units. A via is cheap in lattice cells but
#: expensive in reality (drill cost, stub, reliability). Without a charge a
#: free via action will drill its way out of every problem instead of learning
#: to route. Carried over from NeuroRoute.
VIA_LENGTH_COST = 4.0


@dataclass
class WorldConfig:
    batch_size: int = 8
    max_nets: int = 64
    #: Polyline vertices stored **per frontier**. Each end grows its own
    #: polyline from its own pad and they are concatenated at export, so this
    #: is half what a single-ended router would need for the same route.
    max_vertices: int = 64
    #: Macro-steps before the episode ends. See the horizon arithmetic above:
    #: this is a property of board size, not of net count.
    max_macro_steps: int = 64
    #: Macro-steps one frontier may take before its net is abandoned.
    max_steps_per_frontier: int = 64
    #: Consecutive rejected moves before a wedged net is abandoned. Generous on
    #: purpose: at stages 1+ a frontier may be blocked for several steps by
    #: copper that a rip-up round is about to clear.
    max_stuck_steps: int = 16
    #: Cells within which a frontier may snap onto its target pad, or onto its
    #: partner frontier. MUST be >= max(STEP_LENGTHS) / 2, or a long step jumps
    #: clean over the snap zone and the frontier orbits its target forever -- a
    #: config bug that reads exactly like a learning failure.
    snap_radius: int = 2
    geodesic_downsample: int = 4
    geodesic_iterations: int = 96
    price: PriceRules = None  # type: ignore[assignment]
    ripup: RipupRules = None  # type: ignore[assignment]
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.snap_radius * 2 < max(STEP_LENGTHS):
            raise ValueError(
                f"snap_radius={self.snap_radius} < max step {max(STEP_LENGTHS)}/2; "
                "a frontier would step over its own snap zone"
            )
        if self.price is None:
            self.price = PriceRules()
        if self.ripup is None:
            self.ripup = RipupRules()


@dataclass
class _Plan:
    """What every frontier *would* do. Nothing is written until commit."""

    live: torch.Tensor          # (B*F,) bool
    want_via: torch.Tensor
    want_move: torch.Tensor
    legal: torch.Tensor
    pos: torch.Tensor           # (B*F, 3)
    newpos: torch.Tensor
    seg_len: torch.Tensor
    claim_flat: torch.Tensor    # (B*F, X)
    claim_valid: torch.Tensor


@dataclass
class StepResult:
    """Outcome of one macro-step. Frontier tensors are (B, F); board (B,)."""

    rejected: torch.Tensor
    moved: torch.Tensor
    via_placed: torch.Tensor
    progress: torch.Tensor
    live: torch.Tensor
    #: (B,) counts
    nets_done: torch.Tensor
    nets_failed: torch.Tensor
    #: (B,) present congestion summed over the board, and its change. The
    #: reward charges the *change* so a policy is debited for creating
    #: contention and credited for resolving it.
    congestion: torch.Tensor
    congestion_delta: torch.Tensor
    #: (B,) frontiers whose move was arbitrated away by another net this step.
    contended: torch.Tensor
    #: (B, F) how sharply this frontier turned, in 45-degree octants: 0 straight,
    #: 1 = 45, 2 = 90, 3 = 135, 4 = reversal. Zero for a frontier that did not
    #: move or has no previous heading. Fab practice routes 45-degree bends and
    #: avoids right angles -- a 90-degree corner in copper is an acid trap when
    #: etched and an impedance discontinuity when driven -- so the reward can
    #: price a corner without the action space needing to change.
    turn: torch.Tensor


class SimultaneousRouterWorld:
    """Every net on every board grows at once, on a DRC-legal lattice."""

    def __init__(self, spec: BoardSpec, cfg: WorldConfig):
        self.spec = spec
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tables = geo.build_tables(spec.rules, self.device)

        B, N = cfg.batch_size, cfg.max_nets
        F = N * 2 * NUM_ENDS
        L, H, W = spec.num_layers, spec.height_cells, spec.width_cells
        ds = cfg.geodesic_downsample
        h, w = max(1, H // ds), max(1, W // ds)
        dev = self.device
        self.F = F
        self._geo_shape = (h, w)

        z = lambda *s, dt=torch.int64: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731

        self.occ = z(B, L, H, W, dt=torch.int16)
        self.static = z(B, L, H, W, dt=torch.int16)
        self.pour = torch.zeros(B, L, H, W, dtype=torch.bool, device=dev)

        # -- net table --
        #: (B, N, leg, end, 3) -- pad cell of each end of each leg.
        self.net_pad = z(B, N, 2, NUM_ENDS, 3)
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
        self.ripup_count = z(B)

        #: (B, N, leg) -- does this leg exist, and is it connected end-to-end?
        self.leg_valid = torch.zeros(B, N, 2, dtype=torch.bool, device=dev)
        self.leg_done = torch.zeros(B, N, 2, dtype=torch.bool, device=dev)

        # -- frontier table --
        self.fr_pos = z(B, F, 3)
        self.fr_alive = torch.zeros(B, F, dtype=torch.bool, device=dev)
        self.fr_steps = z(B, F)
        #: Consecutive macro-steps this frontier's move was rejected. A frontier
        #: whose every direction is blocked re-picks the same illegal move each
        #: step -- nothing is written, so the next observation is identical and
        #: a deterministic policy is trapped until the episode ends. Measured on
        #: stage 0: a frozen run of 48 of 48 steps. Counting consecutive
        #: rejections lets `step()` retire the net instead of spinning.
        self.fr_stuck = z(B, F)
        #: Last accepted absolute heading, or -1 before a frontier has moved.
        #: A via does not change it: a layer change is not a corner.
        self.fr_dir = torch.full((B, F), -1, dtype=torch.long, device=dev)
        self.fr_prev = torch.zeros(B, F, dtype=torch.float32, device=dev)
        # fp16: a coarse field's values are small (a 128-cell board at ds=4 is
        # ~32 coarse cells across) so half precision is ample, and the cache is
        # the dominant memory term once every net is live at once -- (B, F, L,
        # h, w) grows with net count where NeuroRoute's grew with slot count.
        self.fr_geo = torch.full(
            (B, F, L, h, w), float("inf"), dtype=torch.float16, device=dev
        )

        # -- per-frontier polyline --
        self.route_v = z(B, F, cfg.max_vertices, 3, dt=torch.int16)
        self.route_n = z(B, F)

        self.price = CongestionPrice((B, L, H, W), cfg.price, dev)
        self.step_count = 0
        self._prev_congestion = torch.zeros(B, dtype=torch.float32, device=dev)
        self._claim_buf = torch.empty(B * L * H * W, dtype=torch.int32, device=dev)

    # -- shape helpers ------------------------------------------------------

    @property
    def num_layers(self) -> int:
        return self.spec.num_layers

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.spec.num_layers, self.spec.height_cells, self.spec.width_cells

    def frontier_index(self, net: torch.Tensor, leg: torch.Tensor, end: torch.Tensor):
        """(net, leg, end) -> flat frontier index."""
        return (net * 2 + leg) * NUM_ENDS + end

    def _fr_view(self, t: torch.Tensor) -> torch.Tensor:
        """(B, F, ...) -> (B, N, 2, 2, ...), for leg- or end-wise reductions."""
        B = t.shape[0]
        return t.view(B, self.cfg.max_nets, 2, NUM_ENDS, *t.shape[2:])

    # -- loading ------------------------------------------------------------

    def load(self, boards: Sequence[GeneratedBoard]) -> None:
        """Install a batch of boards and make every net live at once.

        There is no assignment step and no pending queue: `load()` ends with
        every valid net ROUTING and every frontier sitting on its own pad. That
        absence *is* the architecture.
        """
        cfg = self.cfg
        if len(boards) != cfg.batch_size:
            raise ValueError(f"expected {cfg.batch_size} boards, got {len(boards)}")
        B, N = cfg.batch_size, cfg.max_nets

        self.net_valid.zero_()
        self.net_status.fill_(STATUS_DONE)
        self.net_len.zero_()
        self.net_target_len.fill_(-1.0)
        self.net_vias.zero_()
        self.net_split.zero_()
        self.net_coupled_steps.zero_()
        self.net_group.fill_(-1)
        self.ripup_count.zero_()
        self.leg_valid.zero_()
        self.leg_done.zero_()
        self.fr_alive.zero_()
        self.fr_steps.zero_()
        self.fr_stuck.zero_()
        self.fr_dir.fill_(-1)
        self.fr_prev.zero_()
        self.route_n.zero_()
        self.price.reset()
        self._prev_congestion.zero_()
        self.step_count = 0

        static = np.stack([b.static for b in boards], axis=0)
        pour = np.stack([b.pour_mask for b in boards], axis=0)
        self.static.copy_(torch.from_numpy(static).to(self.device))
        self.pour.copy_(torch.from_numpy(pour).to(self.device))
        self.occ.copy_(self.static)

        pad = np.zeros((B, N, 2, NUM_ENDS, 3), dtype=np.int64)
        kind = np.zeros((B, N), dtype=np.int64)
        width = np.zeros_like(kind)
        group = np.full_like(kind, -1)
        valid = np.zeros((B, N), dtype=bool)
        leg_valid = np.zeros((B, N, 2), dtype=bool)

        for bi, board in enumerate(boards):
            for ni, net in enumerate(board.netlist.nets[:N]):
                for li, (s, d) in enumerate(net.endpoints()):
                    pad[bi, ni, li, END_SRC] = s
                    pad[bi, ni, li, END_DST] = d
                    leg_valid[bi, ni, li] = True
                kind[bi, ni] = net.kind
                width[bi, ni] = net.width_class
                group[bi, ni] = net.group_id
                valid[bi, ni] = True

        t = lambda a: torch.from_numpy(a).to(self.device)  # noqa: E731
        self.net_pad.copy_(t(pad))
        self.net_kind.copy_(t(kind))
        self.net_width.copy_(t(width))
        self.net_group.copy_(t(group))
        self.net_valid.copy_(t(valid))
        self.leg_valid.copy_(t(leg_valid))
        self.net_status = torch.where(
            self.net_valid,
            torch.full_like(self.net_status, STATUS_ROUTING),
            torch.full_like(self.net_status, STATUS_DONE),
        )

        self._seed_frontiers(self.net_valid)
        self._refresh_geodesic(self.fr_alive)

    def _seed_frontiers(self, net_mask: torch.Tensor) -> None:
        """Place every live net's frontiers on their pads and start polylines.

        Frontier ``(net, leg, end)`` starts on that end's own pad; its *target*
        is the opposite end's pad, so the two frontiers of a leg grow toward
        each other and meet in the middle.
        """
        B, N = net_mask.shape
        alive = net_mask.view(B, N, 1, 1) & self.leg_valid.view(B, N, 2, 1)
        alive = alive.expand(B, N, 2, NUM_ENDS).reshape(B, self.F)
        pos = self.net_pad.view(B, self.F, 3)

        self.fr_alive = torch.where(alive, torch.ones_like(alive), self.fr_alive & ~alive)
        self.fr_pos = torch.where(alive.unsqueeze(-1), pos, self.fr_pos)
        self.fr_steps = torch.where(alive, torch.zeros_like(self.fr_steps), self.fr_steps)
        self.fr_stuck = torch.where(alive, torch.zeros_like(self.fr_stuck), self.fr_stuck)
        self.fr_dir = torch.where(alive, torch.full_like(self.fr_dir, -1), self.fr_dir)
        self.route_n = torch.where(alive, torch.ones_like(self.route_n), self.route_n)
        self.route_v[:, :, 0] = torch.where(
            alive.unsqueeze(-1), pos.to(torch.int16), self.route_v[:, :, 0]
        )

    def _target_pad(self) -> torch.Tensor:
        """(B, F, 3) -- each frontier's target: the opposite end's pad."""
        B = self.cfg.batch_size
        pad = self.net_pad  # (B, N, 2, ends, 3)
        return pad.flip(dims=[3]).reshape(B, self.F, 3)

    def _refresh_geodesic(self, mask: torch.Tensor) -> None:
        """Recompute cached cost-to-go for the masked frontiers.

        The field is computed against the occupancy *as it is now* and then
        cached, not refreshed per step. That is deliberate and is a real
        separation of concerns in this design: the **geodesic field carries
        static reachability and direction**, and the **congestion price carries
        dynamic contention**. Recomputing a per-frontier global relaxation every
        macro-step for every net is unaffordable once all nets are live, and it
        would be duplicating what the price field already measures -- from
        actual contention rather than from a re-derived estimate.
        """
        if not bool(mask.any()):
            return
        B = mask.shape[0]
        ds = self.cfg.geodesic_downsample
        bb = torch.arange(B, device=self.device).view(B, 1).expand(B, self.F)

        b_i = bb[mask]
        f_i = torch.arange(self.F, device=self.device).view(1, self.F).expand(B, self.F)[mask]
        n_i = f_i // (2 * NUM_ENDS)
        tgt = self._target_pad()[mask]

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
        self.fr_geo[b_i, f_i] = fld.to(self.fr_geo.dtype)
        self.fr_prev[b_i, f_i] = self._geo_at(fld, self.fr_pos[b_i, f_i])

    def _geo_at(self, coarse: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Sample a coarse geodesic field at fine-lattice positions, in fine
        cell units.

        Bilinear, not nearest-neighbour: at ``downsample=4`` a nearest lookup is
        constant across each 4x4 block, so the potential-based shaping term
        would be exactly zero for three steps out of four and then jump. The
        policy would see a sparse, jagged reward where a smooth one exists.
        """
        ds = self.cfg.geodesic_downsample
        val = geo.sample_coarse(coarse.float(), pos[:, 0], pos[:, 1], pos[:, 2], ds)
        return torch.nan_to_num(val * ds, posinf=1e6, nan=1e6)

    # -- bearings -----------------------------------------------------------

    def bearings(self) -> torch.Tensor:
        """(B, F) -- each frontier's egocentric reference direction.

        Direction index 0 in the action space is remapped to this bearing, so
        "walk at the target, around obstacles" is action 0 on every board and
        at every pose. That property is what lets a near-zero-init policy start
        *at* the greedy baseline rather than below it, and it is why board-pose
        generalisation never has to be learned.
        """
        B, F = self.cfg.batch_size, self.F
        L = self.num_layers
        M = B * F
        bb = torch.arange(B, device=self.device).view(B, 1).expand(B, F).reshape(M)
        pos = self.fr_pos.reshape(M, 3)
        n_i = (torch.arange(F, device=self.device) // (2 * NUM_ENDS)).view(1, F).expand(B, F).reshape(M)
        w_i = self.net_width.gather(1, n_i.view(B, F)).reshape(M)

        free = geo.raycast(
            self.occ, bb, pos[:, 0], pos[:, 1], pos[:, 2], n_i, self.tables, w_i
        )
        fld = self.fr_geo.reshape(M, L, *self._geo_shape).float()
        return geo.bearing_from_field(
            fld,
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            self.tables,
            self.cfg.geodesic_downsample,
            free_units=free,
        ).view(B, F)

    # -- the macro-step -----------------------------------------------------

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
        """Advance **every live frontier** by one decision.

        All action arguments are ``(B, F)`` int64.

        * `direction` is egocentric: 0 means "down this frontier's own geodesic
          gradient".
        * `layer_action` 0 = stay; ``j > 0`` = place a via and move to layer
          ``j - 1``. A layer change *is* a via -- there is no separate action,
          because there is no such thing as changing layer without one.
        * `couple` only means anything for a differential pair.

        Order is **plan -> arbitrate -> commit** (see the module docstring).
        The arbitration's contention record is also what drives the congestion
        price, so negotiation costs one extra scatter over machinery the step
        needs anyway.
        """
        B, F = self.cfg.batch_size, self.F
        L, H, W = self.shape
        dev = self.device
        M = B * F

        bb = torch.arange(B, device=dev).view(B, 1).expand(B, F)
        f_idx = torch.arange(F, device=dev).view(1, F).expand(B, F)
        net = f_idx // (2 * NUM_ENDS)
        leg = (f_idx // NUM_ENDS) % 2

        live = self.fr_alive & (self.net_status.gather(1, net) == STATUS_ROUTING)
        is_pair = self.net_kind.gather(1, net) == KIND_DIFF_PAIR
        # A net's own required width is a floor, not a suggestion: the policy
        # may widen a trace but never narrow one below its electrical spec.
        wclass = torch.maximum(width_class, self.net_width.gather(1, net))

        # --- differential-pair coupling ------------------------------------
        # Coupled: both legs at the same end travel on one shared decision, so
        # they stay parallel. Split: each leg follows its own gradient, which
        # is what lets the policy fan the legs around an obstacle and
        # re-converge -- the behaviour a solver call cannot express.
        coupled = is_pair & (couple > 0) & live
        v = lambda t: self._fr_view(t)  # noqa: E731
        # Leg 0's action is the shared one when coupled.
        lead = lambda t: v(t)[:, :, 0:1].expand(B, self.cfg.max_nets, 2, NUM_ENDS).reshape(B, F)  # noqa: E731
        cpl_any = v(coupled).any(dim=2, keepdim=True).expand(B, self.cfg.max_nets, 2, NUM_ENDS).reshape(B, F)
        direction = torch.where(cpl_any, lead(direction), direction)
        step_class = torch.where(cpl_any, lead(step_class), step_class)
        layer_action = torch.where(cpl_any, lead(layer_action), layer_action)
        via_class = torch.where(cpl_any, lead(via_class), via_class)
        wclass = torch.where(cpl_any, lead(wclass), wclass)

        pair_live = is_pair & live
        self.net_coupled_steps += v(coupled & pair_live).any(dim=3).float().sum(dim=2) * 0.5
        self.net_split += v(~coupled & pair_live).any(dim=3).float().sum(dim=2) * 0.5

        if bearing is None:
            bearing = self.bearings()
        abs_dir = (bearing + direction) % NUM_DIRECTIONS

        b_flat = bb.reshape(M)
        n_flat = net.reshape(M)
        w_flat = wclass.reshape(M)

        # --- plan: legality only, nothing written ---------------------------
        plan = self._plan(live, n_flat, abs_dir, step_class, layer_action, via_class, wclass)

        go = plan.legal & plan.live
        # A coupled pair moves atomically: both legs or neither.
        go_v = self._fr_view(go.view(B, F))
        both = go_v.all(dim=2, keepdim=True).expand_as(go_v).reshape(B, F)
        go = torch.where(cpl_any, both & live, go.view(B, F)).reshape(M)

        # --- arbitrate between every simultaneous frontier ------------------
        (ok,) = geo.resolve_claims(
            self.occ,
            self._claim_buf,
            [plan.claim_flat],
            [plan.claim_valid & go.view(M, 1)],
            [n_flat],
        )
        ok_v = self._fr_view(ok.view(B, F))
        ok_pair = ok_v.all(dim=2, keepdim=True).expand_as(ok_v).reshape(B, F)
        ok = torch.where(cpl_any, ok_pair, ok.view(B, F)).reshape(M)
        contended = (go & ~ok).view(B, F)
        go = go & ok

        # --- congestion price -----------------------------------------------
        # Fed from the *attempted* claims of frontiers that were otherwise
        # ready to move, not from the accepted ones: a cell two nets both
        # wanted is contended whether or not one of them won it. Scoring only
        # winners would make the price blind to exactly the pressure it exists
        # to report.
        self.price.update(
            [plan.claim_flat],
            [plan.claim_valid & (plan.legal & plan.live).view(M, 1)],
            [n_flat],
        )
        congestion = self.price.total_oversubscription()
        congestion_delta = congestion - self._prev_congestion
        self._prev_congestion = congestion

        # --- commit -----------------------------------------------------------
        progress, moved, via_placed = self._commit(plan, go, b_flat, n_flat, w_flat)
        rejected = (plan.live & ~go).view(B, F)

        # Turn magnitude, measured only on moves that were actually committed.
        # Directions are octants, so the circular distance between headings is
        # the bend in units of 45 degrees.
        moved_f = moved.view(B, F)
        new_dir = abs_dir.view(B, F)
        raw = (new_dir - self.fr_dir).abs()
        oct_dist = torch.minimum(raw, NUM_DIRECTIONS - raw)
        turn = torch.where(moved_f & (self.fr_dir >= 0), oct_dist, torch.zeros_like(oct_dist))
        self.fr_dir = torch.where(moved_f, new_dir, self.fr_dir)

        self.fr_steps = torch.where(live, self.fr_steps + 1, self.fr_steps)
        self.fr_stuck = torch.where(rejected, self.fr_stuck + 1, torch.zeros_like(self.fr_stuck))

        # --- connection: pad snap, then partner meeting ----------------------
        self._try_snap(b_flat, n_flat, w_flat, live.reshape(M))
        self._try_meet(b_flat, n_flat, w_flat)

        # A leg is done when it is connected; a net is done when every valid
        # leg is. Frontiers of a done leg stop.
        leg_live = self.leg_valid & ~self.leg_done
        self.fr_alive &= leg_live.unsqueeze(-1).expand(B, self.cfg.max_nets, 2, NUM_ENDS).reshape(B, F)

        routing = self.net_status == STATUS_ROUTING
        net_done = routing & self.net_valid & (self.leg_done | ~self.leg_valid).all(dim=-1)
        starved = self._fr_view(self.fr_steps).amax(dim=(2, 3)) >= self.cfg.max_steps_per_frontier
        # A frontier with no legal move at all cannot recover on its own. Give it
        # a generous window first -- at stages 1+ a frontier is often blocked for
        # several steps by copper another net will rip up -- then fail the net
        # rather than let it burn the rest of the episode re-picking one illegal
        # move.
        wedged = self._fr_view(self.fr_stuck).amax(dim=(2, 3)) >= self.cfg.max_stuck_steps
        net_failed = routing & self.net_valid & ~net_done & (starved | wedged)

        nets_done = self._retire(net_done, STATUS_DONE)
        nets_failed = self._retire(net_failed, STATUS_FAILED)

        self.step_count += 1
        r = self.cfg.ripup
        if r.interval > 0 and self.step_count % r.interval == 0:
            self.ripup_round()

        return StepResult(
            rejected=rejected,
            moved=moved.view(B, F),
            via_placed=via_placed.view(B, F),
            progress=progress.view(B, F),
            live=live,
            nets_done=nets_done,
            nets_failed=nets_failed,
            congestion=congestion,
            congestion_delta=congestion_delta,
            contended=contended,
            turn=turn.float(),
        )

    # -- plan / commit ------------------------------------------------------

    def _plan(
        self,
        live: torch.Tensor,
        n_flat: torch.Tensor,
        abs_dir: torch.Tensor,
        step_class: torch.Tensor,
        layer_action: torch.Tensor,
        via_class: torch.Tensor,
        width_class: torch.Tensor,
    ) -> _Plan:
        """Work out what every frontier *would* do. Writes nothing."""
        B, F = live.shape
        L, H, W = self.shape
        dev = self.device
        M = B * F

        pos = self.fr_pos.reshape(M, 3)
        b_i = torch.arange(B, device=dev).view(B, 1).expand(B, F).reshape(M)
        d_i = abs_dir.reshape(M)
        s_i = step_class.reshape(M)
        la_i = layer_action.reshape(M)
        vc_i = via_class.reshape(M)
        wc_i = width_class.reshape(M)
        live_f = live.reshape(M)

        want_via = live_f & (la_i > 0) & ((la_i - 1) != pos[:, 0])
        want_move = live_f & ~want_via

        tgt_layer = (la_i - 1).clamp(0, L - 1)
        via_lo, via_hi = self._via_span(pos[:, 0], tgt_layer)
        via_flat, via_valid = geo.via_claims(
            self.occ, b_i, via_lo, via_hi, pos[:, 1], pos[:, 2], vc_i, self.tables
        )
        via_valid = via_valid & want_via.view(M, 1)
        via_legal = geo.claims_passable(self.occ, via_flat, via_valid, n_flat)

        move_flat, move_valid = geo.move_claims(
            self.occ, b_i, pos[:, 0], pos[:, 1], pos[:, 2], d_i, s_i, wc_i, self.tables
        )
        move_valid = move_valid & want_move.view(M, 1)
        # `claims_passable` only sees in-bounds cells, so a move running off the
        # board would trivially pass. `check_moves` counts clear travel units
        # with out-of-bounds treated as blocked, which catches it.
        _, free_units = geo.check_moves(
            self.occ, b_i, pos[:, 0], pos[:, 1], pos[:, 2], d_i, s_i, wc_i, n_flat, self.tables
        )
        move_legal = geo.claims_passable(self.occ, move_flat, move_valid, n_flat)
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

        return _Plan(
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

    def _commit(
        self,
        plan: _Plan,
        go: torch.Tensor,
        b_i: torch.Tensor,
        n_i: torch.Tensor,
        w_i: torch.Tensor,
    ):
        """Write accepted copper and advance the frontiers that moved."""
        B, F = self.cfg.batch_size, self.F
        L = self.num_layers
        M = B * F

        geo.write_claims(self.occ, plan.claim_flat, plan.claim_valid, n_i, go)

        pos = torch.where(go.view(M, 1), plan.newpos, plan.pos)
        did_via = go & plan.want_via
        did_move = go & plan.want_move

        leg = ((torch.arange(F, device=self.device) // NUM_ENDS) % 2).view(1, F).expand(B, F).reshape(M)
        self.net_vias.view(-1).scatter_add_(
            0,
            (b_i * self.cfg.max_nets + n_i)[did_via],
            torch.ones(int(did_via.sum()), dtype=self.net_vias.dtype, device=self.device),
        )
        self.net_len.view(-1).scatter_add_(
            0,
            ((b_i * self.cfg.max_nets + n_i) * 2 + leg)[go],
            plan.seg_len[go],
        )
        self.fr_pos = pos.view(B, F, 3)
        self._append_vertex(pos, go)

        fld = self.fr_geo.reshape(M, L, *self._geo_shape).float()
        new_dist = self._geo_at(fld, pos)
        prev = self.fr_prev.reshape(M)
        prog = torch.where(plan.live, prev - new_dist, torch.zeros_like(prev))
        self.fr_prev = new_dist.view(B, F)
        return prog, did_move, did_via

    def _via_span(self, cur: torch.Tensor, tgt: torch.Tensor):
        if self.spec.layers.through_only:
            return torch.zeros_like(cur), torch.full_like(cur, self.num_layers - 1)
        return torch.minimum(cur, tgt), torch.maximum(cur, tgt)

    def _append_vertex(self, pos: torch.Tensor, mask: torch.Tensor) -> None:
        """Record an accepted frontier position as a polyline vertex.

        The polyline is what the KiCad exporter emits and what the refine phase
        edits, so it is kept in step with `occ` rather than reconstructed from
        it -- reconstructing a route from an occupancy grid is ambiguous once
        two nets touch.
        """
        if not bool(mask.any()):
            return
        B, F = self.cfg.batch_size, self.F
        idx = torch.arange(B * F, device=self.device)[mask]
        b_i, f_i = idx // F, idx % F
        cnt = self.route_n[b_i, f_i]
        room = cnt < self.cfg.max_vertices
        b_i, f_i, cnt = b_i[room], f_i[room], cnt[room]
        self.route_v[b_i, f_i, cnt] = pos[mask][room].to(torch.int16)
        self.route_n[b_i, f_i] = cnt + 1

    # -- connection ---------------------------------------------------------

    def _try_snap(
        self,
        b_i: torch.Tensor,
        n_i: torch.Tensor,
        w_i: torch.Tensor,
        live: torch.Tensor,
    ) -> None:
        """Snap a frontier that reached its target pad. Marks the leg done."""
        B, F = self.cfg.batch_size, self.F
        M = B * F
        pos = self.fr_pos.reshape(M, 3)
        tgt = self._target_pad().reshape(M, 3)
        r = self.cfg.snap_radius
        near = (
            live
            & (pos[:, 0] == tgt[:, 0])
            & ((pos[:, 1] - tgt[:, 1]).abs() <= r)
            & ((pos[:, 2] - tgt[:, 2]).abs() <= r)
        )
        if not bool(near.any()):
            return
        self._connect(b_i, n_i, w_i, pos, tgt, near)

    def _try_meet(self, b_i: torch.Tensor, n_i: torch.Tensor, w_i: torch.Tensor) -> None:
        """Connect a leg whose two frontiers have met in the middle.

        This is the payoff of growing from both ends: a leg finishes when its
        frontiers reach *each other*, which is half the distance each and
        therefore half the macro-steps. Meeting is tested by proximity plus a
        clear connecting segment -- not by the geodesic field, which targets a
        fixed pad and so cannot see the partner move.
        """
        B, F = self.cfg.batch_size, self.F
        N = self.cfg.max_nets
        M = B * F
        r = self.cfg.snap_radius

        pv = self.fr_pos.view(B, N, 2, NUM_ENDS, 3)
        a = pv[:, :, :, END_SRC]  # (B, N, 2, 3)
        c = pv[:, :, :, END_DST]
        alive = self._fr_view(self.fr_alive)
        both = alive[:, :, :, END_SRC] & alive[:, :, :, END_DST]
        near = (
            both
            & self.leg_valid
            & ~self.leg_done
            & (a[..., 0] == c[..., 0])
            & ((a[..., 1] - c[..., 1]).abs() <= r)
            & ((a[..., 2] - c[..., 2]).abs() <= r)
        )
        if not bool(near.any()):
            return
        # Expressed as a connect from the END_SRC frontier to the END_DST
        # frontier's live position, so it shares one code path with pad snap.
        sel = torch.zeros(B, N, 2, NUM_ENDS, dtype=torch.bool, device=self.device)
        sel[:, :, :, END_SRC] = near
        target = torch.zeros_like(pv)
        target[:, :, :, END_SRC] = c
        self._connect(
            b_i, n_i, w_i, self.fr_pos.reshape(M, 3), target.reshape(M, 3), sel.reshape(M)
        )

    def _connect(
        self,
        b_i: torch.Tensor,
        n_i: torch.Tensor,
        w_i: torch.Tensor,
        pos: torch.Tensor,
        tgt: torch.Tensor,
        cand: torch.Tensor,
    ) -> None:
        """Draw the closing segment for candidate frontiers; mark legs done.

        Arbitrated like any other write: two nets closing through the same cell
        in the same macro-step is exactly the race that plan->arbitrate->commit
        exists to prevent, and a closing segment is no more privileged than a
        move.
        """
        B, F = self.cfg.batch_size, self.F
        M = B * F
        sflat, svalid = geo.segment_claims(
            self.occ, b_i, pos[:, 0], pos[:, 1], pos[:, 2], tgt[:, 1], tgt[:, 2], w_i, self.tables
        )
        svalid = svalid & cand.view(M, 1)
        clear = cand & geo.claims_passable(self.occ, sflat, svalid, n_i)
        (won,) = geo.resolve_claims(
            self.occ, self._claim_buf, [sflat], [svalid & clear.view(M, 1)], [n_i]
        )
        done = clear & won
        if not bool(done.any()):
            return
        geo.write_claims(self.occ, sflat, svalid, n_i, done)

        leg = ((torch.arange(F, device=self.device) // NUM_ENDS) % 2).view(1, F).expand(B, F).reshape(M)
        dy = (tgt[:, 1] - pos[:, 1]).float()
        dx = (tgt[:, 2] - pos[:, 2]).float()
        self.net_len.view(-1).scatter_add_(
            0,
            ((b_i * self.cfg.max_nets + n_i) * 2 + leg)[done],
            torch.sqrt(dy[done] ** 2 + dx[done] ** 2),
        )
        newpos = torch.where(done.view(M, 1), tgt, pos)
        self.fr_pos = newpos.view(B, F, 3)
        self._append_vertex(newpos, done)

        d = self._fr_view(done.view(B, F)).any(dim=3)  # (B, N, 2)
        self.leg_done |= d & self.leg_valid

    def _retire(self, net_mask: torch.Tensor, status: int) -> torch.Tensor:
        """Mark finished/failed nets and stop their frontiers. Returns (B,)."""
        if not bool(net_mask.any()):
            return torch.zeros(self.cfg.batch_size, dtype=torch.int64, device=self.device)
        B, N = net_mask.shape
        self.net_status = torch.where(
            net_mask, torch.full_like(self.net_status, status), self.net_status
        )
        stop = net_mask.view(B, N, 1, 1).expand(B, N, 2, NUM_ENDS).reshape(B, self.F)
        self.fr_alive &= ~stop
        return net_mask.sum(dim=1)

    # -- rip-up -------------------------------------------------------------

    def ripup_round(self) -> torch.Tensor:
        """Rip up the most-congested fraction of unfinished nets and regrow.

        PathFinder rips up and reroutes **every** net each iteration; a fraction
        is the affordable middle ground here, since a macro-step already moves
        every frontier and a full rip-up each round would discard most of the
        episode's work.

        Historical price is deliberately **not** cleared. That is the whole
        mechanism: a corridor that has been fought over stays expensive after
        the retreat, so the nets that regrow into it have a reason to go
        somewhere else. Clearing history here would make rip-up a reset rather
        than a negotiation, and the nets would re-collide in the same place.

        Fixed-rule, not learned. Making this a learned decision from day one
        would reintroduce the pointer-over-nets credit-assignment problem that
        received zero gradient for NeuroRoute's entire history. The offline-RL
        precedent in DESIGN.md section 15 says the *cost schedule* is the part
        worth learning later, once this is stable.

        Returns (B,) counts of nets ripped up.
        """
        B, N = self.cfg.batch_size, self.cfg.max_nets
        r = self.cfg.ripup
        eligible = self.net_valid & (self.net_status == STATUS_ROUTING)
        n_elig = int(eligible.sum(dim=1).max())
        k = int(math.floor(n_elig * r.fraction))
        if k <= 0:
            return torch.zeros(B, dtype=torch.int64, device=self.device)

        # Score each net by the price of the cells it currently occupies: the
        # nets sitting in the most-contested copper are the ones whose retreat
        # frees the most room.
        price = self.price.field()  # (B, L, H, W)
        occ = self.occ.long()
        score = torch.zeros(B, N + 1, dtype=torch.float32, device=self.device)
        score.scatter_add_(1, occ.clamp_min(0).flatten(1), price.flatten(1))
        score = score[:, 1:]
        score = torch.where(eligible, score, torch.full_like(score, -1.0))

        pick = score.topk(min(k, N), dim=1).indices
        sel = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        sel.scatter_(1, pick, True)
        sel &= eligible & (score > 0)
        if not bool(sel.any()):
            return torch.zeros(B, dtype=torch.int64, device=self.device)

        self._clear_nets(sel)
        self.ripup_count += sel.sum(dim=1)
        return sel.sum(dim=1)

    def _clear_nets(self, sel: torch.Tensor) -> None:
        """Erase selected nets' copper and restart their frontiers at the pads."""
        B, N = sel.shape
        own = torch.zeros(B, N + 1, dtype=torch.bool, device=self.device)
        own[:, 1:] = sel
        mine = own.gather(1, self.occ.long().clamp_min(0).flatten(1)).view_as(self.occ)
        # Clear this net's copper but restore its pads from the static layer --
        # a ripped-up net still has to start somewhere next time.
        cleared = torch.where(mine, torch.zeros_like(self.occ), self.occ)
        static_mine = own.gather(1, self.static.long().clamp_min(0).flatten(1)).view_as(self.occ)
        self.occ = torch.where(static_mine, self.static, cleared)

        self.net_len = torch.where(sel.unsqueeze(-1), torch.zeros_like(self.net_len), self.net_len)
        self.net_vias = torch.where(sel, torch.zeros_like(self.net_vias), self.net_vias)
        self.net_split = torch.where(sel, torch.zeros_like(self.net_split), self.net_split)
        self.net_coupled_steps = torch.where(
            sel, torch.zeros_like(self.net_coupled_steps), self.net_coupled_steps
        )
        self.leg_done &= ~sel.unsqueeze(-1)
        self.net_status = torch.where(
            sel, torch.full_like(self.net_status, STATUS_ROUTING), self.net_status
        )
        self._seed_frontiers(sel)
        stop = sel.view(B, N, 1, 1).expand(B, N, 2, NUM_ENDS).reshape(B, self.F)
        self._refresh_geodesic(stop & self.fr_alive)

    # -- metrics ------------------------------------------------------------

    def completion(self) -> torch.Tensor:
        """(B,) fraction of valid nets connected.

        A board with no nets scores 1.0, not 0.0. The generator can quietly
        give up when component placement runs short of pins, and scoring those
        boards 0% lets a generator bug hide inside a training curve.
        """
        valid = self.net_valid.sum(dim=1)
        done = (self.net_valid & (self.net_status == STATUS_DONE)).sum(dim=1)
        return torch.where(valid > 0, done.float() / valid.clamp_min(1), torch.ones_like(done, dtype=torch.float32))

    def board_stats(self) -> dict[str, torch.Tensor]:
        valid = self.net_valid
        nv = valid.sum(dim=1).clamp_min(1)
        return {
            "completion": self.completion(),
            "vias": (self.net_vias * valid).sum(dim=1).float(),
            "length": (self.net_len.sum(dim=-1) * valid).sum(dim=1),
            "ripups": self.ripup_count.float(),
            "congestion": self.price.total_oversubscription(),
            "live_frontiers": self.fr_alive.sum(dim=1).float(),
            "failed": (valid & (self.net_status == STATUS_FAILED)).sum(dim=1).float() / nv,
        }

    def episode_done(self) -> bool:
        """True when nothing can advance: every net settled, or out of steps."""
        if self.step_count >= self.cfg.max_macro_steps:
            return True
        return not bool(self.fr_alive.any())
