"""Sequential expert routers: plain ordered, and PathFinder-negotiated.

`mzr/DESIGN.md` build order step 3. One piece of work with two jobs:

1. **Stage 1's bar.** The gate is "beat sequential **plus PathFinder
   negotiation** by >= 10 points", not "beat naive greedy". Concurrent routing
   is historically judged too expensive and the field retreated to
   sequential-plus-negotiation, so that retreat is what simultaneous growth has
   to actually be better than. Measuring against greedy would be marking our
   own homework.
2. **The behaviour-cloning demonstration source.** PRIMAL -- the closest
   published analogue to stage 1, and the strongest evidence the shared-weights
   design scales -- did *not* reach 1024 agents with pure RL. Its authors credit
   "demonstrations of an expert MAPF planner during training, as well as careful
   reward shaping". `neuroroute/` was pure RL and plateaued at 60-65%.

**Why this is a planner and not a policy.** It routes one net at a time on a
Dijkstra field, which is exactly the greedy net-by-net behaviour this project
is trying to get away from. That is the point: it is the *incumbent*.
PathFinder's negotiation is what makes the incumbent strong -- rip up every net
each iteration and let historical congestion push them apart, so ordering stops
mattering as much as it does for a single-pass sequential router.

**Engine-consistent by construction.** Paths come out as unit lattice steps and
are stamped through `geometry.move_claims` / `via_claims` -- the same functions
`engine.step()` uses. So the expert's copper is byte-identical to what a policy
taking the same actions would write, which is what makes a plan replayable as a
demonstration rather than merely a picture of a good route.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import mzr.world.geometry as geo
from mzr.world.spec import (
    DIRECTION_VECTORS,
    END_DST,
    END_SRC,
    NUM_ENDS,
    OCC_FREE,
    BoardSpec,
)

#: Cost of a layer change, in cell units. Matches `engine.VIA_LENGTH_COST` so
#: the expert and the learned policy optimise comparable objectives.
VIA_COST = 4.0


@dataclass
class ExpertConfig:
    #: PathFinder outer iterations. 1 == plain sequential, no negotiation.
    #:
    #: PathFinder's own schedule runs until nothing is over-subscribed; a fixed
    #: small budget is the honest analogue here, because on a hard board it
    #: never fully converges and an unbounded loop would just be a slower way to
    #: report the same number.
    iterations: int = 6
    #: Added to a cell's history each iteration it is over-subscribed. Higher
    #: than the live engine's `PriceRules.history_rate` on purpose: the expert
    #: gets a handful of iterations to negotiate, where the live price
    #: accumulates over every macro-step of an episode.
    history_rate: float = 0.5
    max_history: float = 16.0
    #: Relaxation sweeps for the distance field. 0 = auto (H + W), the longest
    #: 8-connected path a board can hold.
    relax_iterations: int = 0
    #: Give up extracting a path after this many cells. 0 = auto.
    max_path_cells: int = 0


@dataclass
class ExpertResult:
    """Routed paths for one board, plus what it took to get them."""

    #: (net, leg) -> list of (layer, y, x), source pad first. Completed legs only.
    paths: dict[tuple[int, int], list[tuple[int, int, int]]] = field(default_factory=dict)
    completed: set[tuple[int, int]] = field(default_factory=set)
    attempted: int = 0
    iterations_used: int = 0
    #: Cells still contested when the loop stopped. > 0 means negotiation did
    #: not converge -- a real property of a hard board, reported not hidden.
    unresolved: int = 0
    length: float = 0.0
    vias: int = 0

    @property
    def completion(self) -> float:
        return len(self.completed) / max(1, self.attempted)


_DIR_LOOKUP = {(int(dy), int(dx)): i for i, (dy, dx) in enumerate(DIRECTION_VECTORS)}


def _direction_index(dy: int, dx: int) -> int | None:
    return _DIR_LOOKUP.get((int(dy), int(dx)))


class _BoardRouter:
    """One board's occupancy grid plus what a sequential router needs to do."""

    def __init__(self, spec: BoardSpec, static: torch.Tensor, tables, device):
        self.spec = spec
        self.tables = tables
        self.device = device
        self.L, self.H, self.W = spec.num_layers, spec.height_cells, spec.width_cells
        self.static = static.view(1, self.L, self.H, self.W).clone()
        self.occ = self.static.clone()

    def reset(self) -> None:
        self.occ = self.static.clone()

    def hard_blocked(self, net: int, width_class: int = 0) -> torch.Tensor:
        """Cells no amount of congestion price can buy: keepouts, and other
        nets' pads.

        Deliberately does **not** include other nets' routed copper. That is the
        crux of PathFinder and the thing this implementation got wrong first
        time round: if each net treats already-routed copper as an obstacle, no
        cell is ever over-subscribed, history never rises, and "negotiation"
        silently degrades into a plain ordered sequential router. Measured, that
        bug produced byte-identical results for 1 iteration and 6, with
        `unresolved` stuck at 0.

        Nets must be allowed to *share* a cell within a pass, so that the
        resulting overuse is what prices the next one.
        """
        own = net + 1
        return self._dilate((self.static != OCC_FREE) & (self.static != own), width_class)

    def blocked_for(self, net: int, width_class: int = 0) -> torch.Tensor:
        """Hard obstacles only -- used when committing a pass to real copper."""
        own = net + 1
        return self._dilate((self.occ != OCC_FREE) & (self.occ != own), width_class)

    def _dilate(self, blocked: torch.Tensor, width_class: int) -> torch.Tensor:
        """Grow obstacles by the footprint a trace centred on a cell occupies.

        Without this the planner and the stamper disagree about what is legal:
        the field only forbids a trace's *centre line* from crossing an
        obstacle, while `stamp` writes the trace dilated by its width radius
        plus the diagonal corner guards. A route planned one cell from another
        net's pad therefore plans perfectly and then cannot be written.

        Measured: every one of 24 legs planned successfully and only 46% of them
        stamped -- the entire shortfall, and it made the "expert" score *below*
        the layer_hop baseline it is supposed to be the bar for.

        Radius is the trace's own width radius plus one, the extra cell covering
        the corner guards a 45-degree move reserves beside itself.
        """
        r = int(self.spec.rules.width_radius_cells(width_class)) + 1
        k = 2 * r + 1
        L, H, W = blocked.shape[1:]
        grown = F.max_pool2d(blocked.reshape(L, 1, H, W).float(), k, stride=1, padding=r)
        return grown.reshape(1, L, H, W) > 0.5

    def field(
        self,
        net: int,
        target: tuple[int, int, int],
        price: torch.Tensor,
        cfg: ExpertConfig,
        *,
        width_class: int = 0,
        soft: bool = True,
    ) -> torch.Tensor:
        """Price-weighted cost-to-go to `target`, at **full** lattice resolution.

        Full resolution, unlike the engine's cached coarse field: the engine
        only needs a gradient to shape a reward, whereas extracting an actual
        route by descent needs the field exact at cell granularity. A
        4x-downsampled field is piecewise-constant over each 4x4 block, so a
        descent walk has no gradient to follow in three steps out of four.

        `price` multiplies the cost of *entering* a cell, which is the whole of
        PathFinder: a contested channel is not forbidden, just expensive. Nair's
        predecessor made over-used resources infinitely costly; PathFinder's
        contribution was the gradual penalty, so nets negotiate rather than
        hard-fail.
        """
        blocked = (
            self.hard_blocked(net, width_class)
            if soft
            else self.blocked_for(net, width_class)
        )
        iters = cfg.relax_iterations or (self.H + self.W)
        dist = torch.full(
            (1, self.L, self.H, self.W), float("inf"), dtype=torch.float32, device=self.device
        )
        tl, ty, tx = target
        dist[0, tl, ty, tx] = 0.0
        cost = price.view(1, self.L, self.H, self.W)
        neg_inf = -1e9

        for _ in range(iters):
            nd = torch.where(torch.isinf(dist), torch.full_like(dist, neg_inf), -dist)
            pooled = F.max_pool2d(
                nd.reshape(self.L, 1, self.H, self.W), 3, stride=1, padding=1
            )
            cand = -pooled.reshape(1, self.L, self.H, self.W) + cost
            if self.L > 1:
                up = torch.full_like(dist, float("inf"))
                dn = torch.full_like(dist, float("inf"))
                up[:, :-1] = dist[:, 1:] + VIA_COST
                dn[:, 1:] = dist[:, :-1] + VIA_COST
                cand = torch.minimum(cand, torch.minimum(up, dn))
            cand = torch.where(blocked, torch.full_like(cand, float("inf")), cand)
            new = torch.minimum(dist, cand)
            new[0, tl, ty, tx] = 0.0
            if torch.equal(new, dist):
                break
            dist = new
        return dist

    def extract(
        self,
        fld: torch.Tensor,
        src: tuple[int, int, int],
        dst: tuple[int, int, int],
        cfg: ExpertConfig,
    ) -> list[tuple[int, int, int]] | None:
        """Walk downhill from `src` to `dst`, one lattice step at a time.

        Emits **unit** steps only -- one of the 8 in-plane directions, or a
        single layer change -- so every step maps onto exactly one engine
        action. A planner emitting arbitrary jumps would produce a route the
        policy could not imitate.
        """
        f = fld[0]
        cap = cfg.max_path_cells or (4 * (self.H + self.W))
        cur = tuple(int(v) for v in src)
        goal = tuple(int(v) for v in dst)
        path = [cur]
        seen = {cur}
        for _ in range(cap):
            if cur == goal:
                return path
            l, y, x = cur
            best = None
            best_v = float(f[l, y, x])
            for dy, dx in DIRECTION_VECTORS:
                ny, nx = y + int(dy), x + int(dx)
                if not (0 <= ny < self.H and 0 <= nx < self.W):
                    continue
                v = float(f[l, ny, nx])
                if v < best_v and (l, ny, nx) not in seen:
                    best, best_v = (l, ny, nx), v
            for nl in (l - 1, l + 1):
                if not (0 <= nl < self.L):
                    continue
                v = float(f[nl, y, x])
                if v < best_v and (nl, y, x) not in seen:
                    best, best_v = (nl, y, x), v
            if best is None:
                return None
            cur = best
            seen.add(cur)
            path.append(cur)
        return None

    def stamp(self, net: int, path: list[tuple[int, int, int]], width_class: int) -> bool:
        """Write a path's copper using the engine's own primitives.

        All-or-nothing: returns False and writes nothing if any step is
        illegal, so a partially stamped route can never exist.
        """
        claims: list[tuple[torch.Tensor, torch.Tensor]] = []

        def t(v):
            return torch.tensor([v], device=self.device, dtype=torch.long)

        drilled: set[tuple[int, int]] = set()
        for (l0, y0, x0), (l1, y1, x1) in zip(path, path[1:]):
            if l0 != l1:
                # A through via spans every layer, so consecutive layer hops at
                # one (y, x) are the same hole -- drill it once, exactly as the
                # exporter emits it once.
                if (y0, x0) in drilled:
                    continue
                drilled.add((y0, x0))
                if self.spec.layers.through_only:
                    lo, hi = 0, self.L - 1
                else:
                    lo, hi = min(l0, l1), max(l0, l1)
                claims.append(
                    geo.via_claims(
                        self.occ, t(0), t(lo), t(hi), t(y0), t(x0), t(0), self.tables
                    )
                )
                continue
            d = _direction_index(y1 - y0, x1 - x0)
            if d is None:
                return False
            claims.append(
                geo.move_claims(
                    self.occ, t(0), t(l0), t(y0), t(x0), t(d), t(0), t(width_class), self.tables
                )
            )

        if not claims:
            return True
        flat = torch.cat([c[0] for c in claims], dim=1)
        valid = torch.cat([c[1] for c in claims], dim=1)
        nid = t(net)
        if not bool(geo.claims_passable(self.occ, flat, valid, nid)):
            return False
        geo.write_claims(
            self.occ, flat, valid, nid, torch.ones(1, dtype=torch.bool, device=self.device)
        )
        return True


def route_board(
    spec: BoardSpec,
    static: torch.Tensor,
    legs: list[tuple[int, int, tuple, tuple, int]],
    cfg: ExpertConfig | None = None,
    *,
    negotiate: bool = True,
    device: torch.device | str = "cpu",
) -> ExpertResult:
    """Route one board's legs sequentially, optionally with PathFinder negotiation.

    `legs` is ``(net, leg, src, dst, width_class)``.

    With ``negotiate=False`` this is a single ordered pass -- the classic
    net-by-net router, whose ordering-sensitivity is the disease this project
    exists to cure. With ``negotiate=True`` it is PathFinder: route everything,
    find what is over-subscribed, raise those cells' historical cost, **rip up
    every net**, and route again. After a few iterations ordering matters far
    less, because history rather than sequence decides who gets the contested
    channel.
    """
    cfg = cfg or ExpertConfig()
    device = torch.device(device)
    tables = geo.build_tables(spec.rules, device)
    br = _BoardRouter(spec, static.to(device), tables, device)
    L, H, W = spec.num_layers, spec.height_cells, spec.width_cells

    history = torch.zeros(L, H, W, dtype=torch.float32, device=device)
    # Shortest-first. A heuristic, and a weak one -- which is exactly why
    # negotiation matters more than ordering does.
    order = sorted(
        legs, key=lambda leg: abs(leg[2][1] - leg[3][1]) + abs(leg[2][2] - leg[3][2])
    )

    best_paths: dict[tuple[int, int], list] | None = None
    best_over = None
    iters = cfg.iterations if negotiate else 1
    used = 0

    for it in range(iters):
        used = it + 1
        # A negotiation pass routes every leg against the SAME board -- pads and
        # keepouts only. Nets may share cells; the overuse that produces is the
        # signal, not an error.
        price = 1.0 + history
        demand = torch.zeros(L, H, W, dtype=torch.float32, device=device)
        paths: dict[tuple[int, int], list] = {}

        for net, leg, src, dst, wclass in order:
            # Present congestion, PathFinder's `p`: within a pass, a cell
            # already claimed by an earlier net costs more. This is what stops
            # every net piling into the same channel before history has had a
            # chance to build up over iterations.
            fld = br.field(net, dst, price * (1.0 + demand), cfg, width_class=wclass, soft=True)
            if not torch.isfinite(fld[0, src[0], src[1], src[2]]):
                continue
            path = br.extract(fld, src, dst, cfg)
            if path is None:
                continue
            paths[(net, leg)] = path
            for l, y, x in path:
                demand[l, y, x] += 1.0

        over = (demand > 1.0).float()
        n_over = int(over.sum())
        if best_over is None or n_over < best_over:
            best_over, best_paths = n_over, paths
        if n_over == 0:
            break
        history = (history + over * cfg.history_rate).clamp_(0.0, cfg.max_history)

    # --- commit ---------------------------------------------------------
    # The negotiated paths may still overlap if the loop ran out of iterations,
    # and centre-line disjointness does not guarantee the *dilated* footprints
    # clear each other either. So the board is built by stamping in order and
    # dropping whatever will not fit -- and completion is counted from what
    # actually stamped, never from what was planned.
    br.reset()
    result = ExpertResult(
        attempted=len(order), iterations_used=used, unresolved=int(best_over or 0)
    )
    for net, leg, src, dst, wclass in order:
        path = (best_paths or {}).get((net, leg))
        if path is None:
            continue
        if not br.stamp(net, path, wclass):
            # Re-plan this one leg against the copper that is actually down.
            fld = br.field(net, dst, 1.0 + history, cfg, width_class=wclass, soft=False)
            if not torch.isfinite(fld[0, src[0], src[1], src[2]]):
                continue
            path = br.extract(fld, src, dst, cfg)
            if path is None or not br.stamp(net, path, wclass):
                continue
        result.paths[(net, leg)] = path
        result.completed.add((net, leg))
        result.vias += sum(1 for a, b in zip(path, path[1:]) if a[0] != b[0])
        result.length += float(len(path) - 1)

    return result


def route_world_board(
    world,
    board_index: int,
    cfg: ExpertConfig | None = None,
    *,
    negotiate: bool = True,
) -> ExpertResult:
    """Run the expert over one board of a loaded `SimultaneousRouterWorld`.

    Reads the board's *static* layer -- pads and keepouts -- never its live
    occupancy, so the expert always plans from a clean board regardless of what
    the world has already routed. That is what makes it a comparable baseline
    rather than a cleanup pass over someone else's work.
    """
    b = board_index
    legs = []
    for n in range(world.cfg.max_nets):
        if not bool(world.net_valid[b, n]):
            continue
        for leg in range(2):
            if not bool(world.leg_valid[b, n, leg]):
                continue
            src = tuple(int(v) for v in world.net_pad[b, n, leg, END_SRC])
            dst = tuple(int(v) for v in world.net_pad[b, n, leg, END_DST])
            legs.append((n, leg, src, dst, int(world.net_width[b, n])))
    return route_board(
        world.spec, world.static[b], legs, cfg, negotiate=negotiate, device=world.device
    )


def route_world_board_live(
    world,
    board_index: int,
    cfg: ExpertConfig | None = None,
    *,
    negotiate: bool = False,
) -> ExpertResult:
    """Re-plan a board's **still-open** legs from where their frontiers actually
    are, against **live occupancy**.

    `route_world_board` above plans from the static layer once, which is what
    makes it a fair baseline. It is the wrong thing to *demonstrate* from. In a
    simultaneous world the other nets lay copper as the episode runs, so a plan
    fixed at reset goes stale mid-episode and starts pointing the frontier into
    copper that now exists. Measured on stage 1 with `--bc-coef 0.5`, cloning
    that stale plan made the run oscillate 0.755 / 0.693 / 0.823 / 0.599 while
    `dir_d0_frac` swung 0.32 -> 0.01 -> 0.65 -> 0.07: the policy was being pulled
    between a dead plan and the live field.

    Three differences from the static version, and each one matters:

    * **Live occupancy, not static.** `_BoardRouter.hard_blocked` blocks cells
      that are neither free nor this net's own, so handing it `world.occ[b]`
      makes other nets' committed copper a hard obstacle while the net's own
      trail stays passable. Within a pass the remaining legs may still share
      cells with *each other* -- that is PathFinder's over-subscription signal
      and it must not be suppressed (see `hard_blocked`).
    * **Only open legs.** A finished leg needs no demonstration, and a failed
      net is gone. Re-planning them wastes the budget this runs on.
    * **From the frontier, not the pad.** A leg is planned trunk-end -> live
      frontier, so `bc.py`'s "END_DST walks the path BACKWARDS" rule carries the
      frontier toward the copper it still has to reach, from where it actually
      stands rather than from a pad it left twenty steps ago.

    `negotiate` defaults to **False** here, unlike the reset-time plan. Already
    committed copper has resolved most of the contention the negotiation loop
    exists to find, and this runs inside the collection loop: a negotiating pass
    is `ExpertConfig.iterations` (6) full passes over every leg.
    """
    b = board_index
    w = world
    legs = []
    per_leg = NUM_ENDS
    for n in range(w.cfg.max_nets):
        if not bool(w.net_valid[b, n]) or int(w.net_status[b, n]) != 0:
            continue
        for leg in range(w.cfg.max_legs):
            if not bool(w.leg_valid[b, n, leg]) or bool(w.leg_done[b, n, leg]):
                continue
            base = (n * w.cfg.max_legs + leg) * per_leg
            src = tuple(int(v) for v in w.fr_pos[b, base + END_SRC])
            dst = tuple(int(v) for v in w.fr_pos[b, base + END_DST])
            if src == dst:
                continue
            legs.append((n, leg, src, dst, int(w.net_width[b, n])))
    if not legs:
        return ExpertResult()
    return route_board(
        w.spec, w.occ[b], legs, cfg, negotiate=negotiate, device=w.device
    )
