"""Cost-to-go over free space: the potential that makes a detour pay.

## Why this exists

`LineRouteEnv`'s shaping used `phi = -straight_line_distance`. Potential-based
shaping is policy-invariant (Ng et al.), so that choice cannot change which
policy is optimal -- but policy-invariance is a statement about the optimum,
not about whether gradient descent can find it. Measured on the real reward:

    step toward the pad (into the obstacle) : +0.0545
    step sideways (contouring)              : +0.0050
    step away (rounding a wide obstacle)    : -0.0445

A straight-line potential pays +0.099/step to drive INTO whatever is in the
way. The only thing that ever said otherwise was the +25 at completion,
arriving 20-40 steps later and only if the entire detour succeeded, so half a
detour scored strictly worse than no detour and the agent never accumulated a
completed one to learn from.

It didn't. Two runs, two reward configurations, 600k steps, benchmarked
against the greedy straight-line router on the same 25 boards:

    greedy a=0          62.90%  (78/124)
    300k policy         62.10%  (77/124)     wirelength 1.06x

Identical within a 4.3-point standard error, and a 1.06x wirelength ratio
says the learned policy was still drawing straight lines. No amount of
network capacity fixes a reward whose local gradient opposes the required
manoeuvre on every single step.

Replacing the potential with the length of the shortest obstacle-free path
inverts those three numbers: rounding an obstacle now decreases the potential,
so every step of a correct detour is paid for immediately, and the sparse
completion bonus stops being the only signal that a detour was worth making.

## Why a field rather than a path

A per-step A* would re-plan on every move. Instead this relaxes a single
cost-to-go field outward from the target once, after which each step is a
lookup. That is affordable exactly because `_rebuild_obstacles()` already
caches the obstacle set once per net rather than once per step -- the field
has the same lifetime as the cache it is built from, and rides along with it.

## Pure numpy, deliberately

Same reasoning as line_obs.py: no bridge import, so it runs and unit-tests
anywhere, and no scipy/skimage either, so requirements.txt stays honest and
Colab needs nothing extra. The relaxation below is the one non-obvious part
and is pinned by tests that check it against hand-computed geometry.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from pcbworld.env.line_obs import KIND_EDGE, KIND_GHOST, MM, Segment

# Orthogonal and diagonal step costs, in cells.
_ORTHO = 1.0
_DIAG = float(np.sqrt(2.0))

# Sweep cycles before giving up. A path needs one cycle per change of
# direction; the cap only binds on a spiral, which a PCB is not.
_MAX_SWEEP_CYCLES = 16


@dataclasses.dataclass(frozen=True)
class GeodesicConfig:
    """Grid resolution and how far copper reaches.

    `cell_nm` at the env's step size is the natural choice: a finer grid buys
    resolution the agent cannot act on, and a coarser one lets the head cross
    a cell without the potential changing, which reintroduces the flat spots
    this is meant to remove.

    `inflation_nm` is added to each obstacle's own half-width. It is NOT just
    the legal clearance, and that distinction is the difference between a plan
    that can be followed and one that cannot.

    The legal minimum is 325 um: the routed track's half-width (125) plus the
    design clearance (200). A plan drawn at exactly that threads corridors
    where the correct path is legal and every path beside it is not -- and the
    head does not follow the correct path. It moves in fixed 0.5 mm steps at a
    continuous heading, chosen by one-step lookahead on a discretised field, so
    it tracks the plan with up to about half a step of error. Following a
    minimum-clearance plan imperfectly means touching copper, and PNS then
    refuses to commit the whole route.

    That is not a hypothesis. Measured on 25 Colab boards, 44 of 45 failures
    were fix() refusals, 100% of them colliding at the call, 100% against
    ANOTHER net's copper, 0% near either pad -- the head arriving dirty after
    a 0.99x (i.e. dead straight) route. Reproduced in simulation and swept:

        inflation | blocked nets committed clean | projected overall
          325 um  |            21.7%             |      59.5%   <- as shipped
          450 um  |            35.7%             |      66.8%
          600 um  |            65.2%             |      82.0%
          700 um  |            69.6%             |      84.2%   <- peak
          800 um  |            66.2%             |      82.5%
         1000 um  |            64.3%             |      81.5%

    So the margin is legal clearance PLUS the follower's tracking error, and
    700 um is roughly 325 + 0.75 of a step. Past the peak the plan turns down
    corridors that are genuinely passable and reachability starts costing more
    than the cleanliness buys.

    Grid resolution is NOT the lever here -- 250 um and 125 um cells scored
    worse at the same margin, because a finer grid is better at finding the
    tight gap the head then fails to thread.

    `max_cells_per_axis` is a guard, not a tuning knob. It caps the work on a
    board far larger than the curriculum's 35 mm, at the cost of resolution.
    """

    cell_nm: float = 500_000.0
    inflation_nm: float = 700_000.0   # 325um legal + ~375um follow margin
    max_cells_per_axis: int = 256
    margin_nm: float = 2.0 * MM

    def resolved_cell_nm(self, width_nm: float, height_nm: float) -> float:
        """cell_nm, coarsened if the board would otherwise exceed the cap."""
        span = max(width_nm, height_nm, 1.0)
        return max(self.cell_nm, span / self.max_cells_per_axis)


class GeodesicField:
    """Shortest obstacle-free distance from any point to one net's target.

    `cost_to_go` returns nanometres, or `inf` when the target cannot be
    reached from that point through free space. Callers are expected to fall
    back to straight-line distance on `inf` rather than to treat it as a
    penalty -- an unreachable target is a fact about the board, and charging
    the agent for it would punish it for geometry it did not create.
    """

    __slots__ = ("cost", "origin_x", "origin_y", "cell", "nx", "ny", "reachable")

    def __init__(self, cost, origin_x, origin_y, cell, reachable):
        self.cost = cost
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.cell = cell
        self.ny, self.nx = cost.shape
        self.reachable = reachable

    # -- construction ----------------------------------------------------

    @classmethod
    def build(
        cls,
        segments: list[Segment],
        *,
        head: tuple[float, float],
        target: tuple[float, float],
        config: GeodesicConfig | None = None,
    ) -> "GeodesicField":
        cfg = config or GeodesicConfig()

        xs = [head[0], target[0]]
        ys = [head[1], target[1]]
        for s in segments:
            xs.extend((s.x1, s.x2))
            ys.extend((s.y1, s.y2))

        min_x, max_x = min(xs) - cfg.margin_nm, max(xs) + cfg.margin_nm
        min_y, max_y = min(ys) - cfg.margin_nm, max(ys) + cfg.margin_nm
        cell = cfg.resolved_cell_nm(max_x - min_x, max_y - min_y)

        nx = max(2, int(np.ceil((max_x - min_x) / cell)) + 1)
        ny = max(2, int(np.ceil((max_y - min_y) / cell)) + 1)

        cx = min_x + (np.arange(nx, dtype=np.float64) + 0.5) * cell
        cy = min_y + (np.arange(ny, dtype=np.float64) + 0.5) * cell
        gx, gy = np.meshgrid(cx, cy)

        blocked = cls._blocked_mask(segments, gx, gy, cfg)

        field = cls(
            np.full((ny, nx), np.inf, dtype=np.float64),
            min_x, min_y, cell, True,
        )
        ti, tj = field._cell_of(*target)
        hi, hj = field._cell_of(*head)

        # The net's own endpoints are targets, not obstacles. They are already
        # excluded from `segments`, but a neighbouring pad's inflation can
        # still cover them -- and a target sitting in a blocked cell makes the
        # whole field unreachable, which is the one failure that silently
        # turns this back into the straight-line potential it replaces.
        blocked[ti, tj] = False
        blocked[hi, hj] = False

        cost = cls._relax(blocked, ti, tj, cell)
        field.reachable = bool(np.isfinite(cost[hi, hj]))
        field.cost = cls._fill_blocked(cost, blocked, cell)
        return field

    @staticmethod
    def _blocked_mask(segments, gx, gy, cfg: GeodesicConfig) -> np.ndarray:
        """Cells whose centre is inside an obstacle's inflated footprint.

        Ghost segments are skipped: an unrouted net is a plan, not copper, and
        planning around one would make a target unreachable whenever two nets
        want the same corridor -- which is most of what the later stages are
        about. They stay in the observation, where the policy can weigh them,
        rather than in the potential, where they would be hard constraints.
        """
        blocked = np.zeros(gx.shape, dtype=bool)

        # The outline bounds the board; it is not a wall standing in it. It has
        # to be handled as a box rather than as line obstacles because
        # generate_board.py emits SHAPE_T_RECT and the bridge reports a rect by
        # its two opposite corners -- so rasterising the "segment" would lay a
        # diagonal straight across the middle of the board and make most nets
        # look unroutable. Taking the bounding box of every edge item is
        # correct whether the outline arrives as one rect or as four sides.
        edges = [e for e in segments if e.kind == KIND_EDGE]
        if edges:
            ex = [v for e in edges for v in (e.x1, e.x2)]
            ey = [v for e in edges for v in (e.y1, e.y2)]
            inset = cfg.inflation_nm
            blocked |= (
                (gx < min(ex) + inset) | (gx > max(ex) - inset)
                | (gy < min(ey) + inset) | (gy > max(ey) - inset)
            )

        for s in segments:
            if s.kind in (KIND_GHOST, KIND_EDGE):
                continue
            radius = cfg.inflation_nm + 0.5 * s.width
            dx, dy = s.x2 - s.x1, s.y2 - s.y1
            len_sq = dx * dx + dy * dy
            if len_sq > 0:
                t = ((gx - s.x1) * dx + (gy - s.y1) * dy) / len_sq
                np.clip(t, 0.0, 1.0, out=t)
                px, py = s.x1 + t * dx, s.y1 + t * dy
            else:
                px, py = s.x1, s.y1
            blocked |= (gx - px) ** 2 + (gy - py) ** 2 <= radius * radius
        return blocked

    @staticmethod
    def _relax(blocked: np.ndarray, ti: int, tj: int, cell: float) -> np.ndarray:
        """Dijkstra as Bellman-Ford with a directional sweep order.

        The obvious formulation -- relax all eight neighbours over the whole
        grid, repeat -- advances the frontier by exactly one cell per pass, so
        it needs as many passes as the grid is wide. Measured on a 79x79 board
        that was 141 passes and 70 ms, against a rollout that only takes 1.4 s
        in total; the shaping would have cost more than the router.

        Sweeping instead carries a value the full length of the grid in a
        single pass: the downward sweep reads only the row above, so it can be
        vectorised across the row while still being sequential in the
        direction that matters. Four sweeps (down, up, right, left) per cycle,
        and a path only needs another cycle when it changes direction --
        which on a board of scattered pads is two or three times, not eighty.
        Bellman-Ford converges under any edge order, so this is exact once the
        passes stop changing anything, not an approximation.

        float32 throughout: the field is a shaping potential divided by a
        10 mm length scale, and nanometre-exact cost-to-go would be precision
        spent on a quantity the grid has already discretised.
        """
        ny, nx = blocked.shape
        cost = np.full((ny, nx), np.inf, dtype=np.float32)
        cost[ti, tj] = 0.0

        ortho = np.float32(_ORTHO * cell)
        diag = np.float32(_DIAG * cell)

        def carry(dst, src, blocked_line):
            """Relax one row/column from its neighbour, including diagonals."""
            cand = src + ortho
            np.minimum(cand[1:], src[:-1] + diag, out=cand[1:])
            np.minimum(cand[:-1], src[1:] + diag, out=cand[:-1])
            np.minimum(dst, cand, out=dst)
            dst[blocked_line] = np.inf

        for _ in range(_MAX_SWEEP_CYCLES):
            before = cost.copy()

            for i in range(1, ny):
                carry(cost[i], cost[i - 1], blocked[i])
            for i in range(ny - 2, -1, -1):
                carry(cost[i], cost[i + 1], blocked[i])
            for j in range(1, nx):
                carry(cost[:, j], cost[:, j - 1], blocked[:, j])
            for j in range(nx - 2, -1, -1):
                carry(cost[:, j], cost[:, j + 1], blocked[:, j])

            cost[ti, tj] = 0.0
            if np.array_equal(before, cost):
                break

        return cost

    @staticmethod
    def _fill_blocked(cost, blocked, cell, passes: int = 4):
        """Give blocked cells a finite cost, bleeding outward from free space.

        The head spends 10-45% of its steps inside an obstacle -- PNS marks
        the collision but still advances the full step -- so "the head is in a
        blocked cell" is the common case, not the edge case. Left at inf the
        potential would jump to the straight-line fallback for those steps and
        back again afterwards, and a potential that switches definition
        mid-net produces a reward spike on the switch, which is exactly the
        kind of farmable artifact potential-based shaping exists to avoid.

        Only cells that are still infinite are written, so the free-space
        field stays exact; a value can bleed through a thin wall, but only
        into cells that are inside copper, whose cost is a fallback rather
        than a route. Four passes is 2 mm at a 0.5 mm cell -- enough to cover
        any pad the head can be inside of.
        """
        filled = cost.copy()
        ny, nx = filled.shape
        ortho = np.float32(_ORTHO * cell)
        diag = np.float32(_DIAG * cell)
        padded = np.empty((ny + 2, nx + 2), dtype=np.float32)

        for _ in range(passes):
            gaps = blocked & ~np.isfinite(filled)
            if not gaps.any():
                break
            padded.fill(np.inf)
            padded[1:-1, 1:-1] = filled
            best = np.minimum.reduce([
                padded[0:-2, 1:-1] + ortho, padded[2:, 1:-1] + ortho,
                padded[1:-1, 0:-2] + ortho, padded[1:-1, 2:] + ortho,
                padded[0:-2, 0:-2] + diag, padded[0:-2, 2:] + diag,
                padded[2:, 0:-2] + diag, padded[2:, 2:] + diag,
            ])
            filled = np.where(gaps, best, filled)

        return filled

    # -- queries ---------------------------------------------------------

    def _cell_of(self, x: float, y: float) -> tuple[int, int]:
        j = int(np.clip((x - self.origin_x) / self.cell, 0, self.nx - 1))
        i = int(np.clip((y - self.origin_y) / self.cell, 0, self.ny - 1))
        return i, j

    def cost_to_go(self, x: float, y: float) -> float:
        """Bilinearly interpolated distance to the target, in nm.

        Interpolated rather than nearest-cell because the head advances one
        cell per step by construction: sampling the nearest cell would make
        the potential a staircase, and every step that landed inside the
        current cell would earn exactly zero shaping. Unreachable neighbours
        are dropped from the average rather than counted as a large finite
        cost, so a cell beside a wall is not dragged upward by the wall.
        """
        fx = (x - self.origin_x) / self.cell - 0.5
        fy = (y - self.origin_y) / self.cell - 0.5
        j0 = int(np.floor(fx))
        i0 = int(np.floor(fy))
        tx, ty = fx - j0, fy - i0

        total_w = 0.0
        total_c = 0.0
        for di, wy in ((0, 1.0 - ty), (1, ty)):
            for dj, wx in ((0, 1.0 - tx), (1, tx)):
                w = wx * wy
                if w <= 0.0:
                    continue
                i, j = i0 + di, j0 + dj
                if not (0 <= i < self.ny and 0 <= j < self.nx):
                    continue
                c = self.cost[i, j]
                if np.isfinite(c):
                    total_w += w
                    total_c += w * c

        if total_w <= 0.0:
            i, j = self._cell_of(x, y)
            return float(self.cost[i, j])
        return float(total_c / total_w)
