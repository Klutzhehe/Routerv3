"""Board, netlist, design-rule and congestion-price specification.

Ported from `neuroroute/world/spec.py`, whose lattice-pitch reasoning was
verified against KiCad 9.0.2's own DRC_ENGINE (0 legality violations over 192
routed nets, 4 board configs). **Do not re-derive the pitch or pad-size logic**
-- see `mzr/DESIGN.md` section 11.

The load-bearing idea is the **routing lattice pitch**.

`docs/RL_PLAN.md` rejects rasters with a correct argument: 256 px over a 50 mm
board is 0.195 mm/px while clearance is 0.2 mm, so the legality margin is
sub-pixel and a raster cannot represent it. That argument is about
*rasterising continuous geometry*. It does not apply to a lattice whose pitch
is **defined** as ``min_track_width + min_clearance``: there, one cell is one
routing track slot, two tracks in adjacent cells are exactly one clearance
apart by construction, and legality reduces to cell occupancy -- which is
exact, not approximate.

What is new here relative to NeuroRoute is everything about *simultaneity*:
`FrontierEnd`, and `PriceRules` (section "Congestion price" below).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Net kinds. Kept as small ints because they live in a (B, N, F) float tensor
# as a one-hot slice -- see world/engine.py's net table layout.
# ---------------------------------------------------------------------------
KIND_SINGLE = 0
KIND_DIFF_PAIR = 1
KIND_LENGTH_GROUP = 2
NUM_KINDS = 3

# Routing phase per net. CONNECT grows copper toward the target; REFINE edits
# an already-connected polyline (see DESIGN.md section 12).
PHASE_CONNECT = 0
PHASE_REFINE = 1
PHASE_DONE = 2

# Occupancy sentinel values. Positive entries are ``net_index + 1`` so that 0
# can mean "free" without a separate mask.
OCC_FREE = 0
OCC_KEEPOUT = -1

# ---------------------------------------------------------------------------
# Frontier ends.
#
# Every net grows from BOTH pads inward, and the two frontiers meet in the
# middle. This halves per-net episode depth, which is half of why the
# macro-episode is ~48 steps deep regardless of net count (DESIGN.md section 1).
#
# Each frontier's *target* is the OPPOSITE pad's fixed cell -- not the other
# frontier's live position. That keeps the geodesic field static per frontier,
# so it is computed once per net rather than every macro-step. The two
# frontiers therefore descend (roughly) the same corridor from opposite ends
# and meet naturally; meeting is detected by proximity, not by the field.
# ---------------------------------------------------------------------------
END_SRC = 0
END_DST = 1
NUM_ENDS = 2


@dataclass(frozen=True)
class DesignRules:
    """Fabrication constraints, in millimetres.

    Defaults are the values `get_design_rules()` returns on this repo's
    generated boards (0.2 mm track / 0.2 mm clearance / 0.6 mm via, measured
    [LIVE], see docs/ROUTER_CAPABILITIES.md).
    """

    min_track_width: float = 0.2
    min_clearance: float = 0.2
    track_widths: Sequence[float] = (0.2, 0.3, 0.5, 0.8)
    via_diameters: Sequence[float] = (0.6, 0.8, 1.0, 1.2)
    via_drills: Sequence[float] = (0.3, 0.4, 0.5, 0.6)
    min_hole_to_hole: float = 0.25
    #: Pad side length, in millimetres. A pad is physically larger than a
    #: trace, so it must reserve more than one lattice cell -- reserving one
    #: cell while drawing a `pad_size` square is exactly the inconsistency that
    #: produced 0.1 mm pad-to-track clearance violations in KiCad's own DRC.
    #: The lattice footprint and the exported geometry are derived from this
    #: single number so they cannot drift apart again.
    pad_size: float = 0.4

    def __post_init__(self) -> None:
        if len(self.via_diameters) != len(self.via_drills):
            raise ValueError("via_diameters and via_drills must be the same length")
        if min(self.track_widths) < self.min_track_width - 1e-9:
            raise ValueError("a track width class is narrower than min_track_width")

    @property
    def num_width_classes(self) -> int:
        return len(self.track_widths)

    @property
    def num_via_classes(self) -> int:
        return len(self.via_diameters)

    @property
    def pitch_mm(self) -> float:
        """The lattice pitch: one track plus one clearance.

        This is the single number that makes cell occupancy equivalent to a
        clearance check. Do not set it independently of the rules -- derive it,
        so the two can never drift apart.
        """
        return self.min_track_width + self.min_clearance

    def _radius_cells(self, extent_mm: float) -> int:
        """Cells to dilate by, so a feature of diameter `extent_mm` plus one
        clearance fits inside ``2 * r + 1`` cells.

        Rounds up: a lattice router that under-dilates produces DRC errors, one
        that over-dilates only wastes space. The sim-to-real DRC gap is the
        check on this being right.
        """
        span_cells = (extent_mm + self.min_clearance) / self.pitch_mm
        return max(0, math.ceil((math.ceil(span_cells - 1e-9) - 1) / 2))

    def width_radius_cells(self, width_class: int) -> int:
        """Lateral dilation radius for a track of the given width class."""
        return self._radius_cells(self.track_widths[width_class])

    def via_radius_cells(self, via_class: int) -> int:
        """Dilation radius for a via of the given class, applied on every layer
        the via spans."""
        return self._radius_cells(self.via_diameters[via_class])

    def width_radii(self) -> np.ndarray:
        return np.array(
            [self.width_radius_cells(c) for c in range(self.num_width_classes)],
            dtype=np.int64,
        )

    def pad_radius_cells(self) -> int:
        """Cells a pad occupies on each side of its centre."""
        return self._radius_cells(self.pad_size)

    def via_radii(self) -> np.ndarray:
        return np.array(
            [self.via_radius_cells(c) for c in range(self.num_via_classes)],
            dtype=np.int64,
        )


@dataclass(frozen=True)
class LayerStack:
    """Copper stack-up. `num_layers` is a real parameter here (2 -> 8), unlike
    the PNS thread in this repo, where it was pinned at 2 by `switch_layer()`
    being 0-for-32 (docs/RL_PLAN.md, Gate A)."""

    num_layers: int = 8
    #: Layers a blind/buried via may terminate on. `None` = every layer, i.e.
    #: only through-vias are modelled. Through-only is the safe default: it is
    #: what cheap fabrication actually allows.
    through_only: bool = True

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")

    def via_span(self, layer_a: int, layer_b: int) -> tuple[int, int]:
        """Layer range a via between two layers actually occupies."""
        if self.through_only:
            return 0, self.num_layers - 1
        lo, hi = sorted((int(layer_a), int(layer_b)))
        return lo, hi


@dataclass
class BoardSpec:
    """A board's fixed geometry: lattice size, stack-up and rules.

    Grid indices are ``(layer, y, x)`` throughout, matching the tensor layout
    in world/engine.py, which matches numpy/torch image conventions so the
    convolutional encoder needs no transposes.
    """

    height_cells: int = 128
    width_cells: int = 128
    layers: LayerStack = field(default_factory=LayerStack)
    rules: DesignRules = field(default_factory=DesignRules)
    #: Board-edge margin in cells kept permanently keepout.
    edge_margin_cells: int = 2

    @property
    def num_layers(self) -> int:
        return self.layers.num_layers

    @property
    def pitch_mm(self) -> float:
        return self.rules.pitch_mm

    @property
    def extent_mm(self) -> tuple[float, float]:
        """(width, height) of the board in millimetres."""
        return (self.width_cells * self.pitch_mm, self.height_cells * self.pitch_mm)

    def cell_to_mm(self, x_cell: np.ndarray | float, y_cell: np.ndarray | float):
        """Lattice cell centre -> millimetres. The only conversion the KiCad
        exporter needs."""
        half = 0.5 * self.pitch_mm
        return (x_cell * self.pitch_mm + half, y_cell * self.pitch_mm + half)

    def mm_to_cell(self, x_mm: np.ndarray | float, y_mm: np.ndarray | float):
        """Millimetres -> nearest lattice cell. Used when ingesting a real
        `.kicad_pcb`, where pad centres will not sit exactly on the lattice."""
        return (
            np.rint(np.asarray(x_mm) / self.pitch_mm - 0.5).astype(np.int64),
            np.rint(np.asarray(y_mm) / self.pitch_mm - 0.5).astype(np.int64),
        )


@dataclass
class NetSpec:
    """One net to route.

    A differential pair is **one** NetSpec with `kind == KIND_DIFF_PAIR` and
    both legs' endpoints populated -- not two nets that a solver later
    discovers belong together by name. That is what lets the policy emit a
    single `couple` decision per step (DESIGN.md section 12).
    """

    src: tuple[int, int, int]  # (layer, y, x)
    dst: tuple[int, int, int]
    kind: int = KIND_SINGLE
    width_class: int = 0
    #: Second leg, diff pairs only.
    src_n: tuple[int, int, int] | None = None
    dst_n: tuple[int, int, int] | None = None
    #: Nominal edge-to-edge pair gap, in cells. 1 = adjacent lattice tracks.
    pair_gap_cells: int = 1
    #: Length-matched group id, or -1. Members of a group are tuned to the
    #: longest member's routed length -- resolved at runtime, not here.
    group_id: int = -1
    #: Absolute target length in cells, or <0 for "match the group".
    target_length_cells: float = -1.0
    #: Larger = routed earlier by the sequential baseline. The simultaneous
    #: policy has no ordering to prioritise, so this is baseline-only.
    priority: float = 1.0

    @property
    def is_pair(self) -> bool:
        return self.kind == KIND_DIFF_PAIR

    def endpoints(self) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        """(src, dst) per leg. One entry for a single net, two for a pair."""
        legs = [(self.src, self.dst)]
        if self.is_pair:
            if self.src_n is None or self.dst_n is None:
                raise ValueError("a diff-pair NetSpec needs src_n and dst_n")
            legs.append((self.src_n, self.dst_n))
        return legs


@dataclass
class Netlist:
    """The nets on one board, plus derived group bookkeeping."""

    nets: list[NetSpec] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.nets)

    @property
    def num_legs(self) -> int:
        """Legs across all nets. A diff pair has two."""
        return sum(2 if n.is_pair else 1 for n in self.nets)

    @property
    def num_frontiers(self) -> int:
        """Frontiers if every net were live at once.

        This is `M` in DESIGN.md section 5, and it is the number that must not
        appear in any model dimension: two per leg, because every leg grows
        from both ends.
        """
        return self.num_legs * NUM_ENDS

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i, n in enumerate(self.nets):
            if n.group_id >= 0:
                out.setdefault(n.group_id, []).append(i)
        return out


# ---------------------------------------------------------------------------
# Congestion price -- the negotiation substrate (DESIGN.md section 3)
# ---------------------------------------------------------------------------
#
# PathFinder (McMurchie & Ebeling, FPGA '95):
#
#     cost(c) = base(c) * (1 + h(c)) * (1 + p(c))
#
#       p(c)  present congestion  -- contention for c right now
#       h(c)  historical congestion -- accumulated over iterations
#
# Its predecessor (Nair 1987) assigned *infinite* cost to over-capacity
# resources. PathFinder's contribution was making the penalty **gradual**, so
# nets negotiate rather than hard-fail -- which is exactly what makes it usable
# as an observation channel a policy learns to read, rather than a constraint
# that masks actions.
#
# Where the contention signal comes from, in this engine: the arbitration step
# already computes, exactly, which cells more than one net claimed in the same
# macro-step (`geometry.resolve_claims`). That is *measured* contention, not an
# estimate from overlapping straight-line demand -- so the price field is a
# by-product of machinery the engine needs anyway.


@dataclass(frozen=True)
class PriceRules:
    """Congestion-price dynamics.

    Every default here is a starting point, not a tuned value. The offline-RL
    result cited in DESIGN.md section 15 (CQL selecting rip-up cost weights per
    iteration, 5-31% fewer iterations on unseen designs) says these are exactly
    the knobs worth *learning* once the hand-set version is stable. Do not
    hand-tune them past "obviously reasonable" -- that is the wrong use of a
    training budget.
    """

    #: Added to h(c) each macro-step a cell is over-subscribed. PathFinder
    #: raises history slowly so early contention does not permanently poison a
    #: corridor before the policy has had a chance to route around it.
    history_rate: float = 0.2
    #: Multiplicative decay applied to h(c) every macro-step. Slightly < 1 so a
    #: corridor that stops being contested eventually becomes cheap again;
    #: without decay, early thrash is remembered forever.
    history_decay: float = 0.995
    #: Multiplicative decay applied to p(c) every macro-step. Much faster than
    #: history: present congestion is meant to be a snapshot.
    present_decay: float = 0.5
    #: Weight of one contending net in p(c).
    present_rate: float = 1.0
    #: Ceiling on h(c) and p(c). Prevents a single pathological cell from
    #: dominating the observation's dynamic range.
    max_history: float = 8.0
    max_present: float = 8.0

    def __post_init__(self) -> None:
        if not 0.0 < self.history_decay <= 1.0:
            raise ValueError("history_decay must be in (0, 1]")
        if not 0.0 < self.present_decay <= 1.0:
            raise ValueError("present_decay must be in (0, 1]")


@dataclass(frozen=True)
class RipupRules:
    """Scheduled rip-up-and-regrow (DESIGN.md section 3).

    Nothing commits permanently early: every `interval` macro-steps, the
    most-congested unfinished nets give their copper back and regrow from their
    pads, while historical price stays elevated. This is the mechanical form of
    "let all nets branch out slowly so the AI can route things clearly."

    Whole nets are ripped, not partial frontier retractions. That is what
    PathFinder does (it rips and reroutes *every* net each iteration), it is
    unambiguous to implement, and a partial retraction would need to unstamp a
    polyline suffix from the occupancy grid -- reconstructing which cells a
    route suffix owns is exactly the ambiguity the per-frontier polyline exists
    to avoid.

    Fixed-rule at first, deliberately. Making it a learned decision from day
    one would reintroduce the pointer-over-nets credit-assignment problem that
    received zero gradient for NeuroRoute's entire history.
    """

    #: Macro-steps between rip-up rounds. 0 disables rip-up entirely (stage 0,
    #: where there is one net and nothing to negotiate with).
    interval: int = 8
    #: Fraction of unfinished nets ripped per round, most-congested first.
    fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.interval < 0:
            raise ValueError("ripup interval must be >= 0")
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("ripup fraction must be in [0, 1]")


# ---------------------------------------------------------------------------
# Action space. Factorised deliberately: the flat product is unlearnable, while
# the factorisation is ~32 logits per frontier. See DESIGN.md section 2.
# ---------------------------------------------------------------------------

#: 8 compass directions, egocentric -- index 0 is *down the geodesic gradient
#: toward the target*, not board-north. Carried over from the raster thread's
#: `_bearing_vector`, the one piece of that env measured to remove board-pose
#: generalisation entirely [LIVE].
NUM_DIRECTIONS = 8

#: Cells advanced per frontier per macro-step.
#:
#: NeuroRoute used (1, 2, 4, 8). This is (1, 2, 4) deliberately: with every
#: frontier moving each macro-step, negotiation granularity matters more than
#: raw reach, and the shorter maximum lets `snap_radius` be 2 cells instead of
#: 4 (see WorldConfig), which matters for landing accurately on a pad.
#:
#: The horizon arithmetic in DESIGN.md section 1 assumes this: a ~120-cell net
#: at a mean step of ~2.5 is ~48 macro-steps, independent of net count.
STEP_LENGTHS = (1, 2, 4)
NUM_STEPS = len(STEP_LENGTHS)

#: Perpendicular vertex-drag offsets available in the refine phase, in cells.
REFINE_OFFSETS = (-4, -2, -1, 0, 1, 2, 4)
NUM_REFINE_OFFSETS = len(REFINE_OFFSETS)


def direction_vectors() -> np.ndarray:
    """(NUM_DIRECTIONS, 2) unit (dy, dx) lattice offsets, counter-clockwise
    from +x. Diagonals are full (±1, ±1) steps -- a 45-degree lattice move,
    which the engine treats as occupying both orthogonal neighbours for
    clearance (see geometry's CELLS_PER_STEP)."""
    vecs = []
    for i in range(NUM_DIRECTIONS):
        angle = 2.0 * math.pi * i / NUM_DIRECTIONS
        dx = int(round(math.cos(angle)))
        dy = int(round(math.sin(angle)))
        vecs.append((dy, dx))
    return np.array(vecs, dtype=np.int64)


DIRECTION_VECTORS = direction_vectors()
