"""Procedural board and netlist generation.

Boards are generated as **components with pin arrays**, not as scattered random
pads. That matters more than it looks: the hard part of real routing is *pin
escape* -- getting out of a dense pin field before you can go anywhere -- and a
board of uniformly-scattered pads never poses that problem. `docs/RL_PLAN.md`
records the consequence of not having it: on this repo's scattered-pad boards,
15/24 nets were unreachable on one layer and 1-2 mm detours rescued 0 of them,
because pads are obstacles from the very first net. That is a *board*
pathology as much as a router one.

Generation runs once per episode, in numpy, on the CPU. It is not on the hot
path -- `step()` is (see world/engine.py) -- so clarity wins over speed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mzr.world.spec import (
    KIND_DIFF_PAIR,
    KIND_LENGTH_GROUP,
    KIND_SINGLE,
    BoardSpec,
    NetSpec,
    Netlist,
)


@dataclass
class GeneratorConfig:
    """Difficulty knobs. The curriculum in `training/curriculum.py` moves these
    and nothing else -- the model has no stage-dependent parameters."""

    num_nets: int = 20
    num_components: int = 6
    #: Pins per component, as (rows, cols) sampled from this range.
    pin_rows: tuple[int, int] = (2, 6)
    pin_cols: tuple[int, int] = (2, 6)
    #: Lattice cells between adjacent pins of one component. Must exceed the
    #: pad footprint (2*pad_radius+1) or neighbouring pads merge into a solid
    #: block with no escape route between them.
    pin_pitch_cells: int = 4
    #: Fraction of nets that are differential pairs.
    diff_pair_frac: float = 0.0
    #: Fraction of nets that belong to a length-matched group.
    length_group_frac: float = 0.0
    length_group_size: int = 4
    #: Fraction of nets asking for a non-minimum track width.
    wide_net_frac: float = 0.0
    #: Rectangular keepouts (mounting holes, connectors, antenna clearance).
    num_keepouts: int = 0
    keepout_max_cells: int = 10
    #: Copper pours, as (layer, y0, y1, x0, x1). A pour is an obstacle to other
    #: nets and a valid terminal for its own net.
    num_pours: int = 0
    #: Pins may be placed on any layer when True; otherwise only outer layers,
    #: which is what real SMD boards look like and forces via usage.
    pins_on_inner_layers: bool = False


@dataclass
class GeneratedBoard:
    """One board, ready to be loaded into a `BatchedRouterWorld` slot."""

    spec: BoardSpec
    netlist: Netlist
    #: (L, H, W) int16 -- pads and keepouts pre-stamped, using the same
    #: net_id+1 / OCC_KEEPOUT encoding as the live occupancy grid.
    static: np.ndarray
    #: (L, H, W) bool -- cells belonging to a copper pour, per layer.
    pour_mask: np.ndarray


def _pins_required(cfg: GeneratorConfig) -> int:
    """Pins the requested netlist needs. A differential pair needs four."""
    pairs = int(round(cfg.num_nets * cfg.diff_pair_frac))
    return 2 * (cfg.num_nets - pairs) + 4 * pairs


def _component_pins(
    rng: np.random.Generator,
    spec: BoardSpec,
    cfg: GeneratorConfig,
    min_pins: int,
) -> list[tuple[int, int, int]]:
    """Place components and return every pin as (layer, y, x).

    Components are placed by rejection sampling against already-placed
    bounding boxes, so pin fields never overlap -- overlapping footprints
    would make some nets unroutable for a reason that has nothing to do with
    routing skill.

    Placement continues past `cfg.num_components` if the netlist still needs
    pins. Without that, a run of unlucky rejections silently yields a board
    with **fewer nets than asked for, or none at all** -- and a board with zero
    nets scores 0% completion, so a handful of them quietly drags an otherwise
    healthy metric down and reads as a learning failure. That happened; this
    loop is the fix.
    """
    H, W = spec.height_cells, spec.width_cells
    margin = spec.edge_margin_cells + 2
    placed: list[tuple[int, int, int, int]] = []
    pins: list[tuple[int, int, int]] = []

    outer_layers = [0] if spec.num_layers == 1 else [0, spec.num_layers - 1]
    budget = max(cfg.num_components * 8, 64)

    while budget > 0 and (len(placed) < cfg.num_components or len(pins) < min_pins):
        budget -= 1
        rows = int(rng.integers(cfg.pin_rows[0], cfg.pin_rows[1] + 1))
        cols = int(rng.integers(cfg.pin_cols[0], cfg.pin_cols[1] + 1))
        h = (rows - 1) * cfg.pin_pitch_cells + 1
        w = (cols - 1) * cfg.pin_pitch_cells + 1
        if h >= H - 2 * margin or w >= W - 2 * margin:
            continue

        for _attempt in range(48):
            y0 = int(rng.integers(margin, H - margin - h))
            x0 = int(rng.integers(margin, W - margin - w))
            box = (y0 - 2, x0 - 2, y0 + h + 2, x0 + w + 2)
            if all(
                box[2] <= q[0] or q[2] <= box[0] or box[3] <= q[1] or q[3] <= box[1]
                for q in placed
            ):
                placed.append(box)
                if cfg.pins_on_inner_layers:
                    layer = int(rng.integers(0, spec.num_layers))
                else:
                    layer = int(rng.choice(outer_layers))
                for r in range(rows):
                    for c in range(cols):
                        pins.append(
                            (layer, y0 + r * cfg.pin_pitch_cells, x0 + c * cfg.pin_pitch_cells)
                        )
                break

    return pins


def generate_board(
    spec: BoardSpec,
    cfg: GeneratorConfig,
    seed: int,
) -> GeneratedBoard:
    """Build one board. Deterministic in `seed`, so a failing board is always
    reproducible from its seed alone -- the same property that made the raster
    thread's 10 known-hard seeds usable as a regression set."""
    rng = np.random.default_rng(seed)
    H, W, L = spec.height_cells, spec.width_cells, spec.num_layers

    static = np.zeros((L, H, W), dtype=np.int16)
    pour_mask = np.zeros((L, H, W), dtype=bool)

    # Board edge margin: permanently keepout on every layer.
    em = spec.edge_margin_cells
    if em > 0:
        static[:, :em, :] = -1
        static[:, -em:, :] = -1
        static[:, :, :em] = -1
        static[:, :, -em:] = -1

    for _ in range(cfg.num_keepouts):
        kh = int(rng.integers(3, cfg.keepout_max_cells + 1))
        kw = int(rng.integers(3, cfg.keepout_max_cells + 1))
        y0 = int(rng.integers(em, max(em + 1, H - em - kh)))
        x0 = int(rng.integers(em, max(em + 1, W - em - kw)))
        static[:, y0 : y0 + kh, x0 : x0 + kw] = -1

    pins = _component_pins(rng, spec, cfg, _pins_required(cfg))
    if len(pins) < 4:
        # Should be unreachable now that placement keeps going until the pin
        # budget is met; kept as a guard because a zero-net board silently
        # corrupts the completion metric rather than raising.
        return GeneratedBoard(spec, Netlist([]), static, pour_mask)

    pin_arr = np.array(pins, dtype=np.int64)
    order = rng.permutation(len(pins))

    n_pairs = int(round(cfg.num_nets * cfg.diff_pair_frac))
    n_group = int(round(cfg.num_nets * cfg.length_group_frac))

    nets: list[NetSpec] = []
    cursor = 0

    def take(k: int) -> np.ndarray | None:
        nonlocal cursor
        if cursor + k > len(order):
            return None
        sel = pin_arr[order[cursor : cursor + k]]
        cursor += k
        return sel

    # --- differential pairs: four pins, two at each end ----------------------
    for _ in range(n_pairs):
        sel = take(4)
        if sel is None:
            break
        # Pair the two closest pins as one end so the pair starts coupled.
        d = np.linalg.norm(sel[:, 1:][:, None, :] - sel[:, 1:][None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        a, b = np.unravel_index(np.argmin(d), d.shape)
        rest = [i for i in range(4) if i not in (a, b)]
        nets.append(
            NetSpec(
                src=tuple(int(v) for v in sel[a]),
                dst=tuple(int(v) for v in sel[rest[0]]),
                src_n=tuple(int(v) for v in sel[b]),
                dst_n=tuple(int(v) for v in sel[rest[1]]),
                kind=KIND_DIFF_PAIR,
                width_class=0,
                pair_gap_cells=1,
                priority=2.0,
            )
        )

    # --- length-matched groups ----------------------------------------------
    gid = 0
    made = 0
    while made < n_group:
        size = min(cfg.length_group_size, n_group - made)
        if size < 2:
            break
        ok = True
        for _ in range(size):
            sel = take(2)
            if sel is None:
                ok = False
                break
            nets.append(
                NetSpec(
                    src=tuple(int(v) for v in sel[0]),
                    dst=tuple(int(v) for v in sel[1]),
                    kind=KIND_LENGTH_GROUP,
                    group_id=gid,
                    priority=1.5,
                )
            )
            made += 1
        gid += 1
        if not ok:
            break

    # --- plain single nets ---------------------------------------------------
    while len(nets) < cfg.num_nets:
        sel = take(2)
        if sel is None:
            break
        wide = rng.random() < cfg.wide_net_frac
        wc = int(rng.integers(1, spec.rules.num_width_classes)) if wide else 0
        nets.append(
            NetSpec(
                src=tuple(int(v) for v in sel[0]),
                dst=tuple(int(v) for v in sel[1]),
                kind=KIND_SINGLE,
                width_class=wc,
            )
        )

    netlist = Netlist(nets)

    # --- copper pours --------------------------------------------------------
    # A pour belongs to a net (usually ground). It blocks other nets and is a
    # legal terminal for its own -- both are just occupancy, so a pour needs no
    # special case anywhere else in the engine.
    for _ in range(cfg.num_pours):
        if not nets:
            break
        layer = int(rng.integers(0, L))
        ph = int(rng.integers(H // 6, max(H // 6 + 1, H // 3)))
        pw = int(rng.integers(W // 6, max(W // 6 + 1, W // 3)))
        y0 = int(rng.integers(em, max(em + 1, H - em - ph)))
        x0 = int(rng.integers(em, max(em + 1, W - em - pw)))
        owner = int(rng.integers(0, len(nets)))
        region = static[layer, y0 : y0 + ph, x0 : x0 + pw]
        free = region == 0
        region[free] = np.int16(owner + 1)
        pour_mask[layer, y0 : y0 + ph, x0 : x0 + pw] |= free

    # --- stamp pads ----------------------------------------------------------
    # Pads go in last so they win over a pour that happens to cover them, and
    # they are stamped at their REAL size (`DesignRules.pad_size`), not as a
    # single cell. A one-cell pad lets a trace run in the neighbouring cell,
    # 0.4 mm from the pad centre -- which is 0.1 mm edge-to-edge against a
    # 0.4 mm pad and a real clearance violation that KiCad's DRC caught.
    #
    # Centres are written first, then the surround, so a pad can never erase a
    # neighbouring pad's centre when two footprints sit close together.
    r = spec.rules.pad_radius_cells()
    for i, net in enumerate(netlist.nets):
        for src, dst in net.endpoints():
            for (ly, py, px) in (src, dst):
                if 0 <= ly < L and 0 <= py < H and 0 <= px < W:
                    static[ly, py, px] = np.int16(i + 1)
    if r > 0:
        for i, net in enumerate(netlist.nets):
            for src, dst in net.endpoints():
                for (ly, py, px) in (src, dst):
                    if not (0 <= ly < L and 0 <= py < H and 0 <= px < W):
                        continue
                    y0, y1 = max(0, py - r), min(H, py + r + 1)
                    x0, x1 = max(0, px - r), min(W, px + r + 1)
                    block = static[ly, y0:y1, x0:x1]
                    block[block == 0] = np.int16(i + 1)

    return GeneratedBoard(spec, netlist, static, pour_mask)


def straight_line_demand(spec: BoardSpec, netlist: Netlist) -> np.ndarray:
    """(L, H, W) float32 estimate of routing demand from unrouted nets.

    Each net contributes a rasterised straight line between its endpoints,
    spread over the layers its endpoints touch. This is the **baseline the
    learned `FutureFieldPredictor` has to beat** (DESIGN.md section 5's gate):
    if a forecast trained on real rollouts cannot outperform drawing straight
    lines, it has learned nothing worth carrying, and stage 2 does not start.
    """
    H, W, L = spec.height_cells, spec.width_cells, spec.num_layers
    field = np.zeros((L, H, W), dtype=np.float32)
    for net in netlist.nets:
        for src, dst in net.endpoints():
            l0, y0, x0 = src
            l1, y1, x1 = dst
            n = max(abs(y1 - y0), abs(x1 - x0), 1)
            ys = np.rint(np.linspace(y0, y1, n + 1)).astype(np.int64).clip(0, H - 1)
            xs = np.rint(np.linspace(x0, x1, n + 1)).astype(np.int64).clip(0, W - 1)
            for ly in {int(l0), int(l1)}:
                if 0 <= ly < L:
                    np.add.at(field[ly], (ys, xs), 1.0)
    return field
