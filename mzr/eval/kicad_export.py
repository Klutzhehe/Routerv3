"""Export a routed lattice to a real `.kicad_pcb`.

Written as plain s-expression text, **not** through `pcbnew`. That is a
deliberate constraint: `ROADMAP.md`'s hard constraint 1 says the compiled
`pcbworld_pns_bridge` and the system `pcbnew` module can never share a process
(both define KiCad's one-instance-per-process globals; loading both crashes
with no Python-catchable exception). A text writer has no such constraint,
needs no KiCad install to run, and needs no subprocess dance -- so the export
path can be exercised anywhere, including on a laptop with no KiCad at all.

Verification happens the other way round: `kicad-cli pcb drc` reads the file we
wrote and judges it. That is a stronger check than round-tripping through
`pcbnew`, because it is KiCad's own `DRC_ENGINE` judging the copper rather than
our geometry judging itself.

Ported from `neuroroute/eval/kicad_export.py`, whose output KiCad 9.0.2 accepted
and found DRC-clean over 192 routed nets. **One real difference**: routes here
are stored per *frontier*, because every net grows from both pads inward, so a
leg's copper is two polylines that meet in the middle. See `_leg_polyline`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mzr.world.engine import STATUS_DONE
from mzr.world.spec import END_DST, END_SRC, KIND_DIFF_PAIR, NUM_ENDS, BoardSpec


def copper_layer_names(num_layers: int) -> list[str]:
    """Lattice layer index -> KiCad layer name.

    Index 0 is top copper and `num_layers - 1` is bottom, with inner layers
    between -- the same ordering the lattice uses.
    """
    if num_layers == 1:
        return ["F.Cu"]
    return ["F.Cu"] + [f"In{i}.Cu" for i in range(1, num_layers - 1)] + ["B.Cu"]


def _layer_ordinal(index: int, num_layers: int) -> int:
    """Ordinal for the `(layers ...)` stack-up block.

    KiCad's *runtime* `PCB_LAYER_ID` numbering changed in 9.0 -- this repo
    measured `F_Cu = 0` but `B_Cu = 2` against a local 9.0 install, which is not
    the historical `B.Cu = 31`. The *file format* has historically used
    F.Cu = 0, In1..InN = 1..N, B.Cu = 31, and KiCad still reads files written
    that way; a file KiCad can load gets normalised on load, whereas one it
    cannot parse fails outright. Verified accepted by 9.0.2 and 8.0.9.
    """
    if index == 0:
        return 0
    if index == num_layers - 1:
        return 31
    return index


@dataclass
class ExportStats:
    tracks: int
    vias: int
    pads: int
    nets: int
    #: Legs exported as two stubs rather than one joined route -- see
    #: `_leg_polyline`. Surfaced because it is the honest count of copper that
    #: will read as `track_dangling` to KiCad.
    stub_legs: int


def _leg_runs(routes, counts, f_src: int, f_dst: int) -> list[list[tuple[int, int, int]]]:
    """One leg's copper, as a list of **separately-drawn** polyline runs.

    Both frontiers start on their own pad and grow toward each other, so a leg's
    copper is ``src-polyline`` and ``dst-polyline reversed``.

    When the two frontiers *met*, those two halves are contiguous and become one
    run. `_connect` appended the meeting point to the src frontier, so the join
    point appears in both halves; the duplicate is a zero-length segment and the
    writer drops it.

    When one frontier instead reached the far *pad*, that half is already a
    complete route and the other half is a **stub** hanging off its own pad --
    and the two ends are nowhere near each other. They must be emitted as
    **two runs**.

    Joining them unconditionally was a real bug, caught by this gate on its
    first run: it drew a phantom segment straight across the board from the far
    pad to the stub's tip, backed by no copper in `occ`. KiCad reported it as
    30 `clearance`, 11 `shorting_items` and 5 `tracks_crossing` violations. The
    lattice was innocent; the exporter was drawing a board that had never been
    routed. Contiguity is therefore *tested*, never assumed.

    Both halves are still exported, stub included: the exported board must be
    the board the engine actually checked, or this gate validates a different
    board than the one that was routed. KiCad reports a stub's far end as
    `track_dangling`, classified as *completeness* rather than legality -- the
    honest description of a route that was never fully used.
    """
    a = [tuple(int(v) for v in routes[f_src, i]) for i in range(int(counts[f_src]))]
    b = [tuple(int(v) for v in routes[f_dst, i]) for i in range(int(counts[f_dst]))]
    if not a:
        return [b[::-1]] if b else []
    if not b:
        return [a]
    tail, head = a[-1], b[-1]
    contiguous = (
        tail[0] == head[0]
        and abs(tail[1] - head[1]) <= 1
        and abs(tail[2] - head[2]) <= 1
    )
    return [a + b[::-1]] if contiguous else [a, b[::-1]]


def export_board(
    world,
    board_index: int,
    path: str | Path,
    net_prefix: str = "net",
    routed_only: bool = True,
) -> ExportStats:
    """Write board `board_index` of a batched world to `path`.

    `routed_only` emits copper only for nets that actually connected. A net that
    ran out of budget has stubs leading from both its pads into empty space, and
    exporting those produces a `track_dangling` violation per segment -- a true
    statement about the file that says nothing about whether the lattice model
    is legal. Unrouted nets still get their pads and their net entry, so KiCad
    reports them as `unconnected_items`, which is the honest way for an
    incomplete board to be incomplete. (NeuroRoute exported them and drew 478
    `track_dangling` before this rule.)
    """
    spec: BoardSpec = world.spec
    L = spec.num_layers
    names = copper_layer_names(L)
    rules = spec.rules
    pitch = spec.pitch_mm

    def mm(cell: int) -> float:
        return round(float(cell) * pitch + 0.5 * pitch, 4)

    b = board_index
    valid = world.net_valid[b].cpu()
    net_ids = [i for i in range(valid.shape[0]) if bool(valid[i])]
    # KiCad reserves net 0 for "no net", so board nets start at 1.
    net_number = {n: i + 1 for i, n in enumerate(net_ids)}

    out: list[str] = []
    out.append('(kicad_pcb (version 20221018) (generator "mzr")')
    out.append("  (general (thickness 1.6))")
    out.append('  (paper "A4")')
    out.append("  (layers")
    for i, name in enumerate(names):
        out.append(f'    ({_layer_ordinal(i, L)} "{name}" signal)')
    out.append('    (44 "Edge.Cuts" user)')
    out.append("  )")
    out.append("  (setup")
    out.append("    (pad_to_mask_clearance 0)")
    out.append("    (stackup)")
    out.append("  )")

    out.append('  (net 0 "")')
    for n in net_ids:
        out.append(f'  (net {net_number[n]} "{net_prefix}_{n}")')

    # Board outline. Without an Edge.Cuts loop KiCad reports the board as having
    # no outline and several DRC rules silently do not run -- which would turn
    # this gate into a rubber stamp.
    w_mm = round(spec.width_cells * pitch, 4)
    h_mm = round(spec.height_cells * pitch, 4)
    corners = [(0.0, 0.0), (w_mm, 0.0), (w_mm, h_mm), (0.0, h_mm), (0.0, 0.0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        out.append(
            f"  (gr_line (start {x1} {y1}) (end {x2} {y2}) "
            f'(stroke (width 0.05) (type solid)) (layer "Edge.Cuts"))'
        )

    # --- pads, as one single-pad footprint each ---------------------------
    pad_count = 0
    pads = world.net_pad[b].cpu()
    kind = world.net_kind[b].cpu()
    for n in net_ids:
        legs = 2 if int(kind[n]) == KIND_DIFF_PAIR else 1
        for leg in range(legs):
            for tag, end in (("S", END_SRC), ("D", END_DST)):
                pt = pads[n, leg, end]
                ly, py, px = int(pt[0]), int(pt[1]), int(pt[2])
                layer = names[min(ly, L - 1)]
                # The same number the lattice reserved cells for. When these
                # two drifted apart, KiCad reported 0.100 mm pad-to-track
                # against a 0.200 mm rule -- see DesignRules.pad_size.
                size = round(rules.pad_size, 4)
                out.append(
                    f'  (footprint "mzr:pad" (layer "{layer}") '
                    f"(at {mm(px)} {mm(py)}) "
                    f'(attr smd) (fp_text reference "P{n}{tag}{leg}" (at 0 0) (layer "F.SilkS") hide '
                    f"(effects (font (size 0.5 0.5) (thickness 0.1)))) "
                    f'(pad "1" smd rect (at 0 0) (size {size} {size}) (layers "{layer}") '
                    f'(net {net_number[n]} "{net_prefix}_{n}")))'
                )
                pad_count += 1

    # --- tracks and vias, walked off the stored polylines -----------------
    # The polyline is the authority, not the occupancy grid: reconstructing a
    # route from occupancy is ambiguous the moment two nets touch.
    routes = world.route_v[b].cpu()
    counts = world.route_n[b].cpu()
    status = world.net_status[b].cpu()
    leg_done = world.leg_done[b].cpu()
    track_count = via_count = stub_legs = 0
    widths = list(rules.track_widths)
    via_d, via_dr = rules.via_diameters[0], rules.via_drills[0]

    for n in net_ids:
        if routed_only and int(status[n]) != STATUS_DONE:
            continue
        wclass = int(world.net_width[b, n])
        width = widths[min(wclass, len(widths) - 1)]
        legs = 2 if int(kind[n]) == KIND_DIFF_PAIR else 1
        for leg in range(legs):
            if not bool(leg_done[n, leg]):
                continue
            f_src = (n * 2 + leg) * NUM_ENDS + END_SRC
            f_dst = (n * 2 + leg) * NUM_ENDS + END_DST
            runs = _leg_runs(routes, counts, f_src, f_dst)
            if len(runs) > 1:
                stub_legs += 1
            # A through via spans every layer, so two vias at one cell on the
            # same net are the same hole drilled twice -- KiCad calls that
            # `holes_co_located`. Emit each distinct position once.
            drilled: set[tuple[int, int]] = set()
            for pts in runs:
                for i in range(len(pts) - 1):
                    l0, y0, x0 = pts[i]
                    l1, y1, x1 = pts[i + 1]
                    if l0 != l1:
                        if (y0, x0) not in drilled:
                            drilled.add((y0, x0))
                            out.append(
                                f"  (via (at {mm(x0)} {mm(y0)}) (size {via_d}) "
                                f"(drill {via_dr}) "
                                f'(layers "{names[0]}" "{names[-1]}") '
                                f"(net {net_number[n]}))"
                            )
                            via_count += 1
                        continue
                    if (y0, x0) == (y1, x1):
                        continue
                    out.append(
                        f"  (segment (start {mm(x0)} {mm(y0)}) (end {mm(x1)} {mm(y1)}) "
                        f'(width {width}) (layer "{names[min(l0, L - 1)]}") '
                        f"(net {net_number[n]}))"
                    )
                    track_count += 1

    out.append(")")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return ExportStats(
        tracks=track_count,
        vias=via_count,
        pads=pad_count,
        nets=len(net_ids),
        stub_legs=stub_legs,
    )
