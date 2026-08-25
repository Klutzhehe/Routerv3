"""Export a routed lattice to a real `.kicad_pcb`.

Written as plain s-expression text, **not** through `pcbnew`. That is a
deliberate constraint, not laziness: `ROADMAP.md`'s hard constraint 1 says the
compiled `pcbworld_pns_bridge` and the system `pcbnew` module can never share a
process (both define KiCad's one-instance-per-process globals; loading both
crashes with no Python-catchable exception). A text writer has no such
constraint, needs no KiCad install to run, and needs no subprocess dance --
which means the export path can be unit-tested anywhere, including here on a
laptop with no KiCad at all.

Verification happens the other way round: `kicad-cli pcb drc` reads the file we
wrote and tells us whether it is both *valid* and *legal*. That is a stronger
check than round-tripping through `pcbnew` would be, because it is KiCad's own
`DRC_ENGINE` judging the copper, not our own geometry judging itself.

**Status: [UNVERIFIED].** Nothing here has been run against a real KiCad. The
file-format details most likely to be wrong are called out inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from neuroroute.world.engine import BatchedRouterWorld
from neuroroute.world.spec import BoardSpec


def copper_layer_names(num_layers: int) -> list[str]:
    """Lattice layer index -> KiCad layer name.

    Index 0 is the top copper and index `num_layers - 1` is the bottom, with
    inner layers in between, which is the same ordering the lattice uses.
    """
    if num_layers == 1:
        return ["F.Cu"]
    return ["F.Cu"] + [f"In{i}.Cu" for i in range(1, num_layers - 1)] + ["B.Cu"]


def _layer_ordinal(index: int, num_layers: int) -> int:
    """Ordinal for the `(layers ...)` stack-up block.

    **Most likely thing in this file to be wrong.** KiCad's *runtime*
    `PCB_LAYER_ID` numbering changed in 9.0 -- this repo measured `F_Cu = 0`
    but `B_Cu = 2` against a local 9.0 install (docs/RL_PLAN.md's Gate A layer
    sweep), which is not the historical `B.Cu = 31`. The *file format* has
    historically used F.Cu = 0, In1..InN = 1..N, B.Cu = 31, and KiCad still
    reads files written that way.

    Emitting the historical numbering is the safer bet, because a file KiCad
    can load gets normalised on load, whereas a file it cannot parse fails
    outright. `kicad-cli pcb drc` refusing to open the file is the signal that
    this guess was wrong -- and that is precisely the first thing the Colab
    validation run checks.
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


def export_board(
    world: BatchedRouterWorld,
    board_index: int,
    path: str | Path,
    net_prefix: str = "net",
    routed_only: bool = True,
) -> ExportStats:
    """Write board `board_index` of a batched world to `path`.

    `routed_only` emits copper for nets that actually completed. A net that
    ran out of budget has a *stub* -- copper leading from its pad into empty
    space -- and exporting it produces a `track_dangling` violation per
    segment, which is a true statement about the file but says nothing about
    whether the lattice model is legal. Unrouted nets still get their pads and
    their net entry, so KiCad reports them as `unconnected_items`, which is the
    honest way for an incomplete board to be incomplete.
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
    n_nets = int(valid.sum())
    net_ids = [i for i in range(valid.shape[0]) if bool(valid[i])]
    # KiCad reserves net 0 for "no net", so board nets start at 1.
    net_number = {n: i + 1 for i, n in enumerate(net_ids)}

    out: list[str] = []
    out.append('(kicad_pcb (version 20221018) (generator "neuroroute")')
    out.append("  (general (thickness 1.6))")
    out.append('  (paper "A4")')
    out.append("  (layers")
    for i, name in enumerate(names):
        out.append(f'    ({_layer_ordinal(i, L)} "{name}" signal)')
    out.append('    (44 "Edge.Cuts" user)')
    out.append("  )")
    out.append("  (setup")
    out.append(f"    (pad_to_mask_clearance 0)")
    out.append("    (stackup)")
    out.append("  )")

    out.append('  (net 0 "")')
    for n in net_ids:
        out.append(f'  (net {net_number[n]} "{net_prefix}_{n}")')

    # Board outline. Without an Edge.Cuts loop KiCad reports the board as
    # having no outline and several DRC rules silently do not run.
    w_mm = round(spec.width_cells * pitch, 4)
    h_mm = round(spec.height_cells * pitch, 4)
    corners = [(0.0, 0.0), (w_mm, 0.0), (w_mm, h_mm), (0.0, h_mm), (0.0, 0.0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        out.append(
            f'  (gr_line (start {x1} {y1}) (end {x2} {y2}) '
            f'(stroke (width 0.05) (type solid)) (layer "Edge.Cuts"))'
        )

    # --- pads, as one single-pad footprint each ---------------------------
    pad_count = 0
    src = world.net_src[b].cpu()
    dst = world.net_dst[b].cpu()
    kind = world.net_kind[b].cpu()
    for n in net_ids:
        legs = 2 if int(kind[n]) == 1 else 1
        for leg in range(legs):
            for tag, pt in (("S", src[n, leg]), ("D", dst[n, leg])):
                ly, py, px = int(pt[0]), int(pt[1]), int(pt[2])
                layer = names[min(ly, L - 1)]
                # Same number the lattice reserved cells for -- see
                # DesignRules.pad_size.
                size = round(rules.pad_size, 4)
                out.append(
                    f'  (footprint "neuroroute:pad" (layer "{layer}") '
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
    track_count = via_count = 0
    widths = list(rules.track_widths)
    via_d, via_dr = rules.via_diameters[0], rules.via_drills[0]

    for n in net_ids:
        if routed_only and int(status[n]) != 2:  # STATUS_DONE
            continue
        wclass = int(world.net_width[b, n])
        width = widths[min(wclass, len(widths) - 1)]
        legs = 2 if int(kind[n]) == 1 else 1
        for leg in range(legs):
            k = int(counts[n, leg])
            pts = routes[n, leg, :k]
            for i in range(k - 1):
                l0, y0, x0 = (int(v) for v in pts[i])
                l1, y1, x1 = (int(v) for v in pts[i + 1])
                if l0 != l1:
                    out.append(
                        f"  (via (at {mm(x0)} {mm(y0)}) (size {via_d}) (drill {via_dr}) "
                        f'(layers "{names[0]}" "{names[-1]}") '
                        f'(net {net_number[n]}))'
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
    return ExportStats(tracks=track_count, vias=via_count, pads=pad_count, nets=n_nets)
