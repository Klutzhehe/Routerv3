"""Generate a small D2-like synthetic board for routing training data.

Same constraints as scripts/make_toy_board.py, which this generalizes:
uses the *system* pcbnew module (never pcbworld_pns_bridge -- see
docs/performance.md on why those two must never share a process), so it
must always run as its own process, never inline in a bridge-loaded
interpreter. Run standalone or via subprocess, same as make_toy_board.py's
own usage in notebooks/00_setup.ipynb.

Deliberately simple relative to the PCBWorld paper's full D2 spec: every
net is a two-terminal net (matching make_toy_board.py's existing pattern)
rather than the paper's arbitrary-fanout nets. Multi-pad nets are a
plausible next step once the two-terminal case is routing reliably through
the Gym env (pcbworld/env/) -- not attempted here.

Three net kinds, distinguished by name convention (there's no separate
metadata channel -- callers recover grouping by parsing net names, same as
everything else that consumes NetPads()/get_board_geometry()):
  - Plain two-terminal nets:            "net_<i>"
  - Differential pairs (positive/negative legs, routed via
    PNS_BRIDGE::SetMode(MODE_ROUTE_DIFF_PAIR) -- see ROADMAP.md item 7):
                                         "diffpair_<i>_P" / "diffpair_<i>_N"
  - Length-matched groups (route via MODE_TUNE_SINGLE against the group's
    longest member, or MODE_TUNE_DIFF_PAIR_SKEW if a group's members are
    themselves diff pairs -- not currently combined with diff pairs here):
                                         "lengthgrp_<g>_<member>"

Pad attachment type is a whole-board switch (`pad_type`), not per-net: SMD
(front copper only, the default and what every existing board here used) or
THT/plated-through (spans both copper layers). The THT option exists for
scripts/diagnose_layer_switch.py, which needs to vary exactly one thing --
whether the route's start item spans the layer being switched to -- to test
why switch_layer() has never once succeeded against the real bridge. See
that script's hypothesis table.
"""

import argparse
import math
import random

import pcbnew


def _perp_unit(dx: float, dy: float) -> tuple[float, float]:
    """Unit vector perpendicular to (dx, dy); arbitrary if it's zero-length."""
    length = math.hypot(dx, dy)
    if length == 0:
        return (1.0, 0.0)
    return (-dy / length, dx / length)


def generate_synthetic_board(
    path: str,
    num_nets: int = 5,
    num_diff_pairs: int = 0,
    num_length_matched_groups: int = 0,
    length_matched_group_size: int = 2,
    seed: int | None = None,
    board_size_mm: tuple[float, float] = (50.0, 50.0),
    pad_size_mm: float = 1.0,
    diff_pair_pad_size_mm: float = 0.3,
    diff_pair_pitch_mm: float = 1.0,
    min_spacing_mm: float = 3.0,
    margin_mm: float = 5.0,
    pad_type: str = "smd",
    tht_drill_mm: float = 0.5,
) -> None:
    """Writes a gridless 2-layer board mixing plain, diff-pair, and
    length-matched-group nets (see module docstring for the naming
    convention each kind uses).

    "Gridless" here means pad positions are arbitrary floats (not snapped
    to any placement grid), matching the paper's D2-style boards -- unlike
    make_toy_board.py's two hand-picked positions, these are randomly
    sampled subject to a minimum pairwise spacing so pads don't overlap.
    Diff-pair legs are the one exception: they're deliberately placed
    closer than `min_spacing_mm` (at `diff_pair_pitch_mm`) to each other,
    since that's what makes them routable as a coupled pair -- everything
    else still respects `min_spacing_mm` against every pad placed so far,
    diff-pair legs included.

    `pad_type` is "smd" (front copper only -- the default, and what every
    board generated before this option existed used) or "tht" (plated
    through-hole, layer set spanning F_Cu *and* B_Cu, with a drill). The
    drill is `min(tht_drill_mm, size_mm * 0.6)` so a smaller pad still
    keeps an annular ring rather than becoming a hole with no copper --
    which matters because diff-pair pads default to 0.3mm, far smaller
    than tht_drill_mm's 0.5mm default. THT + diff pairs is not a
    combination anything here has verified; the option exists for plain
    nets (see the module docstring).
    """
    if pad_type not in ("smd", "tht"):
        raise ValueError(f"pad_type must be 'smd' or 'tht', got {pad_type!r}")

    rng = random.Random(seed)

    board = pcbnew.BOARD()
    board.SetCopperLayerCount(2)

    width_mm, height_mm = board_size_mm
    min_x, max_x = margin_mm, width_mm - margin_mm
    min_y, max_y = margin_mm, height_mm - margin_mm

    if min_x >= max_x or min_y >= max_y:
        raise ValueError("margin_mm too large for board_size_mm")

    placed: list[tuple[float, float]] = []

    def sample_position(max_attempts: int = 500) -> tuple[float, float]:
        for _ in range(max_attempts):
            x = rng.uniform(min_x, max_x)
            y = rng.uniform(min_y, max_y)
            if all((x - px) ** 2 + (y - py) ** 2 >= min_spacing_mm ** 2 for px, py in placed):
                placed.append((x, y))
                return x, y
        raise RuntimeError(
            f"couldn't place a pad with {min_spacing_mm}mm spacing after "
            f"{max_attempts} attempts -- fewer nets, more margin, or a "
            f"larger board"
        )

    fp_index = 0

    def add_pad(x_mm: float, y_mm: float, net, size_mm: float) -> None:
        nonlocal fp_index
        pos = pcbnew.VECTOR2I(pcbnew.FromMM(x_mm), pcbnew.FromMM(y_mm))

        fp_index += 1
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(f"J{fp_index}")
        fp.SetPosition(pos)
        board.Add(fp)

        # See make_toy_board.py for the caveats on this pad-construction
        # sequence (KiCad 9 PADSTACK move, LSET construction via
        # base_seqVect) -- identical here, not re-derived.
        pad = pcbnew.PAD(fp)
        pad.SetNumber("1")
        pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
        pad.SetSize(pcbnew.VECTOR2I(pcbnew.FromMM(size_mm), pcbnew.FromMM(size_mm)))

        # LSET is built via base_seqVect either way rather than
        # LSET.AllCuMask() -- the seqVect construction is the one already
        # proven against this KiCad version here and in make_toy_board.py,
        # and introducing a second LSET-construction API is exactly the
        # kind of unforced pcbnew-surface risk this file avoids.
        layer_vec = pcbnew.base_seqVect()
        layer_vec.append(pcbnew.F_Cu)

        if pad_type == "tht":
            pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
            layer_vec.append(pcbnew.B_Cu)
            drill_mm = min(tht_drill_mm, size_mm * 0.6)
            pad.SetDrillSize(
                pcbnew.VECTOR2I(pcbnew.FromMM(drill_mm), pcbnew.FromMM(drill_mm))
            )
        else:
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)

        pad.SetLayerSet(pcbnew.LSET(layer_vec))
        pad.SetPosition(pos)
        pad.SetNet(net)
        fp.Add(pad)

    def add_two_terminal_net(net_name: str) -> None:
        net = pcbnew.NETINFO_ITEM(board, net_name)
        board.Add(net)
        for _ in range(2):
            x_mm, y_mm = sample_position()
            add_pad(x_mm, y_mm, net, pad_size_mm)

    for net_idx in range(num_nets):
        add_two_terminal_net(f"net_{net_idx}")

    diff_pair_half_pitch = diff_pair_pitch_mm / 2.0
    for pair_idx in range(num_diff_pairs):
        # Two base positions (the pair's two ends, e.g. driver/receiver),
        # each split into a P/N leg offset perpendicular to the line
        # between the ends -- so the two legs run parallel to each other,
        # matching how a diff pair is actually routed.
        base_a = sample_position()
        base_b = sample_position()
        ox, oy = _perp_unit(base_b[0] - base_a[0], base_b[1] - base_a[1])

        pos_p_a = (base_a[0] + ox * diff_pair_half_pitch, base_a[1] + oy * diff_pair_half_pitch)
        pos_n_a = (base_a[0] - ox * diff_pair_half_pitch, base_a[1] - oy * diff_pair_half_pitch)
        pos_p_b = (base_b[0] + ox * diff_pair_half_pitch, base_b[1] + oy * diff_pair_half_pitch)
        pos_n_b = (base_b[0] - ox * diff_pair_half_pitch, base_b[1] - oy * diff_pair_half_pitch)
        # Register the actual leg positions (not just the base points) so
        # later sample_position() calls -- for other nets, other pairs,
        # length-matched groups -- keep min_spacing_mm away from them too.
        placed.extend([pos_p_a, pos_n_a, pos_p_b, pos_n_b])

        net_p = pcbnew.NETINFO_ITEM(board, f"diffpair_{pair_idx}_P")
        net_n = pcbnew.NETINFO_ITEM(board, f"diffpair_{pair_idx}_N")
        board.Add(net_p)
        board.Add(net_n)
        add_pad(*pos_p_a, net_p, diff_pair_pad_size_mm)
        add_pad(*pos_n_a, net_n, diff_pair_pad_size_mm)
        add_pad(*pos_p_b, net_p, diff_pair_pad_size_mm)
        add_pad(*pos_n_b, net_n, diff_pair_pad_size_mm)

    for group_idx in range(num_length_matched_groups):
        for member_idx in range(length_matched_group_size):
            add_two_terminal_net(f"lengthgrp_{group_idx}_{member_idx}")

    outline_start = pcbnew.VECTOR2I(pcbnew.FromMM(0), pcbnew.FromMM(0))
    outline_end = pcbnew.VECTOR2I(pcbnew.FromMM(width_mm), pcbnew.FromMM(height_mm))
    outline = pcbnew.PCB_SHAPE(board)
    outline.SetShape(pcbnew.SHAPE_T_RECT)
    outline.SetLayer(pcbnew.Edge_Cuts)
    outline.SetStart(outline_start)
    outline.SetEnd(outline_end)
    board.Add(outline)

    board.BuildListOfNets()
    board.BuildConnectivity()
    board.Save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="output .kicad_pcb path")
    parser.add_argument("--num-nets", type=int, default=5)
    parser.add_argument("--num-diff-pairs", type=int, default=0)
    parser.add_argument("--num-length-matched-groups", type=int, default=0)
    parser.add_argument("--length-matched-group-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width-mm", type=float, default=50.0)
    parser.add_argument("--height-mm", type=float, default=50.0)
    parser.add_argument(
        "--pad-type",
        choices=("smd", "tht"),
        default="smd",
        help="smd (front copper only, default) or tht (plated through-hole, "
        "spans both copper layers -- see scripts/diagnose_layer_switch.py)",
    )
    parser.add_argument("--tht-drill-mm", type=float, default=0.5)
    args = parser.parse_args()

    generate_synthetic_board(
        args.path,
        num_nets=args.num_nets,
        num_diff_pairs=args.num_diff_pairs,
        num_length_matched_groups=args.num_length_matched_groups,
        length_matched_group_size=args.length_matched_group_size,
        seed=args.seed,
        board_size_mm=(args.width_mm, args.height_mm),
        pad_type=args.pad_type,
        tht_drill_mm=args.tht_drill_mm,
    )
    print(
        f"wrote {args.path} ({args.num_nets} plain nets, "
        f"{args.num_diff_pairs} diff pairs, "
        f"{args.num_length_matched_groups} length-matched groups of "
        f"{args.length_matched_group_size}, {args.pad_type.upper()} pads)"
    )
