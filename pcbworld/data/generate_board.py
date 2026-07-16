"""Generate a small D2-like synthetic board for routing training data.

Same constraints as scripts/make_toy_board.py, which this generalizes:
uses the *system* pcbnew module (never pcbworld_pns_bridge -- see
docs/performance.md on why those two must never share a process), so it
must always run as its own process, never inline in a bridge-loaded
interpreter. Run standalone or via subprocess, same as make_toy_board.py's
own usage in notebooks/00_setup.ipynb.

Deliberately simple relative to the PCBWorld paper's full D2 spec: each net
here is a single pad-to-pad two-terminal net (matching make_toy_board.py's
existing pattern) rather than the paper's arbitrary-fanout nets. Multi-pad
nets are a plausible next step once the two-terminal case is routing
reliably through the Gym env (pcbworld/env/) -- not attempted here to keep
this generator's pcbnew API usage close to the already-verified toy board
script.
"""

import argparse
import random

import pcbnew


def generate_synthetic_board(
    path: str,
    num_nets: int = 5,
    seed: int | None = None,
    board_size_mm: tuple[float, float] = (50.0, 50.0),
    pad_size_mm: float = 1.0,
    min_spacing_mm: float = 3.0,
    margin_mm: float = 5.0,
) -> None:
    """Writes a gridless 2-layer board with `num_nets` two-pad nets.

    "Gridless" here means pad positions are arbitrary floats (not snapped
    to any placement grid), matching the paper's D2-style boards -- unlike
    make_toy_board.py's two hand-picked positions, these are randomly
    sampled subject to a minimum pairwise spacing so pads don't overlap.
    """
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

    for net_idx in range(num_nets):
        net_name = f"net_{net_idx}"
        net = pcbnew.NETINFO_ITEM(board, net_name)
        board.Add(net)

        for _pad_in_net in range(2):
            x_mm, y_mm = sample_position()
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
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
            pad.SetSize(pcbnew.VECTOR2I(pcbnew.FromMM(pad_size_mm), pcbnew.FromMM(pad_size_mm)))
            layer_vec = pcbnew.base_seqVect()
            layer_vec.append(pcbnew.F_Cu)
            pad.SetLayerSet(pcbnew.LSET(layer_vec))
            pad.SetPosition(pos)
            pad.SetNet(net)
            fp.Add(pad)

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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width-mm", type=float, default=50.0)
    parser.add_argument("--height-mm", type=float, default=50.0)
    args = parser.parse_args()

    generate_synthetic_board(
        args.path,
        num_nets=args.num_nets,
        seed=args.seed,
        board_size_mm=(args.width_mm, args.height_mm),
    )
    print(f"wrote {args.path} ({args.num_nets} nets)")
