"""Generate a minimal 2-pad, 1-net .kicad_pcb test fixture.

Deliberately uses the *system* pcbnew module (the standard SWIG bindings
that ship with any KiCad install, e.g. via `apt install kicad`) rather than
our custom-built pcbworld_pns_bridge -- this keeps the fixture generator
independent of the code we're trying to test, so a passing round-trip test
(generate -> route via our bridge -> reload here and check) isn't just
checking our own code against itself.

Unverified like the rest of this repo: written from KiCad's documented
scripting API, not yet run. Likely candidates for a first-run fix: exact
PAD_ATTRIB/PAD_SHAPE enum names can drift between KiCad versions.
"""

import argparse

import pcbnew


def make_toy_board(path: str) -> None:
    board = pcbnew.BOARD()

    net = pcbnew.NETINFO_ITEM(board, "toy_net")
    board.Add(net)

    positions = [pcbnew.VECTOR2I(pcbnew.FromMM(0), pcbnew.FromMM(0)),
                 pcbnew.VECTOR2I(pcbnew.FromMM(20), pcbnew.FromMM(0))]

    for i, pos in enumerate(positions):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(f"J{i + 1}")
        fp.SetPosition(pos)
        board.Add(fp)

        # NOTE: KiCad 9 moved shape/size onto a per-layer PADSTACK
        # (pcbnew/pad.h: SetShape(layer, shape), SetSize(layer, size) etc).
        # The calls below use the pre-padstack single-arg convenience forms
        # that SWIG has historically kept for backwards compat -- unverified
        # against KiCad 9's actual SWIG interface, likely first thing to
        # fix if this script errors.
        pad = pcbnew.PAD(fp)
        pad.SetNumber("1")
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
        pad.SetSize(pcbnew.VECTOR2I(pcbnew.FromMM(1.5), pcbnew.FromMM(1.5)))
        pad.SetLayerSet(pcbnew.LSET([pcbnew.F_Cu]))
        pad.SetPosition(pos)
        pad.SetNet(net)
        fp.Add(pad)

    outline_start = pcbnew.VECTOR2I(pcbnew.FromMM(-5), pcbnew.FromMM(-5))
    outline_end = pcbnew.VECTOR2I(pcbnew.FromMM(25), pcbnew.FromMM(5))
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
    args = parser.parse_args()
    make_toy_board(args.path)
    print(f"wrote {args.path}")
