"""Independent check that a board file has a real track on a given net.

Deliberately a separate script/process, run via subprocess -- not a cell
importing pcbnew inline in the same Jupyter kernel that already has
pcbworld_pns_bridge loaded. Those two must never coexist in one process:
both statically link large, overlapping chunks of KiCad's own C++ code and
both define KiCad's "exactly one instance per process" globals (Kiface(),
GFootprintTable -- see kicad_headless_mocks.cpp and
include/fp_lib_table.h:294 "KIFACE scope"). Loading both into the same
process crashes it (confirmed: crashes when run right after the bridge is
loaded, works fine standalone) -- this is a real architectural constraint,
not a bug to paper over, and it applies to the future RL environment too
(never import both in the same worker process; see docs/performance.md).
"""

import argparse

import pcbnew


def verify_routed_board(path: str, net_name: str) -> None:
    board = pcbnew.LoadBoard(path)
    tracks = list(board.GetTracks())
    net_tracks = [t for t in tracks if t.GetNetname() == net_name]

    print(f"{len(tracks)} total track/via items, {len(net_tracks)} on {net_name}")
    assert net_tracks, f"no track found on {net_name} -- commit didn't reach disk"
    print(f"VERIFIED: a real track on {net_name} round-tripped to disk.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help=".kicad_pcb file to check")
    parser.add_argument("net_name", nargs="?", default="toy_net")
    args = parser.parse_args()
    verify_routed_board(args.path, args.net_name)
