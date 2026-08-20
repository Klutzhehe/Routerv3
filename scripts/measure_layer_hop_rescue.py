"""Does a via/layer change rescue the nets that same-layer waypoint-following
cannot route?

measure_waypoint_fidelity.py's own data (three Colab runs) found that every
same-layer-failing net failed *identically* across all 6 attempted paths --
straight line, a 5-point polyline, and 4 perpendicular detours in both
directions at two offsets. Failing the same way regardless of approach
direction points at contention concentrated at/near the target pad itself
(most likely an earlier-routed net's copper sitting close to it, since nets
are routed in order and later nets see more clutter), not an obstacle
blocking one particular route -- which a same-layer detour can never route
around, because every attempt still ends by pushing to the exact same
(x, y). A via to the other copper layer can, in principle: docs/
AI_ARCHITECTURE.md's field already carries a plane per copper layer for
exactly this reason (num_field_planes = num_copper_layers + 1).

Two phases against the same board (density accumulates realistically,
same as measure_waypoint_fidelity.py):
  1. Route every net same-layer via _route_one_net() (imported directly --
     no reason to re-derive already-verified plumbing).
  2. For whichever nets failed phase 1, try ONE via/layer-hop: push to the
     straight-line midpoint, switch to the back layer, push to the target,
     fix(). Two insertion methods are tried per net (see below) since it
     isn't certain which is right.

UNVERIFIED API SURFACE -- READ BEFORE INTERPRETING A NULL RESULT: unlike
push()/fix()/query_hover_items() (Colab-verified repeatedly, including by
measure_waypoint_fidelity.py's own three runs), switch_layer() and
toggle_via_placement() have never been exercised against the real bridge
anywhere in this repo. Their exact push()-interaction semantics are
inferred from their names and C++ signatures, not confirmed. The two
possible outcomes here are therefore NOT symmetrically trustworthy:
  - Rescues some nets -> strong positive evidence. An API call driven
    incorrectly is very unlikely to accidentally produce a real routed
    connection with 0.0000mm endpoint deviation.
  - Rescues nothing -> AMBIGUOUS. Could mean vias genuinely don't help
    here (the congestion isn't actually solvable by a layer change), or it
    could mean this script is driving switch_layer()/toggle_via_placement()
    wrong. Do not read a null result as "vias don't work" -- read it as
    "inconclusive, needs someone who knows PNS::ROUTER::SwitchLayer's real
    contract to look at this."

Bridge-only, like every other script here that touches pcbworld_pns_bridge:
never import pcbnew (the system module) in this process (see
docs/performance.md).

Usage (after notebooks/00_setup.ipynb has built the bridge):
    python3 scripts/print_back_layer_id.py                 # run in a
        # SEPARATE process, system pcbnew -- prints an int, e.g. 31
    python3 pcbworld/data/generate_board.py board.kicad_pcb --num-nets 24 --seed 0
    python3 scripts/measure_layer_hop_rescue.py board.kicad_pcb --back-layer <N>
"""

from __future__ import annotations

import sys
from pathlib import Path

# Sibling import -- reuse measure_waypoint_fidelity.py's already-verified
# bridge-loading, candidate-picking, and per-net routing logic rather than
# re-deriving it. Needs this script's own directory on sys.path since
# scripts/ isn't a package (no __init__.py), matching how the rest of this
# repo's scripts are run standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

from measure_waypoint_fidelity import (  # noqa: E402
    SNAP_RADIUS_NM,
    TRACK_WIDTH_NM,
    _load_bridge,
    _pick_pad_candidate,
    _route_one_net,
)

FRONT_LAYER = 0  # Colab-verified elsewhere (simple_route_env.py, pcb_route_env.py)


def _midpoint(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] + b[0]) // 2, (a[1] + b[1]) // 2


def _try_layer_hop(
    bridge,
    start_xy: tuple[int, int],
    target_xy: tuple[int, int],
    start_id: int,
    target_id: int,
    back_layer: int,
    method: str,
) -> tuple[bool, str]:
    """One rescue attempt: push to the midpoint, change layer via `method`,
    push to the target, fix(). Returns (succeeded, reason)."""
    if not bridge.start_route(start_xy[0], start_xy[1], start_id, 0):
        return False, "start_route failed"

    mid = _midpoint(start_xy, target_xy)
    if not bridge.push(mid[0], mid[1], -1):
        bridge.stop_routing()
        return False, "push(midpoint) rejected"

    if method == "switch_layer":
        if not bridge.switch_layer(back_layer):
            bridge.stop_routing()
            return False, "switch_layer() rejected"
    elif method == "toggle_via_placement":
        bridge.toggle_via_placement()  # no return value to check (void in bindings.cpp)
    else:
        raise ValueError(f"unknown method {method!r}")

    if not bridge.push(target_xy[0], target_xy[1], -1):
        bridge.stop_routing()
        return False, f"push(target) after {method} rejected"

    if not bridge.fix(target_xy[0], target_xy[1], target_id, True, True):
        bridge.stop_routing()
        return False, "fix() rejected"

    bridge.commit_routing()
    return True, "ok"


def run(board_path: str, num_nets: int, back_layer: int, bridge_dir: str | None) -> dict[str, str]:
    bridge_module = _load_bridge(bridge_dir)
    bridge = bridge_module.PNSBridge()

    assert bridge.load_board(board_path), f"load_board failed: {board_path}"
    bridge.set_mode(bridge_module.MODE_ROUTE_SINGLE)
    bridge.set_collision_mode(bridge_module.RM_MARK_OBSTACLES)
    bridge.set_track_width(TRACK_WIDTH_NM)

    pads = bridge.net_pads()
    available = sorted(
        {p.net for p in pads if p.net and p.net.startswith("net_")},
        key=lambda name: int(name.split("_")[1]),
    )
    assert available, "no plain 'net_*' nets found"
    net_names = available[:num_nets]

    print(f"board: {board_path}  back_layer: {back_layer}")
    print(f"Phase 1: same-layer baseline over {len(net_names)} nets...")
    warnings: list[str] = []
    baseline = [_route_one_net(bridge, bridge_module, pads, net, warnings) for net in net_names]
    failed = [r.net for r in baseline if not r.reached_target]
    print(
        f"  {len(net_names) - len(failed)}/{len(net_names)} succeeded same-layer, "
        f"{len(failed)} failed: {failed}\n"
    )

    if not failed:
        print("Nothing to rescue -- every net already succeeded same-layer.")
        return {}

    print(f"Phase 2: attempting a via/layer-hop rescue on the {len(failed)} failures...")
    rescued_by: dict[str, str] = {}
    for net in failed:
        net_pads = [p for p in pads if p.net == net]
        start_pad, target_pad = net_pads
        start_xy, target_xy = (start_pad.x, start_pad.y), (target_pad.x, target_pad.y)

        start_c = bridge.query_hover_items(*start_xy, layer=FRONT_LAYER, slop_radius=SNAP_RADIUS_NM)
        target_c = bridge.query_hover_items(*target_xy, layer=FRONT_LAYER, slop_radius=SNAP_RADIUS_NM)
        start_id = _pick_pad_candidate(start_c, f"{net}/start", warnings).id
        target_id = _pick_pad_candidate(target_c, f"{net}/target", warnings).id

        for method in ("switch_layer", "toggle_via_placement"):
            ok, why = _try_layer_hop(
                bridge, start_xy, target_xy, start_id, target_id, back_layer, method
            )
            print(f"  {net:<10} {method:<20} {'RESCUED' if ok else 'no: ' + why}")
            if ok:
                rescued_by[net] = method
                break

    print(f"\n{'=' * 70}")
    print(f"RESULT: {len(rescued_by)}/{len(failed)} previously-failing nets rescued by a layer hop")
    for net, method in rescued_by.items():
        print(f"  {net}: rescued via {method}")
    if not rescued_by:
        print(
            "  Inconclusive by design, not a negative result -- see this script's module\n"
            "  docstring on why a null result here does NOT confirm vias don't help:\n"
            "  switch_layer()/toggle_via_placement() are unverified API surface in this\n"
            "  repo, unlike push()/fix()/query_hover_items()."
        )
    if warnings:
        print(f"\n({len(warnings)} candidate-resolution warnings during phase 2 -- see "
              f"measure_waypoint_fidelity.py's _pick_pad_candidate for what these mean)")
    print(f"{'=' * 70}")

    return rescued_by


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("board_path", help=".kicad_pcb generated by generate_board.py")
    parser.add_argument("--num-nets", type=int, default=24)
    parser.add_argument(
        "--back-layer",
        type=int,
        required=True,
        help="numeric PCB_LAYER_ID for the back copper layer -- get this from "
        "scripts/print_back_layer_id.py (system pcbnew, separate process). Do not guess it.",
    )
    parser.add_argument("--bridge-dir", default=None)
    args = parser.parse_args()
    run(args.board_path, args.num_nets, args.back_layer, args.bridge_dir)
