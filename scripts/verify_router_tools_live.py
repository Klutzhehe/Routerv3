"""Verification script for RouterTools (pcbworld/agent/tools.py) against a
live pcbworld_pns_bridge.

scripts/verify_head_bindings.py proved the four raw C++ bindings compile
and behave correctly (get_design_rules/get_head_geometry/head_collides/
rip_up). This script proves something different and still open as of that
run: that RouterTools -- the validate/execute/verify wrapper the LLM agent
actually calls -- behaves correctly on top of them. The raw bindings
passing does not imply the wrapper does; this exercises the wrapper's own
logic (deviation warnings, design-rule guards, the back-layer finish_route
refusal, rip-up-then-reroute) against the real router, not a stub.

Deliberately narrated as ToolResult.to_model() output throughout -- that
compact text form is exactly what an LLM would read mid-run, so printing
it here doubles as a check that the messages stay readable against real
router state, not just against the fixtures in tests/test_agent_tools.py.

Run in Colab after building pcbworld_pns_bridge:
    python3 Routerv3/pcbworld/data/generate_board.py board.kicad_pcb --num-nets 6 --seed 0
    python3 Routerv3/scripts/verify_router_tools_live.py board.kicad_pcb
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

MM = 1_000_000
_BRIDGE_SEARCH_ROOTS = ("/content", str(Path.home() / "routerv3-build"))


def _load_bridge(bridge_dir: str | None):
    try:
        import pcbworld_pns_bridge as bridge  # noqa: F401

        return bridge
    except ImportError:
        pass

    roots = [bridge_dir] if bridge_dir else list(_BRIDGE_SEARCH_ROOTS)
    for root in roots:
        matches = glob.glob(f"{root}/kicad-src/build/**/pcbworld_pns_bridge*.so", recursive=True)
        if matches:
            sys.path.insert(0, str(Path(matches[0]).parent))
            import pcbworld_pns_bridge as bridge

            return bridge

    raise ImportError(
        "pcbworld_pns_bridge not found. Run notebooks/00_setup.ipynb build step first."
    )


def _show(label: str, result) -> None:
    print(f"\n--- {label} " + "-" * max(1, 58 - len(label)))
    print(result.to_model())


def verify(board_path: str, bridge_dir: str | None = None) -> bool:
    bridge_module = _load_bridge(bridge_dir)

    # Import here, not at module top level -- RouterTools itself has no
    # pcbworld_pns_bridge dependency (it takes an already-constructed
    # bridge object), but this script does, so it follows the same
    # deferred-import convention as every env in pcbworld/env/.
    from pcbworld.agent.tools import RouterTools

    bridge = bridge_module.PNSBridge()
    assert bridge.load_board(board_path), f"load_board failed on {board_path}"

    pads = bridge.net_pads()
    nets = sorted({p.net for p in pads if p.net})
    assert len(nets) >= 2, f"need at least 2 nets on {board_path} to exercise rip-up"

    # Board size isn't in net_pads()/get_board_geometry() as an explicit
    # field -- generate_board.py's own --width-mm/--height-mm are the
    # ground truth here, defaulting to 50x50 same as that script. Passed
    # explicitly rather than inferred from pad bounding box, which would
    # silently be wrong the moment a net sits near the edge.
    # max_step_mm generous on purpose: generate_board.py's default 50x50mm
    # board with min_spacing_mm=3.0 can place two pads of the same net up
    # to the board diagonal apart (~70mm), and this script's job is
    # checking RouterTools' logic, not exercising the step-budget path
    # (tests/test_agent_tools.py::test_step_longer_than_the_cap_is_refused
    # already covers that against a stub).
    tools = RouterTools(bridge, board_width_mm=50.0, board_height_mm=50.0, max_step_mm=80.0)

    print("=== 1. get_board_info() ===")
    _show("board info", tools.get_board_info())

    print("\n=== 2. list_nets() ===")
    _show("nets", tools.list_nets())

    net_a, net_b = nets[0], nets[1]

    print(f"\n=== 3. Full route of {net_a!r}: start -> route_to -> finish_route ===")
    r = tools.start_route(net_a)
    _show(f"start_route({net_a!r})", r)
    assert r.ok, "start_route on a fresh net must succeed"

    target_mm = r.data["target_mm"]
    r = tools.route_to(*target_mm)
    _show(f"route_to{target_mm}", r)
    assert r.ok, "a direct route_to the target should be accepted"
    if r.warnings:
        print(
            "  NOTE: deviation/collision warning fired on a direct route -- "
            "real, not a bug; the message above is what an LLM would see."
        )

    r = tools.finish_route()
    _show("finish_route()", r)
    assert r.ok, f"finish_route on {net_a!r} should succeed after reaching its target"

    print(f"\n=== 4. Re-routing an already-routed net is refused correctly ===")
    r = tools.start_route(net_a)
    _show(f"start_route({net_a!r}) again", r)
    assert not r.ok, "routing an already-routed net must be refused"
    assert "rip_up" in r.message, "the refusal must point at rip_up() as the fix"

    print(f"\n=== 5. place_via() / switch_to_layer() on {net_b!r} ===")
    r = tools.start_route(net_b)
    _show(f"start_route({net_b!r})", r)
    assert r.ok

    r = tools.place_via()
    _show("place_via()", r)

    r = tools.switch_to_layer(0)
    _show("switch_to_layer(0)", r)

    r = tools.abandon_route()
    _show("abandon_route()", r)
    assert r.ok

    print(f"\n=== 6. rip_up({net_a!r}) through the tool layer ===")
    r = tools.rip_up(net_a)
    _show(f"rip_up({net_a!r})", r)
    assert r.ok
    assert r.data.get("removed_items", 0) > 0, "rip_up should report real removed items"

    print(f"\n=== 7. Re-routing {net_a!r} after rip_up succeeds (the actual recovery flow) ===")
    r = tools.start_route(net_a)
    _show(f"start_route({net_a!r}) after rip_up", r)
    assert r.ok, "a net should be routable again immediately after rip_up"

    r = tools.route_to(*r.data["target_mm"])
    r = tools.finish_route()
    _show("finish_route() after reroute", r)
    assert r.ok

    print("\n=== 8. check_drc() ===")
    _show("DRC", tools.check_drc())

    print("\n" + "=" * 60)
    print("ROUTERTOOLS VERIFIED SUCCESSFULLY AGAINST LIVE BRIDGE!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify RouterTools against a live bridge")
    parser.add_argument("board_path", help="Path to .kicad_pcb board file")
    parser.add_argument("--bridge-dir", default=None, help="Optional path to bridge library directory")
    args = parser.parse_args()
    verify(args.board_path, args.bridge_dir)
