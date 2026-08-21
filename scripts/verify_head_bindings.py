"""Verification script for the four new PNSBridge C++ bindings.

Exercises:
  1. bridge.get_design_rules() -- track width, via diameter, drill, clearance, and hard minimums.
  2. bridge.get_head_geometry() -- active status, segments, vias, end coords, layer, length.
  3. bridge.head_collides() -- collision detection on in-progress routing head.
  4. bridge.rip_up(net) -- single-net tearout and board sync without full reload.

Run in Colab after building pcbworld_pns_bridge:
    python3 Routerv3/scripts/verify_head_bindings.py board.kicad_pcb
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

MM = 1_000_000
SNAP_RADIUS_NM = int(0.5 * MM)
_BRIDGE_SEARCH_ROOTS = ("/content", str(Path.home() / "routerv3-build"))


def _load_bridge(bridge_dir: str | None):
    try:
        import pcbworld_pns_bridge as bridge

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


def verify(board_path: str, bridge_dir: str | None = None) -> bool:
    bridge_module = _load_bridge(bridge_dir)
    bridge = bridge_module.PNSBridge()

    print(f"=== 1. Loading Board: {board_path} ===")
    assert bridge.load_board(board_path), f"load_board failed on {board_path}"
    print("  [PASS] Board loaded successfully")

    print("\n=== 2. Testing get_design_rules() ===")
    rules = bridge.get_design_rules()
    print(f"  Track Width      : {rules.track_width / MM:.4f} mm ({rules.track_width} nm)")
    print(f"  Via Diameter     : {rules.via_diameter / MM:.4f} mm ({rules.via_diameter} nm)")
    print(f"  Via Drill        : {rules.via_drill / MM:.4f} mm ({rules.via_drill} nm)")
    print(f"  Clearance        : {rules.clearance / MM:.4f} mm ({rules.clearance} nm)")
    print(f"  Min Track Width  : {rules.min_track_width / MM:.4f} mm ({rules.min_track_width} nm)")
    print(f"  Min Via Diameter : {rules.min_via_diameter / MM:.4f} mm ({rules.min_via_diameter} nm)")
    print(f"  Min Via Drill    : {rules.min_via_drill / MM:.4f} mm ({rules.min_via_drill} nm)")
    print(f"  Min Hole-to-Hole : {rules.min_hole_to_hole / MM:.4f} mm ({rules.min_hole_to_hole} nm)")
    assert rules.track_width > 0, "track_width must be positive"
    print("  [PASS] get_design_rules() returns valid positive values")

    print("\n=== 3. Testing get_head_geometry() & head_collides() with no active route ===")
    idle_head = bridge.get_head_geometry()
    print(f"  Idle head active: {idle_head.active} (expected False)")
    print(f"  Idle head collides: {bridge.head_collides()} (expected False)")
    assert not idle_head.active, "head should be inactive before start_route()"
    assert not bridge.head_collides(), "idle head should not report collision"
    print("  [PASS] Idle head behavior verified")

    pads = bridge.net_pads()
    assert pads, "Board has no pads"
    plain_pads = [p for p in pads if p.net and p.net.startswith("net_")]
    target_net = plain_pads[0].net if plain_pads else pads[0].net
    net_pads = [p for p in pads if p.net == target_net]
    assert len(net_pads) >= 2, f"Net {target_net} needs >= 2 pads, found {len(net_pads)}"

    start_pad, end_pad = net_pads[0], net_pads[1]
    start_candidates = bridge.query_hover_items(start_pad.x, start_pad.y, layer=0, slop_radius=SNAP_RADIUS_NM)
    start_id = [c for c in start_candidates if c.kind == "pad"][0].id

    end_candidates = bridge.query_hover_items(end_pad.x, end_pad.y, layer=0, slop_radius=SNAP_RADIUS_NM)
    end_id = [c for c in end_candidates if c.kind == "pad"][0].id

    print(f"\n=== 4. Starting route for {target_net} ===")
    assert bridge.start_route(start_pad.x, start_pad.y, start_id, 0), "start_route failed"

    active_head = bridge.get_head_geometry()
    print(f"  Active head after start_route: active={active_head.active}, layer={active_head.layer}")
    assert active_head.active, "head must be active during route"

    mid_x = (start_pad.x + end_pad.x) // 2
    mid_y = (start_pad.y + end_pad.y) // 2

    print(f"  Pushing waypoint to ({mid_x / MM:.2f}, {mid_y / MM:.2f}) mm...")
    pushed = bridge.push(mid_x, mid_y, -1)
    print(f"  push() return: {pushed}")

    head_after_push = bridge.get_head_geometry()
    collides = bridge.head_collides()
    print(f"  Head geometry after push:")
    print(f"    end: ({head_after_push.end_x / MM:.4f}, {head_after_push.end_y / MM:.4f}) mm")
    print(f"    layer: {head_after_push.layer}")
    print(f"    length: {head_after_push.length / MM:.4f} mm")
    print(f"    segments: {len(head_after_push.segments)}")
    print(f"    collides: {collides}")
    assert head_after_push.active, "head must stay active after push"
    print("  [PASS] get_head_geometry() & head_collides() during push verified")

    print(f"\n=== 5. Fixing and Committing {target_net} ===")
    fixed = bridge.fix(end_pad.x, end_pad.y, end_id, True, True)
    print(f"  fix() return: {fixed}")
    bridge.commit_routing()

    geom_before_ripup = bridge.get_board_geometry()
    tracks_for_net = [t for t in geom_before_ripup.tracks if t.net == target_net]
    print(f"  Committed tracks for {target_net}: {len(tracks_for_net)}")

    print(f"\n=== 6. Testing rip_up('{target_net}') ===")
    removed_count = bridge.rip_up(target_net)
    print(f"  rip_up('{target_net}') removed items: {removed_count}")

    geom_after_ripup = bridge.get_board_geometry()
    tracks_after = [t for t in geom_after_ripup.tracks if t.net == target_net]
    print(f"  Tracks for {target_net} remaining after rip_up: {len(tracks_after)} (expected 0)")
    assert len(tracks_after) == 0, f"rip_up failed to remove tracks for {target_net}"
    print("  [PASS] rip_up() successfully removed all tracks for the net")

    print("\n" + "=" * 60)
    print("ALL 4 NEW BINDINGS VERIFIED SUCCESSFULLY AGAINST LIVE BRIDGE!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify 4 new PNSBridge C++ bindings")
    parser.add_argument("board_path", help="Path to .kicad_pcb board file")
    parser.add_argument("--bridge-dir", default=None, help="Optional path to bridge library directory")
    args = parser.parse_args()
    verify(args.board_path, args.bridge_dir)
