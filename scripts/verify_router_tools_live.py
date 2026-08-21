"""Verification script for RouterTools (pcbworld/agent/tools.py) against a
live pcbworld_pns_bridge.

scripts/verify_head_bindings.py proved the four raw C++ bindings compile
and behave correctly (get_design_rules/get_head_geometry/head_collides/
rip_up). This script proves something different: that RouterTools -- the
validate/execute/verify wrapper the LLM agent actually calls -- behaves
correctly on top of them, and specifically investigates a real finding from
this script's first Colab run rather than assuming it away.

THE FINDING (first run, one net): route_to() pushed the head to sit exactly
on the target pad's own coordinates (deviation 0.000mm, a clean accept).
head_collides() then reported True at that exact position, and
finish_route() failed with HEAD_COLLIDES. That is not obviously a bug --
it is either (a) real contention (something else's copper genuinely at
that point) or (b) an artifact of the head's own uncommitted line touching
the very pad it is about to connect to, which every other proven routing
pattern in this repo avoids by construction: pcb_route_env.py,
diff_pair_route_env.py, and this script's OWN sibling
(scripts/verify_head_bindings.py's net_4 case) all push NEAR the target and
let fix() do the final connect, never push() the head to land exactly on
the destination pad's coordinates before calling fix() there.

The first version of this script used a hard `assert r.ok` on that first
failure, which crashed the whole run before check_drc() (step 8, never
reached) or a second net could add any more evidence. That was the actual
bug -- not in RouterTools, in this script's own diagnostic design: an
unexpected-but-legitimate result destroyed the exact data needed to explain
it. Fixed here: nothing hard-asserts on the exact-vs-near comparison below;
every net contributes a data point (RESULT: line) and a genuine failure
pulls check_drc() and nearby geometry immediately, before moving on --
matching scripts/measure_waypoint_fidelity.py's own established method for
exactly this kind of question (aggregate across many nets, don't stop at
the first one).

Run in Colab after building pcbworld_pns_bridge (no rebuild needed --
same .so as before, this only changes the Python driving it):
    python3 Routerv3/pcbworld/data/generate_board.py board.kicad_pcb --num-nets 6 --seed 0
    python3 Routerv3/scripts/verify_router_tools_live.py board.kicad_pcb
"""

from __future__ import annotations

import argparse
import dataclasses
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


def _dump_nearby(tools, x_mm: float, y_mm: float, radius_mm: float = 3.0) -> None:
    """Prints every pad/track within radius_mm of (x_mm, y_mm). Called only
    when something failed -- this is the "what was actually there" data the
    first version of this script never collected."""
    geometry = tools.bridge.get_board_geometry()

    print(f"  geometry within {radius_mm}mm of ({x_mm:.3f}, {y_mm:.3f}):")
    found = False
    for p in geometry.pads:
        d = ((p.x / MM - x_mm) ** 2 + (p.y / MM - y_mm) ** 2) ** 0.5
        if d <= radius_mm:
            print(f"    pad  {p.pad_name} net={p.net!r} at ({p.x/MM:.3f}, {p.y/MM:.3f}) dist={d:.3f}mm")
            found = True
    for t in geometry.tracks:
        d1 = ((t.x1 / MM - x_mm) ** 2 + (t.y1 / MM - y_mm) ** 2) ** 0.5
        d2 = ((t.x2 / MM - x_mm) ** 2 + (t.y2 / MM - y_mm) ** 2) ** 0.5
        if min(d1, d2) <= radius_mm:
            print(f"    track net={t.net!r} ({t.x1/MM:.3f},{t.y1/MM:.3f})->({t.x2/MM:.3f},{t.y2/MM:.3f})")
            found = True
    if not found:
        print("    (nothing else within radius -- if this net's own target pad is the only "
              "thing here, the collision is most likely the head touching its OWN target pad, "
              "not real contention)")


@dataclasses.dataclass
class NetProbeResult:
    net: str
    strategy: str  # "exact" | "near"
    finished: bool
    collision_warning_during_route: bool
    finish_error_code: str | None
    obstacle_net: str | None = None      # None = no obstacle; "" = obstacle w/ no net
    obstacle_same_net: bool | None = None
    obstacle_kind: str | None = None


def _obstacle_detail(tools, net: str) -> tuple[str | None, bool | None, str | None]:
    """Pulls get_head_obstacle() directly, not just through a message string
    -- this is the decisive signal from the C++ fix that replaced the
    original bare-bool HeadCollides(): does the collision check see the
    SAME net being routed (near-certainly its own start/target pad, not a
    real obstacle) or a genuinely DIFFERENT net (real contention)? Returns
    (obstacle_net_or_None, is_same_net_or_None, kind_or_None)."""
    if not getattr(tools, "has_obstacle_detail", False):
        return None, None, None
    obstacle = tools.bridge.get_head_obstacle()
    if not obstacle.found:
        return None, None, None
    return obstacle.net, (obstacle.net == net), obstacle.kind


def _probe_net(tools, net: str, strategy: str, results: list) -> None:
    """Routes one net using either strategy and records the outcome.

    "exact": route_to() pushes all the way to the target's own coordinates,
        then finish_route() is called -- what the first run of this script
        did, and what triggered HEAD_COLLIDES.
    "near": route_to() stops just short of the target (offset back along
        the approach vector by SNAP_RADIUS's mm equivalent), then
        finish_route() is called without any further push -- the pattern
        every proven-successful route in this repo actually uses, letting
        fix() do the final connect rather than pre-empting it.
    """
    r = tools.start_route(net)
    if not r.ok:
        print(f"RESULT: net={net!r} strategy={strategy} start_route FAILED: {r.message}")
        return

    target_mm = r.data["target_mm"]
    start_mm = r.data["start_mm"]

    if strategy == "near":
        dx, dy = target_mm[0] - start_mm[0], target_mm[1] - start_mm[1]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist > 0.6:  # stop 0.6mm short -- just outside SNAP_RADIUS_NM (0.5mm)
            frac = (dist - 0.6) / dist
            approach = (start_mm[0] + dx * frac, start_mm[1] + dy * frac)
        else:
            approach = target_mm  # pads too close together to meaningfully stop short
        r = tools.route_to(*approach)
    else:
        r = tools.route_to(*target_mm)

    collided_during_route = "head_collides" in r.data or any("colliding" in w for w in r.warnings)
    if r.warnings:
        print(f"  [{net}/{strategy}] route_to warning: {r.warnings}")

    finish = tools.finish_route()
    finished = finish.ok

    obstacle_net, obstacle_same_net, obstacle_kind = _obstacle_detail(tools, net)

    print(
        f"RESULT: net={net!r} strategy={strategy} finished={finished} "
        f"collision_warning_during_route={collided_during_route} "
        f"finish_error={finish.error_code} "
        f"obstacle_net={obstacle_net!r} obstacle_same_net={obstacle_same_net} "
        f"obstacle_kind={obstacle_kind!r}"
    )

    if not finished:
        _show(f"  finish_route() detail ({net}/{strategy})", finish)
        _show(f"  check_drc() right after the failure", tools.check_drc())
        _dump_nearby(tools, *target_mm)
        tools.abandon_route()

    results.append(
        NetProbeResult(
            net, strategy, finished, collided_during_route, finish.error_code,
            obstacle_net, obstacle_same_net, obstacle_kind,
        )
    )


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

    tools = RouterTools(bridge, board_width_mm=50.0, board_height_mm=50.0, max_step_mm=80.0)

    print("=== 1. get_board_info() ===")
    _show("board info", tools.get_board_info())

    print("\n=== 2. list_nets() ===")
    _show("nets", tools.list_nets())

    # -- The actual investigation: does pushing exactly onto the target -----
    # collide, and does stopping short of it (letting fix() close the gap,
    # the pattern every other proven route in this repo uses) avoid that?
    # Alternates strategy per net so both get roughly equal net variety
    # rather than "near" only ever seeing the easy leftover nets.
    print("\n=== 3. Exact-target vs. near-target finishing, across every net ===")
    results: list[NetProbeResult] = []
    for i, net in enumerate(nets):
        strategy = "exact" if i % 2 == 0 else "near"
        _probe_net(tools, net, strategy, results)

    exact = [r for r in results if r.strategy == "exact"]
    near = [r for r in results if r.strategy == "near"]

    def _rate(rs):
        return f"{sum(r.finished for r in rs)}/{len(rs)}" if rs else "n/a"

    print(f"\n  exact-target finish rate: {_rate(exact)}")
    print(f"  near-target  finish rate: {_rate(near)}")
    print(
        "  (exact vs. near is CONFOUNDED with routing order in this script -- "
        "strategy alternates with net index, so a difference here could be "
        "routing order, not the strategy itself. Read this as a secondary "
        "signal, not the primary one.)"
    )

    # -- THE decisive axis: is the collision self-referential (same net as
    # the one being routed -- a head touching its own start/target pad) or
    # a genuine different-net blocker? This is what pcbworld/engine/cpp's
    # GetHeadObstacle() was added specifically to answer, replacing the
    # first run's bare head_collides() bool.
    failed = [r for r in results if not r.finished]
    same_net_failures = [r for r in failed if r.obstacle_same_net is True]
    diff_net_failures = [r for r in failed if r.obstacle_same_net is False]
    unknown_failures = [r for r in failed if r.obstacle_same_net is None]

    print(f"\n  failed nets: {len(failed)}/{len(results)}")
    print(f"    -- colliding with their OWN net (self-touch, likely benign): {len(same_net_failures)}")
    for r in same_net_failures:
        print(f"       {r.net!r} ({r.strategy}) vs. own net {r.obstacle_net!r} ({r.obstacle_kind})")
    print(f"    -- colliding with a DIFFERENT net (real contention): {len(diff_net_failures)}")
    for r in diff_net_failures:
        print(f"       {r.net!r} ({r.strategy}) vs. {r.obstacle_net!r} ({r.obstacle_kind})")
    print(f"    -- obstacle detail unavailable (older bridge, or not a collision failure): {len(unknown_failures)}")

    if same_net_failures and not diff_net_failures:
        print(
            "\n  CONCLUSION: every failure was a same-net (self) collision. This "
            "confirms the GetHeadObstacle() fix's hypothesis directly -- fix "
            "belongs in pcbworld/agent/tools.py's _collision_message() / "
            "router.md guidance (already updated to say 'try finish_route() "
            "again' for this case), not further C++ changes."
        )
    elif diff_net_failures and not same_net_failures:
        print(
            "\n  CONCLUSION: every failure was a genuine different-net "
            "collision. This is real board contention, not a self-touch "
            "artifact -- the rip_up() recovery path is the correct fix, "
            "already reflected in the tool's own message."
        )
    elif same_net_failures and diff_net_failures:
        print(
            "\n  CONCLUSION: both kinds of failure occurred. The self-touch "
            "artifact was real (some failures) AND genuine contention exists "
            "separately (other failures) -- both recovery paths are needed, "
            "which is what the updated _collision_message()/router.md now "
            "provide."
        )

    routed_nets = {r.net for r in results if r.finished}
    unrouted_nets = [n for n in nets if n not in routed_nets]

    # -- place_via/switch_to_layer, on any net still available --------------
    via_net = unrouted_nets[0] if unrouted_nets else None
    if via_net:
        print(f"\n=== 4. place_via() / switch_to_layer() on {via_net!r} ===")
        r = tools.start_route(via_net)
        _show(f"start_route({via_net!r})", r)
        if r.ok:
            _show("place_via()", tools.place_via())
            _show("switch_to_layer(0)", tools.switch_to_layer(0))
            _show("abandon_route()", tools.abandon_route())
    else:
        print("\n=== 4. place_via()/switch_to_layer(): skipped, every net already routed ===")

    # -- Re-routing an already-routed net is refused correctly --------------
    if routed_nets:
        sample = sorted(routed_nets)[0]
        print(f"\n=== 5. Re-routing already-routed {sample!r} is refused correctly ===")
        r = tools.start_route(sample)
        _show(f"start_route({sample!r}) again", r)
        if r.ok:
            print("  UNEXPECTED: this should have been refused as NET_ALREADY_ROUTED")
        elif "rip_up" not in r.message:
            print("  UNEXPECTED: refusal didn't point at rip_up() as the fix")

        # -- rip-up-then-reroute, the actual recovery flow -------------------
        print(f"\n=== 6. rip_up({sample!r}) then reroute -- the real recovery flow ===")
        rip = tools.rip_up(sample)
        _show(f"rip_up({sample!r})", rip)
        if rip.ok and rip.data.get("removed_items", 0) > 0:
            r = tools.start_route(sample)
            if r.ok:
                tools.route_to(*r.data["target_mm"])
                _show(f"finish_route() after rip-up-then-reroute", tools.finish_route())
    else:
        print("\n=== 5/6. rip-up-then-reroute: skipped, no net finished cleanly to rip up ===")

    print("\n=== 7. check_drc() (final board state) ===")
    _show("DRC", tools.check_drc())

    print("\n" + "=" * 60)
    print(f"ROUTERTOOLS LIVE PROBE COMPLETE -- {len(routed_nets)}/{len(nets)} nets routed")
    print("(this is investigative, not pass/fail -- read the RESULT: lines above)")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify RouterTools against a live bridge")
    parser.add_argument("board_path", help="Path to .kicad_pcb board file")
    parser.add_argument("--bridge-dir", default=None, help="Optional path to bridge library directory")
    args = parser.parse_args()
    verify(args.board_path, args.bridge_dir)
