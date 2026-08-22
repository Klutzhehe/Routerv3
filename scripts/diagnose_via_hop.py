"""Does toggle_via_placement() produce a LAYER HOP, or only a via?

Gate A (scripts/diagnose_layer_switch.py) settled two things and left one
open. Settled: switch_layer() is dead as a layer change -- it accepts only a
no-op (switching to the layer the head is already on) and refuses every real
transition, at every candidate PCB_LAYER_ID, with or without via sizes
configured, from an SMD pad and from a THT pad, idle and mid-route. All four
hypotheses eliminated. Also settled: toggle_via_placement() followed by
fix(force_finish=True, force_commit=True) + commit_routing() puts ONE REAL
COMMITTED VIA on the board.

Left open, and it is the half that matters: that run showed

    after toggle_via_placement(): active=True layer=0 head_vias=0
    after next push():            layer=0 head_vias=0
    fix()+commit_routing():       1 committed via for net_0

-- the head layer never changed, and the via never appeared in
get_head_geometry().vias; it materialized only at commit. So we have "a via
can be placed", NOT "a route can change layer and keep going", which is what
docs/RL_PLAN.md's stage 3+ actually needs.

## The hypothesis this tests

That run called fix(force_finish=True, force_commit=True), which ENDS the
route. KiCad's interactive flow is press-V, then CLICK -- and that click is a
FixRoute that commits the current segment plus the via and CONTINUES placing
from it on the far side. A mid-route via therefore probably wants
fix(x, y, -1, force_finish=False, ...), leaving the placer alive, with the
layer flip observable only AFTER that call.

force_finish=True/force_commit=True is the convention everything else in this
repo uses (pcb_route_env.py, diff_pair_route_env.py, commit 7f746b6) for a
very good reason -- it is what makes a route SNAP TO THE TARGET PAD. That is
the right call for the LAST fix of a route and, this script hypothesizes,
exactly the wrong one for a mid-route via.

## What "it works" would mean

Not "fix() returned true". Three things together:
  1. after the mid-route fix, get_head_geometry() reports active=True on a
     DIFFERENT layer -- the placer is alive on the far side;
  2. pushes after that produce copper on that layer;
  3. after the route finishes and commits, get_board_geometry() shows real
     tracks on BOTH layers plus the vias joining them.
Only (3) is copper on disk. (1) and (2) can look right and still commit
nothing, which is exactly how the previous round overclaimed.

Bridge-only: never import pcbnew in this process.

Usage (after notebooks/00_setup.ipynb has built the bridge):
    python3 pcbworld/data/generate_board.py smd1.kicad_pcb --num-nets 1 --seed 0
    python3 scripts/diagnose_via_hop.py smd1.kicad_pcb
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from scripts.measure_waypoint_fidelity import MM, SNAP_RADIUS_NM, _load_bridge, _pick_pad_candidate

TRACK_WIDTH_NM = 250_000
VIA_DIAMETER_NM = 600_000
VIA_DRILL_NM = 300_000
FRONT_LAYER = 0


@dataclasses.dataclass
class HopResult:
    name: str
    config: str
    fix_ok: bool | None = None
    layer_before: int | None = None
    layer_after: int | None = None
    active_after: bool | None = None
    pushes_after_hop: int | None = None      # accepted pushes on the far side
    committed_vias: int | None = None
    tracks_per_layer: dict[int, int] = dataclasses.field(default_factory=dict)
    error: str | None = None
    notes: list[str] = dataclasses.field(default_factory=list)

    @property
    def hopped(self) -> bool:
        """Placer alive, on a different layer. Necessary, not sufficient --
        copper on both layers is what actually settles it."""
        return bool(
            self.active_after
            and self.layer_before is not None
            and self.layer_after is not None
            and self.layer_before != self.layer_after
        )


def _head(bridge):
    h = bridge.get_head_geometry()
    return h.active, h.layer, len(h.vias)


def _setup(bridge, module, board_path: str) -> None:
    assert bridge.load_board(board_path), f"load_board failed: {board_path}"
    bridge.set_mode(module.MODE_ROUTE_SINGLE)
    bridge.set_collision_mode(module.RM_MARK_OBSTACLES)
    bridge.set_track_width(TRACK_WIDTH_NM)
    bridge.set_via_diameter(VIA_DIAMETER_NM)
    bridge.set_via_drill(VIA_DRILL_NM)


def _net_endpoints(bridge) -> tuple[str, tuple[int, int], tuple[int, int]]:
    pads = bridge.net_pads()
    named = sorted({p.net for p in pads if p.net and p.net.startswith("net_")})
    assert named, "no plain 'net_*' nets -- generate with --num-nets >= 1"
    net = named[0]
    a, b = [p for p in pads if p.net == net]
    return net, (a.x, a.y), (b.x, b.y)


def _lerp(a: tuple[int, int], b: tuple[int, int], t: float) -> tuple[int, int]:
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t))


def _start(bridge, start_xy, notes: list[str]) -> int:
    candidates = bridge.query_hover_items(*start_xy, layer=FRONT_LAYER, slop_radius=SNAP_RADIUS_NM)
    assert candidates, "no candidate at the start pad"
    item_id = _pick_pad_candidate(candidates, "start", notes).id
    assert bridge.start_route(start_xy[0], start_xy[1], item_id, FRONT_LAYER), "start_route failed"
    return item_id


def _trial_fix_flags(
    bridge, module, board_path: str, force_finish: bool, force_commit: bool, toggle_first: bool
) -> HopResult:
    """The core question, one (force_finish, force_commit) combination at a
    time: after toggle_via_placement() and a mid-route fix(), is the placer
    still alive on the other layer?

    `toggle_first` also varies WHEN the toggle happens relative to the push,
    since it isn't known whether PNS attaches the pending via to the current
    head or to the next one."""
    order = "toggle,push" if toggle_first else "push,toggle"
    result = HopResult(
        name=f"fix_ff{int(force_finish)}_fc{int(force_commit)}_{order}",
        config=f"force_finish={force_finish} force_commit={force_commit} order={order}",
    )
    try:
        _setup(bridge, module, board_path)
        _, start_xy, target_xy = _net_endpoints(bridge)
        _start(bridge, start_xy, result.notes)

        mid = _lerp(start_xy, target_xy, 0.5)
        if toggle_first:
            bridge.toggle_via_placement()
            bridge.push(mid[0], mid[1], -1)
        else:
            bridge.push(mid[0], mid[1], -1)
            bridge.toggle_via_placement()

        _, result.layer_before, _ = _head(bridge)
        result.fix_ok = bool(bridge.fix(mid[0], mid[1], -1, force_finish, force_commit))
        result.active_after, result.layer_after, head_vias = _head(bridge)
        result.notes.append(
            f"after mid-route fix: ok={result.fix_ok} active={result.active_after} "
            f"layer={result.layer_before}->{result.layer_after} head_vias={head_vias}"
        )

        # If the placer is still alive, can it actually draw on the far side?
        if result.active_after:
            three_q = _lerp(start_xy, target_xy, 0.75)
            result.pushes_after_hop = int(bool(bridge.push(three_q[0], three_q[1], -1)))
            _, layer_now, _ = _head(bridge)
            result.notes.append(
                f"push after the hop: accepted={bool(result.pushes_after_hop)} layer now {layer_now}"
            )

        bridge.stop_routing()
    except Exception as exc:  # noqa: BLE001 -- one dead trial must not kill the run
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _trial_full_two_hop(
    bridge, module, board_path: str, force_finish: bool, force_commit: bool
) -> HopResult:
    """The only trial that proves anything on disk: F_Cu -> via -> far layer
    -> via -> F_Cu -> finish at the target pad -> commit, then count real
    committed geometry per layer.

    Two hops, not one: generate_board.py's pads are SMD on F_Cu only, so a
    route MUST come back to the front layer before it can legally terminate
    at the target pad -- the same reason measure_layer_hop_rescue.py routes
    two hops rather than landing on the back layer."""
    result = HopResult(
        name="full_two_hop",
        config=f"mid-route fix(force_finish={force_finish}, force_commit={force_commit})",
    )
    try:
        _setup(bridge, module, board_path)
        net, start_xy, target_xy = _net_endpoints(bridge)
        target_id = _pick_pad_candidate(
            bridge.query_hover_items(*target_xy, layer=FRONT_LAYER, slop_radius=SNAP_RADIUS_NM),
            "target",
            result.notes,
        ).id
        _start(bridge, start_xy, result.notes)

        _, result.layer_before, _ = _head(bridge)

        # Hop 1: quarter point, onto the far layer.
        p25 = _lerp(start_xy, target_xy, 0.25)
        bridge.push(p25[0], p25[1], -1)
        bridge.toggle_via_placement()
        hop1 = bool(bridge.fix(p25[0], p25[1], -1, force_finish, force_commit))
        active, layer_mid, _ = _head(bridge)
        result.notes.append(f"hop 1 at 25%: fix={hop1} active={active} layer={layer_mid}")
        if not active:
            result.notes.append("placer died at hop 1 -- a two-hop route is not possible this way")
            bridge.stop_routing()
            result.active_after, result.layer_after = active, layer_mid
            return result

        # Cross on the far layer.
        p75 = _lerp(start_xy, target_xy, 0.75)
        crossed = bool(bridge.push(p75[0], p75[1], -1))
        result.notes.append(f"pushed across on layer {layer_mid}: accepted={crossed}")

        # Hop 2: back to the front layer, so the route can end on an SMD pad.
        bridge.toggle_via_placement()
        hop2 = bool(bridge.fix(p75[0], p75[1], -1, force_finish, force_commit))
        active, layer_back, _ = _head(bridge)
        result.notes.append(f"hop 2 at 75%: fix={hop2} active={active} layer={layer_back}")
        result.active_after, result.layer_after = active, layer_back

        # Final approach and finish -- force_finish/force_commit True here is
        # correct and deliberate: this IS the last fix, and it is what snaps
        # the route to the target pad (commit 7f746b6).
        result.fix_ok = bool(bridge.fix(target_xy[0], target_xy[1], target_id, True, True))
        result.notes.append(f"final fix at the target pad: {result.fix_ok}")

        if result.fix_ok:
            bridge.commit_routing()
            geometry = bridge.get_board_geometry()
            result.committed_vias = sum(1 for v in geometry.vias if v.net == net)
            per_layer: dict[int, int] = {}
            for t in geometry.tracks:
                if t.net == net:
                    per_layer[t.layer] = per_layer.get(t.layer, 0) + 1
            result.tracks_per_layer = per_layer
            result.notes.append(
                f"committed: {result.committed_vias} via(s), tracks per layer {per_layer}"
            )
        else:
            bridge.stop_routing()
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _print_row(r: HopResult) -> None:
    if r.error:
        print(f"{r.name:<28} ERROR  {r.error}", flush=True)
    else:
        layers = f"{r.layer_before}->{r.layer_after}"
        print(
            f"{r.name:<28} fix={str(r.fix_ok):<5} active_after={str(r.active_after):<5} "
            f"layer={layers:<10} {r.config}",
            flush=True,
        )
    for note in r.notes:
        print(f"    [{r.name}] {note}", flush=True)


def _verdict(results: list[HopResult]) -> list[str]:
    lines: list[str] = []
    hops = [r for r in results if r.error is None and r.hopped]
    full = next((r for r in results if r.name == "full_two_hop" and r.error is None), None)

    if hops:
        best = hops[0]
        lines.append(
            f"MID-ROUTE HOP WORKS: {best.name} left the placer ALIVE on layer "
            f"{best.layer_after} (from {best.layer_before}). Protocol: "
            f"toggle_via_placement() -> push() -> {best.config}."
        )
    else:
        lines.append(
            "NO MID-ROUTE HOP: no (force_finish, force_commit) combination left the placer "
            "alive on a different layer. toggle_via_placement() can place a via but cannot "
            "continue a route on the far side through this API."
        )

    if full is not None:
        both_layers = len(full.tracks_per_layer) > 1
        if full.committed_vias and both_layers:
            lines.append(
                f"*** LAYER HOPPING CONFIRMED ON DISK: {full.committed_vias} committed via(s) "
                f"and real tracks on {len(full.tracks_per_layer)} layers "
                f"({full.tracks_per_layer}). Two-layer routing is available -- wire a "
                f"place-via action into the env and let stage 3 use it."
            )
        elif full.fix_ok:
            lines.append(
                f"PARTIAL: the two-hop route finished and committed, but geometry shows "
                f"{full.committed_vias} via(s) and tracks on layers "
                f"{sorted(full.tracks_per_layer)} -- not a real hop. The route probably "
                f"stayed on the front layer the whole way."
            )
        else:
            lines.append(
                "The two-hop route did not finish, so nothing was committed to inspect. "
                "Whatever the head state suggested above, no two-layer copper exists."
            )

    if not any(line.startswith("***") for line in lines):
        lines.append(
            "CONCLUSION: stay single-layer. Per docs/RL_PLAN.md, stages 1-3 do not need vias; "
            "train those and revisit two-layer routing only if stage 3 plateaus against the "
            "9/24 baseline."
        )
    return lines


def run(board_path: str, bridge_dir: str | None) -> list[HopResult]:
    module = _load_bridge(bridge_dir)
    bridge = module.PNSBridge()

    print(f"board: {board_path}\n")
    results: list[HopResult] = []

    def record(r: HopResult) -> HopResult:
        results.append(r)
        _print_row(r)
        return r

    # force_finish=True is deliberately NOT swept with toggle order: it is
    # already known to end the route (that is what the previous round did).
    # It appears once, as the control that reproduces that known behavior.
    for force_finish, force_commit, toggle_first in (
        (False, False, True),
        (False, False, False),
        (False, True, True),
        (True, True, True),  # control: the previous round's call
    ):
        record(_trial_fix_flags(bridge, module, board_path, force_finish, force_commit, toggle_first))

    hops = [r for r in results if r.error is None and r.hopped]
    if hops:
        # Reuse whichever flags actually kept the placer alive.
        name = hops[0].name
        force_finish = "ff1" in name
        force_commit = "fc1" in name
        print(f"\n-> two-hop route will use {hops[0].config}\n", flush=True)
    else:
        force_finish, force_commit = False, False
        print(
            "\n-> no combination hopped; running the two-hop route with "
            "(False, False) anyway, so the committed geometry is on record\n",
            flush=True,
        )

    record(_trial_full_two_hop(bridge, module, board_path, force_finish, force_commit))

    print(f"\n{'=' * 96}\nVERDICT\n{'=' * 96}")
    for line in _verdict(results):
        print(f"  {line}\n")
    print("=" * 96)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board_path", help="1-net .kicad_pcb from generate_board.py")
    parser.add_argument("--bridge-dir", default=None)
    args = parser.parse_args()

    run(args.board_path, args.bridge_dir)
    sys.exit(0)
