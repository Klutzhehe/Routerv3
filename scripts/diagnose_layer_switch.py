"""Why has switch_layer() never once succeeded against the real bridge?

0 successes in 32 real attempts (15/15 and 17/17, commits 381e1a7 and
dc9164e). Two Colab rounds have already been spent on this, each testing
exactly ONE hypothesis, each costing a full session. This script tests all
of them in a single run, on a 1-net board so nothing can be blamed on
contention.

What is already known and should not be re-derived:
  - Rejections are POSITION-INDEPENDENT. The 15/15 round rejected at a
    straight-line midpoint in open board space, nowhere near any pad. So
    this is not local geometry/clearance.
  - push()/fix()/query_hover_items() are Colab-verified repeatedly.
    switch_layer() and toggle_via_placement() are not -- they have never
    been exercised successfully anywhere in this repo.

## The hypotheses

  H1  Via size was never configured. switch_layer() places a real via
      internally; with no legal via geometry the refusal would be uniform
      and position-independent, matching the data. A fix (set 0.6mm/0.3mm)
      is already committed in dc9164e but was NEVER RE-RUN -- so this is
      both the cheapest hypothesis to test and the one with an untested
      fix already sitting in the tree.

  H2  LINE_PLACER::SetLayer() refuses unless the route's START ITEM spans
      the layer being switched to. generate_board.py pads are
      PAD_ATTRIB_SMD with a layer set of F_Cu only, so under this
      hypothesis an SMD-started route can never legally reach B_Cu, at any
      position, with any via size. This also explains position-independence
      -- and it is the leading candidate.

  H3  Router state machine: the call is only legal in some state this
      script's call order never reaches (e.g. IDLE, or before the first
      Move()).

  H4  Wrong primitive entirely. Changing layer mid-route in the GUI places
      a via; toggle_via_placement() may be the call that actually does
      this, with switch_layer() being for something else. IF THIS ONE IS
      TRUE THE PROBLEM DISSOLVES -- we never needed switch_layer().

## Two controls that rule out whole classes of false conclusion

  - SAME-LAYER NO-OP: switch_layer() to the layer the head is ALREADY on.
    H1 and H2 both predict this succeeds (no via needed; the start pad
    trivially spans its own layer). A rejection here means the call is
    being refused structurally, before any geometry is considered, and
    knocks out H1 and H2 together.

  - LAYER-ID SWEEP: a wrong PCB_LAYER_ID constant is INDISTINGUISHABLE
    from a structural rejection -- both are just `False`. KiCad 9
    renumbered PCB_LAYER_ID: measured against the local KiCad 9.0 install,
    F_Cu=0 but B_Cu=**2**, not the pre-9 31 and not the 1 that parts of
    this repo's env code informally treat as "the other layer". This
    process cannot ask pcbnew for it directly -- importing the system
    pcbnew module here would crash the process (docs/performance.md, hard
    constraint 1) -- and scripts/measure_layer_hop_rescue.py takes
    --back-layer as a required argument, so nothing in the tree records
    which value the historical 0-for-32 runs actually passed. The sweep
    settles it: it tries each candidate and, if any is accepted, uses that
    one for every later trial automatically.

Bridge-only, like every script here that touches pcbworld_pns_bridge: never
import pcbnew in this process.

Usage (after notebooks/00_setup.ipynb has built the bridge):
    python3 pcbworld/data/generate_board.py smd.kicad_pcb --num-nets 1 --seed 0
    python3 pcbworld/data/generate_board.py tht.kicad_pcb --num-nets 1 --seed 0 \
        --pad-type tht
    python3 scripts/diagnose_layer_switch.py smd.kicad_pcb --tht-board tht.kicad_pcb

The --tht-board argument is what tests H2. Without it the script still runs
and reports everything else, but says plainly that H2 was left untested --
which would waste the run, since H2 is the leading hypothesis.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from scripts.measure_waypoint_fidelity import MM, SNAP_RADIUS_NM, _load_bridge, _pick_pad_candidate

TRACK_WIDTH_NM = 250_000

# Back-copper PCB_LAYER_ID candidates, most likely first (the first entry is
# also the fallback if none is accepted).
#
# 2 is MEASURED, not guessed: `pcbnew.B_Cu` printed 2 against the local
# KiCad 9.0 install (F_Cu=0, B_Cu=2 -- KiCad 9 renumbered PCB_LAYER_ID so
# copper layers are no longer contiguous from 0). 31 is the pre-9 value and
# 1 is what parts of this repo's env code have informally treated as "the
# other layer" (fake_bridge.py's toggle flips 0<->1). The sweep still runs
# all three because the Colab build is a different point release (9.0.8/
# 9.0.9) than the install this was measured on, and because a wrong
# constant and a structural refusal are the same `False`.
LAYER_ID_SWEEP = (2, 1, 31)
FRONT_LAYER = 0


@dataclasses.dataclass
class TrialResult:
    name: str
    hypothesis: str
    config: str
    accepted: bool | None = None       # switch_layer()'s return; None if not called
    layer_before: int | None = None
    layer_after: int | None = None
    active_before: bool | None = None
    active_after: bool | None = None
    head_vias_before: int | None = None
    head_vias_after: int | None = None
    committed_vias: int | None = None  # after fix()+commit_routing(), if reached
    error: str | None = None
    notes: list[str] = dataclasses.field(default_factory=list)

    @property
    def layer_changed(self) -> bool:
        return (
            self.layer_before is not None
            and self.layer_after is not None
            and self.layer_before != self.layer_after
        )

    @property
    def via_appeared(self) -> bool:
        return (
            self.head_vias_before is not None
            and self.head_vias_after is not None
            and self.head_vias_after > self.head_vias_before
        )


def _head_snapshot(bridge) -> tuple[bool | None, int | None, int | None]:
    """(active, layer, via count) from get_head_geometry(), or (None,)*3 if
    this build predates that binding. Every trial reads this BEFORE and
    AFTER its action: a False return with a changed layer -- or a True
    return with an unchanged one -- is itself a finding, and a bare return
    value cannot show either."""
    if not hasattr(bridge, "get_head_geometry"):
        return None, None, None
    head = bridge.get_head_geometry()
    return head.active, head.layer, len(head.vias)


def _prepare(bridge, module, board_path: str, via_sizes: tuple[int, int] | None) -> None:
    """Fresh router state for one trial.

    load_board() constructs a BRAND NEW PNS::ROUTER (pns_bridge.cpp's
    LoadBoard, confirmed by reading it -- not assumed), which is what makes
    the "via sizes never set" trial honest: Sizes() genuinely resets here,
    so a set_via_diameter() call from an earlier trial cannot leak into a
    later one. Everything else the router needs must therefore be re-applied
    after every load, which is why this is one function rather than setup
    done once at the top.
    """
    assert bridge.load_board(board_path), f"load_board failed: {board_path}"
    bridge.set_mode(module.MODE_ROUTE_SINGLE)
    bridge.set_collision_mode(module.RM_MARK_OBSTACLES)
    bridge.set_track_width(TRACK_WIDTH_NM)
    if via_sizes is not None:
        bridge.set_via_diameter(via_sizes[0])
        bridge.set_via_drill(via_sizes[1])


def _first_net_pads(bridge) -> tuple[str, tuple[int, int], tuple[int, int]]:
    pads = bridge.net_pads()
    named = sorted({p.net for p in pads if p.net and p.net.startswith("net_")})
    assert named, "no plain 'net_*' nets on this board -- generate with --num-nets >= 1"
    net = named[0]
    net_pads = [p for p in pads if p.net == net]
    assert len(net_pads) == 2, f"{net!r} has {len(net_pads)} pads, expected 2"
    a, b = net_pads
    return net, (a.x, a.y), (b.x, b.y)


def _open_route(bridge, start_xy, target_xy, push_to_midpoint: bool) -> list[str]:
    """start_route() at the start pad, optionally push to the straight-line
    midpoint -- open board space on a 1-net board, which is where the 15/15
    rejections happened."""
    notes: list[str] = []
    candidates = bridge.query_hover_items(*start_xy, layer=FRONT_LAYER, slop_radius=SNAP_RADIUS_NM)
    assert candidates, "no candidate at the start pad"
    start_id = _pick_pad_candidate(candidates, "start", notes).id
    assert bridge.start_route(start_xy[0], start_xy[1], start_id, FRONT_LAYER), "start_route failed"

    if push_to_midpoint:
        mid = ((start_xy[0] + target_xy[0]) // 2, (start_xy[1] + target_xy[1]) // 2)
        if not bridge.push(mid[0], mid[1], -1):
            notes.append("push() to midpoint REJECTED -- unexpected, push() has never rejected before")
    return notes


def _trial_switch(
    bridge,
    module,
    board_path: str,
    name: str,
    hypothesis: str,
    target_layer: int,
    via_sizes: tuple[int, int] | None,
    start_route: bool = True,
    push_to_midpoint: bool = True,
) -> TrialResult:
    via_desc = "unset" if via_sizes is None else f"{via_sizes[0] / MM:.2f}/{via_sizes[1] / MM:.2f}mm"
    result = TrialResult(
        name=name,
        hypothesis=hypothesis,
        config=f"layer={target_layer} vias={via_desc} "
        f"{'mid-route' if start_route and push_to_midpoint else 'pre-push' if start_route else 'idle'}",
    )
    try:
        _prepare(bridge, module, board_path, via_sizes)
        _, start_xy, target_xy = _first_net_pads(bridge)

        if start_route:
            result.notes.extend(_open_route(bridge, start_xy, target_xy, push_to_midpoint))

        result.active_before, result.layer_before, result.head_vias_before = _head_snapshot(bridge)
        result.accepted = bool(bridge.switch_layer(target_layer))
        result.active_after, result.layer_after, result.head_vias_after = _head_snapshot(bridge)

        bridge.stop_routing()
    except Exception as exc:  # noqa: BLE001 -- a crashed trial must not kill the run
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _trial_toggle_via(bridge, module, board_path: str, via_sizes: tuple[int, int]) -> TrialResult:
    """H4: the GUI changes layer mid-route by placing a via. Walks the whole
    sequence and snapshots the head at each stage, because it isn't known
    which step materializes the via -- toggle_via_placement() itself, or the
    next push() (several PNS placers only attach pending geometry on the
    following Move())."""
    result = TrialResult(
        name="toggle_via_placement",
        hypothesis="H4",
        config=f"vias={via_sizes[0] / MM:.2f}/{via_sizes[1] / MM:.2f}mm mid-route",
    )
    try:
        _prepare(bridge, module, board_path, via_sizes)
        net, start_xy, target_xy = _first_net_pads(bridge)
        result.notes.extend(_open_route(bridge, start_xy, target_xy, push_to_midpoint=True))

        result.active_before, result.layer_before, result.head_vias_before = _head_snapshot(bridge)

        bridge.toggle_via_placement()  # void -- state readback is the only signal
        active, layer, vias = _head_snapshot(bridge)
        result.notes.append(f"after toggle_via_placement(): active={active} layer={layer} head_vias={vias}")

        # Three-quarter point: far enough past the midpoint to force a real
        # Move() without reaching the target pad's snap radius.
        three_q = (
            (start_xy[0] + 3 * target_xy[0]) // 4,
            (start_xy[1] + 3 * target_xy[1]) // 4,
        )
        pushed = bridge.push(three_q[0], three_q[1], -1)
        result.active_after, result.layer_after, result.head_vias_after = _head_snapshot(bridge)
        result.notes.append(
            f"after next push() (accepted={pushed}): layer={result.layer_after} "
            f"head_vias={result.head_vias_after}"
        )

        # Does any of it survive to real copper? A via in the head that never
        # commits would be a false positive for "this primitive works".
        if bridge.fix(target_xy[0], target_xy[1], -1, True, True):
            bridge.commit_routing()
            geometry = bridge.get_board_geometry()
            result.committed_vias = sum(1 for v in geometry.vias if v.net == net)
            result.notes.append(
                f"fix()+commit_routing() succeeded; {result.committed_vias} committed via(s) for {net}"
            )
        else:
            result.notes.append("fix() rejected -- no committed copper to inspect")
            bridge.stop_routing()
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _verdict(results: dict[str, TrialResult], back_layer: int, tht_tested: bool) -> list[str]:
    """Maps the observed pattern onto which hypotheses survive.

    This exists so the run is self-interpreting: per AGENTS.md the Colab
    side reports output rather than diagnosing, so the conclusions have to
    be IN the output, not derived afterwards from a table of booleans.
    """
    lines: list[str] = []

    def ok(name: str) -> bool:
        r = results.get(name)
        return bool(r and r.error is None and r.accepted)

    sweep_hits = [lid for lid in LAYER_ID_SWEEP if ok(f"sweep_layer_{lid}")]
    if sweep_hits:
        lines.append(
            f"LAYER ID: switch_layer() ACCEPTED layer id(s) {sweep_hits} -- so at least some of "
            f"the historical 0-for-32 may have been a wrong PCB_LAYER_ID constant, not a "
            f"structural refusal. Use {sweep_hits[0]} as the back-copper id."
        )
    else:
        lines.append(
            f"LAYER ID: no id in {list(LAYER_ID_SWEEP)} was accepted, so the refusal is NOT a "
            f"wrong-constant artifact. That whole class of explanation is now ruled out."
        )

    noop = results.get("same_layer_noop")
    if noop and noop.error is None:
        if noop.accepted:
            lines.append(
                "CONTROL (same-layer no-op): ACCEPTED. The call works in principle and is "
                "refusing this specific layer TRANSITION -- H1 and H2 both stay alive."
            )
        else:
            lines.append(
                "CONTROL (same-layer no-op): REJECTED -- switching to the layer the head is "
                "ALREADY on was refused. The call is being refused structurally, before any "
                "via geometry or layer-span check could matter. This knocks out H1 and H2 "
                "together and points hard at H3/H4."
            )

    via_trials = [results.get(n) for n in ("vias_060_030", "vias_040_020", "vias_unset")]
    via_trials = [t for t in via_trials if t and t.error is None]
    if len(via_trials) == 3:
        outcomes = {t.accepted for t in via_trials}
        if len(outcomes) == 1 and not outcomes.pop():
            lines.append(
                "H1 (via size) DEAD: 0.6/0.3mm, 0.4/0.2mm, and never-set all produced the "
                "identical rejection. dc9164e's committed fix does not change the outcome."
            )
        else:
            lines.append(
                "H1 (via size) LIVE: the three via configurations did NOT behave identically -- "
                + ", ".join(f"{t.name}={t.accepted}" for t in via_trials)
            )

    idle, mid = results.get("idle_switch"), results.get("vias_060_030")
    if idle and mid and idle.error is None and mid.error is None:
        if idle.accepted and not mid.accepted:
            lines.append(
                "H3 (state machine) SUPPORTED: accepted while IDLE, rejected mid-route. "
                "switch_layer() is a pre-route layer selector, not a mid-route layer change."
            )
        elif not idle.accepted and not mid.accepted:
            lines.append("H3: rejected in BOTH idle and mid-route -- state is not the discriminator.")

    if tht_tested:
        tht, smd = results.get("tht_start"), results.get("vias_060_030")
        if tht and smd and tht.error is None and smd.error is None:
            if tht.accepted and not smd.accepted:
                lines.append(
                    "*** H2 (start item must span the target layer) CONFIRMED: identical trial "
                    "succeeded from a THT pad and failed from an SMD pad. Consequence: on "
                    "SMD-only boards switch_layer() can NEVER work, and no amount of via "
                    "sizing or positioning changes that. Fix: place a via "
                    "(toggle_via_placement) instead, or generate boards with --pad-type tht."
                )
            elif tht.accepted and smd.accepted:
                lines.append("H2: both SMD and THT accepted -- pad type is not the discriminator.")
            else:
                lines.append(
                    "H2 NOT CONFIRMED: the THT-started route was rejected too, so the start "
                    "item's layer span is not what is blocking this."
                )
    else:
        lines.append(
            "H2 UNTESTED -- no --tht-board was passed. H2 is the LEADING hypothesis; this run "
            "leaves it open. Generate one with `--pad-type tht` and re-run."
        )

    toggle = results.get("toggle_via_placement")
    if toggle and toggle.error is None:
        if toggle.committed_vias:
            lines.append(
                f"*** H4 CONFIRMED: toggle_via_placement() produced {toggle.committed_vias} "
                f"REAL committed via(s) and the route finished. switch_layer() is not needed -- "
                f"this is the layer-change primitive. Wire this into the env's action space."
            )
        elif toggle.via_appeared or toggle.layer_changed:
            lines.append(
                "H4 PARTIAL: toggle_via_placement() changed head state "
                f"(layer {toggle.layer_before}->{toggle.layer_after}, "
                f"head vias {toggle.head_vias_before}->{toggle.head_vias_after}) but nothing "
                "committed. The primitive does something; finishing the route is the open part."
            )
        else:
            lines.append(
                "H4 DEAD: toggle_via_placement() changed no head state and committed nothing. "
                "Neither layer-change primitive works on this board."
            )

    if not any(line.startswith("***") for line in lines):
        lines.append(
            "NO PRIMITIVE WORKED. Per docs/RL_PLAN.md this is the point to stop spending "
            "sessions on layer hopping and train single-layer (stage 1-3), revisiting vias "
            "as a second action head later."
        )
    return lines


def run(board_path: str, tht_board: str | None, bridge_dir: str | None) -> list[TrialResult]:
    module = _load_bridge(bridge_dir)
    bridge = module.PNSBridge()

    for required in ("switch_layer", "toggle_via_placement", "get_head_geometry"):
        if not hasattr(bridge, required):
            print(f"WARNING: this bridge build has no {required}() -- trials using it will be skipped")

    print(f"board (SMD): {board_path}")
    print(f"board (THT): {tht_board or '-- not provided, H2 will be left untested --'}\n")

    results: list[TrialResult] = []

    # The sweep runs FIRST so every later trial can use whichever id is real,
    # instead of inheriting a guess.
    for layer_id in LAYER_ID_SWEEP:
        results.append(
            _trial_switch(
                bridge, module, board_path,
                name=f"sweep_layer_{layer_id}",
                hypothesis="layer-id control",
                target_layer=layer_id,
                via_sizes=(600_000, 300_000),
            )
        )

    accepted_ids = [r for r in results if r.accepted]
    back_layer = int(accepted_ids[0].name.rsplit("_", 1)[1]) if accepted_ids else LAYER_ID_SWEEP[0]
    if accepted_ids:
        print(f"-> layer-id sweep accepted {back_layer}; using it for the remaining trials\n")
    else:
        print(f"-> layer-id sweep accepted nothing; using {back_layer} for the remaining trials\n")

    results.append(
        _trial_switch(
            bridge, module, board_path, name="same_layer_noop", hypothesis="control",
            target_layer=FRONT_LAYER, via_sizes=(600_000, 300_000),
        )
    )
    results.append(
        _trial_switch(
            bridge, module, board_path, name="idle_switch", hypothesis="H3",
            target_layer=back_layer, via_sizes=(600_000, 300_000), start_route=False,
            push_to_midpoint=False,
        )
    )
    results.append(
        _trial_switch(
            bridge, module, board_path, name="before_first_push", hypothesis="H3",
            target_layer=back_layer, via_sizes=(600_000, 300_000), push_to_midpoint=False,
        )
    )
    results.append(
        _trial_switch(
            bridge, module, board_path, name="vias_060_030", hypothesis="H1",
            target_layer=back_layer, via_sizes=(600_000, 300_000),
        )
    )
    results.append(
        _trial_switch(
            bridge, module, board_path, name="vias_040_020", hypothesis="H1",
            target_layer=back_layer, via_sizes=(400_000, 200_000),
        )
    )
    results.append(
        _trial_switch(
            bridge, module, board_path, name="vias_unset", hypothesis="H1",
            target_layer=back_layer, via_sizes=None,
        )
    )
    if tht_board:
        results.append(
            _trial_switch(
                bridge, module, tht_board, name="tht_start", hypothesis="H2",
                target_layer=back_layer, via_sizes=(600_000, 300_000),
            )
        )
    results.append(_trial_toggle_via(bridge, module, board_path, (600_000, 300_000)))

    print(f"{'trial':<22} {'H':<5} {'accepted':<9} {'layer':<12} {'head vias':<11} {'config'}")
    print("-" * 100)
    for r in results:
        if r.error:
            print(f"{r.name:<22} {r.hypothesis:<5} {'ERROR':<9} {r.error}")
            continue
        layers = f"{r.layer_before}->{r.layer_after}"
        vias = f"{r.head_vias_before}->{r.head_vias_after}"
        print(
            f"{r.name:<22} {r.hypothesis:<5} {str(r.accepted):<9} {layers:<12} {vias:<11} {r.config}"
        )

    for r in results:
        for note in r.notes:
            print(f"    [{r.name}] {note}")

    print(f"\n{'=' * 100}\nVERDICT\n{'=' * 100}")
    by_name = {r.name: r for r in results}
    for line in _verdict(by_name, back_layer, tht_tested=bool(tht_board)):
        print(f"  {line}\n")

    disagreements = [
        r for r in results
        if r.error is None and r.accepted is not None and r.accepted is not r.layer_changed
        and r.name != "same_layer_noop"  # a no-op switch legitimately changes nothing
    ]
    if disagreements:
        print("  RETURN VALUE vs READBACK DISAGREEMENT (worth more than the table above):")
        for r in disagreements:
            print(
                f"    {r.name}: returned {r.accepted} but layer went "
                f"{r.layer_before}->{r.layer_after}"
            )
    print("=" * 100)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board_path", help="SMD .kicad_pcb from generate_board.py (--num-nets 1)")
    parser.add_argument(
        "--tht-board",
        default=None,
        help="the same board generated with --pad-type tht. Tests H2, the leading hypothesis; "
        "without it this run cannot close the question.",
    )
    parser.add_argument("--bridge-dir", default=None)
    args = parser.parse_args()

    run(args.board_path, args.tht_board, args.bridge_dir)
    sys.exit(0)
