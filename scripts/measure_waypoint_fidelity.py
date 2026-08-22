"""Gates docs/AI_ARCHITECTURE.md's whole design: does push()-ing a polyline
of waypoints actually reproduce that polyline as committed copper, and how
expensive is one net-route (T_pns) in wall-clock terms?

CFP never draws copper itself -- it emits a coarse cost field, an A* planner
turns that into 5-20 waypoints, and pcbworld_pns_bridge's push()/fix() walk
those waypoints into a real route (see docs/AI_ARCHITECTURE.md's diagram).
That pipeline has never been exercised against the real router. If push()
routinely ignores or reroutes around the requested waypoints, the agent's
field has no causal effect on the outcome and nothing can learn. If push()
is a faithful validator but rejects naive straight-line waypoints often, that
is not fatal -- it just means the planner needs real replanning on
rejection, which this script also measures via a hand-authored perpendicular
detour (see _perp_offset() and _route_one_net()'s third attempt).

T_pns -- the wall-clock cost of one net-route -- has never been measured
either, and docs/AI_ARCHITECTURE.md's whole throughput analysis
(scripts/smoke_cfp.py's "pipelined trainer" numbers) is provisional until it
is. The GPU forward pass was measured at 1306 net-decisions/sec on a T4
(fp16, batch 128, 0.766 ms/board); a pipelined trainer keeps up iff
T_pns > 0.766 ms * num_workers -- this script's T_pns histogram is the input
to that inequality.

## Gate B (docs/RL_PLAN.md) -- added later, and now the main reason to run this

The line-geometry RL plan hangs on one measurable question this script is
already 90% of the way to answering, so it answers it here rather than in a
second Colab session:

  - PER-CALL WALL CLOCK, not just per-net. A line-geometry observation calls
    get_board_geometry()/get_head_geometry() EVERY STEP, not once per net,
    so the breakdown decides whether that observation is affordable at all
    and how many worker processes the GPU can feed. See TimingBridge.

  - DOES head_collides() PREDICT A fix() REJECTION? push() accepted 72/72
    net-attempts across three runs while fix() rejected ~67% -- success is
    silent, failure is late. If the collision signal fires during the pushes
    before a rejected fix() and stays quiet before an accepted one, the
    per-step reward in docs/RL_PLAN.md is real signal and the planned 1-D
    heading action space works. If it fires equally in both, one terminal
    bit has to carry ~20 steps of credit and the action granularity must
    change BEFORE a trainer is written. See AttemptRecord.

    Run --no-collision-trace once as a control: probing the router mid-route
    should not change the outcome, and if the direct-success count moves,
    the measurement is perturbing the thing it measures and the correlation
    is void. This repo has been bitten by measurement-side bugs producing
    router "findings" more than once (see this file's own git history).

Deliberately exercises MODE_ROUTE_SINGLE only, not diff pairs or length
tuning: DiffPairRouteEnv's legs decompose into MODE_ROUTE_SINGLE /
MODE_ROUTE_DIFF_PAIR / MODE_TUNE_SINGLE, but the diff-pair and tune
primitives are already Colab-verified (ROADMAP.md item 7) and are not what
the waypoint-follower (build order step 4, not written yet) will drive --
that primitive only ever pushes a single-net polyline, which is exactly
MODE_ROUTE_SINGLE. Plain nets are a representative, sufficient proxy.

Bridge-only, like every other script here that touches pcbworld_pns_bridge:
never import pcbnew (the system module) in this process once the bridge is
loaded (see docs/performance.md) -- generate the board with
pcbworld/data/generate_board.py as a genuinely separate process first.

Usage (after notebooks/00_setup.ipynb has built the bridge):
    python3 pcbworld/data/generate_board.py board.kicad_pcb --num-nets 24 --seed 0
    python3 scripts/measure_waypoint_fidelity.py board.kicad_pcb
    python3 scripts/measure_waypoint_fidelity.py board.kicad_pcb --no-collision-trace

    Leave board size at generate_board.py's own default (50x50mm) unless you
    have deliberately checked the net count against its min_spacing_mm=3.0 --
    e.g. 24 nets (48 pads) on a 30x30mm board hits its packing limit and
    raises RuntimeError before a board even gets written.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

MM = 1_000_000
SNAP_RADIUS_NM = int(0.5 * MM)
TRACK_WIDTH_NM = 250_000

# Known WORKDIR conventions from notebooks/00_setup.ipynb's cell 2 -- used
# only as a fallback when pcbworld_pns_bridge isn't already importable
# (e.g. this script run standalone via %%bash rather than pasted into the
# same kernel cell 8 already loaded the bridge into).
_BRIDGE_SEARCH_ROOTS = ("/content", str(Path.home() / "routerv3-build"))


def _load_bridge(bridge_dir: str | None):
    try:
        import pcbworld_pns_bridge as bridge  # noqa: F401  (may already be on sys.path)

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
        "pcbworld_pns_bridge not found. Run notebooks/00_setup.ipynb through its "
        "build step first, or pass --bridge-dir /path/to/kicad-src's parent."
    )


class TimingBridge:
    """Transparent proxy recording wall-clock per bridge method.

    Per-net elapsed time was already measured; what docs/RL_PLAN.md's Gate B
    needs is the BREAKDOWN -- which call dominates. That decides worker count
    (push() at 0.1ms and push() at 10ms are different training designs) and
    whether per-step observation building is affordable at all, since a
    line-geometry observation calls get_board_geometry()/get_head_geometry()
    every single step, not once per net.

    A proxy rather than call-site instrumentation: every existing call site
    stays untouched, so this cannot change the behavior it is measuring.
    Overhead is one perf_counter() pair per call (~100ns) against calls
    expected to be microseconds at minimum -- negligible, but it IS included
    in the numbers, so treat sub-microsecond means as noise.
    """

    def __init__(self, bridge) -> None:
        self._bridge = bridge
        self.timings: dict[str, list[float]] = defaultdict(list)

    def __getattr__(self, name: str):
        # Only reached when normal lookup fails, so self._bridge/self.timings
        # resolve without recursion.
        attr = getattr(self._bridge, name)
        if not callable(attr):
            return attr

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return attr(*args, **kwargs)
            finally:
                self.timings[name].append(time.perf_counter() - t0)

        return timed


@dataclasses.dataclass
class AttemptRecord:
    """One (net, strategy) attempt -- the unit the collision/fix correlation
    is computed over. A net can produce up to 6 of these.

    Exists to answer docs/RL_PLAN.md's Gate B question, which gates the whole
    action-space choice: push() accepted 72/72 while fix() rejected ~67%, so
    success is silent and failure is late. If head_collides() fires during
    the pushes that precede a REJECTED fix() and stays quiet before an
    ACCEPTED one, there is a real dense per-step reward signal and the 1-D
    heading policy works. If it fires equally in both, one terminal bit has
    to carry ~20 steps of credit and the action granularity has to change
    before any trainer is written.
    """

    net: str
    strategy: str
    pushes: int
    first_collision_step: int | None  # index of the first push after which
                                      #   head_collides() was True
    collided_nets: list[str]
    fix_ok: bool

    @property
    def collided(self) -> bool:
        return self.first_collision_step is not None


@dataclasses.dataclass
class NetResult:
    net: str
    reached_target: bool
    strategy: str  # "direct" | "polyline" | "detour(<mm>mm,<side>)" | "failed"
    waypoints_requested: int
    waypoints_accepted: int
    first_rejection_frac: float | None  # fraction along the path of the first
                                        #   push() rejection, None if none
    max_deviation_nm: float | None  # committed track vs. accepted waypoints
    elapsed_s: float


def _straight_waypoints(start: tuple[int, int], target: tuple[int, int], n: int):
    """n evenly spaced interior points on the straight line start->target.
    This is what an A* planner on a near-zero cost field emits -- CFP's
    untrained/near-baseline behavior (see model.py's init discussion)."""
    (x0, y0), (x1, y1) = start, target
    return [
        (int(x0 + (x1 - x0) * i / (n + 1)), int(y0 + (y1 - y0) * i / (n + 1)))
        for i in range(1, n + 1)
    ]


def _perp_offset(
    a: tuple[int, int], b: tuple[int, int], magnitude_nm: int, side: int
) -> tuple[int, int]:
    """A point offset perpendicular to a->b at its midpoint, `side` in
    {-1, +1}. Stands in for what a trained field would produce around an
    obstacle -- this script hand-authors it since no planner exists yet."""
    import math

    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    px, py = -dy / length, dx / length
    mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
    return int(mx + px * magnitude_nm * side), int(my + py * magnitude_nm * side)


def _pick_pad_candidate(candidates: list, label: str, warnings: list[str]):
    """query_hover_items() returns every item within slop_radius, in
    whatever order the router's internal hit-test happens to produce -- NOT
    sorted by kind or distance. On a mostly-empty board candidates[0] is
    almost always the pad, which is why this bug hides on sparse fixtures
    (the toy 1-net board, DiffPairRouteEnv's Colab test). On a board with
    many already-committed nets, an unrelated track can legitimately pass
    within slop_radius of a later net's pad -- KiCad's default clearance is
    far tighter than SNAP_RADIUS_NM's 0.5mm -- and candidates[0] can be that
    track instead of the pad. fix()ing against that item's id then fails
    for a reason that has nothing to do with THIS net's route, and looks
    identical to a real collision in the summary stats.

    simple_route_env.py, pcb_route_env.py and diff_pair_route_env.py all
    have the same `candidates[0]` pattern -- if this hypothesis is
    confirmed here, it likely affects them too, not just this script.
    """
    pads = [c for c in candidates if c.kind == "pad"]
    if pads:
        return pads[0]
    kinds = [c.kind for c in candidates]
    warnings.append(f"{label}: no 'pad' among candidates, got {kinds} -- used candidates[0]")
    return candidates[0]


def _route_one_net(
    bridge,
    module,
    pads: list,
    net: str,
    warnings: list[str],
    attempts_out: list[AttemptRecord] | None = None,
    trace_collisions: bool = True,
) -> NetResult:
    # attempts_out/trace_collisions are appended-to, optional parameters
    # rather than a changed return type: scripts/measure_layer_hop_rescue.py
    # imports this function directly and calls it with the original five
    # arguments.
    net_pads = [p for p in pads if p.net == net]
    assert len(net_pads) == 2, f"{net!r} has {len(net_pads)} pad(s), expected 2 (see generate_board.py)"
    start_pad, target_pad = net_pads
    start_xy = (start_pad.x, start_pad.y)
    target_xy = (target_pad.x, target_pad.y)

    t0 = time.perf_counter()

    start_candidates = bridge.query_hover_items(*start_xy, layer=0, slop_radius=SNAP_RADIUS_NM)
    assert start_candidates, f"no candidate at {net!r}'s start pad"
    start_id = _pick_pad_candidate(start_candidates, f"{net}/start", warnings).id
    assert bridge.start_route(start_xy[0], start_xy[1], start_id, 0), (
        f"start_route failed for {net!r}"
    )
    target_candidates = bridge.query_hover_items(*target_xy, layer=0, slop_radius=SNAP_RADIUS_NM)
    assert target_candidates, f"no candidate at {net!r}'s target pad"
    target_id = _pick_pad_candidate(target_candidates, f"{net}/target", warnings).id

    can_trace = trace_collisions and hasattr(bridge, "head_collides")
    can_name_obstacle = can_trace and hasattr(bridge, "get_head_obstacle")

    def try_push_sequence(
        points: list[tuple[int, int]], collisions: list[tuple[int, str]]
    ) -> tuple[int, int | None]:
        """Pushes each point in order. Returns (accepted_count,
        first_rejection_index or None).

        After each accepted push, asks the router whether the head now
        collides -- appending (step_index, obstacle_net) to `collisions`.
        This is the only early failure signal available: push()'s own return
        value has never once been False against the real bridge.
        """
        for i, (x, y) in enumerate(points):
            if not bridge.push(x, y, -1):
                return i, i
            if can_trace and bridge.head_collides():
                obstacle = ""
                if can_name_obstacle:
                    detail = bridge.get_head_obstacle()
                    obstacle = detail.net if detail.found else ""
                collisions.append((i, obstacle))
        return len(points), None

    # Attempts, in order: one big hop, a 5-point straight polyline (what an
    # untrained/near-flat-field CFP emits, chunked the way the real planner
    # will chunk it -- 5-20 waypoints, see docs/AI_ARCHITECTURE.md), then
    # four hand-authored perpendicular detours at increasing offsets (a
    # stand-in for what a trained field would produce; not exhaustive, its
    # only job is "is a rescue possible at all", not "is this a good plan").
    #
    # Retries on EITHER a push() rejection OR a fix() rejection -- not just
    # push(). A Colab run showed push() accepting all 72/72 net-attempts
    # across three runs while fix() rejected ~67% of them even with
    # force_finish/force_commit=True, which an earlier version of this loop
    # never retried past: it only escalated to polyline/detour when push()
    # itself failed, so with push() apparently never rejecting a single big
    # hop, the whole retry ladder was dead code -- polyline and detour had
    # never once run against the real bridge. This also directly tests
    # something the old structure couldn't: whether fix() accepts a route
    # built from several incremental push()es (matching how a real
    # waypoint-follower will actually drive the router) more often than one
    # push() straight to the target, for the SAME net.
    attempts: list[tuple[str, list[tuple[int, int]]]] = [
        ("direct", [target_xy]),
        ("polyline", _straight_waypoints(start_xy, target_xy, 5) + [target_xy]),
    ]
    for magnitude_mm, side in ((1, 1), (1, -1), (2, 1), (2, -1)):
        detour_point = _perp_offset(start_xy, target_xy, int(magnitude_mm * MM), side)
        attempts.append((f"detour({magnitude_mm}mm,{'+' if side > 0 else '-'})", [detour_point, target_xy]))

    strategy = "failed"
    reached = False
    accepted = requested = 0
    rejected_at: int | None = None

    for attempt_name, waypoints in attempts:
        assert bridge.start_route(start_xy[0], start_xy[1], start_id, 0), (
            f"start_route (re-)failed for {net!r} on attempt {attempt_name!r}"
        )
        requested = len(waypoints)
        collisions: list[tuple[int, str]] = []
        accepted, rejected_at = try_push_sequence(waypoints, collisions)

        def record(fix_ok: bool) -> None:
            if attempts_out is None:
                return
            attempts_out.append(
                AttemptRecord(
                    net=net,
                    strategy=attempt_name,
                    pushes=accepted,
                    first_collision_step=collisions[0][0] if collisions else None,
                    collided_nets=sorted({n for _, n in collisions if n}),
                    fix_ok=fix_ok,
                )
            )

        if rejected_at is not None:
            # No fix() was reached, so this attempt carries no information
            # about the collision/fix correlation -- deliberately not
            # recorded as fix_ok=False, which would poison the statistic
            # with attempts that never got to try.
            bridge.stop_routing()
            continue

        # force_finish=True, force_commit=True: matches the Colab-verified
        # convention in pcb_route_env.py / diff_pair_route_env.py (commit
        # 7f746b6) -- an earlier version of this script called fix() with
        # (False, False) and every "failed" net had accepted==requested
        # (push() always reached the target; only this call rejected it).
        fixed = bridge.fix(target_xy[0], target_xy[1], target_id, True, True)
        record(fix_ok=bool(fixed))
        if fixed:
            strategy = attempt_name
            reached = True
            break
        bridge.stop_routing()

    max_deviation_nm = None
    if reached:
        bridge.commit_routing()
        geometry = bridge.get_board_geometry()
        net_tracks = [t for t in geometry.tracks if t.net == net]
        assert net_tracks, (
            f"{net!r}: fix()/commit_routing() reported success but "
            f"get_board_geometry() has no track for it -- the router "
            f"and the geometry readback disagree"
        )
        # The endpoint that matters: does the committed track actually
        # reach the target pad, not just wherever push() last left the
        # head. This is the one thing RM_MARK_OBSTACLES's push()-as-
        # validator design does NOT guarantee by construction -- closure.
        end_dist = min(
            (t.x2 - target_xy[0]) ** 2 + (t.y2 - target_xy[1]) ** 2 for t in net_tracks
        ) ** 0.5
        max_deviation_nm = end_dist
    else:
        bridge.stop_routing()

    elapsed = time.perf_counter() - t0
    first_rejection_frac = None if rejected_at is None or requested == 0 else rejected_at / requested

    return NetResult(
        net=net,
        reached_target=reached,
        strategy=strategy,
        waypoints_requested=requested,
        waypoints_accepted=accepted,
        first_rejection_frac=first_rejection_frac,
        max_deviation_nm=max_deviation_nm,
        elapsed_s=elapsed,
    )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def run(
    board_path: str,
    num_nets: int,
    bridge_dir: str | None,
    save_image: str | None = None,
    trace_collisions: bool = True,
) -> list[NetResult]:
    bridge_module = _load_bridge(bridge_dir)
    bridge = TimingBridge(bridge_module.PNSBridge())

    assert bridge.load_board(board_path), f"load_board failed: {board_path}"
    bridge.set_mode(bridge_module.MODE_ROUTE_SINGLE)
    # The mode the CFP design actually commits to: push() must be a pure
    # validator (accept/reject a straight segment) rather than a router that
    # quietly shoves other tracks to make room -- see simple_route_env.py's
    # docstring and PNS_BRIDGE::SetCollisionMode's doc comment for why.
    bridge.set_collision_mode(bridge_module.RM_MARK_OBSTACLES)
    bridge.set_track_width(TRACK_WIDTH_NM)

    pads = bridge.net_pads()
    # Numeric, not the lexicographic order plain sorted() would give
    # ("net_10" < "net_2" as strings) -- routing order matters here (each
    # net is routed against whatever earlier nets already committed), so it
    # should match the net_<i> index a reader actually sees, not string order.
    available = sorted(
        {p.net for p in pads if p.net and p.net.startswith("net_")},
        key=lambda name: int(name.split("_")[1]),
    )
    assert available, "no plain 'net_*' nets found -- was the board generated with --num-nets > 0?"
    net_names = available[:num_nets]

    print(f"board: {board_path}")
    print(f"routing {len(net_names)} of {len(available)} available plain nets, in order "
          f"(collision mode RM_MARK_OBSTACLES, track width {TRACK_WIDTH_NM / MM:.2f}mm)\n")

    warnings: list[str] = []
    attempts: list[AttemptRecord] = []
    results = [
        _route_one_net(bridge, bridge_module, pads, net, warnings, attempts, trace_collisions)
        for net in net_names
    ]

    violations = bridge.run_drc()

    if save_image:
        # Deferred import -- pcbworld.viz has no pcbnew/bridge dependency of
        # its own (matplotlib only), so this only costs anything for callers
        # who actually want an image. Reads the SAME bridge instance's live
        # state (get_board_geometry() + net_pads()), not a reload -- this
        # process's board never touches disk, and matplotlib doesn't
        # conflict with pcbworld_pns_bridge the way system pcbnew would
        # (see docs/performance.md), so no second process is needed.
        import matplotlib

        matplotlib.use("Agg")  # headless -- this script has no display
        from pcbworld.viz import render_board

        geometry = bridge.get_board_geometry()
        ax = render_board(
            geometry,
            net_pads=pads,
            title=f"{board_path}: {sum(1 for r in results if r.reached_target)}/"
            f"{len(results)} nets routed",
        )
        ax.figure.savefig(save_image, dpi=150, bbox_inches="tight")
        print(f"\nsaved board image: {save_image}")

    print(f"{'net':<10} {'reached':<8} {'strategy':<9} {'accepted/req':<13} "
          f"{'end_dev(mm)':<12} {'time(ms)':<9}")
    for r in results:
        dev = "n/a" if r.max_deviation_nm is None else f"{r.max_deviation_nm / MM:.4f}"
        print(
            f"{r.net:<10} {str(r.reached_target):<8} {r.strategy:<16} "
            f"{r.waypoints_accepted}/{r.waypoints_requested:<11} {dev:<12} "
            f"{r.elapsed_s * 1e3:.2f}"
        )

    reached = [r for r in results if r.reached_target]
    direct = [r for r in results if r.strategy == "direct" and r.reached_target]
    polyline_rescued = [r for r in results if r.strategy == "polyline" and r.reached_target]
    detour_rescued = [r for r in results if r.strategy.startswith("detour") and r.reached_target]
    rescued = polyline_rescued + detour_rescued
    failed = [r for r in results if not r.reached_target]
    times_ms = [r.elapsed_s * 1e3 for r in results]
    devs_mm = [r.max_deviation_nm / MM for r in reached if r.max_deviation_nm is not None]

    print(f"\n{'=' * 70}")
    print("FIDELITY")
    print(f"  direct straight-push success : {len(direct)}/{len(results)}")
    print(f"  rescued by polyline/detour   : {len(rescued)}/{len(results)}")
    print(f"  unreachable after all retries: {len(failed)}/{len(results)}  {[r.net for r in failed]}")
    if devs_mm:
        print(f"  final-endpoint deviation (mm): mean={statistics.mean(devs_mm):.4f} "
              f"max={max(devs_mm):.4f}")
        print(f"    -> should be ~0: RM_MARK_OBSTACLES makes push() a validator, so an "
              f"accepted\n       waypoint sequence should land exactly on the target it fix()ed to. "
              f"A\n       nonzero number here means push()/fix() altered geometry beyond what "
              f"was\n       requested -- the actual fidelity risk this script exists to catch.")
    print(f"\nCANDIDATE RESOLUTION")
    print(f"  query_hover_items() calls where no 'pad' was among the hits: {len(warnings)}")
    if warnings:
        print(f"    -> a fix() failure on one of these nets may not be a real collision at "
              f"all: it was\n       handed some OTHER item's id (an unrelated already-committed "
              f"track, most likely) and\n       correctly refused to finish there. See "
              f"_pick_pad_candidate()'s docstring.")
        for w in warnings[:10]:
            print(f"    {w}")
        if len(warnings) > 10:
            print(f"    ... and {len(warnings) - 10} more")
    else:
        print(f"    -> every start/target query resolved cleanly to a pad. If nets still "
              f"failed,\n       that failure is NOT explained by candidate-id confusion -- "
              f"push()/fix() are\n       rejecting for some other reason (real DRC/clearance, "
              f"most likely).")

    print(f"\nDRC violations after commit  : {len(violations)}")
    for v in violations[:10]:
        print(f"    [{v.severity}] {v.message} @ ({v.x / MM:.2f}, {v.y / MM:.2f})")

    print(f"\n{'=' * 70}")
    print("GATE B -- IS THERE A DENSE PER-STEP SIGNAL? (docs/RL_PLAN.md)")
    if not trace_collisions:
        print("  collision tracing DISABLED (--no-collision-trace). This is the control run:")
        print("  its direct-success count should match a traced run's. If it doesn't, calling")
        print("  head_collides() mid-route perturbs the router and the traced numbers are void.")
    elif not attempts:
        print("  no attempt reached a fix() call -- nothing to correlate.")
    else:
        failed = [a for a in attempts if not a.fix_ok]
        succeeded = [a for a in attempts if a.fix_ok]

        def collided_frac(group: list[AttemptRecord]) -> float:
            return sum(1 for a in group if a.collided) / len(group) if group else float("nan")

        p_fail, p_ok = collided_frac(failed), collided_frac(succeeded)
        print(f"  attempts that reached a fix() call: {len(attempts)} "
              f"({len(succeeded)} accepted, {len(failed)} rejected)")
        print(f"  P(head_collides fired | fix REJECTED) = {p_fail:.2f}  (n={len(failed)})")
        print(f"  P(head_collides fired | fix ACCEPTED) = {p_ok:.2f}  (n={len(succeeded)})")

        lead = [a.pushes - a.first_collision_step for a in failed if a.collided]
        if lead:
            print(f"  lead time on rejected attempts: the signal first fired "
                  f"{statistics.mean(lead):.1f} pushes (mean) before fix() was called "
                  f"-- that is how much credit-assignment distance it buys")

        blamed = sorted({n for a in attempts for n in a.collided_nets})
        if blamed:
            print(f"  obstacle nets named by get_head_obstacle(): {blamed[:10]}"
                  f"{' ...' if len(blamed) > 10 else ''}")

        if len(failed) < 5 or len(succeeded) < 5:
            print(f"\n  -> INCONCLUSIVE: fewer than 5 samples in a bucket. Re-run with more nets.")
        elif p_fail - p_ok >= 0.3:
            print(f"\n  -> SEPARATION ({p_fail - p_ok:+.2f}). head_collides() predicts fix()")
            print(f"     rejection, so the per-step collision penalty in docs/RL_PLAN.md's reward")
            print(f"     is real signal and the 1-D heading action space is viable as specified.")
        else:
            print(f"\n  -> NO SEPARATION ({p_fail - p_ok:+.2f}). head_collides() does NOT predict")
            print(f"     fix() rejection. One terminal bit would have to carry ~20 steps of credit.")
            print(f"     Per docs/RL_PLAN.md this blocks the trainer: revisit action granularity")
            print(f"     (macro waypoints, or a learned value on committed geometry) FIRST.")

    print(f"\n{'=' * 70}")
    print("PER-CALL WALL CLOCK -- ms (sets worker count; a line-geometry observation calls")
    print("get_board_geometry()/get_head_geometry() every step, not once per net)")
    total_s = sum(sum(v) for v in bridge.timings.values()) or 1.0
    print(f"  {'call':<22} {'n':>6} {'mean':>9} {'median':>9} {'p90':>9} {'max':>9} {'share':>7}")
    for name, samples in sorted(bridge.timings.items(), key=lambda kv: -sum(kv[1])):
        ms = [s * 1e3 for s in samples]
        print(f"  {name:<22} {len(ms):>6} {statistics.mean(ms):>9.3f} "
              f"{statistics.median(ms):>9.3f} {_percentile(ms, 0.9):>9.3f} {max(ms):>9.3f} "
              f"{sum(samples) / total_s:>6.1%}")

    print(f"\nT_pns (wall clock per net, includes query/start_route/all pushes/fix/commit)")
    print(f"  n={len(times_ms)}  mean={statistics.mean(times_ms):.2f}ms  "
          f"median={statistics.median(times_ms):.2f}ms  "
          f"p90={_percentile(times_ms, 0.9):.2f}ms  max={max(times_ms):.2f}ms")

    print(f"\nThroughput comparison (scripts/smoke_cfp.py measured 1306 net-decisions/s, "
          f"0.766ms/board, T4 fp16 batch 128):")
    t_pns_mean = statistics.mean(times_ms)
    for workers in (4, 8, 16, 32):
        bound = 0.766 * workers
        verdict = "GPU keeps up" if t_pns_mean > bound else "GPU is the bottleneck"
        print(f"  {workers:3d} workers: need T_pns > {bound:6.2f}ms, "
              f"measured mean {t_pns_mean:.2f}ms -> {verdict}")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board_path", help=".kicad_pcb generated by generate_board.py")
    parser.add_argument("--num-nets", type=int, default=24)
    parser.add_argument(
        "--bridge-dir",
        default=None,
        help="parent of kicad-src/ if pcbworld_pns_bridge isn't already importable "
        "(defaults to the Colab/notebook WORKDIR conventions)",
    )
    parser.add_argument(
        "--save-image",
        default=None,
        help="PNG path to save a rendered view of the final board state to "
        "(requires matplotlib; not installed by default outside Colab)",
    )
    parser.add_argument(
        "--no-collision-trace",
        action="store_true",
        help="skip the per-push head_collides() probe. Run this ONCE as a control: if the "
        "direct-success count differs from a traced run's, the probe itself is perturbing "
        "the router and the Gate B correlation is void.",
    )
    args = parser.parse_args()
    run(
        args.board_path,
        args.num_nets,
        args.bridge_dir,
        args.save_image,
        trace_collisions=not args.no_collision_trace,
    )
