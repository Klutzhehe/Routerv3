"""Verification script for RoutingAgent (pcbworld/agent/loop.py) against a
live pcbworld_pns_bridge.

Everything verified so far in this investigation
(scripts/verify_head_bindings.py, scripts/verify_router_tools_live.py)
tested RouterTools directly. The loop ORCHESTRATING RouterTools --
RoutingAgent, with its step budgets, stuck-loop detector, PNG rendering,
and JSON transcript -- has only ever run against ScriptedBackend +
tests/fake_bridge.py's fake, which always accepts every push/fix and never
models a real router refusal (see fake_bridge.py's own docstring: "passing
tests here only mean the Python glue doesn't crash"). This script closes
that gap before spending Colab GPU time on a real model (QwenBackend):
does the loop's own machinery survive a real success, a real HEAD_COLLIDES
failure, and a real repeated-failure stuck-loop, driven against the actual
compiled router -- not just its own control-flow logic in isolation.

Builds a deterministic per-net ScriptedBackend script from
bridge.net_pads() (known pad coordinates), not by parsing model-facing
text out of the message history. This script's job is proving loop
MECHANICS survive real bridge timing/state, not policy intelligence, so
there is no reason to route through the harder, LLM-shaped text-parsing
path to get there -- that is what an actual model run (QwenBackend) is
for, later.

One net is deliberately scripted to call finish_route() three times in a
row without changing anything in between, specifically to exercise the
stuck-loop detector against a REAL HEAD_COLLIDES failure -- every prior
test of that path (tests/test_agent_loop.py::test_stuck_loop_detection_fires)
used the fake bridge, which cannot produce a genuine collision to retry
against.

Run in Colab after building pcbworld_pns_bridge (no rebuild needed --
same compiled .so this whole investigation has been using):
    python3 Routerv3/pcbworld/data/generate_board.py board.kicad_pcb --num-nets 6 --seed 0
    python3 Routerv3/scripts/verify_agent_loop_live.py board.kicad_pcb --output-dir /content/agent_run
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


def _build_net_targets(bridge) -> dict[str, tuple[float, float]]:
    """net name -> its second pad's (x_mm, y_mm), i.e. what start_route()
    will report as target_mm. Matches RouterTools._start_next_leg's own
    "pads[0] is start, pads[1] is target" convention."""
    pads_by_net: dict[str, list] = {}
    for pad in bridge.net_pads():
        if pad.net:
            pads_by_net.setdefault(pad.net, []).append(pad)

    targets = {}
    for net, pads in pads_by_net.items():
        if len(pads) >= 2:
            targets[net] = (pads[1].x / MM, pads[1].y / MM)
    return targets


def _make_reactive_backend(backends_mod, targets: dict[str, tuple[float, float]], stuck_test_net: str):
    """A single reactive callable, not a fixed-length flat script.

    ScriptedBackend's step index is global across the WHOLE run, not scoped
    per net -- a fixed-length per-net script that finishes early (net
    routes successfully sooner than its scripted step count) leaks its
    leftover queued calls into the NEXT net's turn, which is not a
    hypothetical: it happened on the first version of this script's dry
    run against the fake bridge (net_0 finished in 3 steps, its two
    leftover finish_route()/abandon_route() calls fired against net_1's
    fresh conversation instead, producing misleading NO_ROUTE_IN_PROGRESS
    noise in the transcript). A reactive callable that decides its next
    move from what ACTUALLY happened -- not a step count fixed in advance
    -- cannot leak this way: RoutingAgent.route_net() simply stops calling
    the backend once a net reaches a terminal state, and the callable never
    assumes a fixed number of steps a net "should" take.

    Net name is recovered from the user-prompt message route_net() always
    puts at messages[1] ("Please route net {net!r}...") rather than passed
    in externally -- this callable is net-agnostic, driven purely by the
    per-net conversation ScriptedBackend/route_net() already scope
    correctly (a fresh net_messages list per route_net() call, confirmed
    by reading loop.py directly).
    """
    ToolCall = backends_mod.ToolCall
    BackendReply = backends_mod.BackendReply
    import re

    def reactive(messages, tools, think):
        net_match = re.search(r"route net '([^']+)'", messages[1]["content"])
        net = net_match.group(1) if net_match else None

        tool_results = [m for m in messages if m.get("role") == "tool"]
        last = tool_results[-1] if tool_results else None
        last_was_finish = last is not None and last.get("name") == "finish_route"
        last_ok = last is not None and last["content"].startswith("OK")

        finish_attempts = sum(1 for m in tool_results if m.get("name") == "finish_route")

        already_abandoned = last is not None and last.get("name") == "abandon_route" and last_ok

        if already_abandoned:
            # This callable has nothing further scripted after a deliberate
            # give-up (it is not a real recovery policy, just a loop-
            # mechanics probe). Going quiet here matters: the first version
            # of this check was missing, and re-calling abandon_route() has
            # nothing to act on the second time (RouterTools correctly
            # errors "nothing to abandon" -- see tools.py), which without
            # this guard repeats for the rest of the per-net step budget,
            # spamming real (if harmless) tool errors and repeated stuck-
            # loop warnings into the transcript. route_net()'s own
            # "no tool call -> prompt, consume a step, continue" handling
            # absorbs this cleanly until the budget catches it, which is
            # the correct way for a scripted policy to run out of ideas.
            return BackendReply(
                text="No further scripted action for this net.",
                reasoning="reactive live-probe step: out of ideas" if think else None,
                tool_calls=[],
            )

        if last is None:
            call = ToolCall(name="start_route", arguments={"net": net})
        elif last.get("name") == "start_route":
            x_mm, y_mm = targets[net]
            call = ToolCall(name="route_to", arguments={"x_mm": x_mm, "y_mm": y_mm})
        elif last.get("name") == "route_to":
            call = ToolCall(name="finish_route", arguments={})
        elif last_was_finish and not last_ok and net == stuck_test_net and finish_attempts < 3:
            # Deliberately repeat the SAME failing call -- this is the
            # stuck-loop probe. Only for the designated net, only while a
            # real finish_route() keeps genuinely failing (not scripted to
            # fail -- whether it fails at all depends on real router
            # state, which is the point).
            call = ToolCall(name="finish_route", arguments={})
        else:
            # Either finished, or gave up retrying -- clean exit either way.
            call = ToolCall(name="abandon_route", arguments={})

        return BackendReply(
            text=f"Calling {call.name}",
            reasoning="reactive live-probe step" if think else None,
            tool_calls=[call],
        )

    # Repeated, not a single entry: ScriptedBackend consumes one script
    # item per chat() call regardless of content, so the SAME reactive
    # function has to be available at every step index across the whole
    # run for every net to reach it. 200 is comfortably above this run's
    # max_total_steps (100).
    return backends_mod.ScriptedBackend([reactive] * 200)


def verify(board_path: str, output_dir: str | None, bridge_dir: str | None = None) -> bool:
    bridge_module = _load_bridge(bridge_dir)

    # Deferred, same convention as every env/script touching the bridge.
    from pcbworld.agent import backends as backends_mod
    from pcbworld.agent.loop import run_routing_agent
    from pcbworld.agent.tools import RouterTools

    bridge = bridge_module.PNSBridge()
    assert bridge.load_board(board_path), f"load_board failed on {board_path}"

    targets = _build_net_targets(bridge)
    nets = sorted(targets)
    assert len(nets) >= 2, f"need at least 2 routable nets on {board_path}"

    tools = RouterTools(bridge, board_width_mm=50.0, board_height_mm=50.0, max_step_mm=80.0)

    # One reactive backend drives every net -- see _make_reactive_backend's
    # docstring for why this replaced an earlier fixed-length-per-net
    # design that leaked unused scripted steps into the next net's turn.
    # The first net is designated for the stuck-loop probe; whether
    # finish_route() actually fails against the real bridge for it isn't
    # known ahead of time, which is what makes this a real test.
    backend = _make_reactive_backend(backends_mod, targets, stuck_test_net=nets[0])

    print(f"Routing {len(nets)} nets: {nets}")
    print(f"(net {nets[0]!r} additionally scripted to retry finish_route() 3x -- stuck-loop probe)\n")

    summary = run_routing_agent(
        tools=tools,
        backend=backend,
        net_order=nets,
        max_steps_per_net=10,
        max_total_steps=100,
        stuck_threshold=3,
        output_dir=output_dir,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print(f"AGENT LOOP LIVE PROBE COMPLETE")
    print(f"  nets routed:   {summary.routed_nets}/{summary.total_nets}")
    print(f"  total steps:   {summary.total_steps}")
    print(f"  drc errors:    {summary.drc_errors}")
    print(f"  wall time:     {summary.wall_time_s:.2f}s")
    if output_dir:
        print(f"  transcript:    {summary.transcript_path}")
        print(f"  PNGs written to: {output_dir}")
    print("=" * 60)
    print(
        "\nThis is investigative, not pass/fail: what matters is that the loop "
        "completed cleanly (no crash, every net reached a terminal state, "
        "budgets/stuck-detection behaved sanely) against REAL router "
        "responses -- not that every net actually routed. Read the "
        "per-net summary and transcript for what happened to each."
    )

    for s in summary.net_summaries:
        status = "ROUTED" if s.routed else "NOT ROUTED"
        print(f"  {s.net!r}: {status} ({s.steps} steps) -- {s.failure_reason or 'success'}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify RoutingAgent against a live bridge")
    parser.add_argument("board_path", help="Path to .kicad_pcb board file")
    parser.add_argument(
        "--output-dir", default=None, help="Directory for PNG renders + transcript.json"
    )
    parser.add_argument("--bridge-dir", default=None, help="Optional path to bridge library directory")
    args = parser.parse_args()
    verify(args.board_path, args.output_dir, args.bridge_dir)
