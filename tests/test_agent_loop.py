"""Tests for the LLM routing agent loop (offline verification with ScriptedBackend).

Verifies:
1. Clean multi-net routing run to completion.
2. Per-net budget exhaustion triggering automatic route abandonment.
3. Stuck-loop detection firing on repeated identical failing calls and injecting recovery guidance.
4. Malformed tool arguments fed back as structured errors without crashing.
5. Rip-up-then-reroute recovery flow.
6. Visualizer PNG rendering and JSON transcript serialization.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

from pcbworld.agent.backends import BackendReply, ScriptedBackend, ToolCall
from pcbworld.agent.loop import RoutingAgent, run_routing_agent
from pcbworld.agent.tools import ErrorCode, RouterTools
from tests.test_agent_tools import DeviatingBridge, StubBridge


def make_agent(
    backend: ScriptedBackend,
    bridge=None,
    output_dir=None,
    **kwargs,
) -> RoutingAgent:
    b = bridge or StubBridge()
    tools = RouterTools(b, 50.0, 50.0)
    return RoutingAgent(
        tools=tools,
        backend=backend,
        output_dir=output_dir,
        verbose=False,
        **kwargs,
    )


def test_clean_multi_net_run():
    # Net 0: start -> route_to (15, 5) -> route_to (25, 5) -> finish
    # Net 1: start -> route_to (15, 15) -> route_to (25, 15) -> finish
    script = [
        ToolCall("start_route", {"net": "net_0"}),
        ToolCall("route_to", {"x_mm": 15.0, "y_mm": 5.0}),
        ToolCall("route_to", {"x_mm": 25.0, "y_mm": 5.0}),
        ToolCall("finish_route", {}),
        ToolCall("start_route", {"net": "net_1"}),
        ToolCall("route_to", {"x_mm": 15.0, "y_mm": 15.0}),
        ToolCall("route_to", {"x_mm": 25.0, "y_mm": 15.0}),
        ToolCall("finish_route", {}),
    ]
    backend = ScriptedBackend(script)
    bridge = StubBridge()
    agent = make_agent(backend, bridge)

    summary = agent.run(net_order=["net_0", "net_1"])

    assert summary.total_nets == 2
    assert summary.routed_nets == 2
    assert summary.drc_errors == 0
    assert len(summary.net_summaries) == 2
    assert summary.net_summaries[0].routed
    assert summary.net_summaries[1].routed
    assert len(bridge.committed) == 2


def test_budget_exhaustion_forces_abandon():
    # Net 0 only makes intermediate moves and never finishes -> exhausts budget (max 3 steps)
    script = [
        ToolCall("start_route", {"net": "net_0"}),
        ToolCall("route_to", {"x_mm": 10.0, "y_mm": 5.0}),
        ToolCall("route_to", {"x_mm": 15.0, "y_mm": 5.0}),
        # 3rd step exhausts max_steps_per_net=3
        # Net 1 follows and routes cleanly
        ToolCall("start_route", {"net": "net_1"}),
        ToolCall("route_to", {"x_mm": 25.0, "y_mm": 15.0}),
        ToolCall("finish_route", {}),
    ]
    backend = ScriptedBackend(script)
    bridge = StubBridge()
    agent = make_agent(backend, bridge, max_steps_per_net=3)

    summary = agent.run(net_order=["net_0", "net_1"])

    assert summary.routed_nets == 1
    assert not summary.net_summaries[0].routed
    assert "budget" in summary.net_summaries[0].failure_reason.lower()
    assert summary.net_summaries[1].routed
    assert agent.tools._active_net is None  # no dangling route


def test_stuck_loop_detection_fires():
    # Model attempts the exact same failing move 3 times
    class RejectionBridge(StubBridge):
        def push(self, x, y, item_id=-1):
            return False

    script = [
        ToolCall("start_route", {"net": "net_0"}),
        ToolCall("route_to", {"x_mm": 10.0, "y_mm": 5.0}),
        ToolCall("route_to", {"x_mm": 10.0, "y_mm": 5.0}),
        ToolCall("route_to", {"x_mm": 10.0, "y_mm": 5.0}),
        # 4th action: model reads stuck warning, abandons route
        ToolCall("abandon_route", {}),
    ]
    backend = ScriptedBackend(script)
    agent = make_agent(backend, RejectionBridge(), stuck_threshold=3)

    summary = agent.run(net_order=["net_0"])

    assert summary.routed_nets == 0
    # Check that stuck detection message was injected into conversation history
    last_call_messages = backend.call_history[-1]["messages"]
    stuck_msgs = [m for m in last_call_messages if "STUCK LOOP DETECTED" in m.get("content", "")]
    assert len(stuck_msgs) >= 1


def test_malformed_tool_call_handled_gracefully():
    # Backend emits a malformed tool call with parse error
    script = [
        ToolCall("start_route", {"net": "net_0"}),
        ToolCall("route_to", {"_parse_error": "Malformed JSON in argument", "raw": "{'x': 10}"}),
        # Model gets error and sends valid call
        ToolCall("route_to", {"x_mm": 25.0, "y_mm": 5.0}),
        ToolCall("finish_route", {}),
    ]
    backend = ScriptedBackend(script)
    agent = make_agent(backend)

    summary = agent.run(net_order=["net_0"])

    assert summary.routed_nets == 1
    # Check that error was reported back in step records
    step_results = agent.step_records[1].tool_results
    assert not step_results[0]["ok"]
    assert "parse" in step_results[0]["message"].lower()


def test_rip_up_and_reroute_flow():
    # Net 0 routes
    # Net 1 starts, detects target collision, abandons route, rips up net_0, routes net_1, then reroutes net_0
    script = [
        # 1. Route net_0
        ToolCall("start_route", {"net": "net_0"}),
        ToolCall("route_to", {"x_mm": 25.0, "y_mm": 5.0}),
        ToolCall("finish_route", {}),
        # 2. Net 1 realizes conflict, rips up net_0, routes net_1
        ToolCall("start_route", {"net": "net_1"}),
        ToolCall("abandon_route", {}),
        ToolCall("rip_up", {"net": "net_0"}),
        ToolCall("start_route", {"net": "net_1"}),
        ToolCall("route_to", {"x_mm": 25.0, "y_mm": 15.0}),
        ToolCall("finish_route", {}),
    ]
    backend = ScriptedBackend(script)
    bridge = StubBridge()
    agent = make_agent(backend, bridge)

    summary = agent.run(net_order=["net_0", "net_1"])

    assert "net_0" in bridge.ripped
    assert summary.net_summaries[1].routed


def test_transcript_and_board_rendering_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        script = [
            ToolCall("start_route", {"net": "net_0"}),
            ToolCall("route_to", {"x_mm": 25.0, "y_mm": 5.0}),
            ToolCall("finish_route", {}),
        ]
        backend = ScriptedBackend(script)
        agent = make_agent(backend, output_dir=out_dir)

        summary = agent.run(net_order=["net_0"])

        # Check transcript JSON exists and has valid contents
        transcript_file = out_dir / "transcript.json"
        assert transcript_file.exists()
        data = json.loads(transcript_file.read_text(encoding="utf-8"))
        assert data["run_info"]["routed_nets"] == 1
        assert len(data["steps"]) >= 3

        # Check board PNGs were created
        png_files = list(out_dir.glob("*.png"))
        assert len(png_files) >= 1


def test_warning_on_an_ok_call_escalates_thinking_for_the_next_turn():
    """Live Colab run (Qwen3-4B): a real different-net collision surfaced as
    a WARNING on an otherwise-ok=True route_to() result (push() succeeded;
    the collision is informational, not a call failure). Under the
    error-only version of the think-escalation check, that warning never
    turned thinking back on -- the model kept issuing plain route_to()
    calls for several more turns after the collision first appeared, only
    reacting (if at all) once its step budget was nearly gone. A warning
    must escalate the NEXT turn's thinking the same way a hard error does,
    without being counted as a failing call for stuck-loop-detection
    purposes (DeviatingBridge's warning fires on an ACCEPTED push, not a
    rejected one -- it must not pollute failing_call_history)."""
    script = [
        ToolCall("start_route", {"net": "net_0"}),
        ToolCall("route_to", {"x_mm": 10.0, "y_mm": 5.0}),  # DeviatingBridge: ok=True + warning
        ToolCall("route_to", {"x_mm": 15.0, "y_mm": 5.0}),
        ToolCall("finish_route", {}),
    ]
    backend = ScriptedBackend(script)
    agent = make_agent(backend, DeviatingBridge())

    agent.run(net_order=["net_0"])

    steps = agent.step_records
    assert len(steps) >= 3
    # Step 0: start_route, think=False (nothing has happened yet).
    assert steps[0].think is False
    # Step 1: the route_to that produced the deviation warning -- itself
    # unescalated, since escalation only applies to the turn AFTER a
    # warning/error was seen.
    assert steps[1].think is False
    assert steps[1].tool_results[0]["ok"] is True
    assert steps[1].tool_results[0]["warnings"]
    # Step 2: the very next turn -- this is what the fix is about.
    assert steps[2].think is True

    # The warning-only call must not have been treated as a stuck-loop
    # failure: two DIFFERENT route_to() calls ran back to back with no
    # stuck-loop warning ever injected.
    for s in steps:
        for tr in s.tool_results:
            assert "STUCK LOOP" not in tr["message"]
