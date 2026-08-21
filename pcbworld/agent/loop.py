"""The LLM routing agent loop for RouterV3.

Drives KiCad's push-and-shove router headlessly via `RouterTools` and a `Backend`.
Includes:
- Net-by-net sequential routing according to priority order.
- Adaptive thinking latency lever (think=False by default, think=True after errors).
- Per-net and total step budgets with automatic route abandonment on exhaustion.
- Stuck-loop detector injecting guidance on repeated failing calls.
- Safe structured error feedback preventing crashes on malformed model outputs.
- Comprehensive live visibility, per-run JSON transcripts, post-net board rendering,
  and end-of-run diagnostic summaries.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import time
from typing import Any, Sequence

from pcbworld.agent.backends import Backend, BackendReply, ToolCall, parse_tool_call_arguments
from pcbworld.agent.tools import ErrorCode, RouterTools, TOOL_SCHEMAS, ToolResult

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "router.md"


@dataclasses.dataclass
class StepRecord:
    """Detailed record of one agent turn."""

    step_index: int
    net: str | None
    think: bool
    reasoning: str | None
    text: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    timestamp: float


@dataclasses.dataclass
class NetRunSummary:
    """Summary of routing for a single net."""

    net: str
    routed: bool
    steps: int
    failure_reason: str | None = None
    wall_time_s: float = 0.0


@dataclasses.dataclass
class RunSummary:
    """End-of-run board routing summary."""

    total_nets: int
    routed_nets: int
    attempted_nets: int
    total_steps: int
    drc_errors: int
    wall_time_s: float
    net_summaries: list[NetRunSummary]
    transcript_path: str | None = None


class RoutingAgent:
    """Orchestrates LLM-driven board routing over RouterTools."""

    def __init__(
        self,
        tools: RouterTools,
        backend: Backend,
        system_prompt: str | None = None,
        max_steps_per_net: int = 30,
        max_total_steps: int = 300,
        stuck_threshold: int = 3,
        output_dir: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.tools = tools
        self.backend = backend
        self.max_steps_per_net = max_steps_per_net
        self.max_total_steps = max_total_steps
        self.stuck_threshold = stuck_threshold
        self.output_dir = Path(output_dir) if output_dir else None
        self.verbose = verbose

        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif DEFAULT_SYSTEM_PROMPT_PATH.exists():
            self.system_prompt = DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        else:
            self.system_prompt = "You are a PCB routing agent driving KiCad's push-and-shove router."

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.transcript: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.step_records: list[StepRecord] = []

    def _log(self, text: str) -> None:
        if self.verbose:
            print(text)

    def _execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Dispatches a ToolCall to RouterTools with structured error handling."""
        name = tool_call.name
        args = tool_call.arguments

        if "_parse_error" in args:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.BAD_COORDINATE,
                message=f"Failed to parse tool arguments for {name!r}: {args.get('error', 'unknown error')}",
                data={"raw": args.get("raw", "")},
            )

        if not hasattr(self.tools, name) or name.startswith("_"):
            return ToolResult(
                ok=False,
                error_code=ErrorCode.ROUTER_REJECTED,
                message=f"Unknown tool {name!r}. Legal tools: {[s['function']['name'] for s in TOOL_SCHEMAS]}",
            )

        method = getattr(self.tools, name)
        try:
            return method(**args)
        except TypeError as err:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.BAD_COORDINATE,
                message=f"Invalid arguments for {name}: {err}. Arguments received: {args!r}",
            )
        except Exception as ex:
            return ToolResult(
                ok=False,
                error_code=ErrorCode.ROUTER_REJECTED,
                message=f"Tool execution exception in {name}: {ex}",
            )

    def _render_and_save(self, net_idx: int, net_name: str, state: str) -> str | None:
        """Renders the current board geometry to PNG if output_dir is configured."""
        if self.output_dir is None:
            return None

        try:
            import matplotlib.pyplot as plt
            from pcbworld.viz.render_board import render_board

            if not hasattr(self.tools.bridge, "get_board_geometry"):
                return None

            geometry = self.tools.bridge.get_board_geometry()
            net_pads = self.tools.bridge.net_pads() if hasattr(self.tools.bridge, "net_pads") else None

            fig, ax = plt.subplots(figsize=(8, 8))
            render_board(
                geometry,
                net_pads=net_pads,
                ax=ax,
                title=f"Net [{net_idx}] {net_name} ({state})",
            )
            out_file = self.output_dir / f"net_{net_idx:02d}_{net_name}_{state}.png"
            fig.savefig(out_file, dpi=120, bbox_inches="tight")
            plt.close(fig)
            return str(out_file)
        except Exception as e:
            self._log(f"[Warning] Failed to render board PNG: {e}")
            return None

    def route_net(
        self,
        net: str,
        net_idx: int = 1,
        total_steps_so_far: int = 0,
    ) -> tuple[NetRunSummary, int]:
        """Routes a single net using the backend with budgeting and stuck detection."""
        start_time = time.time()
        self._log(f"\n================================================================================")
        self._log(f" ROUTING NET [{net_idx}]: {net}")
        self._log(f"================================================================================")

        # Net-specific sub-conversation
        net_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Please route net {net!r}. Use start_route({net!r}) to begin, route_to to place tracks, and finish_route() on Layer 0 to complete.",
            },
        ]

        net_steps = 0
        think_next = False  # Latency lever: think=False by default
        failing_call_history: list[tuple[str, str]] = []  # (name, json_args)
        failure_reason: str | None = None
        routed = False

        while net_steps < self.max_steps_per_net and (total_steps_so_far + net_steps) < self.max_total_steps:
            net_steps += 1
            curr_total = total_steps_so_far + net_steps

            # 1. Backend generation
            reply: BackendReply = self.backend.chat(
                messages=net_messages,
                tools=TOOL_SCHEMAS,
                think=think_next,
            )

            # Log reasoning & text
            if reply.reasoning:
                self._log(f"[Thinking (turn {net_steps})]:\n{reply.reasoning}")
            if reply.text:
                self._log(f"[Model]: {reply.text}")

            step_tool_calls_rec: list[dict[str, Any]] = []
            step_tool_results_rec: list[dict[str, Any]] = []

            # If no tool calls emitted, prompt the model to make a tool call
            if not reply.tool_calls:
                self._log(f"[Agent]: No tool call returned. Prompting model for action.")
                net_messages.append({"role": "assistant", "content": reply.text})
                net_messages.append(
                    {
                        "role": "user",
                        "content": "Please call a tool to make progress (e.g. start_route, route_to, or finish_route).",
                    }
                )
                think_next = True
                continue

            # Assistant response message
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": reply.text,
                "tool_calls": [
                    {
                        "id": tc.call_id or f"call_{net_steps}_{i}",
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for i, tc in enumerate(reply.tool_calls)
                ],
            }
            net_messages.append(assistant_msg)

            any_error_in_turn = False

            # 2. Execute each tool call
            for i, tool_call in enumerate(reply.tool_calls):
                call_id = tool_call.call_id or f"call_{net_steps}_{i}"
                self._log(f"[Tool Call]: {tool_call.name}({tool_call.arguments})")

                result = self._execute_tool_call(tool_call)
                rendered_result = result.to_model()
                self._log(f"[Result]:\n{rendered_result}")

                step_tool_calls_rec.append(
                    {"name": tool_call.name, "arguments": tool_call.arguments, "call_id": call_id}
                )
                step_tool_results_rec.append(
                    {
                        "ok": result.ok,
                        "error_code": result.error_code,
                        "message": result.message,
                        "data": result.data,
                        "warnings": result.warnings,
                        "rendered": rendered_result,
                    }
                )

                # Feed result back to model
                net_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_call.name,
                        "content": rendered_result,
                    }
                )

                # Track failure & stuck detection
                if not result.ok:
                    any_error_in_turn = True
                    call_sig = (tool_call.name, json.dumps(tool_call.arguments, sort_keys=True))
                    failing_call_history.append(call_sig)

                    # Check consecutive identical failing calls
                    if len(failing_call_history) >= self.stuck_threshold:
                        recent = failing_call_history[-self.stuck_threshold :]
                        if all(sig == call_sig for sig in recent):
                            stuck_warning = (
                                f"STUCK LOOP DETECTED: You have attempted {tool_call.name}({tool_call.arguments}) "
                                f"{self.stuck_threshold} times in a row, and it continues to fail. "
                                f"Options: (1) Route in a different direction / detour, (2) Place a via with place_via() "
                                f"or switch_to_layer(1) to cross on the bottom layer, (3) Call abandon_route() if unroutable, "
                                f"or (4) Check DRC with check_drc() and rip_up conflicting nets."
                            )
                            self._log(f"\n[Stuck Detector]: {stuck_warning}\n")
                            net_messages.append({"role": "user", "content": stuck_warning})
                else:
                    # Successful call clears stuck history for this specific action
                    failing_call_history = []

                # Check if net is finished
                if tool_call.name == "finish_route" and result.ok:
                    routed = True
                    break

                if tool_call.name == "abandon_route" and result.ok:
                    failure_reason = "Model voluntarily abandoned route."
                    break

            # Record step transcript
            self.step_records.append(
                StepRecord(
                    step_index=curr_total,
                    net=net,
                    think=think_next,
                    reasoning=reply.reasoning,
                    text=reply.text,
                    tool_calls=step_tool_calls_rec,
                    tool_results=step_tool_results_rec,
                    timestamp=time.time(),
                )
            )

            if routed or failure_reason is not None:
                break

            # Latency lever: Think next turn only if this turn encountered an error
            think_next = any_error_in_turn

        # Teardown / budget handling
        if not routed:
            if net_steps >= self.max_steps_per_net:
                failure_reason = f"Per-net step budget ({self.max_steps_per_net}) exhausted."
            elif (total_steps_so_far + net_steps) >= self.max_total_steps:
                failure_reason = f"Total board step budget ({self.max_total_steps}) exhausted."

            # Ensure no dangling route session
            if self.tools._active_net is not None:
                self._log(f"[Teardown]: Abandoning incomplete route on {net!r}")
                self.tools.abandon_route()

        state = "ROUTED" if routed else "FAILED"
        self._render_and_save(net_idx, net, state)

        elapsed = time.time() - start_time
        self._log(f"\nNet {net!r} finished: {state} in {net_steps} steps ({elapsed:.2f}s). Reason: {failure_reason or 'Success'}")

        summary = NetRunSummary(
            net=net,
            routed=routed,
            steps=net_steps,
            failure_reason=failure_reason,
            wall_time_s=elapsed,
        )
        return summary, net_steps

    def run(self, net_order: Sequence[str] | None = None) -> RunSummary:
        """Executes full board routing across all nets."""
        start_time = time.time()
        self._log("================================================================================")
        self._log(" STARTING ROUTERV3 LLM AGENT ROUTING RUN")
        self._log("================================================================================")

        # 1. Inspect board
        board_info = self.tools.get_board_info()
        self._log(f"Board Info:\n{board_info.to_model()}")

        # 2. Determine nets
        if net_order is not None:
            nets_to_route = list(net_order)
        else:
            # Query pads from bridge
            pads = self.tools.bridge.net_pads() if hasattr(self.tools.bridge, "net_pads") else []
            nets_to_route = sorted({p.net for p in pads if p.net})

        self._log(f"Target nets ({len(nets_to_route)}): {nets_to_route}")

        net_summaries: list[NetRunSummary] = []
        total_steps = 0

        # Initial board render
        self._render_and_save(0, "initial", "unrouted")

        for idx, net in enumerate(nets_to_route, start=1):
            if total_steps >= self.max_total_steps:
                self._log(f"\nTotal step budget ({self.max_total_steps}) reached. Skipping remaining nets.")
                net_summaries.append(
                    NetRunSummary(
                        net=net,
                        routed=False,
                        steps=0,
                        failure_reason="Total step budget exhausted before start.",
                    )
                )
                continue

            summary, steps_used = self.route_net(net=net, net_idx=idx, total_steps_so_far=total_steps)
            net_summaries.append(summary)
            total_steps += steps_used

        # 3. Final DRC Check
        drc_result = self.tools.check_drc()
        violations = self.tools.bridge.run_drc() if hasattr(self.tools.bridge, "run_drc") else []
        drc_error_count = sum(1 for v in violations if getattr(v, "severity", "") == "error")

        wall_time = time.time() - start_time
        routed_count = sum(1 for s in net_summaries if s.routed)

        # 4. Save JSON transcript
        transcript_path: str | None = None
        if self.output_dir is not None:
            transcript_file = self.output_dir / "transcript.json"
            transcript_data = {
                "run_info": {
                    "total_nets": len(nets_to_route),
                    "routed_nets": routed_count,
                    "total_steps": total_steps,
                    "drc_errors": drc_error_count,
                    "wall_time_s": wall_time,
                },
                "steps": [dataclasses.asdict(r) for r in self.step_records],
                "net_summaries": [dataclasses.asdict(s) for s in net_summaries],
            }
            transcript_file.write_text(json.dumps(transcript_data, indent=2), encoding="utf-8")
            transcript_path = str(transcript_file)

        # 5. Print summary
        self._print_summary(len(nets_to_route), routed_count, total_steps, drc_error_count, wall_time, net_summaries)

        return RunSummary(
            total_nets=len(nets_to_route),
            routed_nets=routed_count,
            attempted_nets=len(net_summaries),
            total_steps=total_steps,
            drc_errors=drc_error_count,
            wall_time_s=wall_time,
            net_summaries=net_summaries,
            transcript_path=transcript_path,
        )

    def _print_summary(
        self,
        total_nets: int,
        routed_nets: int,
        total_steps: int,
        drc_errors: int,
        wall_time: float,
        summaries: list[NetRunSummary],
    ) -> None:
        self._log("\n" + "=" * 80)
        self._log(" ROUTERV3 AGENT RUN SUMMARY")
        self._log("=" * 80)
        self._log(f"  Nets Routed / Total: {routed_nets} / {total_nets} ({100.0 * routed_nets / max(total_nets, 1):.1f}%)")
        self._log(f"  Total Tool Steps:    {total_steps}")
        self._log(f"  DRC Error Count:     {drc_errors}")
        self._log(f"  Total Wall Clock:    {wall_time:.2f}s")
        self._log("-" * 80)
        self._log(f"  {'Net':<16} | {'Status':<8} | {'Steps':<6} | {'Time (s)':<8} | Failure Reason")
        self._log("-" * 80)
        for s in summaries:
            status = "ROUTED" if s.routed else "FAILED"
            reason = s.failure_reason or "-"
            self._log(f"  {s.net:<16} | {status:<8} | {s.steps:<6} | {s.wall_time_s:<8.2f} | {reason}")
        self._log("=" * 80 + "\n")


def run_routing_agent(
    tools: RouterTools,
    backend: Backend,
    net_order: Sequence[str] | None = None,
    max_steps_per_net: int = 30,
    max_total_steps: int = 300,
    stuck_threshold: int = 3,
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> RunSummary:
    """Convenience entrypoint for running the LLM routing agent loop."""
    agent = RoutingAgent(
        tools=tools,
        backend=backend,
        max_steps_per_net=max_steps_per_net,
        max_total_steps=max_total_steps,
        stuck_threshold=stuck_threshold,
        output_dir=output_dir,
        verbose=verbose,
    )
    return agent.run(net_order=net_order)
