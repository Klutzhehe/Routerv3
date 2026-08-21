"""First real LLM-driven routing run: QwenBackend + RoutingAgent against a
live pcbworld_pns_bridge.

Everything up to this point (verify_head_bindings.py,
verify_router_tools_live.py, verify_agent_loop_live.py) proved the ENGINE
side -- bindings correct, tool layer correct, loop orchestration correct --
all driven by ScriptedBackend, which never once involved a real model or
GPU. This script is the other half: does a real Qwen3 model, given the
real tool schemas and the real system prompt, actually drive the router.

Unverified going in -- stated plainly, not assumed away:
  - Whether Qwen3's chat template output, run through
    pcbworld/agent/backends.py's _parse_tool_calls_from_text(), actually
    produces tool calls RouterTools can execute. The parser matches Qwen's
    documented <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    format, but has never seen real model output -- only ScriptedBackend's
    synthetic replies.
  - Whether a small (4B, Q4) model is capable enough to make real
    progress, as opposed to calling tools invalidly until the stuck
    detector or step budget catches it. That is a legitimate possible
    outcome of this run, not a bug in the harness if it happens.
  - Wall-clock cost per net and per board -- unmeasured until this runs.
    docs/AI_ARCHITECTURE.md's "T_pns still unmeasured" note is about the
    router side; this is the model side of the same open question.

Deliberately small on this first run: few nets (pass a board generated
with --num-nets 3, not 6+), small step budgets, think starts False and
only escalates after an error (see RoutingAgent's own latency lever) --
a failure here should be cheap to see and iterate on, not expensive to
wait out.

Colab setup (before running this script):
    !pip install -q unsloth transformers accelerate bitsandbytes

Run:
    python3 Routerv3/pcbworld/data/generate_board.py board_small.kicad_pcb --num-nets 3 --seed 0
    python3 Routerv3/scripts/run_qwen_agent_live.py board_small.kicad_pcb \
        --output-dir /content/qwen_run --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
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


def run(
    board_path: str,
    output_dir: str | None,
    model_name: str,
    load_in_4bit: bool,
    max_steps_per_net: int,
    max_total_steps: int,
    bridge_dir: str | None = None,
) -> None:
    bridge_module = _load_bridge(bridge_dir)

    # Deferred, same convention as every env/script touching the bridge.
    from pcbworld.agent.backends import QwenBackend
    from pcbworld.agent.loop import run_routing_agent
    from pcbworld.agent.tools import RouterTools

    bridge = bridge_module.PNSBridge()
    assert bridge.load_board(board_path), f"load_board failed on {board_path}"

    pads = bridge.net_pads()
    nets = sorted({p.net for p in pads if p.net})
    assert nets, f"no routable nets on {board_path}"

    print(f"Board: {board_path}, nets: {nets}")
    print(f"Model: {model_name} (4bit={load_in_4bit})")
    print("Loading model -- first call downloads weights, can take a few minutes...")

    t0 = time.time()
    backend = QwenBackend(model_name=model_name, load_in_4bit=load_in_4bit)
    backend._lazy_init()  # pay the load cost here, not silently inside the first chat() call
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    tools = RouterTools(bridge, board_width_mm=50.0, board_height_mm=50.0, max_step_mm=8.0)

    t0 = time.time()
    summary = run_routing_agent(
        tools=tools,
        backend=backend,
        net_order=nets,
        max_steps_per_net=max_steps_per_net,
        max_total_steps=max_total_steps,
        stuck_threshold=3,
        output_dir=output_dir,
        verbose=True,
    )
    wall_time = time.time() - t0

    print("\n" + "=" * 60)
    print("FIRST REAL MODEL RUN COMPLETE")
    print(f"  nets routed:      {summary.routed_nets}/{summary.total_nets}")
    print(f"  total tool steps: {summary.total_steps}")
    print(f"  drc errors:       {summary.drc_errors}")
    print(f"  wall time:        {wall_time:.1f}s ({wall_time / max(1, len(nets)):.1f}s/net avg)")
    if output_dir:
        print(f"  transcript:       {summary.transcript_path}")
        print(f"  PNGs written to:  {output_dir}")
    print("=" * 60)
    print(
        "\nRead the transcript for HOW it got there, not just the numbers -- "
        "the reasoning traces and tool-call sequence are the actual point "
        "of this whole approach over the abandoned RL one."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board_path", help="Path to .kicad_pcb board file")
    parser.add_argument("--output-dir", default=None, help="Directory for PNG renders + transcript.json")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HF model id (Qwen3 family)")
    parser.add_argument("--no-4bit", action="store_true", help="Load in fp16 instead of 4-bit (needs more VRAM)")
    parser.add_argument("--max-steps-per-net", type=int, default=15, help="Small on purpose for a first run")
    parser.add_argument("--max-total-steps", type=int, default=60, help="Small on purpose for a first run")
    parser.add_argument("--bridge-dir", default=None, help="Optional path to bridge library directory")
    args = parser.parse_args()
    run(
        args.board_path,
        args.output_dir,
        args.model,
        not args.no_4bit,
        args.max_steps_per_net,
        args.max_total_steps,
        args.bridge_dir,
    )
