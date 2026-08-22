"""Randomized Single-Board Test & Visualizer for Routerv3 Policies.

Generates a fresh random board with specified seed and net configuration,
routes it using a trained policy checkpoint (with dynamic rip-up & reroute),
prints detailed metrics, and saves/renders a 3-panel layer-split PNG.

Usage:
    PYTHONPATH=".:build/pcbworld_bridge" python3 scripts/random_test_router.py \
        --checkpoint /path/to/policy_latest.pt \
        --seed 12345 \
        --num-nets 6 \
        --num-diff-pairs 2 \
        --num-length-groups 2 \
        --output-dir ./random_test_output \
        --show
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    from pcbworld.agents.line_policy import LineActorCritic, RunningMeanStd
except ImportError:
    torch = None

from pcbworld.env.line_route_env import LineRouteEnv
from pcbworld.viz.render_board import render_board_layers_split

MM = 1_000_000


def run_random_test(
    checkpoint_path: str | None = None,
    seed: int | None = None,
    num_nets: int = 6,
    num_diff_pairs: int = 2,
    num_length_groups: int = 2,
    length_group_size: int = 2,
    width_mm: float = 35.0,
    height_mm: float = 35.0,
    enable_ripup: bool = True,
    max_ripups: int = 6,
    output_dir: str = "/content/random_test_output",
    show_plot: bool = True,
) -> dict:
    if seed is None:
        seed = random.randint(1000, 999999)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    board_path = str(out_path / f"random_test_seed{seed}.kicad_pcb")
    routed_board_path = str(out_path / f"random_test_seed{seed}_routed.kicad_pcb")
    render_png_path = str(out_path / f"random_test_seed{seed}_layer_split.png")

    print("=" * 80)
    print(f"               ROUTERV3 RANDOM TEST RUN (SEED: {seed})")
    print("=" * 80)
    print(f"Board Dimensions:       {width_mm:.1f}mm x {height_mm:.1f}mm")
    print(f"Net Configuration:      {num_nets} Plain Nets | {num_diff_pairs} Diff Pairs | {num_length_groups} Length Groups (Size {length_group_size})")

    # Generate board
    gen_script = str(Path(__file__).resolve().parent.parent / "pcbworld" / "data" / "generate_board.py")
    cmd = [
        sys.executable,
        gen_script,
        board_path,
        "--num-nets", str(num_nets),
        "--num-diff-pairs", str(num_diff_pairs),
        "--num-length-matched-groups", str(num_length_groups),
        "--length-matched-group-size", str(length_group_size),
        "--width-mm", str(width_mm),
        "--height-mm", str(height_mm),
        "--seed", str(seed),
    ]
    subprocess.run(cmd, check=True)
    print(f"Generated Board:        {board_path}")

    # Load policy
    policy = None
    rms = None
    if checkpoint_path and torch is not None:
        chk = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        policy = LineActorCritic(action_dim=1)
        policy.load_state_dict(chk["policy_state_dict"])
        policy.eval()
        rms = RunningMeanStd()
        if chk.get("rms_mean") is not None:
            rms.mean = chk["rms_mean"]
            rms.var = chk["rms_var"]
            rms.count = chk["rms_count"]
        print(f"Loaded Policy:          {checkpoint_path}")
    else:
        print("Running Baseline:       Greedy Straight-Line (a = 0)")

    env = LineRouteEnv(
        board_path,
        enable_ripup=enable_ripup,
        max_ripups_per_episode=max_ripups,
        step_size_nm=800_000,
        snap_radius_nm=600_000,
        max_steps_per_net=100,
    )

    t0 = time.perf_counter()
    obs, info = env.reset()
    all_nets = list(env._nets)
    terminated = False
    steps = 0
    colliding_steps = 0

    print("\nStarting Routing...")
    while not terminated and steps < len(all_nets) * 120 * 2:
        if policy is not None:
            norm_obs = obs.copy()
            if rms is not None:
                norm_obs[:8] = rms.normalize(obs[:8])
            obs_t = torch.as_tensor(norm_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist, _ = policy.forward(obs_t)
                action = dist.mean.squeeze(0).numpy()
        else:
            action = np.array([0.0], dtype=np.float32)

        obs, reward, terminated, _, info = env.step(action)
        if info.get("collides", False):
            colliding_steps += 1
        steps += 1

    t1 = time.perf_counter()
    wall_clock_ms = (t1 - t0) * 1000.0

    completed = info["completed"]
    failed = info["failed"]
    total_completed = len(completed)
    total_nets = len(all_nets)
    completion_rate = (total_completed / max(1, total_nets)) * 100.0

    # Save routed board
    env.bridge.save_board(routed_board_path)
    print(f"Saved Routed Board:     {routed_board_path}")

    # Geometry & DRC
    geometry = env.bridge.get_board_geometry()
    pads = env.bridge.net_pads()
    total_routed_nm = sum(float(np.hypot(s.x2 - s.x1, s.y2 - s.y1)) for s in geometry.tracks)
    drc_violations = len(env.bridge.run_drc()) if hasattr(env.bridge, "run_drc") else 0

    print("\n" + "-" * 80)
    print(f"RESULTS FOR SEED {seed}:")
    print(f"  * Total Steps:         {steps} steps ({wall_clock_ms:.1f}ms total, {wall_clock_ms/max(1,steps):.3f}ms/step)")
    print(f"  * Completed Nets:      {total_completed}/{total_nets} ({completion_rate:.1f}%)")
    print(f"  * Nets Routed:         {completed}")
    print(f"  * Failed Nets:         {failed}")
    print(f"  * Dynamic Rip-Ups:     {info.get('ripup_count', 0)} ({info.get('ripups_performed', [])})")
    print(f"  * Colliding Steps:     {colliding_steps}/{steps} ({colliding_steps/max(1,steps)*100:.1f}%)")
    print(f"  * Total Routed Length: {total_routed_nm / MM:.2f} mm")
    print(f"  * DRC Violations:      {drc_violations}")
    print("-" * 80)

    # Render 3-panel layer split
    fig, axes = render_board_layers_split(
        geometry,
        net_pads=pads,
        save_path=render_png_path,
        dpi=150,
        title=f"Routerv3 Random Test (Seed {seed}) -- {total_completed}/{total_nets} Nets Routed",
    )
    print(f"Saved Layer Render:     {render_png_path}")

    if show_plot:
        plt.show()
    plt.close(fig)

    return {
        "seed": seed,
        "total_nets": total_nets,
        "completed_nets": total_completed,
        "completion_rate": completion_rate,
        "completed": completed,
        "failed": failed,
        "ripup_count": info.get("ripup_count", 0),
        "drc_violations": drc_violations,
        "routed_board_path": routed_board_path,
        "render_png_path": render_png_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Test and visualize router on a random board.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (generated if omitted)")
    parser.add_argument("--num-nets", type=int, default=6, help="Number of plain single-ended nets")
    parser.add_argument("--num-diff-pairs", type=int, default=2, help="Number of differential pairs")
    parser.add_argument("--num-length-groups", type=int, default=2, help="Number of length-matched groups")
    parser.add_argument("--length-group-size", type=int, default=2, help="Size of length-matched groups")
    parser.add_argument("--output-dir", type=str, default="./random_test_output", help="Output directory")
    parser.add_argument("--no-show", action="store_true", help="Don't display matplotlib window")
    args = parser.parse_args()

    run_random_test(
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        num_nets=args.num_nets,
        num_diff_pairs=args.num_diff_pairs,
        num_length_groups=args.num_length_groups,
        length_group_size=args.length_group_size,
        output_dir=args.output_dir,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
