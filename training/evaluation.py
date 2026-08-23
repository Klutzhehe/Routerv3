"""Evaluation and Benchmark Suite for PCBRouterNet.

Computes:
- Net completion rate %
- Mean wirelength ratio (actual / straight-line Manhattan/Euclidean)
- Collision rate %
- Via counts & DRC legality
"""

from __future__ import annotations

import math
from typing import Dict, Any, List
import numpy as np
import torch

from pcbworld.environment import PCBRouterEnv
from models.router_policy import PCBRouterNet, select_deterministic_action


def evaluate_policy(
    model: PCBRouterNet,
    num_eval_episodes: int = 50,
    grid_size: int = 256,
    num_nets: int = 1,
    num_obstacles: int = 0,
    enable_layer_via: bool = True,
    max_steps_per_net: int = 120,
    max_net_restarts: int = 0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run deterministic evaluation of policy over test boards."""
    model.eval()
    dev = torch.device(device)

    env = PCBRouterEnv(
        grid_size=grid_size,
        num_nets=num_nets,
        num_obstacles=num_obstacles,
        max_steps_per_net=max_steps_per_net,
        max_net_restarts=max_net_restarts,
        snap_radius=6,
        enable_layer_via=enable_layer_via,
    )

    total_nets = 0
    completed_nets = 0
    total_wirelength = 0.0
    straight_line_dist = 0.0
    total_vias = 0
    total_collisions = 0

    for ep in range(num_eval_episodes):
        obs_np, info = env.reset(seed=9000 + ep)
        done = False

        # Compute straight line reference
        for net in env.board.nets:
            straight_line_dist += math.hypot(
                net.target_pad.x - net.source_pad.x,
                net.target_pad.y - net.source_pad.y,
            )

        # Rejected-action-avoidance state: reset whenever the head actually
        # moves. See select_deterministic_action's docstring -- plain argmax
        # retries an identical rejected move forever.
        forbidden: set[int] = set()
        prev_head = env.head_x, env.head_y
        while not done:
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=dev).unsqueeze(0)
            with torch.no_grad():
                dist, _ = model(obs_t)
                action = select_deterministic_action(dist, forbidden)

            obs_np, reward, term, trunc, step_info = env.step(action)
            done = term or trunc
            new_head = env.head_x, env.head_y
            if new_head == prev_head:
                forbidden.add(action)
            else:
                forbidden = set()
            prev_head = new_head

        completed_nets += step_info.get("completed_nets", 0)
        total_nets += step_info.get("total_nets", 0)
        total_vias += step_info.get("vias", 0)
        total_wirelength += step_info.get("total_wirelength", 0.0)
        if step_info.get("failed_nets", 0) > 0:
            total_collisions += 1

    completion_rate = (completed_nets / max(1, total_nets)) * 100.0
    wl_ratio = (total_wirelength / max(1e-3, straight_line_dist)) if straight_line_dist > 0 else 1.0

    print("=" * 70)
    print("                    AI PCB ROUTER EVALUATION REPORT")
    print("=" * 70)
    print(f"Total Boards Evaluated:       {num_eval_episodes}")
    print(f"Total Nets Evaluated:         {total_nets}")
    print(f"Completion Rate:              {completion_rate:.2f}% ({completed_nets}/{total_nets} nets)")
    print(f"Wirelength Ratio:             {wl_ratio:.2f}x (actual / straight-line)")
    print(f"Total Vias Used:              {total_vias}")
    print(f"Collision / Failure Rate:     {(total_collisions / num_eval_episodes) * 100.0:.2f}%")
    print("=" * 70)

    return {
        "completion_rate": completion_rate,
        "wirelength_ratio": wl_ratio,
        "total_vias": total_vias,
        "completed_nets": completed_nets,
        "total_nets": total_nets,
    }
