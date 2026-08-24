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
from models.router_policy import PCBRouterNet, select_deterministic_action, lookahead_select_action
from models.fast_lookahead import FastDistancePredictor, fast_lookahead_select_action
from models.analytic_lookahead import analytic_lookahead_select_action


def evaluate_policy(
    model: PCBRouterNet,
    num_eval_episodes: int = 50,
    grid_size: int = 256,
    num_nets: int = 1,
    num_obstacles: int = 0,
    enable_layer_via: bool = True,
    max_steps_per_net: int = 120,
    max_net_restarts: int = 0,
    max_no_progress_steps: int = 20,
    eval_seed_offset: int = 9000,
    device: str = "cpu",
    use_lookahead: bool = False,
    lookahead_top_k: int = 4,
    lookahead_horizon: int = 4,
    use_fast_lookahead: bool = False,
    fast_lookahead_predictor: "FastDistancePredictor | None" = None,
    fast_lookahead_top_k: int = 4,
    use_analytic_lookahead: bool = False,
    analytic_lookahead_top_k: int = 4,
) -> Dict[str, Any]:
    """Run deterministic evaluation of policy over test boards.

    eval_seed_offset picks WHICH num_eval_episodes-sized block of board
    seeds to test on. The default (9000) is the block every prior stage-2
    result in this project's history has been measured against -- a policy
    scoring well there has been checked against those specific boards, not
    against boards in general. Passing a different offset (e.g. 20000) is
    the cheap way to ask "does this generalize, or did it just get good at
    the 50 boards we keep testing on."

    use_lookahead swaps select_deterministic_action's single-step argmax
    for lookahead_select_action's shallow forward search -- see that
    function's docstring. Confirmed on render_episode.py traces to resolve
    oscillation-loop failures a plain argmax cannot escape; use_lookahead
    here is what checks whether that holds at benchmark scale, not just on
    the handful of seeds it was diagnosed against. Materially slower per
    step (~lookahead_top_k*lookahead_horizon extra env steps and forward
    passes per real decision) -- expect a full 1000-board sweep to take
    noticeably longer than the plain-argmax version.

    use_fast_lookahead swaps in fast_lookahead_select_action instead (see
    models/fast_lookahead.py) -- a small trained MLP predicts each candidate
    action's future distance-to-target instead of simulating it, so this
    should run close to plain-argmax speed. Not proven as reliable as
    use_lookahead yet; requires fast_lookahead_predictor.

    use_analytic_lookahead swaps in analytic_lookahead_select_action instead
    (see models/analytic_lookahead.py) -- scores each candidate by replaying
    the environment's own deterministic movement/collision math against the
    already-computed geodesic field, no learned predictor and no simulated
    env.step() involved, so this should also run close to plain-argmax
    speed. Superseded models/fast_lookahead.py after that approach's own
    real-checkpoint validation (jepa/README.md, models/fast_lookahead.py's
    docstring) found distance-to-target isn't decodable from ANY embedding
    this encoder produces -- this sidesteps that question by not decoding
    anything. Only one of use_lookahead / use_fast_lookahead /
    use_analytic_lookahead may be set at a time (the caller picks one).
    """
    model.eval()
    dev = torch.device(device)

    env = PCBRouterEnv(
        grid_size=grid_size,
        num_nets=num_nets,
        num_obstacles=num_obstacles,
        max_steps_per_net=max_steps_per_net,
        max_net_restarts=max_net_restarts,
        max_no_progress_steps=max_no_progress_steps,
        snap_radius=6,
        enable_layer_via=enable_layer_via,
    )

    total_nets = 0
    completed_nets = 0
    total_wirelength = 0.0
    straight_line_dist = 0.0
    total_vias = 0
    total_collisions = 0
    failed_seeds: List[int] = []

    for ep in range(num_eval_episodes):
        seed = eval_seed_offset + ep
        obs_np, info = env.reset(seed=seed)
        done = False

        # Compute straight line reference
        for net in env.board.nets:
            straight_line_dist += math.hypot(
                net.target_pad.x - net.source_pad.x,
                net.target_pad.y - net.source_pad.y,
            )

        # Rejected-action-avoidance state: reset whenever a net's head
        # actually moves. See select_deterministic_action's docstring --
        # plain argmax retries an identical rejected move forever. Keyed per
        # net (not one shared set) because round-robin makes a different net
        # current_net_idx on every step() call.
        forbidden_by_net: Dict[int, set] = {}
        while not done:
            acting_idx = env.current_net_idx
            prev_head = (env.head_x, env.head_y)
            forbidden = forbidden_by_net.get(acting_idx, set())
            if use_lookahead:
                action = lookahead_select_action(
                    model, env, obs_np, device, forbidden,
                    top_k=lookahead_top_k, horizon=lookahead_horizon,
                )
            elif use_fast_lookahead:
                action = fast_lookahead_select_action(
                    model, fast_lookahead_predictor, env, obs_np, device, forbidden,
                    top_k=fast_lookahead_top_k,
                )
            elif use_analytic_lookahead:
                action = analytic_lookahead_select_action(
                    model, env, obs_np, device, forbidden,
                    top_k=analytic_lookahead_top_k,
                )
            else:
                obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=dev).unsqueeze(0)
                with torch.no_grad():
                    dist, _ = model(obs_t)
                    action = select_deterministic_action(dist, forbidden)

            obs_np, reward, term, trunc, step_info = env.step(action)
            done = term or trunc
            new_head = step_info["acted_head_pos"][:2]
            if new_head == prev_head:
                forbidden_by_net[acting_idx] = forbidden_by_net.get(acting_idx, set()) | {action}
            else:
                forbidden_by_net[acting_idx] = set()

        completed_nets += step_info.get("completed_nets", 0)
        total_nets += step_info.get("total_nets", 0)
        total_vias += step_info.get("vias", 0)
        total_wirelength += step_info.get("total_wirelength", 0.0)
        if step_info.get("failed_nets", 0) > 0:
            total_collisions += 1
            failed_seeds.append(seed)

    completion_rate = (completed_nets / max(1, total_nets)) * 100.0
    wl_ratio = (total_wirelength / max(1e-3, straight_line_dist)) if straight_line_dist > 0 else 1.0

    print("=" * 70)
    print("                    AI PCB ROUTER EVALUATION REPORT")
    print("=" * 70)
    print(f"Board Seeds:                  {eval_seed_offset}-{eval_seed_offset + num_eval_episodes - 1}")
    print(f"Total Boards Evaluated:       {num_eval_episodes}")
    print(f"Total Nets Evaluated:         {total_nets}")
    print(f"Completion Rate:              {completion_rate:.2f}% ({completed_nets}/{total_nets} nets)")
    print(f"Wirelength Ratio:             {wl_ratio:.2f}x (actual / straight-line)")
    print(f"Total Vias Used:              {total_vias}")
    print(f"Collision / Failure Rate:     {(total_collisions / num_eval_episodes) * 100.0:.2f}%")
    if failed_seeds:
        print(f"Failed Board Seeds:           {failed_seeds}")
        print(f"  -> inspect with: python scripts/render_episode.py --seed {failed_seeds[0]} --checkpoint <path> --stage <N>")
    print("=" * 70)

    return {
        "completion_rate": completion_rate,
        "wirelength_ratio": wl_ratio,
        "total_vias": total_vias,
        "completed_nets": completed_nets,
        "total_nets": total_nets,
        "failed_seeds": failed_seeds,
    }
