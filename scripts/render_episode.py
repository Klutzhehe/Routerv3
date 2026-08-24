"""Render one evaluation episode from a trained checkpoint.

The canonical way to visualize what a policy actually does, so "why did it
stop at step 32" doesn't have to mean digging through an ad-hoc notebook cell
that capped the step budget independently of training and evaluate_policy
(both of which use max_steps_per_net=120) -- here it's one explicit flag.

Also settles sampling-vs-mode questions cheaply: --stochastic renders one
sampled rollout (what training's collect loop does, dist.sample()); the
default is deterministic action selection, the SAME select_deterministic_action
evaluate_policy uses (argmax, but skipping actions already rejected at the
current position -- plain argmax repeats an identical rejected move forever).
If a route zigzags under --stochastic but not by default, that was sampling
noise from rendering the wrong action mode, not a policy bug.
"""

from __future__ import annotations

import argparse
import os

import torch

from pcbworld.environment import PCBRouterEnv
from pcbworld.renderer import render_grid_board
from models.router_policy import PCBRouterNet, select_deterministic_action, lookahead_select_action
from models.fast_lookahead import fast_lookahead_select_action
from models.analytic_lookahead import analytic_lookahead_select_action
from scripts.train_ai_router import STAGE_CONFIG, action_dim_for_stage, _load_fast_lookahead_predictor


def main():
    parser = argparse.ArgumentParser(description="Render one PCBRouterEnv episode from a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint saved by train_single_net_policy")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4], help="Must match the stage the checkpoint was trained on (action_dim/env config)")
    parser.add_argument("--seed", type=int, default=0, help="Board seed -- vary this to look at different boards")
    parser.add_argument("--max-steps", type=int, default=120, help="Per-net step budget for this render. Training and evaluate_policy both use 120 -- pass something lower only to deliberately reproduce a truncated view.")
    parser.add_argument("--max-net-restarts", type=int, default=0, help="Restart the whole net from its source pad this many times on a jam, instead of failing immediately. 0 (default) matches evaluate_policy's default.")
    parser.add_argument("--max-no-progress-steps", type=int, default=20, help="Give up (or restart) after this many consecutive steps without the cost-to-go improving -- catches oscillation between two valid cells, which never trips the collision-based jam check.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions like training's rollout collection does, instead of the deterministic argmax evaluate_policy uses.")
    parser.add_argument("--raw", action="store_true", help="Draw the raw rasterized copper (every stepped cell) instead of the simplified straight-segment trace. Use this to compare against the cleaned-up default.")
    parser.add_argument("--out", default="episode_render.png")
    parser.add_argument("--verbose", action="store_true", help="Print a step-by-step trace (position, decoded action, distance-to-target, reward, status) instead of just the final image -- the same evidence format used to diagnose every jam/trap bug in this project so far.")
    parser.add_argument("--lookahead", action="store_true", help="Use lookahead_select_action instead of plain deterministic argmax: simulates --lookahead-horizon steps ahead for the top --lookahead-top-k candidate actions (continuing greedily with the same policy) and commits to whichever gets closest to the target, instead of the single best immediate action. Meant for targeted investigation of boards where the plain policy gets stuck oscillating -- materially slower per step, not for bulk benchmarking. Incompatible with --stochastic.")
    parser.add_argument("--lookahead-top-k", type=int, default=4, help="How many of the policy's top candidate actions to simulate forward under --lookahead.")
    parser.add_argument("--lookahead-horizon", type=int, default=4, help="How many steps to simulate forward per candidate under --lookahead.")
    parser.add_argument("--fast-lookahead", action="store_true", help="Use fast_lookahead_select_action instead of plain deterministic argmax -- see models/fast_lookahead.py. Scores each candidate action with a small trained MLP against the already-computed encoder output instead of --lookahead's real simulation. Requires --fast-lookahead-checkpoint. Incompatible with --stochastic and --lookahead.")
    parser.add_argument("--fast-lookahead-checkpoint", type=str, default=None, help="Path to a FastDistancePredictor checkpoint saved by scripts/train_fast_lookahead.py.")
    parser.add_argument("--fast-lookahead-top-k", type=int, default=4, help="How many of the policy's top candidate actions to score under --fast-lookahead.")
    parser.add_argument("--analytic-lookahead", action="store_true", help="Use analytic_lookahead_select_action instead of plain deterministic argmax -- see models/analytic_lookahead.py. Scores each candidate by replaying the environment's own deterministic movement/collision math against the already-computed geodesic distance field -- no learned predictor, no simulated env.step(). Superseded --fast-lookahead after real-checkpoint validation showed distance-to-target isn't decodable from this encoder's embeddings at all. Incompatible with --stochastic, --lookahead, and --fast-lookahead.")
    parser.add_argument("--analytic-lookahead-top-k", type=int, default=4, help="How many of the policy's top candidate actions to score under --analytic-lookahead.")
    args = parser.parse_args()
    selector_flags = [args.lookahead, args.fast_lookahead, args.analytic_lookahead]
    if sum(bool(f) for f in selector_flags) > 1:
        raise SystemExit("--lookahead, --fast-lookahead, and --analytic-lookahead are mutually exclusive -- pick one action selector.")
    if any(selector_flags) and args.stochastic:
        raise SystemExit("--stochastic is incompatible with any lookahead flag -- lookahead is a deterministic search over the policy's own distribution.")
    if args.fast_lookahead and not args.fast_lookahead_checkpoint:
        raise SystemExit("--fast-lookahead requires --fast-lookahead-checkpoint")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    stage_cfg = STAGE_CONFIG[args.stage]
    action_dim = action_dim_for_stage(stage_cfg)

    model = PCBRouterNet(in_channels=10, action_dim=action_dim, d_model=256, num_transformer_layers=2, num_heads=4)
    chk = torch.load(args.checkpoint, map_location=device_str, weights_only=False)
    model.load_state_dict(chk["model_state_dict"])
    model.to(device_str)
    model.eval()

    fast_lookahead_predictor = None
    if args.fast_lookahead:
        fast_lookahead_predictor = _load_fast_lookahead_predictor(args.fast_lookahead_checkpoint, device_str)

    env = PCBRouterEnv(
        grid_size=256,
        max_steps_per_net=args.max_steps,
        max_net_restarts=args.max_net_restarts,
        max_no_progress_steps=args.max_no_progress_steps,
        snap_radius=6,
        **stage_cfg,
    )
    obs_np, info = env.reset(seed=args.seed)

    done = False
    # Round-robin means a DIFFERENT net can be current_net_idx on every
    # step() call, so retry-avoidance has to be tracked per net -- a
    # forbidden set keyed by whichever net just acted, not one shared set
    # that silently mixes different nets' rejected actions together.
    forbidden_by_net: dict[int, set[int]] = {}
    step_num = 0
    if args.verbose:
        print(f"{'Step':>5} | {'Net':>3} | {'Pos':>12} | {'Action':>6} (dir,dist) | {'Forbid':>6} | {'ToTarget':>8} | {'Reward':>9} | Status")
    while not done:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device_str).unsqueeze(0)
        acting_idx = env.current_net_idx
        acting_state = env.net_states[acting_idx]
        prev_head = (env.head_x, env.head_y)
        prev_completed = len(env.completed_nets)
        prev_failed = len(env.failed_nets)
        prev_restarts = acting_state.restart_count
        forbidden = forbidden_by_net.get(acting_idx, set())
        if args.lookahead:
            action = lookahead_select_action(
                model, env, obs_np, device_str, forbidden,
                top_k=args.lookahead_top_k, horizon=args.lookahead_horizon,
            )
        elif args.fast_lookahead:
            action = fast_lookahead_select_action(
                model, fast_lookahead_predictor, env, obs_np, device_str, forbidden,
                top_k=args.fast_lookahead_top_k,
            )
        elif args.analytic_lookahead:
            action = analytic_lookahead_select_action(
                model, env, obs_np, device_str, forbidden,
                top_k=args.analytic_lookahead_top_k,
            )
        else:
            with torch.no_grad():
                dist, _ = model(obs_t)
                action = int(dist.sample().item()) if args.stochastic else select_deterministic_action(dist, forbidden)
        dir_idx, dist_idx, _, _ = env.decode_action(action)
        obs_np, reward, term, trunc, info = env.step(action)
        done = term or trunc
        new_head = info["acted_head_pos"][:2]
        if not args.stochastic:
            if new_head == prev_head:
                forbidden_by_net[acting_idx] = forbidden_by_net.get(acting_idx, set()) | {action}
            else:
                forbidden_by_net[acting_idx] = set()
        if args.verbose:
            if len(env.completed_nets) > prev_completed:
                status = "CONNECTED"
            elif acting_state.restart_count > prev_restarts:
                status = "RESTARTED"
            elif len(env.failed_nets) > prev_failed:
                status = "FAILED (timeout/jammed-out)"
            elif new_head == prev_head:
                status = "REJECTED (collision)"
            else:
                status = "GROW"
            dist_to_target = env._geo_dist_at(acting_state.geodesic_cache, new_head[0], new_head[1])
            forbidden_count = len(forbidden_by_net.get(acting_idx, set())) if not args.stochastic else 0
            print(f"{step_num:5d} | {info['acted_net_id']:3d} | {str(new_head):>12} | "
                  f"{action:3d} ({dir_idx},{dist_idx}) | {forbidden_count:6d} | {dist_to_target:8.1f} | "
                  f"{reward:+9.1f} | {status}")
        step_num += 1

    print(f"completed_nets={info['completed_nets']}/{info['total_nets']}  "
          f"failed_nets={info['failed_nets']}  wirelength={info['total_wirelength']:.1f}  "
          f"vias={info['vias']}  mode={'stochastic' if args.stochastic else 'deterministic'}")

    pads = [p for net in env.board.nets for p in (net.source_pad, net.target_pad)]
    simplified_paths = None
    if not args.raw:
        simplified_paths = {nid: env.simplify_net_path(nid) for nid in env.completed_net_paths}
    render_grid_board(
        copper_grid=env.board.copper_grid,
        pads=pads,
        obstacles=env.board.obstacles,
        heads=[],
        save_path=args.out,
        title=f"Stage {args.stage} -- seed {args.seed} -- {'stochastic' if args.stochastic else 'deterministic'}",
        simplified_paths=simplified_paths,
    )
    print(f"saved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
