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
from models.router_policy import PCBRouterNet, select_deterministic_action
from scripts.train_ai_router import STAGE_CONFIG, action_dim_for_stage


def main():
    parser = argparse.ArgumentParser(description="Render one PCBRouterEnv episode from a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint saved by train_single_net_policy")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4], help="Must match the stage the checkpoint was trained on (action_dim/env config)")
    parser.add_argument("--seed", type=int, default=0, help="Board seed -- vary this to look at different boards")
    parser.add_argument("--max-steps", type=int, default=120, help="Per-net step budget for this render. Training and evaluate_policy both use 120 -- pass something lower only to deliberately reproduce a truncated view.")
    parser.add_argument("--max-net-restarts", type=int, default=0, help="Restart the whole net from its source pad this many times on a jam, instead of failing immediately. 0 (default) matches evaluate_policy's default.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions like training's rollout collection does, instead of the deterministic argmax evaluate_policy uses.")
    parser.add_argument("--out", default="episode_render.png")
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    stage_cfg = STAGE_CONFIG[args.stage]
    action_dim = action_dim_for_stage(stage_cfg)

    model = PCBRouterNet(in_channels=10, action_dim=action_dim, d_model=256, num_transformer_layers=2, num_heads=4)
    chk = torch.load(args.checkpoint, map_location=device_str, weights_only=False)
    model.load_state_dict(chk["model_state_dict"])
    model.to(device_str)
    model.eval()

    env = PCBRouterEnv(
        grid_size=256,
        max_steps_per_net=args.max_steps,
        max_net_restarts=args.max_net_restarts,
        snap_radius=6,
        **stage_cfg,
    )
    obs_np, info = env.reset(seed=args.seed)

    done = False
    forbidden: set[int] = set()
    prev_head = env.head_x, env.head_y
    while not done:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device_str).unsqueeze(0)
        with torch.no_grad():
            dist, _ = model(obs_t)
            action = int(dist.sample().item()) if args.stochastic else select_deterministic_action(dist, forbidden)
        obs_np, reward, term, trunc, info = env.step(action)
        done = term or trunc
        new_head = env.head_x, env.head_y
        if not args.stochastic:
            forbidden = forbidden | {action} if new_head == prev_head else set()
        prev_head = new_head

    print(f"completed_nets={info['completed_nets']}/{info['total_nets']}  "
          f"failed_nets={info['failed_nets']}  wirelength={info['total_wirelength']:.1f}  "
          f"vias={info['vias']}  mode={'stochastic' if args.stochastic else 'deterministic'}")

    pads = [p for net in env.board.nets for p in (net.source_pad, net.target_pad)]
    render_grid_board(
        copper_grid=env.board.copper_grid,
        pads=pads,
        obstacles=env.board.obstacles,
        heads=[],
        save_path=args.out,
        title=f"Stage {args.stage} -- seed {args.seed} -- {'stochastic' if args.stochastic else 'deterministic'}",
    )
    print(f"saved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
