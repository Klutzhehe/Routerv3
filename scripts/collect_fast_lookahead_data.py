"""Data collection for the fast (learned) lookahead distance predictor
(see models/fast_lookahead.py).

Collects ordinary supervised (state, action, future distance) triples by
replaying rollouts from a trained checkpoint's exploring top-k behavior
policy (same behavior policy jepa/collect_transitions.py used, proven to
reach 997/1000 completions on the v7 checkpoint) and reading off the REAL
geodesic distance PCBRouterEnv._geo_dist_at already computes -- no
self-supervised target, no predictor-of-a-predictor: the label is a real
number that already existed in the environment.

For each of a net's own steps (round-robin skips other nets' steps in
between, kept per net_id so they don't mix), records the per-token
head/target features from BEFORE the action, the action taken, and -- once
--horizon further real steps of THAT SAME net are known -- the geodesic
distance --horizon steps later. If the net connects before --horizon steps
run out, the label is 0.0 for every remaining query (distance stays 0 once
connected -- a safe extrapolation). If the net fails or times out first,
those trailing queries are simply left unlabeled rather than guessed.

Does NOT modify pcbworld/environment.py, models/router_policy.py, or
scripts/train_ai_router.py -- only imports the latter's STAGE_CONFIG /
action_dim_for_stage to stay in sync with the stage definitions instead of
re-declaring a second copy that drifts.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from pcbworld.environment import PCBRouterEnv
from models.router_policy import PCBRouterNet
from models.fast_lookahead import extract_head_target_tokens
from scripts.train_ai_router import STAGE_CONFIG, action_dim_for_stage


def collect(
    checkpoint: str,
    stage: int,
    num_episodes: int,
    seed_offset: int,
    max_steps: int,
    max_net_restarts: int,
    max_no_progress_steps: int,
    top_k: int,
    explore_eps: float,
    horizon: int,
    output_dir: str,
    shard_size: int,
    device_str: str,
    log_every: int,
) -> None:
    stage_cfg = STAGE_CONFIG[stage]
    action_dim = action_dim_for_stage(stage_cfg)

    print(f"Loading checkpoint from {checkpoint} ...")
    sys.stdout.flush()
    device = torch.device(device_str)
    model = PCBRouterNet(in_channels=10, action_dim=action_dim, d_model=256, num_transformer_layers=2, num_heads=4)
    chk = torch.load(checkpoint, map_location=device_str, weights_only=False)
    model.load_state_dict(chk["model_state_dict"])
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    print(f"Checkpoint loaded onto {device_str}. Building environment (stage {stage}: {stage_cfg}) ...")
    sys.stdout.flush()

    env = PCBRouterEnv(
        grid_size=256,
        max_steps_per_net=max_steps,
        max_net_restarts=max_net_restarts,
        max_no_progress_steps=max_no_progress_steps,
        snap_radius=6,
        **stage_cfg,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting collection: {num_episodes} episodes, seeds {seed_offset}-{seed_offset + num_episodes - 1}, "
          f"horizon={horizon}, writing shards to {out_dir}")
    sys.stdout.flush()

    buf: Dict[str, List] = {"head_token": [], "target_token": [], "action": [], "label_dist": [], "episode_idx": []}
    shard_idx = 0
    total_labeled = 0
    total_completed_episodes = 0
    start_time = time.time()

    def flush():
        nonlocal buf, shard_idx
        if not buf["action"]:
            return
        path = out_dir / f"shard_{shard_idx:05d}.npz"
        np.savez_compressed(
            path,
            head_token=np.stack(buf["head_token"]).astype(np.float32),
            target_token=np.stack(buf["target_token"]).astype(np.float32),
            action=np.asarray(buf["action"], dtype=np.int64),
            label_dist=np.asarray(buf["label_dist"], dtype=np.float32),
            episode_idx=np.asarray(buf["episode_idx"], dtype=np.int32),
        )
        print(f"  wrote {path} ({len(buf['action'])} labeled examples)")
        sys.stdout.flush()
        shard_idx += 1
        for k in buf:
            buf[k] = []

    for ep in range(num_episodes):
        seed = seed_offset + ep
        obs_np, info = env.reset(seed=seed)
        done = False
        forbidden_by_net: Dict[int, set] = {}

        # Per-net_id trace of this episode's own steps, in order, so
        # round-robin (num_nets > 1) doesn't mix different nets' sequences.
        traces: Dict[int, Dict[str, list]] = {}

        while not done:
            idx = env.current_net_idx
            state = env.net_states[idx]
            net = env.board.nets[idx]
            forbidden = forbidden_by_net.get(idx, set())

            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                pooled, tokens = model.encoder(obs_t)
                action_logits = model.policy_head(pooled)
            logits = action_logits.squeeze(0)
            ranked = [a for a in torch.argsort(logits, descending=True).tolist() if a not in forbidden]
            if not ranked:
                ranked = torch.argsort(logits, descending=True).tolist()
            candidates = ranked[:top_k]

            # Behavior policy: mostly greedy top-1 (matches real deployment
            # trajectories), sometimes another top-k candidate -- covers the
            # (state, action) pairs the fast selector will actually query at
            # inference, not just the on-policy trajectory.
            if len(candidates) > 1 and random.random() < explore_eps:
                action = random.choice(candidates[1:])
            else:
                action = candidates[0]

            head_tok, target_tok = extract_head_target_tokens(
                tokens, state.head_x, state.head_y, net.target_pad.x, net.target_pad.y, env.grid_size,
            )

            trace = traces.setdefault(idx, {"head_token": [], "target_token": [], "action": [], "dist_after": []})
            trace["head_token"].append(head_tok.squeeze(0).cpu().numpy())
            trace["target_token"].append(target_tok.squeeze(0).cpu().numpy())
            trace["action"].append(action)

            prev_head = (state.head_x, state.head_y)
            obs_next_np, reward, term, trunc, step_info = env.step(action)
            done = term or trunc
            new_head = step_info["acted_head_pos"][:2]
            if new_head == prev_head:
                forbidden_by_net[idx] = forbidden_by_net.get(idx, set()) | {action}
            else:
                forbidden_by_net[idx] = set()

            # state is the SAME _NetState object env.step() just mutated in
            # place (matches jepa/collect_transitions.py's dist_next_val
            # pattern) -- valid even if this step restarted or completed the
            # net, since neither replaces the object, only mutates it.
            dist_after = env._geo_dist_at(state.geodesic_cache, state.head_x, state.head_y)
            trace["dist_after"].append(dist_after)

            obs_np = obs_next_np

        # Episode over -- every net is now either completed or failed, so
        # emit labeled examples per net's own trace.
        for idx, trace in traces.items():
            net_id = env.board.nets[idx].net_id
            net_completed = net_id in env.completed_nets
            dist_after = trace["dist_after"]
            L = len(dist_after)
            for i in range(L):
                if net_completed:
                    j = min(i + horizon - 1, L - 1)
                else:
                    j = i + horizon - 1
                    if j >= L:
                        continue
                buf["head_token"].append(trace["head_token"][i])
                buf["target_token"].append(trace["target_token"][i])
                buf["action"].append(trace["action"][i])
                buf["label_dist"].append(dist_after[j])
                buf["episode_idx"].append(ep)
                total_labeled += 1

        if len(buf["action"]) >= shard_size:
            flush()

        if step_info.get("completed_nets", 0) > 0:
            total_completed_episodes += 1

        if (ep + 1) % log_every == 0:
            elapsed = time.time() - start_time
            eta_sec = (num_episodes - (ep + 1)) * (elapsed / (ep + 1))
            print(f"[{ep + 1}/{num_episodes}] labeled_examples={total_labeled} "
                  f"completed_episodes={total_completed_episodes}/{ep + 1} "
                  f"elapsed={elapsed:.0f}s ETA={eta_sec:.0f}s")
            sys.stdout.flush()

    flush()
    print("=" * 70)
    print(f"Done. {total_labeled} labeled examples across {num_episodes} episodes "
          f"({total_completed_episodes} completed) written to {out_dir}")
    print("=" * 70)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Collect (head_token, target_token, action, future distance) triples for the fast lookahead predictor"
    )
    parser.add_argument("--checkpoint", type=str, default="/content/drive/MyDrive/pcb_ai_router/checkpoints_stage2_v7/single_net_router_latest.pt")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed-offset", type=int, default=300000, help="Disjoint from the canonical eval block (9000-9999), the known-hard-seed list, and the earlier jepa/ attempt's own seed blocks (100000+, 200000+).")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-net-restarts", type=int, default=2, help="Matches the setting the stage-2 100%% benchmark used.")
    parser.add_argument("--max-no-progress-steps", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=4, help="Behavior-policy candidate pool size -- match this to --fast-lookahead-top-k at inference.")
    parser.add_argument("--explore-eps", type=float, default=0.3, help="Probability of taking a non-greedy top-k action, so the dataset covers what the fast selector will query at decision time, not just the on-policy trajectory.")
    parser.add_argument("--horizon", type=int, default=4, help="How many real steps ahead each label looks -- matches lookahead_select_action's default horizon, so the two approaches answer comparable questions.")
    parser.add_argument("--output-dir", type=str, default="/content/drive/MyDrive/pcb_ai_router/fast_lookahead_data")
    parser.add_argument("--shard-size", type=int, default=5000, help="Labeled examples per .npz shard file.")
    parser.add_argument("--seed-py", type=int, default=0, help="Python `random` seed for the exploration coin flips, for reproducibility.")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed_py)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_str}")
    sys.stdout.flush()

    collect(
        checkpoint=args.checkpoint,
        stage=args.stage,
        num_episodes=args.num_episodes,
        seed_offset=args.seed_offset,
        max_steps=args.max_steps,
        max_net_restarts=args.max_net_restarts,
        max_no_progress_steps=args.max_no_progress_steps,
        top_k=args.top_k,
        explore_eps=args.explore_eps,
        horizon=args.horizon,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        device_str=device_str,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
