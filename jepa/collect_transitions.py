"""Log (state, action, next_state) transitions from a trained checkpoint's
rollouts, for training the JEPA-style dynamics predictor (see jepa/README.md).

Data-pipeline design decision (jepa/README.md point 5): rather than storing
raw (10, 256, 256) observations (~2.6MB each -- prohibitive at the transition
counts a dynamics dataset needs), this script embeds each observation through
the ALREADY-TRAINED, FROZEN PCBEncoder at collection time and stores only the
256-d (or whatever --stage's checkpoint used) embedding vectors. This also
means there is no online encoder left to fine-tune/collapse in
jepa/train_dynamics.py -- only the new dynamics predictor is trained there.
See jepa/README.md for the full rationale and what would change if the
encoder ever needs fine-tuning instead.

Does NOT modify pcbworld/environment.py, models/router_policy.py, or
scripts/train_ai_router.py -- only imports from the latter (STAGE_CONFIG,
action_dim_for_stage) to stay in sync with the stage definitions instead of
re-declaring a second copy that drifts.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from pcbworld.environment import PCBRouterEnv
from models.router_policy import PCBRouterNet
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
    output_dir: str,
    shard_size: int,
    device_str: str,
) -> None:
    stage_cfg = STAGE_CONFIG[stage]
    action_dim = action_dim_for_stage(stage_cfg)
    enable_layer_via = stage_cfg["enable_layer_via"]

    device = torch.device(device_str)
    model = PCBRouterNet(in_channels=10, action_dim=action_dim, d_model=256, num_transformer_layers=2, num_heads=4)
    chk = torch.load(checkpoint, map_location=device_str, weights_only=False)
    model.load_state_dict(chk["model_state_dict"])
    model.to(device)
    model.eval()
    model.requires_grad_(False)

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

    # Column buffers, flushed to a shard file every `shard_size` transitions.
    buf: Dict[str, List] = {
        "episode_idx": [], "z_t": [], "action": [], "z_next": [],
        "dist_t": [], "dist_next": [], "done": [], "completed": [], "failed": [],
    }
    shard_idx = 0
    total_transitions = 0
    total_completed_episodes = 0
    action_hist = np.zeros(action_dim, dtype=np.int64)

    def flush():
        nonlocal buf, shard_idx
        if not buf["action"]:
            return
        path = out_dir / f"shard_{shard_idx:05d}.npz"
        np.savez_compressed(
            path,
            episode_idx=np.asarray(buf["episode_idx"], dtype=np.int32),
            z_t=np.stack(buf["z_t"]).astype(np.float32),
            action=np.asarray(buf["action"], dtype=np.int64),
            z_next=np.stack(buf["z_next"]).astype(np.float32),
            dist_t=np.asarray(buf["dist_t"], dtype=np.float32),
            dist_next=np.asarray(buf["dist_next"], dtype=np.float32),
            done=np.asarray(buf["done"], dtype=np.bool_),
            completed=np.asarray(buf["completed"], dtype=np.bool_),
            failed=np.asarray(buf["failed"], dtype=np.bool_),
        )
        print(f"  wrote {path} ({len(buf['action'])} transitions)")
        shard_idx += 1
        for k in buf:
            buf[k] = []

    start_time = time.time()
    for ep in range(num_episodes):
        # Seed block deliberately disjoint from the canonical eval block
        # (9000-9999) and the known-hard-seed list -- training the fast
        # selector on the exact boards it must later be VALIDATED against
        # would make that validation meaningless.
        seed = seed_offset + ep
        obs_np, info = env.reset(seed=seed)
        done = False
        forbidden_by_net: Dict[int, set] = {}

        while not done:
            idx = env.current_net_idx
            state = env.net_states[idx]
            forbidden = forbidden_by_net.get(idx, set())

            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                pcb_latent, _ = model.encoder(obs_t)
                action_logits = model.policy_head(pcb_latent)
            z_t = pcb_latent.squeeze(0).cpu().numpy()
            dist_t_val = env._geo_dist_at(state.geodesic_cache, state.head_x, state.head_y)

            logits = action_logits.squeeze(0)
            ranked = [a for a in torch.argsort(logits, descending=True).tolist() if a not in forbidden]
            if not ranked:
                ranked = torch.argsort(logits, descending=True).tolist()
            candidates = ranked[:top_k]

            # Behavior policy: mostly greedy top-1 (matches real deployment
            # trajectories), sometimes one of the OTHER top-k candidates --
            # exactly the set jepa_lookahead will later query at inference,
            # so the dataset covers what it needs to score without wasting
            # capacity on actions that will never be candidates.
            if len(candidates) > 1 and random.random() < explore_eps:
                action = random.choice(candidates[1:])
            else:
                action = candidates[0]
            action_hist[action] += 1

            prev_head = (state.head_x, state.head_y)
            obs_next_np, reward, term, trunc, step_info = env.step(action)
            done = term or trunc
            new_head = step_info["acted_head_pos"][:2]
            if new_head == prev_head:
                forbidden_by_net[idx] = forbidden_by_net.get(idx, set()) | {action}
            else:
                forbidden_by_net[idx] = set()

            acted_net_id = step_info["acted_net_id"]
            completed = acted_net_id in env.completed_nets
            failed = acted_net_id in env.failed_nets
            # Only reasons about ONE net's own trajectory, same limitation
            # lookahead_select_action documents -- if round-robin rotated
            # control to a different net (num_nets > 1 only), this step's
            # "next state" isn't this net's, so skip logging it rather than
            # attribute someone else's observation to this net's transition.
            # NOTE (num_nets > 1 only -- irrelevant to stage 2's num_nets=1):
            # this also silently drops a transition where THIS net just
            # completed/failed while OTHER nets are still active (net_done
            # pops it from active_order without setting env-level `done`,
            # so current_net_idx moves to a different net next). Harmless
            # today since done always coincides with this net's own outcome
            # when it's the only net, but revisit before relying on this for
            # stage 3+.
            rotated_away = (not done) and (env.current_net_idx != idx)
            if rotated_away:
                obs_np = obs_next_np
                continue

            if done:
                # _build_observation() with current_net_idx=None returns a
                # blank/default render, not a meaningful "next state" -- keep
                # dist_next (still computable from state, which step() has
                # already mutated in place) but do not store a z_next to
                # train the predictive loss against. train_dynamics.py
                # filters these out by default (see --include-terminal).
                z_next = z_t
            else:
                obs_next_t = torch.as_tensor(obs_next_np, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    pcb_latent_next, _ = model.encoder(obs_next_t)
                z_next = pcb_latent_next.squeeze(0).cpu().numpy()

            dist_next_val = env._geo_dist_at(state.geodesic_cache, state.head_x, state.head_y)

            buf["episode_idx"].append(ep)
            buf["z_t"].append(z_t)
            buf["action"].append(action)
            buf["z_next"].append(z_next)
            buf["dist_t"].append(dist_t_val)
            buf["dist_next"].append(dist_next_val)
            buf["done"].append(done)
            buf["completed"].append(completed)
            buf["failed"].append(failed)
            total_transitions += 1

            if len(buf["action"]) >= shard_size:
                flush()

            obs_np = obs_next_np

        if step_info.get("completed_nets", 0) > 0:
            total_completed_episodes += 1

        if (ep + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"[{ep + 1}/{num_episodes}] transitions={total_transitions} "
                f"completed_episodes={total_completed_episodes}/{ep + 1} "
                f"elapsed={elapsed:.0f}s"
            )

    flush()
    print("=" * 70)
    print(f"Done. {total_transitions} transitions across {num_episodes} episodes "
          f"({total_completed_episodes} completed) written to {out_dir}")
    print(f"Action histogram (top 10 by count): "
          f"{sorted(enumerate(action_hist.tolist()), key=lambda kv: -kv[1])[:10]}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Collect (state, action, next_state) transitions for JEPA dynamics training")
    parser.add_argument("--checkpoint", type=str, default="/content/drive/MyDrive/pcb_ai_router/checkpoints_stage2_v7/single_net_router_latest.pt", help="Trained PCBRouterNet checkpoint to embed observations and act with (frozen -- see module docstring)")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed-offset", type=int, default=100000, help="First board seed. Must stay disjoint from the canonical eval block (9000-9999) and the known-hard-seed list (docs/... see the session context) -- collecting on those would contaminate the later correctness validation.")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-net-restarts", type=int, default=2, help="Matches the setting the stage-2 100%% benchmark used.")
    parser.add_argument("--max-no-progress-steps", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=4, help="Behavior-policy candidate pool size -- match this to the --lookahead-top-k the fast selector will eventually use.")
    parser.add_argument("--explore-eps", type=float, default=0.3, help="Probability of taking a non-greedy top-k action instead of the greedy top-1, so the dataset covers what the fast selector will query, not just the on-policy trajectory.")
    parser.add_argument("--output-dir", type=str, default="/content/drive/MyDrive/pcb_ai_router/jepa_data")
    parser.add_argument("--shard-size", type=int, default=5000, help="Transitions per .npz shard file.")
    parser.add_argument("--seed-py", type=int, default=0, help="Python `random` seed for the exploration coin flips, for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed_py)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_str}")

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
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        device_str=device_str,
    )


if __name__ == "__main__":
    main()
