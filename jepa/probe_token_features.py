"""Small-scale, isolated validation of the "per-token features preserve
positional info the mean-pool destroys" hypothesis -- BEFORE committing to
rewriting collect_transitions.py / dynamics_model.py / train_dynamics.py
around a new representation. Same "verify cheaply before building"
discipline that already caught two previous ideas (input-scale fix,
value_head anchor) not working out before any large rewrite was based on
them.

Background: probe_distance_from_embedding.py and inspect_value_head.py both
showed the encoder's POOLED global_latent doesn't support decoding
distance-to-target -- not via a fresh linear/MLP probe, not even via the
network's own co-trained value_head (which turned out to be functionally
constant, 0.2014-0.2088 across the whole real dataset). The chosen next
hypothesis: mean-pooling over PCBEncoder's 256 spatial patch tokens
(16x16 grid, see models/pcb_encoder.py) is what destroys the local/spatial
facts (where's the head, where's the target) needed to compute distance --
the PRE-pool `encoded_tokens` might retain that.

Rather than store all 256 tokens (256x more storage than the pooled vector,
and a much bigger predictor input than needed), this extracts just the TWO
tokens that matter for distance-to-target: whichever patch-grid cell
contains the head's current position, and whichever contains the target
pad's position (both known exactly at collection time -- no learning needed
to find them). Tests three representations against the same ridge/MLP probe
methodology as probe_distance_from_embedding.py:
  1. global_latent (pooled) -- included as a same-run control, to confirm
     this smaller sample reproduces the known failure before trusting
     anything new measured on it.
  2. head_token + target_token concatenated (2 * d_model).
  3. head_token alone (d_model) -- isolates whether the head's own position
     carries enough signal by itself, or whether the target's token is
     load-bearing too.

Runs FRESH short episodes with the trained checkpoint's plain deterministic
policy (no lookahead, no action-exploration complexity needed -- this only
needs (representation, ground-truth distance) pairs, not full transitions
for dynamics training) -- does not read or write collect_transitions.py's
existing shards, and does not modify anything outside this file.
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from pcbworld.environment import PCBRouterEnv
from models.router_policy import PCBRouterNet, select_deterministic_action
from scripts.train_ai_router import STAGE_CONFIG, action_dim_for_stage
from jepa.probe_distance_from_embedding import ridge_probe, SimpleDistanceProbe
from jepa.train_dynamics import MAX_GEO_DIST, episode_split

import torch.optim as optim
import torch.nn.functional as F

PATCH_GRID = 16  # 256x256 CNN downsample factor -- see models/pcb_encoder.py's 4 stride-2 stages


def patch_token_index(x: int, y: int, grid_size: int) -> int:
    """Which of the 256 (16x16) patch-grid tokens covers pixel (x, y).

    Mirrors PCBEncoder's own flatten: features (B, C, 16, 16) -> flatten(2)
    merges (row, col) in row-major order, so token index = row*16 + col.
    """
    downsample = grid_size // PATCH_GRID
    col = min(PATCH_GRID - 1, max(0, x // downsample))
    row = min(PATCH_GRID - 1, max(0, y // downsample))
    return row * PATCH_GRID + col


def collect_probe_data(
    checkpoint: str,
    stage: int,
    num_episodes: int,
    seed_offset: int,
    max_steps: int,
    max_net_restarts: int,
    max_no_progress_steps: int,
    device_str: str,
) -> Dict[str, np.ndarray]:
    stage_cfg = STAGE_CONFIG[stage]
    action_dim = action_dim_for_stage(stage_cfg)
    device = torch.device(device_str)

    print(f"Loading checkpoint from {checkpoint} ...")
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

    pooled_list, head_tok_list, target_tok_list, dist_list, episode_idx_list = [], [], [], [], []

    print(f"Collecting {num_episodes} episodes (deterministic policy, no exploration -- "
          f"this probe only needs representative (state, distance) pairs) ...")
    for ep in range(num_episodes):
        seed = seed_offset + ep
        obs_np, info = env.reset(seed=seed)
        done = False
        forbidden_by_net: Dict[int, set] = {}
        while not done:
            idx = env.current_net_idx
            state = env.net_states[idx]
            net = env.board.nets[idx]
            forbidden = forbidden_by_net.get(idx, set())

            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                pooled, tokens = model.encoder(obs_t)  # pooled (1,d), tokens (1,256,d)
                action_logits = model.policy_head(pooled)
            action = select_deterministic_action(Categorical(logits=action_logits), forbidden)

            dist_t_val = env._geo_dist_at(state.geodesic_cache, state.head_x, state.head_y)
            h_idx = patch_token_index(state.head_x, state.head_y, env.grid_size)
            t_idx = patch_token_index(net.target_pad.x, net.target_pad.y, env.grid_size)

            pooled_list.append(pooled.squeeze(0).cpu().numpy())
            head_tok_list.append(tokens[0, h_idx].cpu().numpy())
            target_tok_list.append(tokens[0, t_idx].cpu().numpy())
            dist_list.append(dist_t_val)
            episode_idx_list.append(ep)

            prev_head = (state.head_x, state.head_y)
            obs_next_np, reward, term, trunc, step_info = env.step(action)
            done = term or trunc
            new_head = step_info["acted_head_pos"][:2]
            if new_head == prev_head:
                forbidden_by_net[idx] = forbidden_by_net.get(idx, set()) | {action}
            else:
                forbidden_by_net[idx] = set()
            obs_np = obs_next_np

        if (ep + 1) % max(1, num_episodes // 10) == 0:
            print(f"  [{ep + 1}/{num_episodes}] {len(dist_list)} timesteps collected so far")

    return {
        "pooled": np.stack(pooled_list).astype(np.float32),
        "head_token": np.stack(head_tok_list).astype(np.float32),
        "target_token": np.stack(target_tok_list).astype(np.float32),
        "dist_t": np.asarray(dist_list, dtype=np.float32),
        "episode_idx": np.asarray(episode_idx_list, dtype=np.int32),
    }


def probe_one(name: str, z: np.ndarray, y: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray,
              l2: float, mlp_epochs: int, batch_size: int, lr: float, seed: int, device_str: str) -> None:
    z_train, y_train = z[train_mask], y[train_mask]
    z_val, y_val = z[val_mask], y[val_mask]
    baseline_mae = float(np.mean(np.abs(y_train.mean() - y_val)))

    ridge_mae = ridge_probe(z_train, y_train, z_val, y_val, l2)

    device = torch.device(device_str)
    z_train_t = torch.as_tensor(z_train, device=device)
    y_train_t = torch.as_tensor(y_train, device=device)
    z_val_t = torch.as_tensor(z_val, device=device)
    y_val_t = torch.as_tensor(y_val, device=device)

    probe = SimpleDistanceProbe(d_model=z.shape[-1]).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    n_train = z_train_t.shape[0]
    best_mlp_mae = float("inf")
    for epoch in range(mlp_epochs):
        probe.train()
        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            pred = probe(z_train_t[idx])
            loss = F.mse_loss(pred, y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        probe.eval()
        with torch.no_grad():
            val_mae = (probe(z_val_t) - y_val_t).abs().mean().item()
        best_mlp_mae = min(best_mlp_mae, val_mae)

    beat = ridge_mae < baseline_mae * 0.9 or best_mlp_mae < baseline_mae * 0.9
    print(f"[{name:>14s}] dim={z.shape[-1]:4d}  baseline={baseline_mae:.4f}  "
          f"ridge={ridge_mae:.4f}  mlp_best={best_mlp_mae:.4f}"
          f"  {'*** BEATS BASELINE ***' if beat else 'does not meaningfully beat baseline'}")


def run(checkpoint: str, stage: int, num_episodes: int, seed_offset: int, max_steps: int,
        max_net_restarts: int, max_no_progress_steps: int, val_frac: float, l2: float,
        mlp_epochs: int, batch_size: int, lr: float, seed: int, device_str: str) -> None:
    data = collect_probe_data(checkpoint, stage, num_episodes, seed_offset, max_steps,
                               max_net_restarts, max_no_progress_steps, device_str)
    n = len(data["dist_t"])
    print(f"\nCollected {n} timesteps across {num_episodes} episodes.")

    dist_norm = np.clip(data["dist_t"] / MAX_GEO_DIST, 0.0, 1.0)
    train_mask, val_mask = episode_split(data["episode_idx"], val_frac, seed)
    print(f"Episode split: {train_mask.sum()} train / {val_mask.sum()} val timesteps\n")

    head_target = np.concatenate([data["head_token"], data["target_token"]], axis=-1)

    print("Probing each representation's ability to decode dist_t (lower is better; "
          "baseline = predict train-set mean):")
    probe_one("pooled (control)", data["pooled"], dist_norm, train_mask, val_mask, l2, mlp_epochs, batch_size, lr, seed, device_str)
    probe_one("head_token", data["head_token"], dist_norm, train_mask, val_mask, l2, mlp_epochs, batch_size, lr, seed, device_str)
    probe_one("target_token", data["target_token"], dist_norm, train_mask, val_mask, l2, mlp_epochs, batch_size, lr, seed, device_str)
    probe_one("head+target", head_target, dist_norm, train_mask, val_mask, l2, mlp_epochs, batch_size, lr, seed, device_str)

    print("\n" + "=" * 70)
    print("Read this as: did head_token / target_token / head+target clearly beat")
    print("'pooled (control)'? If yes, per-token features are worth rebuilding the JEPA")
    print("data pipeline around. If everything is equally unable to beat baseline, the")
    print("issue is not specific to pooling -- reconsider the exit-criteria options in")
    print("jepa/README.md instead of a bigger rewrite.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Validate whether per-token (pre-pool) features decode distance-to-target better than the pooled embedding")
    parser.add_argument("--checkpoint", type=str, default="/content/drive/MyDrive/pcb_ai_router/checkpoints_stage2_v7/single_net_router_latest.pt")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--num-episodes", type=int, default=200, help="Small-scale validation, not a full dataset -- default is deliberately modest.")
    parser.add_argument("--seed-offset", type=int, default=200000, help="Disjoint from both the eval block (9000-9999) and collect_transitions.py's own block (100000+).")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-net-restarts", type=int, default=2)
    parser.add_argument("--max-no-progress-steps", type=int, default=20)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_str}")

    run(
        checkpoint=args.checkpoint, stage=args.stage, num_episodes=args.num_episodes,
        seed_offset=args.seed_offset, max_steps=args.max_steps, max_net_restarts=args.max_net_restarts,
        max_no_progress_steps=args.max_no_progress_steps, val_frac=args.val_frac, l2=args.l2,
        mlp_epochs=args.mlp_epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        device_str=device_str,
    )


if __name__ == "__main__":
    main()
