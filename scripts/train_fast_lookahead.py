"""Supervised training for FastDistancePredictor (models/fast_lookahead.py)
against (head_token, target_token, action, future distance) triples written
by scripts/collect_fast_lookahead_data.py.

Ordinary regression -- plain MSE against a real, known, varying label (no
self-supervised objective, no EMA target encoder, no stop-gradient). Unlike
the JEPA attempt's embedding-prediction loss, there is no trivial "predict a
constant" shortcut that minimizes this: the label varies board-to-board and
action-to-action, so a predictor that isn't learning anything shows up
directly as a val MAE close to the "predict the training-set mean" baseline,
no separate collapse diagnostic needed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.fast_lookahead import FastDistancePredictor, MAX_GEO_DIST
from scripts.train_ai_router import STAGE_CONFIG, action_dim_for_stage


def load_shards(data_dir: str) -> Dict[str, np.ndarray]:
    paths = sorted(Path(data_dir).glob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"No shard_*.npz files found in {data_dir}")
    cols: Dict[str, list] = {"head_token": [], "target_token": [], "action": [], "label_dist": [], "episode_idx": []}
    for p in paths:
        d = np.load(p)
        for k in cols:
            cols[k].append(d[k])
    return {k: np.concatenate(v, axis=0) for k, v in cols.items()}


def episode_split(episode_idx: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split by whole EPISODE, not individual example -- consecutive labels
    from the same episode are correlated (same board, overlapping horizon
    windows), so a per-example split would leak train information into val."""
    unique_eps = np.unique(episode_idx)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_eps)
    n_val = max(1, int(len(unique_eps) * val_frac))
    val_eps = set(unique_eps[:n_val].tolist())
    val_mask = np.array([e in val_eps for e in episode_idx])
    return ~val_mask, val_mask


def train(
    data_dir: str,
    stage: int,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    seed: int,
    use_target_token: bool,
    hidden_dim: int,
    out_path: str,
    device_str: str,
) -> None:
    stage_cfg = STAGE_CONFIG[stage]
    action_dim = action_dim_for_stage(stage_cfg)
    device = torch.device(device_str)

    print(f"Loading shards from {data_dir} ...")
    sys.stdout.flush()
    data = load_shards(data_dir)
    n = len(data["action"])
    d_model = data["head_token"].shape[-1]
    print(f"Loaded {n} labeled examples, d_model={d_model}, action_dim={action_dim}")
    sys.stdout.flush()

    label_norm = np.clip(data["label_dist"] / MAX_GEO_DIST, 0.0, 1.0).astype(np.float32)
    train_mask, val_mask = episode_split(data["episode_idx"], val_frac, seed)
    print(f"Episode split: {int(train_mask.sum())} train / {int(val_mask.sum())} val examples")

    def to_tensors(mask):
        return (
            torch.as_tensor(data["head_token"][mask], device=device),
            torch.as_tensor(data["target_token"][mask], device=device),
            torch.as_tensor(data["action"][mask], dtype=torch.long, device=device),
            torch.as_tensor(label_norm[mask], device=device),
        )

    head_tr, target_tr, action_tr, label_tr = to_tensors(train_mask)
    head_val, target_val, action_val, label_val = to_tensors(val_mask)

    baseline_mae = float(np.abs(label_norm[train_mask].mean() - label_norm[val_mask]).mean())
    print(f"Baseline (predict train-set mean) val MAE: {baseline_mae:.4f}")
    sys.stdout.flush()

    predictor = FastDistancePredictor(
        d_model=d_model, action_dim=action_dim, use_target_token=use_target_token, hidden_dim=hidden_dim,
    ).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=lr)

    rng = np.random.RandomState(seed)
    n_train = head_tr.shape[0]
    best_val_mae = float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        predictor.train()
        perm = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            pred = predictor(head_tr[idx], action_tr[idx], target_tr[idx] if use_target_token else None)
            loss = F.mse_loss(pred, label_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        predictor.eval()
        with torch.no_grad():
            val_pred = predictor(head_val, action_val, target_val if use_target_token else None)
            val_mae = (val_pred - label_val).abs().mean().item()
        best_val_mae = min(best_val_mae, val_mae)
        elapsed = time.time() - start_time
        print(f"[epoch {epoch + 1}/{epochs}] train_mse={epoch_loss / max(1, n_batches):.5f} "
              f"val_mae={val_mae:.4f} (baseline={baseline_mae:.4f}) elapsed={elapsed:.0f}s")
        sys.stdout.flush()

    beat = best_val_mae < baseline_mae * 0.9
    print("=" * 70)
    print(f"Best val MAE: {best_val_mae:.4f} vs baseline {baseline_mae:.4f} "
          f"-- {'*** BEATS BASELINE ***' if beat else 'does NOT meaningfully beat baseline'}")
    print("=" * 70)
    sys.stdout.flush()

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": predictor.state_dict(),
        "d_model": d_model,
        "action_dim": action_dim,
        "use_target_token": use_target_token,
        "hidden_dim": hidden_dim,
        "stage": stage,
        "best_val_mae": best_val_mae,
        "baseline_mae": baseline_mae,
    }, out_path)
    print(f"Saved predictor checkpoint to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train FastDistancePredictor on collected (head_token, target_token, action, future distance) triples"
    )
    parser.add_argument("--data-dir", type=str, default="/content/drive/MyDrive/pcb_ai_router/fast_lookahead_data")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-target-token", action="store_true", help="Ablation: drop target_token, use head_token alone.")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out", type=str, default="/content/drive/MyDrive/pcb_ai_router/checkpoints_fast_lookahead/fast_lookahead_latest.pt")
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_str}")
    sys.stdout.flush()

    train(
        data_dir=args.data_dir,
        stage=args.stage,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        use_target_token=not args.no_target_token,
        hidden_dim=args.hidden_dim,
        out_path=args.out,
        device_str=device_str,
    )


if __name__ == "__main__":
    main()
