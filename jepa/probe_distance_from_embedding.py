"""Diagnostic probe, isolated from the dynamics predictor entirely: does the
frozen encoder's globally-pooled embedding (z_t) support decoding
distance-to-target AT ALL?

Why this exists: two full 50-epoch real runs of train_dynamics.py both
showed the auxiliary DistanceHead's MAE tied to the "predict the dataset
mean" baseline, completely unmoved by an input-normalization fix that should
have helped if the problem were input scale. That result is also consistent
with a different, more fundamental possibility: PCBRouterNet's policy/value
heads were CO-TRAINED with the encoder end-to-end over thousands of PPO
updates, so whatever form the encoder represents distance-relevant
information in only has to be usable by heads trained jointly with it --
there's no guarantee that same representation is easily decodable by a
FRESH head trained in isolation for a few dozen epochs, especially since
mean-pooling over 256 patch tokens could dilute inherently local/spatial
facts (where's the head, where's the target) that a global average doesn't
obviously preserve.

This script removes the predictor, the action, and the "predict the NEXT
state" complexity entirely and asks the simplest possible version of the
question: given z_t alone, can anything recover dist_t (the CURRENT,
already-known distance at the same timestep z_t was computed from)? Two
probes, both diagnostic, neither meant to be a real component of the final
system:
  1. Ridge linear regression (closed-form, no training loop) -- the standard
     "linear probe" methodology from the representation-learning literature
     (used by SimCLR/BYOL/etc. to evaluate representation quality). If even
     a LINEAR map can't beat the naive baseline, the information isn't
     present in an easily-usable linear form.
  2. A small MLP with the same LayerNorm-input pattern as DistanceHead, for
     comparison -- if the MLP does much better than the linear probe, that's
     evidence of usable but nonlinearly-encoded information; if it does no
     better, that corroborates the linear probe's verdict.

Does NOT modify jepa/train_dynamics.py, jepa/dynamics_model.py, or anything
outside this file -- reuses load_shards/episode_split from train_dynamics.py
by import, changes nothing there.
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from jepa.train_dynamics import load_shards, episode_split, MAX_GEO_DIST


class SimpleDistanceProbe(nn.Module):
    """Same input-normalization pattern as jepa/dynamics_model.py's
    DistanceHead, kept deliberately separate (not imported) so this probe's
    result is not entangled with anything that changes over there."""

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_norm(z)).squeeze(-1)


def ridge_probe(z_train: np.ndarray, y_train: np.ndarray, z_val: np.ndarray, y_val: np.ndarray, l2: float) -> float:
    """Closed-form ridge regression: w = (X^T X + l2*I)^-1 X^T y, with a bias
    column appended. Returns val MAE."""
    n_train, d = z_train.shape
    x_train = np.concatenate([z_train, np.ones((n_train, 1), dtype=np.float32)], axis=1)
    x_val = np.concatenate([z_val, np.ones((z_val.shape[0], 1), dtype=np.float32)], axis=1)

    xtx = x_train.T @ x_train
    xtx += l2 * np.eye(d + 1, dtype=np.float32)
    xty = x_train.T @ y_train
    w = np.linalg.solve(xtx, xty)

    pred_val = x_val @ w
    return float(np.mean(np.abs(pred_val - y_val)))


def run(data_dir: str, val_frac: float, l2: float, mlp_epochs: int, batch_size: int, lr: float, seed: int, device_str: str) -> None:
    print(f"Loading shards from {data_dir} ...")
    cols = load_shards(data_dir)
    z_t = cols["z_t"].astype(np.float32)
    dist_t_norm = np.clip(cols["dist_t"] / MAX_GEO_DIST, 0.0, 1.0).astype(np.float32)
    n_total = z_t.shape[0]
    print(f"Loaded {n_total} timesteps (using ALL of them, including terminal steps -- "
          f"z_t and dist_t are both always meaningful regardless of what happens next)")

    train_mask, val_mask = episode_split(cols["episode_idx"], val_frac, seed)
    z_train, y_train = z_t[train_mask], dist_t_norm[train_mask]
    z_val, y_val = z_t[val_mask], dist_t_norm[val_mask]
    print(f"Episode split: {train_mask.sum()} train / {val_mask.sum()} val timesteps")

    baseline_mae = float(np.mean(np.abs(y_train.mean() - y_val)))
    print(f"\nBaseline (predict train-set mean dist): val MAE = {baseline_mae:.4f}")

    ridge_mae = ridge_probe(z_train, y_train, z_val, y_val, l2)
    print(f"Ridge linear probe (l2={l2}):            val MAE = {ridge_mae:.4f}"
          f"  {'*** BEATS BASELINE ***' if ridge_mae < baseline_mae * 0.9 else '*** DOES NOT MEANINGFULLY BEAT BASELINE ***'}")

    device = torch.device(device_str)
    z_train_t = torch.as_tensor(z_train, device=device)
    y_train_t = torch.as_tensor(y_train, device=device)
    z_val_t = torch.as_tensor(z_val, device=device)
    y_val_t = torch.as_tensor(y_val, device=device)

    probe = SimpleDistanceProbe(d_model=z_t.shape[-1]).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    n_train = z_train_t.shape[0]
    best_mlp_mae = float("inf")

    for epoch in range(1, mlp_epochs + 1):
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
        if epoch % max(1, mlp_epochs // 10) == 0 or epoch == mlp_epochs:
            print(f"  [mlp probe epoch {epoch}/{mlp_epochs}] val MAE = {val_mae:.4f} (best so far: {best_mlp_mae:.4f})")

    print(f"\nMLP probe best val MAE:                   {best_mlp_mae:.4f}"
          f"  {'*** BEATS BASELINE ***' if best_mlp_mae < baseline_mae * 0.9 else '*** DOES NOT MEANINGFULLY BEAT BASELINE ***'}")

    print("\n" + "=" * 70)
    print("VERDICT:")
    if ridge_mae < baseline_mae * 0.9 or best_mlp_mae < baseline_mae * 0.9:
        print("  Distance-to-target IS decodable from z_t alone -- the frozen encoder's")
        print("  embedding does carry usable signal. The dynamics predictor's failure to")
        print("  beat the same baseline points at the predictor/z_hat pathway specifically,")
        print("  not at the embedding space itself.")
    else:
        print("  Distance-to-target is NOT meaningfully decodable from z_t alone, even with")
        print("  a linear probe. This is a more fundamental limitation of the frozen,")
        print("  globally-pooled embedding than an input-scale or predictor-architecture")
        print("  issue -- geodesic distance may not be the right auxiliary anchor for THIS")
        print("  representation. Next step: try anchoring against the ALREADY-TRAINED")
        print("  value_head's output instead (co-trained with this exact encoder, so proven")
        print("  to be decodable from it), rather than a from-scratch distance regression.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Probe whether z_t alone supports decoding distance-to-target")
    parser.add_argument("--data-dir", type=str, default="/content/drive/MyDrive/pcb_ai_router/jepa_data")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1.0, help="Ridge regularization strength for the linear probe.")
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_str}")

    run(
        data_dir=args.data_dir,
        val_frac=args.val_frac,
        l2=args.l2,
        mlp_epochs=args.mlp_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device_str=device_str,
    )


if __name__ == "__main__":
    main()
