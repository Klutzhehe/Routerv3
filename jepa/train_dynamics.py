"""Train the JEPA-style action-conditioned dynamics predictor on transitions
logged by jepa/collect_transitions.py.

Combined objective (jepa/README.md point 3, REQUIRED not optional):
  (a) predictive_loss  -- match the frozen encoder's real next-state embedding
      (see jepa/dynamics_model.py's predictive_loss: BYOL-style normalized
      loss with an implicit stop-gradient, since the target embeddings come
      straight from logged data with no graph attached).
  (b) aux distance loss -- decode the PREDICTED embedding into the real,
      verifiable normalized geodesic distance-to-target and match it against
      ground truth. A collapsed/constant z_hat cannot also satisfy this
      per-sample-varying target, which is what makes collapse self-defeating
      here rather than merely discouraged.

Collapse is NOT visible in (a) alone -- a collapsed z_hat can drive (a) to
looking excellent (e.g. if the frozen target embeddings for nearby-in-time
states are themselves close together, "predict near the batch mean" can look
like a good cosine match) while carrying none of the per-action information
the fast selector will need. This script computes and PROMINENTLY PRINTS a
separate diagnostic block every epoch instead of trusting the loss curve --
see `compute_diagnostics` below. Read those numbers before trusting any run
of this script, not just whether the loss went down.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from scripts.train_ai_router import STAGE_CONFIG, action_dim_for_stage
from jepa.dynamics_model import DynamicsPredictor, DistanceHead, predictive_loss

GRID_SIZE = 256  # matches PCBRouterEnv's grid_size default every script in this repo assumes
MAX_GEO_DIST = (GRID_SIZE ** 2 + GRID_SIZE ** 2) ** 0.5  # mirrors environment.py Channel 7's normalization


def load_shards(data_dir: str) -> Dict[str, np.ndarray]:
    paths = sorted(Path(data_dir).glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.npz files found under {data_dir} -- run jepa/collect_transitions.py first.")
    cols: Dict[str, List[np.ndarray]] = {}
    for p in paths:
        with np.load(p) as npz:
            for k in npz.files:
                cols.setdefault(k, []).append(npz[k])
    return {k: np.concatenate(v, axis=0) for k, v in cols.items()}


def episode_split(episode_idx: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split by whole EPISODE, not by individual transition -- consecutive
    steps within one episode are highly correlated, so a transition-level
    random split would leak near-duplicates across train/val and make val
    metrics look better than the predictor actually generalizes."""
    unique_eps = np.unique(episode_idx)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_eps)
    n_val = max(1, int(len(unique_eps) * val_frac))
    val_eps = set(unique_eps[:n_val].tolist())
    val_mask = np.isin(episode_idx, list(val_eps))
    return ~val_mask, val_mask


@torch.no_grad()
def compute_diagnostics(
    predictor: DynamicsPredictor,
    dist_head: DistanceHead,
    z_t: torch.Tensor,
    actions: torch.Tensor,
    z_next: torch.Tensor,
    dist_t_norm: torch.Tensor,
    dist_next_norm: torch.Tensor,
    train_dist_mean: float,
    action_dim: int,
    probe_n: int,
) -> Dict[str, float]:
    """Everything needed to catch a collapsed predictor even when the main
    predictive loss looks fine. Every number here is computed on the VAL
    split by the caller -- these are meant to answer "does this generalize",
    not "did it memorize the train set"."""
    z_hat = predictor(z_t, actions)
    dist_pred = dist_head(z_hat)

    diag: Dict[str, float] = {}
    # --- embedding spread (dataset-level collapse check) ---
    diag["z_hat_std"] = z_hat.std(dim=0).mean().item()
    diag["z_target_std"] = z_next.std(dim=0).mean().item()

    # --- aux head accuracy vs two naive floors ---
    mae_model = (dist_pred - dist_next_norm).abs().mean().item()
    mae_baseline_mean = (torch.full_like(dist_next_norm, train_dist_mean) - dist_next_norm).abs().mean().item()
    mae_baseline_nochange = (dist_t_norm - dist_next_norm).abs().mean().item()
    diag["aux_mae_model"] = mae_model
    diag["aux_mae_baseline_mean"] = mae_baseline_mean
    diag["aux_mae_baseline_nochange"] = mae_baseline_nochange

    # --- action sensitivity probe: does the SAME state produce different
    # predictions for different actions, at a scale comparable to how much
    # predictions differ ACROSS states? A predictor that learned to ignore
    # its action input can still show healthy z_hat_std purely from varying
    # z_t across the batch -- this probe is the one that actually catches
    # that specific failure mode. ---
    n = min(probe_n, z_t.shape[0])
    probe_z = z_t[:n]  # (n, d)
    all_actions = torch.arange(action_dim, device=z_t.device)
    # (n, action_dim, d): predictor output for every action, per probed state
    rep_z = probe_z.unsqueeze(1).expand(n, action_dim, -1).reshape(n * action_dim, -1)
    rep_a = all_actions.unsqueeze(0).expand(n, action_dim).reshape(-1)
    z_hat_all = predictor(rep_z, rep_a).reshape(n, action_dim, -1)
    action_sensitivity = z_hat_all.std(dim=1).mean().item()  # spread ACROSS actions, same state
    state_sensitivity = z_hat_all.mean(dim=1).std(dim=0).mean().item()  # spread ACROSS states, action-averaged
    diag["action_sensitivity"] = action_sensitivity
    diag["state_sensitivity"] = state_sensitivity
    diag["action_to_state_sensitivity_ratio"] = action_sensitivity / max(1e-8, state_sensitivity)

    return diag


def print_diagnostics(epoch: int, diag: Dict[str, float]) -> None:
    print(f"  [diagnostics epoch {epoch}]")
    print(f"    embedding std   -- z_hat: {diag['z_hat_std']:.4f}  z_target: {diag['z_target_std']:.4f}"
          f"  {'*** POSSIBLE COLLAPSE (z_hat_std near 0) ***' if diag['z_hat_std'] < 1e-3 else ''}")
    print(f"    aux dist MAE    -- model: {diag['aux_mae_model']:.4f}  "
          f"baseline(mean): {diag['aux_mae_baseline_mean']:.4f}  "
          f"baseline(no-change): {diag['aux_mae_baseline_nochange']:.4f}"
          f"  {'*** MODEL NOT BEATING BASELINES ***' if diag['aux_mae_model'] >= min(diag['aux_mae_baseline_mean'], diag['aux_mae_baseline_nochange']) * 0.98 else ''}")
    print(f"    action vs state sensitivity -- action: {diag['action_sensitivity']:.4f}  "
          f"state: {diag['state_sensitivity']:.4f}  ratio: {diag['action_to_state_sensitivity_ratio']:.4f}"
          f"  {'*** PREDICTOR MAY BE IGNORING ACTION INPUT ***' if diag['action_to_state_sensitivity_ratio'] < 0.05 else ''}")
    sys.stdout.flush()


def train(
    data_dir: str,
    stage: int,
    checkpoint_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    aux_weight: float,
    hidden: int,
    val_frac: float,
    include_terminal: bool,
    probe_n: int,
    device_str: str,
    seed: int,
) -> None:
    stage_cfg = STAGE_CONFIG[stage]
    enable_layer_via = stage_cfg["enable_layer_via"]
    action_dim = action_dim_for_stage(stage_cfg)

    # Print BEFORE the (potentially slow, e.g. over a Drive-FUSE mount)
    # shard load, not after -- a silent multi-minute gap here looks
    # identical to "nothing is happening" from outside the process.
    print(f"Loading shards from {data_dir} ...")
    sys.stdout.flush()
    cols = load_shards(data_dir)
    mask = np.ones(len(cols["action"]), dtype=bool)
    if not include_terminal:
        mask &= ~cols["done"]
    for k in cols:
        cols[k] = cols[k][mask]
    n_total = len(cols["action"])
    print(f"Loaded {n_total} transitions ({'including' if include_terminal else 'excluding'} terminal steps) from {data_dir}")
    sys.stdout.flush()

    dist_t_norm_np = np.clip(cols["dist_t"] / MAX_GEO_DIST, 0.0, 1.0).astype(np.float32)
    dist_next_norm_np = np.clip(cols["dist_next"] / MAX_GEO_DIST, 0.0, 1.0).astype(np.float32)

    train_mask, val_mask = episode_split(cols["episode_idx"], val_frac, seed)
    print(f"Episode split: {len(np.unique(cols['episode_idx'][train_mask]))} train episodes, "
          f"{len(np.unique(cols['episode_idx'][val_mask]))} val episodes "
          f"({train_mask.sum()} train / {val_mask.sum()} val transitions)")
    sys.stdout.flush()

    device = torch.device(device_str)

    def to_device(arr_dict, m):
        return {
            "z_t": torch.as_tensor(arr_dict["z_t"][m], dtype=torch.float32, device=device),
            "action": torch.as_tensor(arr_dict["action"][m], dtype=torch.long, device=device),
            "z_next": torch.as_tensor(arr_dict["z_next"][m], dtype=torch.float32, device=device),
            "dist_t_norm": torch.as_tensor(dist_t_norm_np[m], dtype=torch.float32, device=device),
            "dist_next_norm": torch.as_tensor(dist_next_norm_np[m], dtype=torch.float32, device=device),
        }

    train_data = to_device(cols, train_mask)
    val_data = to_device(cols, val_mask)
    train_dist_mean = train_data["dist_next_norm"].mean().item()

    d_model = train_data["z_t"].shape[-1]
    predictor = DynamicsPredictor(d_model=d_model, enable_layer_via=enable_layer_via, hidden=hidden).to(device)
    dist_head = DistanceHead(d_model=d_model).to(device)
    optimizer = optim.Adam(list(predictor.parameters()) + list(dist_head.parameters()), lr=lr)

    chk_path = Path(checkpoint_dir)
    chk_path.mkdir(parents=True, exist_ok=True)

    history: Dict[str, List[float]] = {
        "epoch": [], "train_loss_pred": [], "train_loss_aux": [],
        "val_loss_pred": [], "val_loss_aux": [], "val_aux_mae_model": [],
        "val_aux_mae_baseline_nochange": [], "action_sensitivity": [], "state_sensitivity": [],
    }
    best_val_loss = float("inf")
    n_train = train_data["z_t"].shape[0]
    rng = np.random.RandomState(seed)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        predictor.train()
        dist_head.train()
        perm = rng.permutation(n_train)
        total_loss_pred, total_loss_aux, n_batches = 0.0, 0.0, 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            zb = train_data["z_t"][idx]
            ab = train_data["action"][idx]
            znb = train_data["z_next"][idx]
            dnb = train_data["dist_next_norm"][idx]

            z_hat = predictor(zb, ab)
            loss_pred = predictive_loss(z_hat, znb.detach())
            dist_pred = dist_head(z_hat)
            loss_aux = F.mse_loss(dist_pred, dnb)
            loss = loss_pred + aux_weight * loss_aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_pred += loss_pred.item()
            total_loss_aux += loss_aux.item()
            n_batches += 1

        train_loss_pred = total_loss_pred / max(1, n_batches)
        train_loss_aux = total_loss_aux / max(1, n_batches)

        predictor.eval()
        dist_head.eval()
        with torch.no_grad():
            z_hat_val = predictor(val_data["z_t"], val_data["action"])
            val_loss_pred = predictive_loss(z_hat_val, val_data["z_next"]).item()
            val_loss_aux = F.mse_loss(dist_head(z_hat_val), val_data["dist_next_norm"]).item()

        diag = compute_diagnostics(
            predictor, dist_head,
            val_data["z_t"], val_data["action"], val_data["z_next"],
            val_data["dist_t_norm"], val_data["dist_next_norm"],
            train_dist_mean, action_dim, probe_n,
        )

        elapsed = time.time() - start_time
        print(f"[epoch {epoch}/{epochs}] train: pred_loss={train_loss_pred:.4f} aux_loss={train_loss_aux:.4f} | "
              f"val: pred_loss={val_loss_pred:.4f} aux_loss={val_loss_aux:.4f} | elapsed={elapsed:.0f}s")
        sys.stdout.flush()
        print_diagnostics(epoch, diag)

        history["epoch"].append(epoch)
        history["train_loss_pred"].append(train_loss_pred)
        history["train_loss_aux"].append(train_loss_aux)
        history["val_loss_pred"].append(val_loss_pred)
        history["val_loss_aux"].append(val_loss_aux)
        history["val_aux_mae_model"].append(diag["aux_mae_model"])
        history["val_aux_mae_baseline_nochange"].append(diag["aux_mae_baseline_nochange"])
        history["action_sensitivity"].append(diag["action_sensitivity"])
        history["state_sensitivity"].append(diag["state_sensitivity"])

        val_total = val_loss_pred + aux_weight * val_loss_aux
        save_payload = {
            "epoch": epoch,
            "predictor_state_dict": predictor.state_dict(),
            "dist_head_state_dict": dist_head.state_dict(),
            "d_model": d_model,
            "enable_layer_via": enable_layer_via,
            "action_dim": action_dim,
            "history": history,
            "last_diagnostics": diag,
        }
        torch.save(save_payload, chk_path / "jepa_dynamics_latest.pt")
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(save_payload, chk_path / "jepa_dynamics_best.pt")

        plot_curves(history, chk_path / "jepa_training_curves.png")
        with open(chk_path / "jepa_diagnostics_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("=" * 70)
    print(f"Done. Best val loss (pred + {aux_weight}*aux): {best_val_loss:.4f}")
    print(f"Checkpoints + diagnostics written to {chk_path}")
    print("=" * 70)
    sys.stdout.flush()


def plot_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    if len(history["epoch"]) < 2:
        return
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=110)
    fig.patch.set_facecolor("#101216")
    for ax in axes:
        ax.set_facecolor("#181b22")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    axes[0].plot(epochs, history["train_loss_pred"], color="#00ffcc", label="train")
    axes[0].plot(epochs, history["val_loss_pred"], color="#ff4444", label="val")
    axes[0].set_title("Predictive loss (2-2cos)", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[0].legend(facecolor="#181b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    axes[0].grid(True, alpha=0.15)

    axes[1].plot(epochs, history["val_aux_mae_model"], color="#ffaa00", label="model")
    axes[1].plot(epochs, history["val_aux_mae_baseline_nochange"], color="#aa00ff", label="no-change baseline")
    axes[1].set_title("Aux distance MAE (val) vs baseline", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[1].legend(facecolor="#181b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    axes[1].grid(True, alpha=0.15)

    axes[2].plot(epochs, history["action_sensitivity"], color="#00bbff", label="action sensitivity")
    axes[2].plot(epochs, history["state_sensitivity"], color="#ff0055", label="state sensitivity")
    axes[2].set_title("Collapse diagnostic: action vs state sensitivity", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[2].legend(facecolor="#181b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
    axes[2].grid(True, alpha=0.15)

    plt.tight_layout()
    fig.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the JEPA-style latent dynamics predictor")
    parser.add_argument("--data-dir", type=str, default="/content/drive/MyDrive/pcb_ai_router/jepa_data")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--checkpoint-dir", type=str, default="/content/drive/MyDrive/pcb_ai_router/jepa_checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--aux-weight", type=float, default=1.0, help="Weight of the auxiliary distance-decoding loss relative to the predictive embedding-matching loss. This is the REQUIRED collapse anchor -- do not set to 0.")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden width of the dynamics predictor MLP.")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Fraction of EPISODES (not transitions) held out for validation.")
    parser.add_argument("--include-terminal", action="store_true", help="Include transitions where the net finished this step (z_next would be a blank default observation's embedding -- off by default since that's not a meaningful predictive target).")
    parser.add_argument("--probe-n", type=int, default=64, help="How many val states to use for the action-sensitivity collapse probe each epoch.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_str}")

    train(
        data_dir=args.data_dir,
        stage=args.stage,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        aux_weight=args.aux_weight,
        hidden=args.hidden,
        val_frac=args.val_frac,
        include_terminal=args.include_terminal,
        probe_n=args.probe_n,
        device_str=device_str,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
