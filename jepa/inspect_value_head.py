"""Before switching the auxiliary anchor to the trained value_head's output
(probe_distance_from_embedding.py's suggested next step, after it showed z_t
alone does not support decoding dist_t via a linear probe OR a shallow MLP --
on the FULL real dataset, ridge MAE 0.1261 vs baseline 0.1258, essentially
no signal recovered), verify that hypothesis cheaply rather than assume
"already trained" implies "the specific quantity we want a proxy for is
present." Same evidence-before-architecture-change discipline as everything
else in this project.

Needs the ORIGINAL checkpoint (for value_head's weights) but NOT the
environment or a new rollout -- z_t (already logged by
collect_transitions.py) is exactly value_head's expected input, so this
reuses the existing collected data with zero new environment steps.

Does not modify anything else in jepa/ or elsewhere.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from models.router_policy import PCBRouterNet
from jepa.train_dynamics import load_shards, MAX_GEO_DIST
from scripts.train_ai_router import STAGE_CONFIG, action_dim_for_stage


def run(checkpoint: str, stage: int, data_dir: str, device_str: str) -> None:
    stage_cfg = STAGE_CONFIG[stage]
    action_dim = action_dim_for_stage(stage_cfg)

    print(f"Loading checkpoint from {checkpoint} ...")
    model = PCBRouterNet(in_channels=10, action_dim=action_dim, d_model=256, num_transformer_layers=2, num_heads=4)
    chk = torch.load(checkpoint, map_location=device_str, weights_only=False)
    model.load_state_dict(chk["model_state_dict"])
    model.to(device_str)
    model.eval()
    model.requires_grad_(False)

    print(f"Loading shards from {data_dir} ...")
    cols = load_shards(data_dir)
    z_t = torch.as_tensor(cols["z_t"], dtype=torch.float32, device=device_str)
    dist_t_norm = np.clip(cols["dist_t"] / MAX_GEO_DIST, 0.0, 1.0).astype(np.float64)
    n = z_t.shape[0]
    print(f"Loaded {n} timesteps")

    with torch.no_grad():
        values = model.value_head(z_t).squeeze(-1).cpu().numpy().astype(np.float64)

    print(f"\nvalue_head(z_t) over {n} timesteps:")
    print(f"  std={values.std():.4f}  mean={values.mean():.4f}  min={values.min():.4f}  max={values.max():.4f}")

    corr = float(np.corrcoef(values, dist_t_norm)[0, 1])
    print(f"\nPearson correlation(value_head(z_t), dist_t_norm): {corr:.4f}"
          f"  (expected notably negative if value tracks progress-to-target: closer -> higher value)")

    print("\n" + "=" * 70)
    print("VERDICT:")
    if values.std() < 1e-4:
        print("  value_head's own output is essentially CONSTANT across the dataset -- this")
        print("  points at a broader representational issue (or a stale/mismatched checkpoint),")
        print("  not something specific to distance decoding. Do not use it as an anchor as-is.")
    elif abs(corr) > 0.3:
        print(f"  value_head(z_t) correlates meaningfully with distance (|r|={abs(corr):.2f}) --")
        print("  this IS a usable auxiliary anchor. Proceed with switching to it.")
    else:
        print(f"  value_head varies (std={values.std():.4f}) but barely correlates with distance")
        print(f"  (|r|={abs(corr):.2f}) -- it may be tracking something else (steps remaining,")
        print("  collision risk, etc.) rather than raw distance. Inspect further before committing")
        print("  to it as the distance-anchor replacement -- it may still be USEFUL, just not as")
        print("  a stand-in specifically for geodesic distance.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Check whether the trained value_head produces a distance-correlated signal from logged z_t")
    parser.add_argument("--checkpoint", type=str, default="/content/drive/MyDrive/pcb_ai_router/checkpoints_stage2_v7/single_net_router_latest.pt")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--data-dir", type=str, default="/content/drive/MyDrive/pcb_ai_router/jepa_data")
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_str}")

    run(checkpoint=args.checkpoint, stage=args.stage, data_dir=args.data_dir, device_str=device_str)


if __name__ == "__main__":
    main()
