"""`FutureFieldPredictor` -- the model of what the board will become.

This is the module that answers the "look into the future" requirement, and it
is deliberately **not** the shape of bet that has already failed four times in
this repo.

What failed (`jepa/` x3, `models/fast_lookahead.py`):

    globally mean-pooled embedding  ->  one scalar (distance to target)

One gradient per sample, decoded from a representation that had already thrown
away the spatial structure the answer depended on. `docs/WORLD_MODEL_SPATIAL_DESIGN.md`
diagnosed it correctly: `encoded_tokens.mean(dim=1)` destroys the information
before anything downstream can use it.

What this does instead:

    spatial latent (B, D, L, h, w)  ->  three dense fields at the same shape

Three differences, each of which independently matters:

1. **No pooling on the path.** Input and output are spatially aligned; this is
   a segmentation problem, and a convolution can solve it locally.
2. **Dense supervision.** ~`L*h*w` labelled values per rollout instead of one.
   At `L=8, h=w=32` that is 8192 targets per board per episode.
3. **Free, on-policy labels.** The ground truth is the *terminal state of the
   rollout that was just collected*. There is no separate data-collection
   phase -- which is where `jepa/collect_transitions.py` spent most of its
   effort -- and the labels are always from the current policy's own
   distribution.

What is predicted, and why each is the right question:

* `final_occupancy` -- will this cell hold copper when the board is done?
  This is the direct answer to "am I about to take a cell that something else
  needs", which is the greedy router's central failure. `docs/HANDOVER.md`'s
  open question 2 (15/24 nets unreachable, detours rescued 0) is that failure.
* `contention` -- how many still-unrouted nets will want to cross here?
  Occupancy says *whether*; contention says *how badly*, which is what
  distinguishes a corridor worth avoiding from one worth taking.
* `jam_risk` -- will a net that needs this cell fail? The two above are about
  demand; this one is about *outcome*, and it is the only one that can learn
  "this region looks fine but always ends badly".

The policy consumes all three **gradient-detached**. The forecaster is trained
by its own supervised losses on completed episodes; the RL objective must not
be able to reshape it into something that merely makes the value function's job
easier, and its supervised gradients must not fight the policy's.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

#: How many channels the policy receives back from the forecaster.
FORECAST_CHANNELS = 3


@dataclass
class Forecast:
    """All fields are (B, 1, L, h, w) at the encoder's latent resolution."""

    final_occupancy: torch.Tensor  # logits
    contention: torch.Tensor       # log-rate for a Poisson observation
    jam_risk: torch.Tensor         # logits

    def as_channels(self) -> torch.Tensor:
        """(B, 3, L, h, w) in a bounded, policy-friendly parameterisation."""
        return torch.cat(
            [
                torch.sigmoid(self.final_occupancy),
                torch.tanh(self.contention),
                torch.sigmoid(self.jam_risk),
            ],
            dim=1,
        )


class FutureFieldPredictor(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv3d(latent_dim, hidden, (1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, (3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
        )
        self.occupancy = nn.Conv3d(hidden, 1, 1)
        self.contention = nn.Conv3d(hidden, 1, 1)
        self.jam = nn.Conv3d(hidden, 1, 1)

    def forward(self, latent: torch.Tensor) -> Forecast:
        h = self.trunk(latent)
        return Forecast(
            final_occupancy=self.occupancy(h),
            contention=self.contention(h),
            jam_risk=self.jam(h),
        )


# ---------------------------------------------------------------------------
# Targets and losses
# ---------------------------------------------------------------------------


def build_targets(
    final_occ: torch.Tensor,
    crossings: torch.Tensor,
    failures: torch.Tensor,
    latent_shape: tuple[int, int, int],
) -> dict[str, torch.Tensor]:
    """Downsample terminal-state statistics onto the latent grid.

    Parameters
    ----------
    final_occ : (B, L, H, W) bool -- copper at the end of the episode.
    crossings : (B, L, H, W) float -- how many nets crossed each cell.
    failures  : (B, L, H, W) float -- crossings attributable to nets that
        ultimately failed.
    latent_shape : (L, h, w) -- the encoder's output grid.

    Average-pool, not max-pool: at 1/4 resolution a latent cell covers 16
    lattice cells, and "how full is this region" is a more useful and far more
    learnable target than "does this region contain any copper at all", which
    saturates to 1 almost everywhere on a routed board.
    """
    L, h, w = latent_shape
    B = final_occ.shape[0]

    def pool(t: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool3d(t.float().unsqueeze(1), (L, h, w))

    return {
        "occupancy": pool(final_occ),
        "contention": pool(crossings),
        "jam": pool(failures),
    }


def forecast_losses(
    forecast: Forecast,
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Per-field losses. Kept separate so a single field failing is visible."""
    occ_t = targets["occupancy"].clamp(0.0, 1.0)
    occ_loss = F.binary_cross_entropy_with_logits(forecast.final_occupancy, occ_t)

    # Poisson NLL on a log-rate: contention is a count, and squared error on a
    # count treats "predicted 0 saw 3" and "predicted 30 saw 33" as equally bad.
    rate = forecast.contention.clamp(-8.0, 8.0)
    cont_loss = (rate.exp() - targets["contention"] * rate).mean()

    jam_t = targets["jam"].clamp(0.0, 1.0)
    jam_loss = F.binary_cross_entropy_with_logits(forecast.jam_risk, jam_t)

    return {"occupancy": occ_loss, "contention": cont_loss, "jam": jam_loss}


@torch.no_grad()
def forecast_gate(
    forecast: Forecast,
    targets: dict[str, torch.Tensor],
    demand_baseline: torch.Tensor,
) -> dict[str, float]:
    """**The gate.** Does the learned forecast beat drawing straight lines?

    `demand_baseline` is `world/generator.py::straight_line_demand`, pooled to
    the latent grid: every unrouted net contributes a rasterised straight line
    between its pads. It costs nothing, needs no training, and is a perfectly
    respectable estimate of where copper will end up.

    If the learned occupancy forecast cannot beat it, the forecaster has
    learned nothing worth carrying and **the latent-rollout stage in
    DESIGN.md section 5 does not start.** That gate exists specifically so this
    line of work cannot become negative result #5 through momentum -- four
    previous lookahead efforts in this repo ran well past the point where the
    evidence had already answered the question.

    Returns correlations and errors for both, so the comparison is a number in
    the training log rather than an impression.
    """
    pred = torch.sigmoid(forecast.final_occupancy)
    tgt = targets["occupancy"].clamp(0.0, 1.0)
    base = demand_baseline
    base = base / base.amax().clamp_min(1e-6)

    def corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.reshape(-1) - a.mean()
        b = b.reshape(-1) - b.mean()
        d = (a.norm() * b.norm()).clamp_min(1e-8)
        return float((a @ b) / d)

    model_mae = float((pred - tgt).abs().mean())
    base_mae = float((base - tgt).abs().mean())
    return {
        "forecast_mae": model_mae,
        "baseline_mae": base_mae,
        "forecast_corr": corr(pred, tgt),
        "baseline_corr": corr(base, tgt),
        "beats_baseline": float(model_mae < base_mae),
    }
