"""Action-conditioned latent dynamics predictor (MuZero-style dynamics network
+ JEPA/BYOL-family anti-collapse machinery), for jepa_lookahead's fast action
selector.

This module ONLY defines the new network pieces -- it does not import or
touch pcbworld/environment.py, pcbworld/reward.py, or models/router_policy.py's
existing functions. See jepa/README.md for why this folder is isolated and
for the data-pipeline design decision (frozen encoder, logged embeddings, no
literal EMA schedule needed) that shapes what's built here.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def decode_action_components(actions: torch.Tensor, enable_layer_via: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized mirror of PCBRouterEnv.decode_action (pcbworld/environment.py).

    Duplicated here (rather than imported) so jepa/ has zero dependency on an
    env instance -- decode_action is an instance method only because it reads
    self.enable_layer_via, and this arithmetic is otherwise pure. Keep this in
    sync if that formula ever changes.
    """
    if not enable_layer_via:
        dist_idx = actions % 3
        dir_idx = torch.div(actions, 3, rounding_mode="floor")
        zeros = torch.zeros_like(actions)
        return dir_idx, dist_idx, zeros, zeros
    via_flag = actions % 2
    rem = torch.div(actions, 2, rounding_mode="floor")
    layer_change = rem % 2
    rem = torch.div(rem, 2, rounding_mode="floor")
    dist_idx = rem % 3
    dir_idx = torch.div(rem, 3, rounding_mode="floor")
    return dir_idx, dist_idx, layer_change, via_flag


class ActionEncoder(nn.Module):
    """Embeds a discrete action index by its known factored structure
    (direction x distance x layer-change x via), instead of a flat
    96-way (or 24-way) embedding table -- exploits structure the raw index
    otherwise hides, cheaper to learn from the modest transition counts this
    project's scale produces."""

    def __init__(self, enable_layer_via: bool, dir_dim: int = 16, dist_dim: int = 8, layer_dim: int = 4, via_dim: int = 4):
        super().__init__()
        self.enable_layer_via = enable_layer_via
        self.dir_embed = nn.Embedding(8, dir_dim)
        self.dist_embed = nn.Embedding(3, dist_dim)
        if enable_layer_via:
            self.layer_embed = nn.Embedding(2, layer_dim)
            self.via_embed = nn.Embedding(2, via_dim)
            self.out_dim = dir_dim + dist_dim + layer_dim + via_dim
        else:
            self.layer_embed = None
            self.via_embed = None
            self.out_dim = dir_dim + dist_dim

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        dir_idx, dist_idx, layer_change, via_flag = decode_action_components(actions, self.enable_layer_via)
        parts = [self.dir_embed(dir_idx), self.dist_embed(dist_idx)]
        if self.enable_layer_via:
            parts.append(self.layer_embed(layer_change))
            parts.append(self.via_embed(via_flag))
        return torch.cat(parts, dim=-1)


class DynamicsPredictor(nn.Module):
    """g(z_t, a_t) -> z_hat_{t+1}, in the style of MuZero's dynamics network.

    Residual formulation (predict a DELTA added to z_t, not the raw absolute
    next embedding from scratch): one router step is spatially small relative
    to the whole board, so z_{t+1} is expected to sit close to z_t in
    embedding space, and a residual target is the easier one for a small MLP
    to fit than reconstructing an unrelated-looking 256-d vector per step.
    """

    def __init__(self, d_model: int = 256, enable_layer_via: bool = False, hidden: int = 512):
        super().__init__()
        self.d_model = d_model
        self.action_encoder = ActionEncoder(enable_layer_via)
        in_dim = d_model + self.action_encoder.out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d_model),
        )

    def forward(self, z_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        a_emb = self.action_encoder(actions)
        x = torch.cat([z_t, a_emb], dim=-1)
        delta = self.net(x)
        return z_t + delta


class DistanceHead(nn.Module):
    """Decodes a (predicted or real) latent embedding into the normalized
    geodesic distance-to-target -- the REAL, verifiable quantity used as the
    collapse-defeating auxiliary anchor (see jepa/README.md point 3). Applied
    to the PREDICTOR's output during training, never to the frozen target
    directly, since anchoring the target would do nothing to keep the
    predictor itself honest.
    """

    def __init__(self, d_model: int = 256, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def predictive_loss(z_hat: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """BYOL-style normalized loss: 2 - 2*cosine_similarity, mean over batch.

    Cosine (not raw MSE) so shrinking every embedding's norm toward zero
    cannot trivially lower the loss -- a classic collapse shortcut MSE alone
    would not penalize.
    """
    z_hat_n = F.normalize(z_hat, dim=-1)
    z_target_n = F.normalize(z_target, dim=-1)
    return (2.0 - 2.0 * (z_hat_n * z_target_n).sum(dim=-1)).mean()
