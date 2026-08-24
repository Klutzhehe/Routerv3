"""Fast (learned) lookahead action selection.

An additive, opt-in alternative to lookahead_select_action's real-simulator
forward search (models/router_policy.py) -- that function solved stage 2 to
100% (1000/1000) by, for each of the policy's top-K candidate actions,
copy.deepcopy(env)-ing and running `horizon` REAL environment steps plus
`horizon` full CNN+Transformer forward passes. It works, but costs
~top_k*horizon extra env steps and forward passes per real decision.

This module scores each candidate with ONE cheap MLP forward pass against
the ALREADY-computed encoder output instead: no environment copies, no real
simulation, no repeated full-network passes. The encoder still runs once per
real decision (shared with the policy's own action logits) -- only the
per-candidate scoring step is replaced.

Design note, informed by jepa/'s diagnostic history (see jepa/README.md):
that earlier attempt tried self-supervised embedding prediction against the
encoder's POOLED `global_latent`, and three independent probes there found
distance-to-target is NOT decodable from the pooled vector -- not via a
fresh linear/MLP head, not even via this exact checkpoint's own co-trained
value_head (functionally constant, 0.2014-0.2088 across 26,070 real
timesteps). jepa/probe_token_features.py's smoke test (random-init weights
only, never confirmed on the real checkpoint) found the opposite for a
PER-TOKEN feature: the patch-grid token covering the head's own position
decodes distance well, because the geodesic distance field is a raw input
channel (Channel 7 -- see PCBRouterEnv._build_observation) with near-direct
local access at that exact spot. This predictor is built on that lead --
head_token (+ target_token, effectively free from the same forward pass) --
rather than the pooled embedding the JEPA attempt started from, specifically
to avoid walking into the same wall. If this turns out not to hold on the
real checkpoint either, that is itself useful information.

This is ordinary supervised regression against a real, known, varying label
(geodesic distance a few real steps ahead) -- no self-supervised objective,
no EMA target encoder, no stop-gradient, none of the JEPA machinery. Unlike
embedding-matching, there is no "predict a constant" shortcut that minimizes
this loss, so a predictor that isn't learning anything shows up directly as
a bad MAE, no separate collapse diagnostic needed.

Does NOT touch models/router_policy.py's select_deterministic_action or
lookahead_select_action -- purely additive. If this doesn't pan out,
--lookahead remains the proven fallback.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_GRID = 16  # 256x256 CNN downsample factor -- mirrors models/pcb_encoder.py's 4 stride-2 stages
MAX_GEO_DIST = math.hypot(256, 256)  # mirrors PCBRouterEnv._build_observation's Channel 7 normalization


def patch_token_index(x: float, y: float, grid_size: int) -> int:
    """Which of PCBEncoder's 16x16=256 patch-grid tokens covers pixel (x, y).

    Mirrors PCBEncoder.forward's own flatten: features (B, C, 16, 16) ->
    flatten(2) merges (row, col) in row-major order, so token index =
    row*16 + col.
    """
    downsample = grid_size // PATCH_GRID
    col = min(PATCH_GRID - 1, max(0, int(x) // downsample))
    row = min(PATCH_GRID - 1, max(0, int(y) // downsample))
    return row * PATCH_GRID + col


def extract_head_target_tokens(
    tokens: torch.Tensor, head_x: float, head_y: float, target_x: float, target_y: float, grid_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """tokens: (1, 256, d_model), PCBEncoder.forward's second return value.
    Returns (head_token, target_token), each (1, d_model)."""
    h_idx = patch_token_index(head_x, head_y, grid_size)
    t_idx = patch_token_index(target_x, target_y, grid_size)
    return tokens[:, h_idx, :], tokens[:, t_idx, :]


class FastDistancePredictor(nn.Module):
    """Small supervised regression head: predicts normalized geodesic
    distance-to-target `horizon` real steps after taking `action` from the
    state that produced `head_token` (+ `target_token`).
    """

    def __init__(
        self,
        d_model: int = 256,
        action_dim: int = 24,
        use_target_token: bool = True,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.use_target_token = use_target_token
        in_dim = d_model * (2 if use_target_token else 1) + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        head_token: torch.Tensor,
        action: torch.Tensor,
        target_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            head_token: (B, d_model)
            action: (B,) long action indices
            target_token: (B, d_model), required iff self.use_target_token
        Returns:
            (B,) predicted normalized distance in [0, 1] -- sigmoid-clamped
            to match the [0, 1] clip both Channel 7 and this module's own
            training labels use (see scripts/collect_fast_lookahead_data.py).
        """
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        parts = [head_token]
        if self.use_target_token:
            assert target_token is not None, "use_target_token=True requires target_token"
            parts.append(target_token)
        parts.append(action_onehot)
        x = torch.cat(parts, dim=-1)
        return torch.sigmoid(self.net(x).squeeze(-1))


def fast_lookahead_select_action(
    model,
    predictor: FastDistancePredictor,
    env,
    obs_np,
    device_str: str,
    forbidden: set,
    top_k: int = 4,
) -> int:
    """Same top-K candidate-ranking shape as lookahead_select_action, but
    scores each candidate with the trained `predictor` instead of a real
    forward simulation. No copy.deepcopy(env), no env.step(), no repeated
    encoder passes -- the encoder runs once (shared with the policy's own
    action logits), then `predictor` runs once per candidate action.

    Only reasons about the CURRENT decision -- unlike lookahead_select_action,
    there is no multi-step simulated rollout to interrupt if round-robin
    (num_nets > 1) rotates control elsewhere, since no simulation happens.
    """
    idx = env.current_net_idx
    if idx is None:
        return 0
    state = env.net_states[idx]
    net = env.board.nets[idx]

    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device_str).unsqueeze(0)
    with torch.no_grad():
        pooled, tokens = model.encoder(obs_t)
        action_logits = model.policy_head(pooled)
    logits = action_logits.squeeze(0)
    ranked = [a for a in torch.argsort(logits, descending=True).tolist() if a not in forbidden]
    if not ranked:
        ranked = torch.argsort(logits, descending=True).tolist()
    candidates = ranked[:top_k]

    head_tok, target_tok = extract_head_target_tokens(
        tokens, state.head_x, state.head_y, net.target_pad.x, net.target_pad.y, env.grid_size,
    )

    best_action = candidates[0]
    best_score = float("inf")
    with torch.no_grad():
        for cand in candidates:
            action_t = torch.tensor([cand], device=device_str, dtype=torch.long)
            score = predictor(
                head_tok, action_t, target_tok if predictor.use_target_token else None,
            ).item()
            if score < best_score:
                best_score = score
                best_action = cand

    return best_action
