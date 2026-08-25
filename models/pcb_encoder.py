"""PCB State Encoder: Multi-Scale CNN + Transformer Backbone.

Encodes (B, 10, 256, 256) spatial multi-channel tensor into a latent
embedding using convolutional patch extraction, Transformer self-attention,
and three additional signals that give the policy head local spatial detail
the whole-board mean-pool alone throws away -- see
docs/WORLD_MODEL_SPATIAL_DESIGN.md for the full rationale (stuck-in-corner
2-cycles are a local-information problem, not a missing-information one).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcbworld.environment import DIST_STEPS

# Mirrors pcbworld/environment.py's DIST_STEPS = [2, 4, 8] -- the raycast
# sensor's cap is the same "max useful lookahead distance" scale the env
# itself uses for its largest discrete step.
RAYCAST_MAX_STEPS = 8
# dir_idx has 8 distinct values for both the 24-action space (dir*3+dist)
# and the 96-action space (dir*12 + dist*4 + layer*2 + via) -- see
# environment.py's decode_action and _bearing_vector (45-degree steps,
# 8 * 45 = 360). NOT 12: that's the number of action indices sharing one
# dir_idx in the 96-space (dist(3)*layer(2)*via(2)), not the dir_idx count.
RAYCAST_NUM_DIRS = 8
PATCH_GRID = 16  # 256 / 16 downsample factor baked into cnn_extractor below
LOCAL_CROP_SIZE = 48
LOCAL_CROP_PAD = LOCAL_CROP_SIZE // 2
LOCAL_CROP_OUT_DIM = 128  # local_crop_cnn's final channel count, after pooling
# One safety bit per (direction, real DIST_STEPS distance) combination --
# NOT per direction alone. A direction can be clear for 2 cells and blocked
# by 8: RAYCAST_NUM_DIRS-only granularity cannot express that, and that
# exact gap (a whole direction biased as "fine" while a specific distance
# within it still collides) is what showed up as repeated same-spot
# REJECTED-collision retries in checkpoints_stage2_v8_spatial's eval trace
# on seed 9764 -- see docs/WORLD_MODEL_SPATIAL_DESIGN.md's collision
# reduction addendum.
DIST_SAFETY_DIM = RAYCAST_NUM_DIRS * len(DIST_STEPS)
# Fixed (NOT learned) suppression magnitude for a (direction, distance)
# combination the raycast sensor proves will collide. Deliberately a
# constant, not an nn.Parameter: this project has direct prior history
# (see the policy_head init comment below in router_policy.py) of a
# LEARNED bias/logit-scale relationship drifting so far apart during
# training that a once-meaningful constant became negligible relative to
# the weight-driven logits it was supposed to compete with. A fixed
# constant added at the very end of forward() can't be trained away --
# training can still route AROUND it (e.g. never letting the safe options'
# own logits get so large that avoiding an unsafe one stops mattering) but
# can never erode the suppression itself. If every option in a fully
# boxed-in cell is flagged unsafe, this constant is added uniformly across
# all of them and therefore changes nothing (a uniform shift never changes
# softmax/argmax) -- it only ever discriminates when it has real
# information to add.
#
# Raised 8.0 -> 32.0 (2026-08-25): checkpoints_stage2_v9_collision's
# converged entropy was ~0.001 (near-deterministic), meaning policy_head's
# OWN learned logit gaps between actions can plausibly exceed 8.0 by
# training's end -- in that regime this constant is a true-positive
# collision flag that still loses to a policy_head that has learned (for
# unrelated reasons) to strongly prefer the colliding action anyway, and
# the 1.51% residual Rejected-Action Rate on the 1000-board benchmark could
# be that, rather than the sensor being wrong. 32.0 is deliberately a big
# jump (not a gentle nudge) specifically to falsify-or-confirm that in one
# retrain: if Rejected-Action Rate drops sharply, magnitude was the
# bottleneck; if it barely moves, the sensor itself must be wrong for the
# remaining cases (e.g. its bearing reference is the raw, not smoothed,
# geodesic gradient -- see _raycast_sensor's docstring -- which no amount
# of suppression magnitude can fix, since dist_safe would read "safe" for
# the wrong direction).
DIST_SAFETY_SUPPRESSION = 32.0


def combined_latent_dim(d_model: int) -> int:
    """Width of PCBEncoder.forward's first return value: whole-board
    mean-pool (d_model) + local-attention pool (d_model) + raycast (8) +
    per-(direction,distance) safety mask (DIST_SAFETY_DIM) + local-crop CNN
    (LOCAL_CROP_OUT_DIM). Callers that build policy/value heads sized to
    the encoder's output should derive the width from here rather than
    hardcode it, since different scripts construct PCBRouterNet with
    different d_model."""
    return 2 * d_model + RAYCAST_NUM_DIRS + DIST_SAFETY_DIM + LOCAL_CROP_OUT_DIM


class PCBEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        d_model: int = 512,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Multi-scale Convolutional Patch Extractor (256x256 -> 16x16 feature grid)
        # Downsample factor: 16 (256/16 = 16x16 = 256 patch tokens)
        #
        # GroupNorm, not BatchNorm -- same reasoning as
        # pcbworld/agents/cfp/modules.py's ResBlock, which already documents
        # this: on-policy rollout collection is thousands of batch-of-1
        # forward passes, each one updating BatchNorm's running stats with a
        # noisy single-sample estimate. The same observation then evaluates
        # differently at collection time (train mode, batch stats) than at
        # update time or eval time (running stats, drifted since collection)
        # -- which silently corrupts the PPO ratio during training and, at
        # deploy time, washes out whatever the weights actually learned.
        # Measured: three differently-trained checkpoints produced
        # bit-identical evaluate_policy() stats (37/50 nets, 0.86x, every
        # time) despite clearly different training curves -- the eval-mode
        # running statistics, not the learned weights, were what eval was
        # actually measuring.
        def _norm(channels: int) -> nn.GroupNorm:
            return nn.GroupNorm(min(8, channels), channels)

        self.cnn_extractor = nn.Sequential(
            # Stage 1: 256x256 -> 128x128
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            _norm(64),
            nn.ReLU(inplace=True),

            # Stage 2: 128x128 -> 64x64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            _norm(128),
            nn.ReLU(inplace=True),

            # Stage 3: 64x64 -> 32x32
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            _norm(256),
            nn.ReLU(inplace=True),

            # Stage 4: 32x32 -> 16x16
            nn.Conv2d(256, d_model, kernel_size=3, stride=2, padding=1),
            _norm(d_model),
            nn.ReLU(inplace=True),
        )

        # 2. 2D Learnable Positional Embeddings for 16x16 = 256 patches
        self.num_patches = PATCH_GRID * PATCH_GRID
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        # 3. Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # 4. Global LayerNorm
        self.norm = nn.LayerNorm(d_model)

        # 5. Local-attention pool: a learned query that cross-attends only to
        # the 3x3 patch-token neighborhood around the head's own cell, so
        # "am I boxed in on 3 sides" has somewhere to live that isn't
        # averaged away by the global mean-pool. See
        # docs/WORLD_MODEL_SPATIAL_DESIGN.md Tier 2 item 1.
        self.local_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.local_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )

        # 6. Local-crop CNN: native-resolution near-field detail the 16px/
        # token patch grid structurally cannot represent (two obstacles 3px
        # apart on the raster are invisible past that downsampling). See
        # docs/WORLD_MODEL_SPATIAL_DESIGN.md Tier 2 item 3.
        self.local_crop_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 48->24
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24->12
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12->6
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _head_position(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recover (head_x, head_y) grid coordinates directly from Channel 3
        -- a Gaussian spot peaked exactly at the head's own position (see
        environment.py's _build_observation) -- instead of threading a new
        argument through every caller of PCBEncoder.forward. Integer grid
        coords, one pair per batch item.
        """
        B, _, H, W = x.shape
        head_map = x[:, 3].reshape(B, -1)
        flat_idx = head_map.argmax(dim=1)
        head_y = torch.div(flat_idx, W, rounding_mode="floor")
        head_x = flat_idx % W
        return head_x, head_y

    @staticmethod
    def _gather_at(field: torch.Tensor, y_idx: torch.Tensor, x_idx: torch.Tensor) -> torch.Tensor:
        """field: (B, H, W). y_idx/x_idx: (B,) or (B, N) integer coords
        (already clamped in-bounds). Returns the sampled values, same shape
        as y_idx/x_idx."""
        B, H, W = field.shape
        flat = field.reshape(B, H * W)
        flat_idx = y_idx * W + x_idx
        orig_shape = flat_idx.shape
        gathered = torch.gather(flat, 1, flat_idx.reshape(B, -1))
        return gathered.reshape(orig_shape)

    def _raycast_sensor(
        self, x: torch.Tensor, head_x: torch.Tensor, head_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Non-learned freespace sensor: for each of the 8 dir_idx bearings,
        how many cells (capped at RAYCAST_MAX_STEPS) can the head move along
        that bearing before hitting an obstacle or foreign copper (Channels
        1 and 0). Deterministic geometry read straight off the raster --
        nothing here is decoded from a learned representation, so there is
        no "can a network learn to reconstruct this" question the way there
        was for the two prior (abandoned) lookahead-speedup efforts; see
        docs/WORLD_MODEL_SPATIAL_DESIGN.md's confidence assessment.

        Bearing reference is the RAW (unsmoothed) gradient of Channel 7 (the
        obstacle-aware geodesic field) at the head's position -- the same
        math environment.py's _geo_descent_dir uses, minus the temporal EMA
        _smoothed_descent_dir applies across steps (that EMA depends on
        per-net history not recoverable from a single observation frame).
        This can differ from the env's true action bearing by a few
        degrees, which does not matter at 45-degree bucket resolution.

        Returns (raycast, dist_safe):
        - raycast: (B, 8) continuous freespace in [0,1], per direction only
          (the original signal -- coarse, for the general concatenated
          latent and the softer per-direction logit bias).
        - dist_safe: (B, 8, len(DIST_STEPS)) -- for each direction AND each
          of the environment's actual discrete step distances, whether a
          hop of exactly that length lands before the first blocked cell.
          A direction can read "mostly open" in `raycast` while still being
          blocked at the specific distance an action would actually try --
          this is the granularity `raycast` alone cannot express, and
          exactly the gap that produced repeated same-spot REJECTED-
          collision retries in practice (see DIST_SAFETY_SUPPRESSION).
        """
        with torch.no_grad():
            B, _, H, W = x.shape
            device = x.device

            x0 = (head_x - 1).clamp(0, W - 1)
            x1 = (head_x + 1).clamp(0, W - 1)
            y0 = (head_y - 1).clamp(0, H - 1)
            y1 = (head_y + 1).clamp(0, H - 1)
            field7 = x[:, 7]
            ddx = self._gather_at(field7, head_y, x0) - self._gather_at(field7, head_y, x1)
            ddy = self._gather_at(field7, y0, head_x) - self._gather_at(field7, y1, head_x)
            norm = torch.hypot(ddx, ddy)
            degenerate = norm < 1e-6
            safe_norm = norm.clamp(min=1e-6)
            gdx = torch.where(degenerate, torch.ones_like(ddx), ddx / safe_norm)
            gdy = torch.where(degenerate, torch.zeros_like(ddy), ddy / safe_norm)
            ref_bearing = torch.atan2(gdy, gdx)  # (B,)

            dir_offsets = torch.arange(RAYCAST_NUM_DIRS, device=device, dtype=x.dtype) * (
                math.pi / 4.0
            )
            angles = ref_bearing.unsqueeze(1) + dir_offsets.unsqueeze(0)  # (B, 8)
            dx = torch.cos(angles)
            dy = torch.sin(angles)

            steps = torch.arange(1, RAYCAST_MAX_STEPS + 1, device=device, dtype=x.dtype)  # (8,)
            # (B, dirs, steps)
            sample_x = (head_x.view(B, 1, 1).to(x.dtype) + dx.unsqueeze(2) * steps.view(1, 1, -1))
            sample_y = (head_y.view(B, 1, 1).to(x.dtype) + dy.unsqueeze(2) * steps.view(1, 1, -1))
            sample_x = sample_x.round().long().clamp(0, W - 1)
            sample_y = sample_y.round().long().clamp(0, H - 1)

            blocked_mask = (x[:, 0] > 0.5) | (x[:, 1] > 0.5)  # (B, H, W)
            blocked_flat = blocked_mask.reshape(B, -1).float()
            flat_idx = (sample_y * W + sample_x).reshape(B, -1)  # (B, dirs*steps)
            hit = torch.gather(blocked_flat, 1, flat_idx).reshape(B, RAYCAST_NUM_DIRS, RAYCAST_MAX_STEPS)

            has_block = hit.max(dim=-1).values > 0.5  # (B, dirs)
            first_hit_step0 = hit.argmax(dim=-1)  # 0-indexed step of first block (only valid where has_block)
            first_blocked_step = torch.where(
                has_block,
                (first_hit_step0 + 1).float(),
                torch.full_like(first_hit_step0, RAYCAST_MAX_STEPS + 1, dtype=torch.float32),
            )
            raycast = (first_blocked_step - 1).clamp(0, RAYCAST_MAX_STEPS) / float(RAYCAST_MAX_STEPS)

            # dist_safe[:, d, k] = True iff a hop of DIST_STEPS[k] cells
            # along direction d lands strictly before the first blocked
            # cell (first_blocked_step counts 1..RAYCAST_MAX_STEPS+1, where
            # +1 means "nothing blocked in any sampled cell").
            dist_steps_t = torch.tensor(DIST_STEPS, device=device, dtype=torch.float32)  # (num_dists,)
            dist_safe = (first_blocked_step.unsqueeze(-1) > dist_steps_t.view(1, 1, -1)).float()
            return raycast, dist_safe  # (B, 8), (B, 8, len(DIST_STEPS))

    def _local_crop(self, x: torch.Tensor, head_x: torch.Tensor, head_y: torch.Tensor) -> torch.Tensor:
        """Fixed-size native-resolution crop centered on the head, all 10
        channels, board-edge padded so off-board reads as a wall (obstacle
        channel padded with 1.0) rather than open space (0.0)."""
        pad = LOCAL_CROP_PAD
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
        x_padded[:, 1, :, :pad] = 1.0
        x_padded[:, 1, :, -pad:] = 1.0
        x_padded[:, 1, :pad, :] = 1.0
        x_padded[:, 1, -pad:, :] = 1.0

        B = x.shape[0]
        head_x_list = head_x.tolist()
        head_y_list = head_y.tolist()
        crops = [
            x_padded[i, :, head_y_list[i]: head_y_list[i] + LOCAL_CROP_SIZE,
                      head_x_list[i]: head_x_list[i] + LOCAL_CROP_SIZE]
            for i in range(B)
        ]
        return torch.stack(crops, dim=0)  # (B, 10, 48, 48)

    def _local_attention_pool(
        self, encoded_tokens: torch.Tensor, head_x: torch.Tensor, head_y: torch.Tensor
    ) -> torch.Tensor:
        """Learned query cross-attending only to the 3x3 patch-token
        neighborhood around the head's own grid cell -- coarse (16px/token)
        but local, unlike the whole-board mean-pool."""
        B = encoded_tokens.shape[0]
        device = encoded_tokens.device
        cell_x = torch.div(head_x, PATCH_GRID, rounding_mode="floor").clamp(0, PATCH_GRID - 1)
        cell_y = torch.div(head_y, PATCH_GRID, rounding_mode="floor").clamp(0, PATCH_GRID - 1)

        dy_offsets = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], device=device)
        dx_offsets = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], device=device)
        nb_y = (cell_y.unsqueeze(1) + dy_offsets.unsqueeze(0)).clamp(0, PATCH_GRID - 1)  # (B, 9)
        nb_x = (cell_x.unsqueeze(1) + dx_offsets.unsqueeze(0)).clamp(0, PATCH_GRID - 1)  # (B, 9)
        token_idx = nb_y * PATCH_GRID + nb_x  # (B, 9)

        local_tokens = torch.gather(
            encoded_tokens, 1, token_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        )  # (B, 9, d_model)
        local_query = self.local_query.expand(B, -1, -1)  # (B, 1, d_model)
        local_latent, _ = self.local_attn(local_query, local_tokens, local_tokens)
        return local_latent.squeeze(1)  # (B, d_model)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: (B, 10, 256, 256) spatial observation tensor
        Returns:
            combined_latent: (B, combined_latent_dim(d_model)) global + local
                + raycast + dist_safe + crop
            patch_tokens: (B, 256, d_model) spatial token sequence
            raycast_vector: (B, 8) non-learned per-direction freespace, [0,1]
            dist_safe: (B, 8, len(DIST_STEPS)) non-learned per-(direction,
                distance) collision-free mask, see _raycast_sensor
        """
        head_x, head_y = self._head_position(x)

        # (B, 10, 256, 256) -> (B, d_model, 16, 16)
        features = self.cnn_extractor(x)
        B, C, H, W = features.shape

        # Reshape to token sequence: (B, 256, d_model)
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embedding

        # Transformer self-attention
        encoded_tokens = self.transformer_encoder(tokens)
        encoded_tokens = self.norm(encoded_tokens)

        # Global average pooling across all patch tokens -> (B, d_model)
        mean_pooled = encoded_tokens.mean(dim=1)

        local_attn_latent = self._local_attention_pool(encoded_tokens, head_x, head_y)
        raycast_vector, dist_safe = self._raycast_sensor(x, head_x, head_y)
        local_crop = self._local_crop(x, head_x, head_y)
        local_crop_latent = self.local_crop_cnn(local_crop).mean(dim=(2, 3))  # (B, 128)

        combined_latent = torch.cat(
            [
                mean_pooled,
                local_attn_latent,
                raycast_vector,
                dist_safe.reshape(B, -1),
                local_crop_latent,
            ],
            dim=-1,
        )

        return combined_latent, encoded_tokens, raycast_vector, dist_safe
