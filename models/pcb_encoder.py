"""PCB State Encoder: Multi-Scale CNN + Transformer Backbone.

Encodes (B, 10, 256, 256) spatial multi-channel tensor into 512-dim latent embedding
using convolutional patch extraction and Transformer self-attention.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.cnn_extractor = nn.Sequential(
            # Stage 1: 256x256 -> 128x128
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 2: 128x128 -> 64x64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Stage 3: 64x64 -> 32x32
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Stage 4: 32x32 -> 16x16
            nn.Conv2d(256, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # 2. 2D Learnable Positional Embeddings for 16x16 = 256 patches
        self.num_patches = 16 * 16
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: (B, 10, 256, 256) spatial observation tensor
        Returns:
            global_latent: (B, 512) global PCB representation
            patch_tokens: (B, 256, 512) spatial token sequence
        """
        # (B, 10, 256, 256) -> (B, 512, 16, 16)
        features = self.cnn_extractor(x)
        B, C, H, W = features.shape

        # Reshape to token sequence: (B, 256, 512)
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embedding

        # Transformer self-attention
        encoded_tokens = self.transformer_encoder(tokens)
        encoded_tokens = self.norm(encoded_tokens)

        # Global average pooling across all patch tokens -> (B, 512)
        global_latent = encoded_tokens.mean(dim=1)

        return global_latent, encoded_tokens
