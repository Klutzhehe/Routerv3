"""Net Selector Head.

Predicts routing priority and net selection logits given PCB latent representation
and net attribute embeddings (width, clearance, criticality, length).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NetSelectorHead(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        max_nets: int = 20,
        net_attr_dim: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_nets = max_nets

        # Net attribute embedder
        self.net_embedder = nn.Sequential(
            nn.Linear(net_attr_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )

        # Joint scoring network
        self.scorer = nn.Sequential(
            nn.Linear(d_model + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        pcb_latent: torch.Tensor,  # (B, 512)
        net_attrs: Optional[torch.Tensor] = None,  # (B, N, 16)
        mask: Optional[torch.Tensor] = None,  # (B, N) boolean mask (True for valid unrouted nets)
    ) -> torch.Tensor:
        """
        Compute selection logits across candidate nets.
        Returns:
            logits: (B, N)
        """
        B = pcb_latent.shape[0]
        if net_attrs is None:
            # Fallback simple projection if net attributes are omitted
            return torch.zeros((B, self.max_nets), device=pcb_latent.device)

        N = net_attrs.shape[1]
        net_emb = self.net_embedder(net_attrs)  # (B, N, 64)

        # Broadcast PCB latent to each net: (B, N, 512)
        pcb_expanded = pcb_latent.unsqueeze(1).expand(B, N, self.d_model)

        # Concat: (B, N, 512 + 64)
        joint = torch.cat([pcb_expanded, net_emb], dim=-1)

        # Score logits: (B, N, 1) -> (B, N)
        logits = self.scorer(joint).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        return logits
