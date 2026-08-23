"""PCBRouterNet: Full Reinforcement Learning Actor-Critic Architecture.

Combines:
1. Spatial PCBEncoder (CNN + Transformer Backbone -> 512 latent)
2. NetSelectorHead (Optional Net Attention)
3. 96-Action Router Policy Head (Categorical Action Distribution)
4. Value Critic Head (Scalar Board Return Estimate)
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.distributions import Categorical

from models.pcb_encoder import PCBEncoder
from models.net_selector import NetSelectorHead


class PCBRouterNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        action_dim: int = 96,
        d_model: int = 512,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model

        # 1. State Encoder (CNN + Transformer)
        self.encoder = PCBEncoder(
            in_channels=in_channels,
            d_model=d_model,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
        )

        # 2. Net Selector Head
        self.net_selector = NetSelectorHead(d_model=d_model)

        # 3. Router Action Policy Head (96 Discrete Growth Actions)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )

        # 4. Value Critic Head (Predicts Future Expected PCB Return)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Initialize policy head near zero to prevent violent initial divergence
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.constant_(self.policy_head[-1].bias, 0.0)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[Categorical, torch.Tensor]:
        """
        Forward pass for action distribution and state value.
        Args:
            obs: (B, 10, 256, 256) spatial observation tensor
        Returns:
            dist: Categorical distribution over 96 discrete actions
            value: (B, 1) state value estimate
        """
        pcb_latent, _patch_tokens = self.encoder(obs)

        # Compute Action Logits & Distribution
        action_logits = self.policy_head(pcb_latent)
        dist = Categorical(logits=action_logits)

        # Compute State Value
        value = self.value_head(pcb_latent)

        return dist, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate or sample action for PPO / Actor-Critic step.
        Returns:
            action: (B,) sampled or passed action
            log_prob: (B,) log probability of action
            entropy: (B,) distribution entropy
            value: (B, 1) state value baseline
        """
        dist, value = self.forward(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value
