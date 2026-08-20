"""Building blocks for CFPNet. Nothing router-specific lives here.

Three things the stock torch.nn equivalents don't do, which is why these
are hand-rolled rather than nn.MultiheadAttention calls:

  1. RelationalSelfAttention adds a per-(edge-type, head) bias to the
     attention logits. That's the whole mechanism by which "these two nets
     are a differential pair" reaches the network as a *relation* instead
     of a feature both endpoints happen to share.
  2. CrossAttention takes a precomputed additive bias, used to make a net
     attend preferentially to the region of canvas near its own pads. A
     stock module has nowhere to put that.
  3. Masking here is written to survive fully-padded query rows (all-masked
     softmax -> NaN), which nn.MultiheadAttention does not guarantee. Net
     slots are padded to a fixed width, so this case is routine, not
     exotic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_logit_value(dtype: torch.dtype) -> float:
    """Fill value for masked-out logits, valid in `dtype`.

    Three constraints, and it is easy to satisfy two while breaking the third:

      1. Finite. -inf makes a categorical's entropy term 0 * -inf = NaN, and
         with a masked action space that is most of every row.
      2. exp()s to exactly 0, so a masked entry can never be sampled or
         attended to.
      3. Representable in `dtype` -- and still representable after softmax or
         log_softmax subtracts the row max.

    A hardcoded -1e9 satisfies 1 and 2 but fails 3 in float16, whose max is
    65504: the moment the model runs under autocast, masked_fill raises
    "value cannot be converted to type c10::Half without overflow".
    torch.finfo(dtype).min satisfies 1 and 3 but underflows back to -inf once
    the max is subtracted.

    finfo.max / 4 satisfies all three for float16, bfloat16 and float32.
    """
    return -torch.finfo(dtype).max / 4


def masked_softmax(logits: torch.Tensor, key_mask: torch.Tensor | None) -> torch.Tensor:
    """Softmax over the last dim, with rows that have no valid key made
    uniform-zero rather than NaN.

    logits:   (..., S)
    key_mask: broadcastable to logits, True = attend to this key.
    """
    if key_mask is None:
        return torch.softmax(logits, dim=-1)
    logits = logits.masked_fill(~key_mask, mask_logit_value(logits.dtype))
    weights = torch.softmax(logits, dim=-1)
    # A row whose keys were all masked softmaxes to uniform over garbage;
    # zero it so it contributes nothing downstream.
    any_valid = key_mask.any(dim=-1, keepdim=True)
    return weights * any_valid


class MLP(nn.Module):
    """The standard transformer feed-forward block, GELU, 4x expansion."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelationalSelfAttention(nn.Module):
    """Multi-head self-attention over net slots, biased by edge type.

    The bias is additive-per-head rather than a separate value projection
    per edge type: with 6 edge types and 8 heads that's 48 learned scalars
    instead of 6 full projection matrices, which is the right trade at this
    sample budget (env throughput, not model capacity, is the binding
    constraint -- see docs/AI_ARCHITECTURE.md).
    """

    def __init__(self, dim: int, num_heads: int, num_edge_types: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.edge_bias = nn.Embedding(num_edge_types, num_heads)
        nn.init.zeros_(self.edge_bias.weight)

    def forward(
        self,
        x: torch.Tensor,          # (B, N, D)
        edge_type: torch.Tensor,  # (B, N, N) int64
        key_mask: torch.Tensor,   # (B, N) bool
    ) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, H, N, hd)

        logits = (q @ k.transpose(-2, -1)) * self.scale        # (B, H, N, N)
        logits = logits + self.edge_bias(edge_type).permute(0, 3, 1, 2)

        weights = masked_softmax(logits, key_mask[:, None, None, :])
        out = (weights @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention with an optional additive logit bias.

    bias is (B, H, Q, S) -- the caller builds it; see CFPNet's distance
    bias, which is what gives "a net looks at the canvas near its own pads"
    without the model having to learn 2-D geometry from scratch.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        q_x: torch.Tensor,                      # (B, Q, D)
        kv_x: torch.Tensor,                     # (B, S, D)
        bias: torch.Tensor | None = None,       # (B, H, Q, S)
        key_mask: torch.Tensor | None = None,   # (B, S) bool
    ) -> torch.Tensor:
        b, qlen, d = q_x.shape
        slen = kv_x.shape[1]

        q = self.to_q(q_x).reshape(b, qlen, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.to_kv(kv_x).reshape(b, slen, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)

        logits = (q @ k.transpose(-2, -1)) * self.scale
        if bias is not None:
            logits = logits + bias

        weights = masked_softmax(
            logits, None if key_mask is None else key_mask[:, None, None, :]
        )
        out = (weights @ v).transpose(1, 2).reshape(b, qlen, d)
        return self.proj(out)


class NetSelfBlock(nn.Module):
    """Pre-norm relational self-attention + MLP over net slots."""

    def __init__(self, dim: int, num_heads: int, num_edge_types: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelationalSelfAttention(dim, num_heads, num_edge_types)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(
        self, x: torch.Tensor, edge_type: torch.Tensor, key_mask: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), edge_type, key_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossBlock(nn.Module):
    """Pre-norm cross-attention + MLP."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        bias: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_x = q_x + self.attn(self.norm_q(q_x), self.norm_kv(kv_x), bias, key_mask)
        q_x = q_x + self.mlp(self.norm2(q_x))
        return q_x


class ResBlock(nn.Module):
    """Conv residual block. GroupNorm, not BatchNorm.

    BatchNorm is actively harmful in on-policy RL here: rollout batches are
    small, highly correlated within an episode, and the running statistics
    drift as the policy changes the board distribution it visits, so the
    same observation evaluates differently at collection time and at update
    time -- which silently corrupts the PPO ratio.
    """

    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class FiLM(nn.Module):
    """Feature-wise linear modulation of a (B, D, H, W) map by a (B, D_ctx)
    vector.

    This is how the field head is told *which net it is drawing a field
    for*. Initialized to the identity (zero gamma/beta) so an untrained
    model emits the unconditioned field rather than noise scaled by a
    random projection of the net embedding.
    """

    def __init__(self, ctx_dim: int, channels: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(ctx_dim, 2 * channels)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(ctx).chunk(2, dim=-1)
        return x * (1.0 + scale[..., None, None]) + shift[..., None, None]


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over dim 1 of (B, N, D) using a (B, N) bool mask, safe when a
    row has no valid entries."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2)) -> nn.Module:
    """Orthogonal weights + zero bias, the standard PPO initialization.
    Applied to head outputs with small gain so the initial policy is close
    to uniform / zero-field -- which for CFP means "behave like stock PNS",
    the baseline the agent should start from rather than fall below."""
    for name, param in module.named_parameters():
        if "weight" in name and param.ndim >= 2:
            nn.init.orthogonal_(param, gain)
        elif "bias" in name:
            nn.init.zeros_(param)
    return module
