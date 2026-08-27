"""Field and net encoders.

Two decisions dominate this file.

**Why 3-D and not 2-D-per-layer.** A via is a cross-layer event, and on an
8-layer board choosing the layer *is* most of the routing problem. A stack of
independent 2-D encoders cannot represent "layer 3 is congested right here but
layer 5 is open two cells over", which is exactly the question a via answers.
So the spatial trunk uses factorised 3-D convolution -- ``(1,3,3)`` in-plane
and ``(3,1,1)`` across layers -- plus explicit attention along the layer axis,
which is cheap because ``L <= 8``.

**Why the trunk runs at 1/4 lattice resolution.** At ``B=64, L=8, H=W=128``, a
single full-resolution feature map with 64 channels is 537 M floats. The
encoder's job is *global context* -- congestion, corridors, where demand is
going -- and none of that needs per-cell resolution. The exact local geometry
a routing decision turns on is delivered separately and exactly, as the
raycast / safety / geodesic features in `env/observation.py`, and by a
native-resolution crop around each head (`FrontierCropEncoder`). That split is the
direct lesson of `docs/WORLD_MODEL_SPATIAL_DESIGN.md`: fine detail belongs in
a path that reads pixels, not in one that has already downsampled them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch: int) -> nn.Module:
    # GroupNorm, not BatchNorm: PPO minibatches are small and correlated, and
    # batch statistics across boards at different routing stages are not a
    # meaningful thing to normalise by.
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class FactorisedBlock(nn.Module):
    """In-plane conv, then cross-layer conv, with a residual connection.

    Factorising ``(3,3,3)`` into ``(1,3,3) + (3,1,1)`` costs ~3x fewer
    parameters and multiplies for the same receptive field, which matters
    because the spatial trunk is the memory bottleneck of the whole model.
    """

    def __init__(self, ch_in: int, ch_out: int, stride: int = 1):
        super().__init__()
        self.plane = nn.Conv3d(ch_in, ch_out, (1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1))
        self.n1 = _norm(ch_out)
        self.cross = nn.Conv3d(ch_out, ch_out, (3, 1, 1), padding=(1, 0, 0))
        self.n2 = _norm(ch_out)
        self.skip = (
            nn.Identity()
            if stride == 1 and ch_in == ch_out
            else nn.Conv3d(ch_in, ch_out, 1, stride=(1, stride, stride))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.n1(self.plane(x)))
        h = self.n2(self.cross(h))
        return F.silu(h + self.skip(x))


class LayerAxialAttention(nn.Module):
    """Self-attention along the layer axis, independently per spatial cell.

    This is the module that lets the encoder compare layers at one location --
    "is there a better layer directly below me" -- which is the decision the
    via action makes. It is cheap: the sequence length is `L`, at most 8.
    """

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = x.shape
        t = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, L, C)
        t = self.norm(t)
        out, _ = self.attn(t, t, t, need_weights=False)
        out = out.reshape(B, H, W, L, C).permute(0, 4, 3, 1, 2)
        return x + out


class SpatialAttention(nn.Module):
    """Self-attention over spatial tokens, independently per layer.

    Global receptive field in one hop. At the bottleneck resolution the token
    count is small (``(H/16) * (W/16)`` = 64 for a 128-cell board), so this is
    far cheaper than stacking convolutions until they see the whole board.
    """

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L, H, W = x.shape
        t = x.permute(0, 2, 3, 4, 1).reshape(B * L, H * W, C)
        t = self.norm(t)
        out, _ = self.attn(t, t, t, need_weights=False)
        out = out.reshape(B, L, H, W, C).permute(0, 4, 1, 2, 3)
        return x + out


class FieldEncoder(nn.Module):
    """(B, C_in, L, H, W) -> spatial latent (B, D, L, H/4, W/4) + global vector.

    The returned latent is at **1/4 lattice resolution** and is deliberately
    the only thing the forecaster and the head-gather read; nothing in the
    model consumes a full-resolution feature map.
    """

    def __init__(self, in_channels: int, width: int = 64, attn_heads: int = 4):
        super().__init__()
        d = width
        self.width = d

        # Straight to 1/4 resolution. See the module docstring for why.
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, d, (1, 5, 5), stride=(1, 4, 4), padding=(0, 2, 2)),
            _norm(d),
            nn.SiLU(),
        )
        self.enc0 = FactorisedBlock(d, d)
        self.down1 = FactorisedBlock(d, 2 * d, stride=2)
        self.down2 = FactorisedBlock(2 * d, 4 * d, stride=2)

        self.mid_layer_attn = LayerAxialAttention(4 * d, attn_heads)
        self.mid_spatial_attn = SpatialAttention(4 * d, attn_heads)
        self.mid_block = FactorisedBlock(4 * d, 4 * d)

        self.up1 = FactorisedBlock(4 * d + 2 * d, 2 * d)
        self.up2 = FactorisedBlock(2 * d + d, d)
        self.out_layer_attn = LayerAxialAttention(d, attn_heads)

        self.global_proj = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, 2 * d))
        self.global_dim = 2 * d

    def forward(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = self.enc0(self.stem(field))          # (B, d,   L, H/4,  W/4)
        x1 = self.down1(x0)                       # (B, 2d,  L, H/8,  W/8)
        x2 = self.down2(x1)                       # (B, 4d,  L, H/16, W/16)

        m = self.mid_block(self.mid_spatial_attn(self.mid_layer_attn(x2)))

        u1 = F.interpolate(m, size=x1.shape[2:], mode="nearest")
        u1 = self.up1(torch.cat([u1, x1], dim=1))
        u2 = F.interpolate(u1, size=x0.shape[2:], mode="nearest")
        z = self.out_layer_attn(self.up2(torch.cat([u2, x0], dim=1)))

        g = self.global_proj(m.mean(dim=(2, 3, 4)))
        return z, g


class FrontierCropEncoder(nn.Module):
    """Native-resolution crop around a routing head -> a per-head vector.

    Everything else the model sees about the board has been downsampled by at
    least 4x. Two obstacles three cells apart are indistinguishable at that
    scale, and three cells is the difference between a corridor and a wall.
    `docs/WORLD_MODEL_SPATIAL_DESIGN.md` reached the same conclusion for the
    raster thread and its local-crop branch is what "actually resolves fine
    corner geometry the tokenized path structurally cannot".

    The crop spans **every layer**, not just the head's own, because the layer
    decision needs to see what is directly above and below.
    """

    def __init__(self, in_channels: int, num_layers: int, crop: int = 16, width: int = 32):
        super().__init__()
        self.crop = crop
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, width, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            _norm(width),
            nn.SiLU(),
            nn.Conv3d(width, 2 * width, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            _norm(2 * width),
            nn.SiLU(),
        )
        self.out_dim = 2 * width

    def forward(self, field: torch.Tensor, frontier_pos: torch.Tensor, frontier_mask: torch.Tensor) -> torch.Tensor:
        """field (B, C, L, H, W); frontier_pos (B, K, 3) as (layer, y, x).

        Gathering is done with flat linear indices rather than by slicing a
        per-head view of the board. Indexing `field[b]` with a (B*K,) index
        would materialise `K` copies of every board -- at B=32, K=8, L=8,
        128x128 that is gigabytes for a crop that is three orders of magnitude
        smaller.
        """
        B, C, L, H, W = field.shape
        K = frontier_pos.shape[1]
        r = self.crop // 2

        # Zero-pad rather than clamp: a head near the board edge must see empty
        # space there, not a smeared copy of the nearest real cell.
        padded = F.pad(field, (r, r, r, r), mode="constant", value=0.0)
        Hp, Wp = H + 2 * r, W + 2 * r

        off = torch.arange(self.crop, device=field.device)
        ys = frontier_pos[..., 1].clamp(0, H - 1).unsqueeze(-1) + off.view(1, 1, -1)   # (B, K, crop)
        xs = frontier_pos[..., 2].clamp(0, W - 1).unsqueeze(-1) + off.view(1, 1, -1)

        lin = ys.unsqueeze(-1) * Wp + xs.unsqueeze(-2)          # (B, K, crop, crop)
        lin = lin.reshape(B, 1, -1).expand(B, C * L, K * self.crop * self.crop)

        flat = padded.reshape(B, C * L, Hp * Wp)
        crop = torch.gather(flat, 2, lin)
        crop = crop.view(B, C, L, K, self.crop, self.crop)
        crop = crop.permute(0, 3, 1, 2, 4, 5).reshape(B * K, C, L, self.crop, self.crop)
        crop = crop * frontier_mask.reshape(-1, 1, 1, 1, 1).float()

        h = self.net(crop)
        return h.mean(dim=(2, 3, 4)).view(B, K, self.out_dim)


class NetEncoder(nn.Module):
    """Per-net tokens, contextualised against the board.

    Scales to thousands of nets because attention is over net tokens only
    (a few thousand is nothing) and the board enters through a small set of
    pooled context vectors rather than as full spatial cross-attention.
    """

    def __init__(self, in_features: int, global_dim: int, width: int = 128, heads: int = 4, layers: int = 2):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_features, width), nn.SiLU(), nn.Linear(width, width))
        self.ctx = nn.Linear(global_dim, width)
        enc = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        # enable_nested_tensor=False on purpose: the nested-tensor fast path is
        # incompatible with norm_first=True (which we want for training
        # stability), and leaving it on emits a UserWarning on every
        # construction that reads like a real error in a Colab log.
        self.blocks = nn.TransformerEncoder(enc, num_layers=layers, enable_nested_tensor=False)
        self.out_dim = width

    def forward(self, nets: torch.Tensor, global_vec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        t = self.embed(nets) + self.ctx(global_vec).unsqueeze(1)
        # `src_key_padding_mask` marks positions to IGNORE, so it is the
        # negation of the validity mask. Getting this backwards silently makes
        # every real net invisible and every padding slot attended to.
        out = self.blocks(t, src_key_padding_mask=~mask)
        return torch.nan_to_num(out) * mask.unsqueeze(-1).float()
