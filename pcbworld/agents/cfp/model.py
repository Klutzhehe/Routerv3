"""CFPNet -- the two-tower cost-field policy network.

Shape of the whole thing, for a 256x256 canvas with 2 copper layers:

    canvas (B,12,256,256) ---> CanvasEncoder ---> (B,D,16,16)
                                                      |
    net_feats (B,N,16) -----> NetEncoder ------> (B,N,D)
                                                      |
                              FUSION (x fusion_rounds)
                        nets  <-- cross-attend --  canvas   (distance-biased)
                        canvas <-- cross-attend --  nets
                        nets  <-- relational self-attention
                                                      |
              +---------------------+-----------------+--------------+
              |                     |                                |
        pointer head           field head (FiLM on the             value
        (B, 2, N) logits        *selected* net) -> (B, L+1, 16, 16)  head

Two things about this that are load-bearing rather than decorative:

  * The field head is conditioned on the net the pointer head just picked,
    so sampling is autoregressive within a step: pick the net, then draw
    that net's field. The joint log-prob is the sum of the two, which is
    what policy.py assembles. An unconditioned field would be a single
    global map and could not express "route *this* net around the region
    those three diff pairs need".

  * The last field plane is the *reserve* plane -- it does not affect the
    current net's path, it is added to the cost every later net pays. That
    is the only mechanism in the architecture that can say "keep this area
    empty", which is what length tuning needs and what classical routers
    structurally cannot do (by the time the meander is needed, the space is
    gone). num_field_planes is therefore num_copper_layers + 1.

Untrained behavior is deliberate, not incidental: pointer logits and field
mean both start near zero (small-gain orthogonal init, zero FiLM), so a
fresh network emits a flat field. A flat field makes the downstream A*
planner produce a near-shortest path, which is approximately what stock PNS
does interactively. The agent therefore starts at the classical baseline
instead of below it.

This file has no dependency on pcbworld_pns_bridge or pcbnew and runs
anywhere torch does.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcbworld.agents.cfp.modules import (
    CrossBlock,
    FiLM,
    NetSelfBlock,
    ResBlock,
    masked_mean,
    orthogonal_init,
)
from pcbworld.agents.cfp.spec import (
    NUM_ACTION_KINDS,
    NUM_CANVAS_CHANNELS,
    NUM_EDGE_TYPES,
    NUM_NET_FEATURES,
    CFPObservation,
)

# The canvas encoder halves resolution four times, so a 256x256 canvas
# becomes the 16x16 token grid the fusion and the field head both work on.
CANVAS_DOWNSAMPLE = 16

# Fill value for illegal pointer actions. Deliberately -1e9 and not -inf or
# torch.finfo.min: log_softmax subtracts the row max, and either of those
# underflows to -inf there, which then makes the categorical's entropy term
# 0 * -inf = NaN. -1e9 is small enough that exp() of it is exactly 0 in
# float32 and large enough that the subtraction stays finite.
MASK_LOGIT = -1e9


@dataclasses.dataclass
class CFPConfig:
    dim: int = 256
    num_heads: int = 8
    net_layers: int = 6           # relational self-attention blocks, pre-fusion
    fusion_rounds: int = 2
    canvas_base_channels: int = 64
    # Residual capacity per canvas stage, coarsest-last. Deliberately zero
    # in the two high-resolution stages: measured on a T4, the canvas
    # encoder was 71% of a forward pass and 8.3 GFLOPs/board, of which
    # stage0 (@128x128) and stage1 (@64x64) alone were 68% -- dense 3x3
    # convs over a mostly-empty binary raster. Moving that capacity down to
    # 32x32 and 16x16, where a residual block costs 16-64x less per
    # parameter, halves the FLOPs and *adds* parameters. An int is accepted
    # and broadcast to every stage, which is what the old scalar field did.
    canvas_blocks_per_stage: tuple[int, ...] | int = (0, 0, 1, 2)
    num_copper_layers: int = 2    # matches generate_board.py's 2-layer boards
    field_size: int = 16          # emitted field resolution, per plane
    initial_field_log_std: float = -0.5

    @property
    def num_field_planes(self) -> int:
        return self.num_copper_layers + 1  # + the reserve plane

    def stage_blocks(self, num_stages: int) -> tuple[int, ...]:
        if isinstance(self.canvas_blocks_per_stage, int):
            return (self.canvas_blocks_per_stage,) * num_stages
        assert len(self.canvas_blocks_per_stage) == num_stages, (
            f"canvas_blocks_per_stage has {len(self.canvas_blocks_per_stage)} entries, "
            f"the encoder has {num_stages} stages"
        )
        return tuple(self.canvas_blocks_per_stage)


@dataclasses.dataclass
class Encoded:
    """What one forward pass produces before a net has been chosen."""

    net_h: torch.Tensor         # (B, N, D)  fused per-net embeddings
    canvas_map: torch.Tensor    # (B, D, G, G) fused canvas features
    pointer_logits: torch.Tensor  # (B, NUM_ACTION_KINDS * N), illegal = -inf
    value: torch.Tensor         # (B,)


class CanvasEncoder(nn.Module):
    """Strided conv tower, 256x256 -> 16x16, GroupNorm throughout.

    Four stages, each halving resolution. Residual blocks are placed per
    stage rather than uniformly -- see CFPConfig.canvas_blocks_per_stage for
    the measurement that motivated it.
    """

    NUM_STAGES = 4

    def __init__(
        self, in_channels: int, base: int, dim: int, blocks_per_stage: tuple[int, ...]
    ) -> None:
        super().__init__()
        widths = [base, base * 2, base * 3, dim]
        assert len(widths) == self.NUM_STAGES == len(blocks_per_stage)
        layers: list[nn.Module] = []
        prev = in_channels
        for width, num_blocks in zip(widths, blocks_per_stage):
            layers.append(nn.Conv2d(prev, width, 3, stride=2, padding=1))
            layers.extend(ResBlock(width) for _ in range(num_blocks))
            prev = width
        layers.append(nn.GroupNorm(8, dim))
        layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CFPNet(nn.Module):
    def __init__(self, config: CFPConfig | None = None) -> None:
        super().__init__()
        self.config = config or CFPConfig()
        c = self.config
        d = c.dim

        self.canvas_encoder = CanvasEncoder(
            NUM_CANVAS_CHANNELS,
            c.canvas_base_channels,
            d,
            c.stage_blocks(CanvasEncoder.NUM_STAGES),
        )
        self.net_embed = nn.Sequential(nn.Linear(NUM_NET_FEATURES, d), nn.GELU(), nn.Linear(d, d))
        self.net_blocks = nn.ModuleList(
            NetSelfBlock(d, c.num_heads, NUM_EDGE_TYPES) for _ in range(c.net_layers)
        )

        # Fusion. Each round: nets read the canvas, the canvas reads the
        # nets, then nets re-mix relationally with what they just learned.
        self.net_from_canvas = nn.ModuleList(
            CrossBlock(d, c.num_heads) for _ in range(c.fusion_rounds)
        )
        self.canvas_from_net = nn.ModuleList(
            CrossBlock(d, c.num_heads) for _ in range(c.fusion_rounds)
        )
        self.net_refine = nn.ModuleList(
            NetSelfBlock(d, c.num_heads, NUM_EDGE_TYPES) for _ in range(c.fusion_rounds)
        )

        # Per-head inverse length-scales for the distance bias, one set per
        # direction. Softplus-ed so they stay positive: the bias is
        # -alpha * squared_distance, i.e. attention decays with distance,
        # and each head learns its own radius (some heads end up nearly
        # global, some nearly local -- that's the point of having several).
        self.net2canvas_alpha = nn.Parameter(torch.zeros(c.num_heads))
        self.canvas2net_alpha = nn.Parameter(torch.zeros(c.num_heads))

        self.pointer_norm = nn.LayerNorm(d)
        self.pointer_head = orthogonal_init(nn.Linear(d, NUM_ACTION_KINDS), gain=0.01)

        self.field_film = FiLM(d, d)
        self.field_norm = nn.GroupNorm(8, d)
        self.field_conv = nn.Sequential(
            nn.Conv2d(d, d // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d // 2, d // 4, 3, padding=1),
            nn.SiLU(),
        )
        self.field_out = orthogonal_init(
            nn.Conv2d(d // 4, c.num_field_planes, 3, padding=1), gain=0.01
        )
        # State-independent log_std, one scalar per plane. Deliberately not
        # per-cell and not state-conditioned: exploration noise on a field
        # has to be spatially coherent to mean anything (it should shift a
        # corridor, not dither pixels), and the coarse field_size grid plus
        # the planner's upsampling already provides that coherence.
        self.field_log_std = nn.Parameter(
            torch.full((c.num_field_planes,), c.initial_field_log_std)
        )

        self.value_head = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.SiLU(),
            orthogonal_init(nn.Linear(d, 1), gain=1.0),
        )

    # -- geometry helpers ---------------------------------------------------

    @staticmethod
    def _cell_centers(grid: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """(grid*grid, 2) normalized centers, matching net_xy's [0, 1] frame
        and row-major (y, x) flattening of the canvas map."""
        coords = (torch.arange(grid, device=device, dtype=dtype) + 0.5) / grid
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def _distance_bias(
        self, net_xy: torch.Tensor, grid: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Additive attention bias for both fusion directions.

        Returns (net->canvas bias (B,H,N,S), canvas->net bias (B,H,S,N)).
        """
        cells = self._cell_centers(grid, net_xy.device, net_xy.dtype)     # (S, 2)
        sqdist = torch.cdist(net_xy, cells.expand(net_xy.shape[0], -1, -1)) ** 2  # (B,N,S)
        n2c = -F.softplus(self.net2canvas_alpha)[None, :, None, None] * sqdist[:, None]
        c2n = -F.softplus(self.canvas2net_alpha)[None, :, None, None] * sqdist.transpose(
            1, 2
        )[:, None]
        return n2c, c2n

    # -- forward ------------------------------------------------------------

    def encode(self, obs: CFPObservation) -> Encoded:
        c = self.config
        b, _, h, w = obs.canvas.shape
        assert h % CANVAS_DOWNSAMPLE == 0 and w % CANVAS_DOWNSAMPLE == 0, (
            f"canvas {h}x{w} must be divisible by {CANVAS_DOWNSAMPLE}"
        )

        canvas_map = self.canvas_encoder(obs.canvas)          # (B, D, G, G)
        grid = canvas_map.shape[-1]
        canvas_tok = canvas_map.flatten(2).transpose(1, 2)    # (B, S, D)

        net_h = self.net_embed(obs.net_feats)                 # (B, N, D)
        for block in self.net_blocks:
            net_h = block(net_h, obs.edge_type, obs.net_mask)

        n2c_bias, c2n_bias = self._distance_bias(obs.net_xy, grid)
        for i in range(c.fusion_rounds):
            net_h = self.net_from_canvas[i](net_h, canvas_tok, bias=n2c_bias)
            canvas_tok = self.canvas_from_net[i](
                canvas_tok, net_h, bias=c2n_bias, key_mask=obs.net_mask
            )
            net_h = self.net_refine[i](net_h, obs.edge_type, obs.net_mask)

        # Padded slots contributed nothing to attention (they were masked as
        # keys) but their own rows still carry junk; zero them so neither
        # the value pooling nor a stray gather can pick them up.
        net_h = net_h * obs.net_mask.unsqueeze(-1)
        canvas_map = canvas_tok.transpose(1, 2).reshape(b, c.dim, grid, grid)

        logits = self.pointer_head(self.pointer_norm(net_h))  # (B, N, KINDS)
        logits = logits.permute(0, 2, 1).flatten(1)           # (B, KINDS*N)
        logits = logits.masked_fill(~obs.action_mask.flatten(1), MASK_LOGIT)

        pooled = torch.cat([masked_mean(net_h, obs.net_mask), canvas_map.mean(dim=(2, 3))], -1)
        value = self.value_head(pooled).squeeze(-1)

        return Encoded(net_h=net_h, canvas_map=canvas_map, pointer_logits=logits, value=value)

    def field_params(
        self, encoded: Encoded, net_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian parameters for the cost field of the chosen net.

        net_index: (B,) long, the *net slot* (not the flat action index --
        policy.py splits kind and slot before calling this).

        Returns (mean, log_std), both (B, num_field_planes, F, F).
        """
        c = self.config
        h_sel = encoded.net_h.gather(
            1, net_index[:, None, None].expand(-1, 1, encoded.net_h.shape[-1])
        ).squeeze(1)  # (B, D)

        x = self.field_film(encoded.canvas_map, h_sel)
        x = F.silu(self.field_norm(x))
        if x.shape[-1] != c.field_size:
            x = F.interpolate(x, size=(c.field_size, c.field_size), mode="bilinear",
                              align_corners=False)
        mean = self.field_out(self.field_conv(x))  # (B, P, F, F)
        log_std = self.field_log_std[None, :, None, None].expand_as(mean)
        return mean, log_std

    def forward(self, obs: CFPObservation) -> Encoded:  # convenience alias
        return self.encode(obs)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
