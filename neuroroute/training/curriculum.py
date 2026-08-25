"""Curriculum stages, and the forecaster's supervision targets.

Every stage changes the **data** and nothing else. The model has no
stage-dependent parameters, no head that switches on, no observation channel
that appears at stage 5 -- which is what makes the jump from stage 3 (200 nets)
to stage 8 (3000 nets) a pure generalisation test rather than a new
architecture. If the model needed a structural change to handle more nets, the
scaling claim in DESIGN.md section 1.3 would be false and there would be no
point testing it.

Advancement is on **held-out completion rate**, never on reward.
`docs/RL_PLAN.md` measured the two disagreeing directly: a random policy scored
-330 reward against greedy's -177 and still routed more nets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from neuroroute.env.route_env import EnvConfig
from neuroroute.world.engine import STATUS_FAILED, BatchedRouterWorld, WorldConfig
from neuroroute.world.generator import GeneratorConfig
from neuroroute.world.spec import BoardSpec, LayerStack


@dataclass
class Stage:
    name: str
    board: BoardSpec
    generator: GeneratorConfig
    max_steps_per_net: int = 96
    max_episode_steps: int = 512
    #: Held-out completion rate required to advance.
    gate: float = 0.9
    #: What is new here, for the training log.
    introduces: str = ""


def default_curriculum(layers_max: int = 8, size: int = 128) -> list[Stage]:
    small = BoardSpec(height_cells=64, width_cells=64, layers=LayerStack(num_layers=2))
    mid2 = BoardSpec(height_cells=size, width_cells=size, layers=LayerStack(num_layers=2))
    full = BoardSpec(height_cells=size, width_cells=size, layers=LayerStack(num_layers=layers_max))

    return [
        Stage(
            "0-plumbing", small,
            GeneratorConfig(num_nets=1, num_components=2, pin_rows=(2, 3), pin_cols=(2, 3)),
            max_steps_per_net=64, max_episode_steps=96, gate=0.98,
            introduces="one net on an empty board. Anything below ~100% here is "
                       "broken plumbing, not a hard problem.",
        ),
        Stage(
            "1-congestion", small,
            GeneratorConfig(num_nets=20, num_components=5),
            gate=0.75, introduces="many nets on two layers: congestion and ordering.",
        ),
        Stage(
            "2-layers", mid2,
            GeneratorConfig(num_nets=20, num_components=6),
            gate=0.85, introduces="a bigger board at the same net count.",
        ),
        Stage(
            "3-eight-layers", full,
            GeneratorConfig(num_nets=60, num_components=8),
            gate=0.85, introduces="8 layers. Vias become the main lever -- the "
                                  "capability KiCad's PNS is 0-for-32 on.",
        ),
        Stage(
            "4-scale", full,
            GeneratorConfig(num_nets=200, num_components=14),
            max_steps_per_net=128, max_episode_steps=1024, gate=0.9,
            introduces="200 nets. The learned scheduler starts to matter.",
        ),
        Stage(
            "5-rules", full,
            GeneratorConfig(num_nets=200, num_components=14, wide_net_frac=0.25, num_keepouts=3),
            max_steps_per_net=128, max_episode_steps=1024, gate=0.9,
            introduces="variable trace widths, via classes, keepouts.",
        ),
        Stage(
            "6-diff-pairs", full,
            GeneratorConfig(num_nets=200, num_components=14, wide_net_frac=0.2,
                            diff_pair_frac=0.25, num_keepouts=3),
            max_steps_per_net=128, max_episode_steps=1024, gate=0.85,
            introduces="differential pairs, learned: the `couple` action, gap and "
                       "skew in the reward. No PNS diff-pair solver.",
        ),
        Stage(
            "7-length-groups", full,
            GeneratorConfig(num_nets=200, num_components=14, wide_net_frac=0.2,
                            diff_pair_frac=0.2, length_group_frac=0.25,
                            length_group_size=4, num_keepouts=3),
            max_steps_per_net=160, max_episode_steps=1280, gate=0.85,
            introduces="length-matched groups. Meanders are whatever the refine "
                       "policy learns, not a meander generator.",
        ),
        Stage(
            "8-pours", full,
            GeneratorConfig(num_nets=200, num_components=14, wide_net_frac=0.2,
                            diff_pair_frac=0.2, length_group_frac=0.2,
                            num_keepouts=3, num_pours=2),
            max_steps_per_net=160, max_episode_steps=1280, gate=0.85,
            introduces="copper pours as both obstacle and terminal.",
        ),
    ]


def stage_env_config(stage: Stage, world: WorldConfig, base: EnvConfig) -> EnvConfig:
    return replace(
        base,
        spec=stage.board,
        generator=stage.generator,
        world=replace(
            world,
            max_steps_per_net=stage.max_steps_per_net,
            max_nets=max(world.max_nets, stage.generator.num_nets + 8),
        ),
        max_episode_steps=stage.max_episode_steps,
    )


# ---------------------------------------------------------------------------
# Forecaster targets, read off a finished episode
# ---------------------------------------------------------------------------


@torch.no_grad()
def episode_targets(world: BatchedRouterWorld, latent_shape: tuple[int, int, int]) -> dict[str, torch.Tensor]:
    """Turn the terminal board state into dense supervision.

    The whole point of predicting *fields* rather than a scalar is that this
    function is nearly free and produces ``L*h*w`` labels per board from a
    rollout that was going to be collected anyway. Four previous lookahead
    efforts in this repo needed a separate data-collection phase
    (`jepa/collect_transitions.py`) to produce one label per sample.

    Returns targets already pooled onto the encoder's latent grid.
    """
    occ = world.occ
    B, L, H, W = occ.shape
    dev = occ.device
    Lz, h, w = latent_shape

    occupied = (occ > 0).float()

    # Contention proxy: a cell that is occupied and whose left neighbour has a
    # *different* owner is the edge of a track. Counting edges rather than
    # cells is what separates "one fat trace" from "four thin ones sharing a
    # channel", and the second is what congestion actually means.
    left = F.pad(occ[..., :-1], (1, 0), value=0)
    edges = ((occ > 0) & (occ != left)).float()

    # Jam: where the nets that ultimately FAILED needed to go. Their pads are
    # known, so a straight line between them is a fair statement of the demand
    # that went unserved.
    failed = (world.net_status == STATUS_FAILED) & world.net_valid
    jam = torch.zeros_like(occupied)
    for b in range(B):
        idx = torch.nonzero(failed[b], as_tuple=False).flatten()
        for n in idx.tolist():
            for leg in range(2 if int(world.net_kind[b, n]) == 1 else 1):
                s = world.net_src[b, n, leg]
                t = world.net_dst[b, n, leg]
                steps = int(max(abs(int(t[1] - s[1])), abs(int(t[2] - s[2])), 1))
                ts = torch.linspace(0.0, 1.0, steps + 1, device=dev)
                ys = (s[1] + (t[1] - s[1]) * ts).round().long().clamp(0, H - 1)
                xs = (s[2] + (t[2] - s[2]) * ts).round().long().clamp(0, W - 1)
                for ly in {int(s[0]), int(t[0])}:
                    jam[b, ly, ys, xs] = 1.0

    pool = lambda t: F.adaptive_avg_pool3d(t.unsqueeze(1), (Lz, h, w))  # noqa: E731
    return {"occupancy": pool(occupied), "contention": pool(edges), "jam": pool(jam)}


@torch.no_grad()
def demand_baseline(demand: torch.Tensor, latent_shape: tuple[int, int, int]) -> torch.Tensor:
    """Pool the straight-line demand channel onto the latent grid.

    This is what `models/forecaster.py::forecast_gate` compares against. It is
    the honest bar: it costs nothing, needs no training, and already knows
    where every net's endpoints are.
    """
    Lz, h, w = latent_shape
    return F.adaptive_avg_pool3d(demand.unsqueeze(1), (Lz, h, w))
