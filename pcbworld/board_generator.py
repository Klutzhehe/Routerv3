"""Synthetic Grid Board Generator for AI PCB Autorouter Platform.

Generates 2D/3D multi-layer grid boards with pads, obstacles, and net configurations
compatible with PCBRouterEnv (256x256 resolution).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Pad:
    net_id: int
    x: int
    y: int
    layer: int = 0  # 0: Top (F_Cu), 1: Bottom (B_Cu)
    radius: int = 4  # Radius in grid cells
    is_source: bool = True


@dataclass
class Obstacle:
    x1: int
    y1: int
    x2: int
    y2: int
    layer: int = -1  # -1 means spans all layers, 0 = Top, 1 = Bottom


@dataclass
class NetSpec:
    net_id: int
    name: str
    source_pad: Pad
    target_pad: Pad
    trace_width: int = 2  # Width in grid cells
    clearance: int = 3    # Minimum clearance in cells
    importance: float = 1.0
    is_diff_pair: bool = False
    diff_pair_partner: Optional[int] = None
    target_length: Optional[float] = None


@dataclass
class BoardState:
    grid_size: int = 256
    num_layers: int = 2
    nets: List[NetSpec] = field(default_factory=list)
    obstacles: List[Obstacle] = field(default_factory=list)
    copper_grid: np.ndarray = field(default_factory=lambda: np.zeros((2, 256, 256), dtype=np.int32))


def generate_random_board(
    grid_size: int = 256,
    num_nets: int = 1,
    num_obstacles: int = 0,
    num_layers: int = 2,
    min_pad_dist: int = 30,
    margin: int = 16,
    pad_radius: int = 4,
    trace_width: int = 2,
    clearance: int = 3,
    seed: Optional[int] = None,
) -> BoardState:
    """Generate a randomized PCB grid board with nets and optional obstacles."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    board = BoardState(
        grid_size=grid_size,
        num_layers=num_layers,
        copper_grid=np.zeros((num_layers, grid_size, grid_size), dtype=np.int32),
    )

    placed_pads: List[Tuple[int, int, int]] = []  # (x, y, layer)

    # 1. Place obstacles (if any)
    for _ in range(num_obstacles):
        w = random.randint(15, 45)
        h = random.randint(15, 45)
        ox = random.randint(margin, grid_size - margin - w)
        oy = random.randint(margin, grid_size - margin - h)
        layer = random.choice([-1, 0, 1]) if num_layers > 1 else 0
        board.obstacles.append(Obstacle(x1=ox, y1=oy, x2=ox + w, y2=oy + h, layer=layer))

    def is_valid_pad_pos(x: int, y: int, layer: int) -> bool:
        # Margin check
        if x < margin or x > grid_size - margin or y < margin or y > grid_size - margin:
            return False
        # Obstacle check
        for obs in board.obstacles:
            if (obs.layer == -1 or obs.layer == layer) and (
                obs.x1 - pad_radius <= x <= obs.x2 + pad_radius
                and obs.y1 - pad_radius <= y <= obs.y2 + pad_radius
            ):
                return False
        # Distance to existing pads
        for px, py, pl in placed_pads:
            dist = math.hypot(x - px, y - py)
            if dist < (pad_radius * 2 + clearance * 2):
                return False
        return True

    # 2. Place nets
    for net_idx in range(1, num_nets + 1):
        attempts = 0
        while attempts < 200:
            attempts += 1
            src_layer = random.randint(0, num_layers - 1) if num_layers > 1 else 0
            tgt_layer = src_layer  # Default same layer for basic stages

            sx = random.randint(margin, grid_size - margin)
            sy = random.randint(margin, grid_size - margin)
            tx = random.randint(margin, grid_size - margin)
            ty = random.randint(margin, grid_size - margin)

            if math.hypot(sx - tx, sy - ty) < min_pad_dist:
                continue

            if is_valid_pad_pos(sx, sy, src_layer) and is_valid_pad_pos(tx, ty, tgt_layer):
                src_pad = Pad(net_id=net_idx, x=sx, y=sy, layer=src_layer, radius=pad_radius, is_source=True)
                tgt_pad = Pad(net_id=net_idx, x=tx, y=ty, layer=tgt_layer, radius=pad_radius, is_source=False)
                
                placed_pads.append((sx, sy, src_layer))
                placed_pads.append((tx, ty, tgt_layer))

                net_spec = NetSpec(
                    net_id=net_idx,
                    name=f"NET_{net_idx}",
                    source_pad=src_pad,
                    target_pad=tgt_pad,
                    trace_width=trace_width,
                    clearance=clearance,
                    importance=1.0 if net_idx == 1 else random.uniform(0.3, 1.0),
                )
                board.nets.append(net_spec)
                break

    return board
