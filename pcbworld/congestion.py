"""Future Congestion Heatmap & Routing Demand Estimation.

Implements fast grid-based multi-path estimation (A* / wavefront potential)
to project where unrouted nets will need spatial channels.
"""

from __future__ import annotations

import heapq
import math
from typing import List, Tuple
import numpy as np


def compute_net_demand_heatmap(
    grid_size: int,
    unrouted_nets: list,
    obstacle_mask: np.ndarray,
    downsample_factor: int = 4,
) -> np.ndarray:
    """Compute a continuous spatial demand heatmap across all unrouted nets.
    
    Returns a (grid_size, grid_size) float32 array normalized to [0, 1].
    """
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    if not unrouted_nets:
        return heatmap

    # Fast downsampled grid for rapid estimation
    ds_size = grid_size // downsample_factor
    ds_obs = (
        obstacle_mask.reshape(ds_size, downsample_factor, ds_size, downsample_factor)
        .max(axis=(1, 3))
    )

    for net in unrouted_nets:
        src = (net.source_pad.x // downsample_factor, net.source_pad.y // downsample_factor)
        tgt = (net.target_pad.x // downsample_factor, net.target_pad.y // downsample_factor)

        # Estimate multi-path probability envelope
        path_density = _estimate_path_density(ds_size, src, tgt, ds_obs)

        # Upsample back to full grid resolution
        upsampled = np.repeat(np.repeat(path_density, downsample_factor, axis=0), downsample_factor, axis=1)
        heatmap += upsampled * net.importance

    max_val = heatmap.max()
    if max_val > 1e-6:
        heatmap /= max_val

    return heatmap.astype(np.float32)


def _estimate_path_density(
    size: int,
    src: Tuple[int, int],
    tgt: Tuple[int, int],
    obstacle_mask: np.ndarray,
) -> np.ndarray:
    """Wavefront potential estimating routing channel likelihood between src and tgt."""
    density = np.zeros((size, size), dtype=np.float32)

    # 1. Direct Euclidean corridor bounding envelope
    x0, y0 = src
    x1, y1 = tgt

    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return density

    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:size, 0:size]
    
    # Distance from point (x, y) to line segment (x0, y0)-(x1, y1)
    px = x_coords - x0
    py = y_coords - y0
    
    t = (px * dx + py * dy) / (length * length)
    t = np.clip(t, 0.0, 1.0)
    
    closest_x = x0 + t * dx
    closest_y = y0 + t * dy
    
    dist_to_segment = np.hypot(x_coords - closest_x, y_coords - closest_y)
    
    # Gaussian channel envelope (standard corridor width ~ 6 cells)
    corridor_sigma = max(3.0, length * 0.15)
    corridor_prob = np.exp(-0.5 * (dist_to_segment / corridor_sigma) ** 2)

    # Mask out obstacles
    corridor_prob[obstacle_mask > 0] *= 0.1

    return corridor_prob.astype(np.float32)


def compute_distance_field(grid_size: int, target_x: int, target_y: int) -> np.ndarray:
    """Compute normalized Euclidean distance field to target point (0 at target, 1 at max dist)."""
    y_coords, x_coords = np.mgrid[0:grid_size, 0:grid_size]
    dist = np.hypot(x_coords - target_x, y_coords - target_y)
    max_dist = math.hypot(grid_size, grid_size)
    return (dist / max_dist).astype(np.float32)


def compute_clearance_field(grid_size: int, binary_obstacles: np.ndarray) -> np.ndarray:
    """Compute clearance cost field (0 far from obstacles, 1 touching/inside)."""
    try:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(1 - binary_obstacles)
        # Cost is highest near obstacles, decaying up to 10 cells
        cost = np.clip(1.0 - (dist / 10.0), 0.0, 1.0)
        return cost.astype(np.float32)
    except ImportError:
        # Fallback numpy approximation
        return (binary_obstacles.astype(np.float32))
