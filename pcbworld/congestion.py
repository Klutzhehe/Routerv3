"""Future Congestion Heatmap & Routing Demand Estimation.

Implements fast grid-based multi-path estimation (A* / wavefront potential)
to project where unrouted nets will need spatial channels.
"""

from __future__ import annotations

import heapq
import math
from typing import List, Tuple
import numpy as np

# Orthogonal and diagonal step costs, in cells -- same convention as
# pcbworld/env/geodesic.py's wavefront relaxation.
_ORTHO = 1.0
_DIAG = float(np.sqrt(2.0))


def _bilinear_upsample(coarse: np.ndarray, out_h: int, out_w: int, downsample_factor: int) -> np.ndarray:
    """Smoothly upsample a coarse grid to (out_h, out_w), instead of
    np.repeat's block-nearest-neighbor (every downsample_factor x
    downsample_factor block sharing one identical value). The env reads this
    field's local GRADIENT every step to steer (_geo_descent_dir) -- a
    blocky field's gradient is flat within a block and jumps at block
    boundaries, which is a real, measured source of the "should be a
    straight line but wobbles" artifact, independent of anything the policy
    has or hasn't learned. Bilinear interpolation makes the field, and so
    its gradient, continuous instead."""
    ds_h, ds_w = coarse.shape
    ys = np.clip((np.arange(out_h) + 0.5) / downsample_factor - 0.5, 0, ds_h - 1)
    xs = np.clip((np.arange(out_w) + 0.5) / downsample_factor - 0.5, 0, ds_w - 1)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, ds_h - 1)
    x1 = np.clip(x0 + 1, 0, ds_w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = coarse[y0][:, x0] * (1 - wx) + coarse[y0][:, x1] * wx
    bot = coarse[y1][:, x0] * (1 - wx) + coarse[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


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


def compute_geodesic_distance_field(
    grid_size: int,
    target_x: int,
    target_y: int,
    obstacle_mask: np.ndarray,
    downsample_factor: int = 2,
) -> np.ndarray:
    """Obstacle-aware cost-to-go, in grid cells, via Dijkstra flood from the target.

    A straight-line distance field pays off walking INTO an obstacle right up
    until the collision fires -- the same failure `pcbworld/env/geodesic.py`
    measured on the vector env (600k steps, two reward configs, policy still
    drew straight lines). This is that fix, but grid-native: flood-fill the
    rasterized obstacle mask once per net instead of relaxing a field over
    line segments. Cells the flood cannot reach (sealed pockets) fall back to
    Euclidean distance rather than infinity, so the field stays defined
    everywhere the head might legally be.

    Downsampled for speed -- this runs once per net, not once per step, same
    amortization as `compute_net_demand_heatmap` and `GeodesicField.build`.
    """
    ds_size = max(1, grid_size // downsample_factor)
    ds_obs = (
        obstacle_mask.reshape(ds_size, downsample_factor, ds_size, downsample_factor)
        .max(axis=(1, 3))
    )

    tx = int(np.clip(target_x // downsample_factor, 0, ds_size - 1))
    ty = int(np.clip(target_y // downsample_factor, 0, ds_size - 1))

    dist = np.full((ds_size, ds_size), np.inf, dtype=np.float32)
    visited = np.zeros((ds_size, ds_size), dtype=bool)
    dist[ty, tx] = 0.0
    heap: list = [(0.0, ty, tx)]
    neighbors = (
        (-1, 0, _ORTHO), (1, 0, _ORTHO), (0, -1, _ORTHO), (0, 1, _ORTHO),
        (-1, -1, _DIAG), (-1, 1, _DIAG), (1, -1, _DIAG), (1, 1, _DIAG),
    )
    while heap:
        d, y, x = heapq.heappop(heap)
        if visited[y, x]:
            continue
        visited[y, x] = True
        for dy, dx, cost in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < ds_size and 0 <= nx < ds_size and not visited[ny, nx] and ds_obs[ny, nx] == 0:
                nd = d + cost
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(heap, (nd, ny, nx))

    unreached = ~np.isfinite(dist)
    if unreached.any():
        yy, xx = np.mgrid[0:ds_size, 0:ds_size]
        dist[unreached] = np.hypot(xx - tx, yy - ty)[unreached]

    full = _bilinear_upsample(dist, grid_size, grid_size, downsample_factor) * downsample_factor
    return full.astype(np.float32)


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
