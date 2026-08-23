"""PCB Board Multi-Layer Grid Renderer.

Visualizes 256x256 multi-channel board state, traces, pads, obstacles,
routing heads, and congestion heatmaps into clean publication-quality PNG images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Palette for multi-net copper tracks
NET_COLORS = [
    "#00ffcc",  # Cyan (Net 1)
    "#ff0055",  # Neon Pink (Net 2)
    "#ffaa00",  # Amber (Net 3)
    "#aa00ff",  # Purple (Net 4)
    "#00bbff",  # Sky Blue (Net 5)
    "#33ff33",  # Lime Green (Net 6)
    "#ffff00",  # Bright Yellow (Net 7)
    "#ff7700",  # Orange (Net 8)
]


def render_grid_board(
    copper_grid: np.ndarray,  # (num_layers, H, W) int32 net IDs
    pads: list,
    obstacles: list,
    heads: list,
    congestion_map: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
    title: str = "AI PCB Router Grid State",
    dpi: int = 120,
) -> plt.Figure:
    """Render 3-panel display: Top Layer (F_Cu), Bottom Layer (B_Cu), and Composite/Congestion."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)
    fig.patch.set_facecolor("#101216")

    layers = ["Top Layer (F_Cu)", "Bottom Layer (B_Cu)", "Composite & Congestion"]
    grid_size = copper_grid.shape[1]

    for idx, ax in enumerate(axes):
        ax.set_facecolor("#181b22")
        ax.set_title(f"{title}\n{layers[idx]}", color="#e6edf3", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlim(0, grid_size)
        ax.set_ylim(grid_size, 0)  # Origin at top-left
        ax.set_aspect("equal")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # 1. Draw obstacles on all layers
    for obs in obstacles:
        for l_idx in range(2):
            if obs.layer == -1 or obs.layer == l_idx:
                rect = plt.Rectangle(
                    (obs.x1, obs.y1),
                    obs.x2 - obs.x1,
                    obs.y2 - obs.y1,
                    facecolor="#3d2222",
                    edgecolor="#ff4444",
                    linewidth=1.0,
                    alpha=0.8,
                )
                axes[l_idx].add_patch(rect)
                axes[2].add_patch(plt.Rectangle(
                    (obs.x1, obs.y1), obs.x2 - obs.x1, obs.y2 - obs.y1,
                    facecolor="#3d2222", edgecolor="#ff4444", linewidth=1.0, alpha=0.5
                ))

    # 2. Draw copper traces
    for l_idx in range(min(2, copper_grid.shape[0])):
        c_layer = copper_grid[l_idx]
        unique_nets = np.unique(c_layer)
        for nid in unique_nets:
            if nid <= 0:
                continue
            color = NET_COLORS[(nid - 1) % len(NET_COLORS)]
            y_indices, x_indices = np.where(c_layer == nid)
            axes[l_idx].scatter(x_indices, y_indices, c=color, s=2.0, alpha=0.9, marker="s")
            axes[2].scatter(x_indices, y_indices, c=color, s=1.5, alpha=0.7, marker="s")

    # 3. Draw Pads
    for pad in pads:
        color = NET_COLORS[(pad.net_id - 1) % len(NET_COLORS)]
        circle = plt.Circle(
            (pad.x, pad.y),
            pad.radius,
            facecolor="#ffffff" if pad.is_source else color,
            edgecolor=color,
            linewidth=2.0,
            zorder=10,
        )
        axes[pad.layer].add_patch(circle)
        
        # Composite
        axes[2].add_patch(plt.Circle(
            (pad.x, pad.y), pad.radius, facecolor=color, edgecolor="#ffffff", linewidth=1.5, zorder=10
        ))
        label = f"S{pad.net_id}" if pad.is_source else f"T{pad.net_id}"
        axes[pad.layer].text(pad.x, pad.y, label, color="#000000", fontsize=7, ha="center", va="center", fontweight="bold", zorder=11)

    # 4. Draw Active Routing Heads
    for h in heads:
        color = NET_COLORS[(h["net_id"] - 1) % len(NET_COLORS)]
        hl = h.get("layer", 0)
        axes[hl].scatter([h["x"]], [h["y"]], c=color, s=80, marker="*", edgecolor="#ffffff", linewidth=1.2, zorder=15)
        axes[2].scatter([h["x"]], [h["y"]], c=color, s=80, marker="*", edgecolor="#ffffff", linewidth=1.2, zorder=15)

    # 5. Overlay congestion heatmap in panel 3
    if congestion_map is not None:
        axes[2].imshow(congestion_map, cmap="inferno", alpha=0.45, extent=(0, grid_size, grid_size, 0), zorder=1)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=dpi)

    return fig
