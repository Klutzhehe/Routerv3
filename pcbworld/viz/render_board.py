"""Draws a pcbworld_pns_bridge BoardGeometry with matplotlib.

Takes whatever bridge.get_board_geometry() returns and draws board edge,
tracks, vias, and pads, colored by net. Supports whole-board views, single-layer
views (F_Cu / B_Cu), and multi-panel layer split visualizations.

Coordinates are nm (KiCad internal units, 1mm = 1e6 nm) -- converted to mm for the plot.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

MM = 1_000_000

# Standard KiCad 2-layer indices
LAYER_F_CU = 0
LAYER_B_CU = 31  # KiCad standard B_Cu index (also handles 1 for 2-layer fixtures)


def _net_color(net: str) -> tuple[float, float, float]:
    """Deterministic, visually-distinct color per net name."""
    if not net:
        return (0.55, 0.55, 0.55)  # unnamed/no-net items: neutral gray
    digest = hashlib.sha1(net.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    return colorsys.hsv_to_rgb(hue, 0.65, 0.85)


def _expand_bounds(bounds: list[float] | None, x: float, y: float) -> list[float]:
    """Grows [min_x, max_x, min_y, max_y] to include (x, y)."""
    if bounds is None:
        return [x, x, y, y]
    bounds[0] = min(bounds[0], x)
    bounds[1] = max(bounds[1], x)
    bounds[2] = min(bounds[2], y)
    bounds[3] = max(bounds[3], y)
    return bounds


def _draw_edge(ax, edge, mm: float) -> None:
    x1, y1, x2, y2 = edge.x1 / mm, edge.y1 / mm, edge.x2 / mm, edge.y2 / mm
    shape = getattr(edge, "shape_type", "segment")
    if shape == "circle":
        cx, cy = x1, y1
        r = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor="black", linewidth=1.0, linestyle="--"))
    else:
        ax.add_patch(
            Rectangle(
                (min(x1, x2), min(y1, y2)),
                abs(x2 - x1),
                abs(y2 - y1),
                fill=False,
                edgecolor="black",
                linewidth=1.0,
            )
        )


def _layer_matches(target_layer: int | None, item_layer: int) -> bool:
    if target_layer is None:
        return True
    if target_layer in (LAYER_B_CU, 1) and item_layer in (LAYER_B_CU, 1):
        return True
    return target_layer == item_layer


def _pad_spans_layer(target_layer: int | None, pad) -> bool:
    if target_layer is None:
        return True
    top = getattr(pad, "layer_top", 0)
    bot = getattr(pad, "layer_bottom", 0)
    if target_layer == 0:
        return top == 0 or bot == 0
    # B_Cu
    return top in (LAYER_B_CU, 1) or bot in (LAYER_B_CU, 1) or (top == 0 and bot in (LAYER_B_CU, 1, 2))


def render_board(
    geometry,
    net_pads=None,
    ax: "plt.Axes | None" = None,
    title: str | None = None,
    unit: str = "mm",
    layer: int | None = None,
):
    """Draws `geometry` (a BoardGeometry) onto `ax`.

    layer: optional layer filter (0 for F_Cu, 31 or 1 for B_Cu, None for all layers).
    """
    mm = MM if unit == "mm" else 1.0
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    bounds: list[float] | None = None

    for edge in geometry.board_edge:
        _draw_edge(ax, edge, mm)
        bounds = _expand_bounds(bounds, edge.x1 / mm, edge.y1 / mm)
        bounds = _expand_bounds(bounds, edge.x2 / mm, edge.y2 / mm)

    seen_nets: set[str] = set()
    track_count = 0
    for t in geometry.tracks:
        if not _layer_matches(layer, t.layer):
            continue
        track_count += 1
        color = _net_color(t.net)
        seen_nets.add(t.net)
        ax.plot(
            [t.x1 / mm, t.x2 / mm],
            [t.y1 / mm, t.y2 / mm],
            color=color,
            linewidth=max(0.75, t.width / mm * 2.5),
            solid_capstyle="round",
        )
        bounds = _expand_bounds(bounds, t.x1 / mm, t.y1 / mm)
        bounds = _expand_bounds(bounds, t.x2 / mm, t.y2 / mm)

    via_count = 0
    for v in geometry.vias:
        top = getattr(v, "layer_top", 0)
        bot = getattr(v, "layer_bottom", LAYER_B_CU)
        if layer is not None and not (top <= layer <= bot or _layer_matches(layer, top) or _layer_matches(layer, bot)):
            continue
        via_count += 1
        color = _net_color(v.net)
        seen_nets.add(v.net)
        ax.add_patch(Circle((v.x / mm, v.y / mm), v.diameter / mm / 2, color=color, alpha=0.85))
        ax.add_patch(Circle((v.x / mm, v.y / mm), v.drill / mm / 2, color="white"))
        r = v.diameter / mm / 2
        bounds = _expand_bounds(bounds, v.x / mm - r, v.y / mm - r)
        bounds = _expand_bounds(bounds, v.x / mm + r, v.y / mm + r)

    pad_source = list(geometry.pads) if geometry.pads else list(net_pads or [])
    for p in pad_source:
        if not _pad_spans_layer(layer, p):
            continue
        color = _net_color(p.net)
        seen_nets.add(p.net)
        size_x = getattr(p, "size_x", 0.5 * MM) / mm
        size_y = getattr(p, "size_y", 0.5 * MM) / mm
        ax.add_patch(
            Rectangle(
                (p.x / mm - size_x / 2, p.y / mm - size_y / 2),
                size_x,
                size_y,
                color=color,
            )
        )
        bounds = _expand_bounds(bounds, p.x / mm - size_x / 2, p.y / mm - size_y / 2)
        bounds = _expand_bounds(bounds, p.x / mm + size_x / 2, p.y / mm + size_y / 2)

    if bounds is not None:
        min_x, max_x, min_y, max_y = bounds
        margin_x = max((max_x - min_x) * 0.05, 1.0)
        margin_y = max((max_y - min_y) * 0.05, 1.0)
        ax.set_xlim(min_x - margin_x, max_x + margin_x)
        ax.set_ylim(min_y - margin_y, max_y + margin_y)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")

    routed = sum(1 for n in seen_nets if n)
    layer_name = "All Layers" if layer is None else ("F_Cu (Top)" if layer == 0 else "B_Cu (Bottom)")
    ax.set_title(
        title
        or f"{layer_name}: {track_count} track(s), {via_count} via(s), {routed} net(s)"
    )
    return ax


def render_board_layers_split(
    geometry,
    net_pads=None,
    figsize: tuple[float, float] = (16.0, 5.5),
    save_path: str | None = None,
    dpi: int = 150,
    title: str | None = None,
):
    """Renders a 3-panel layer breakdown: Front Copper (F_Cu), Back Copper (B_Cu), and Composite."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    render_board(geometry, net_pads=net_pads, ax=axes[0], layer=0, title="Top Layer (F_Cu)")
    render_board(geometry, net_pads=net_pads, ax=axes[1], layer=LAYER_B_CU, title="Bottom Layer (B_Cu)")
    render_board(geometry, net_pads=net_pads, ax=axes[2], layer=None, title="Composite (All Layers)")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved layer split visualization to {save_path}")

    return fig, axes


def main():
    parser = argparse.ArgumentParser(description="Render PCB geometry and export layer-split visualizations.")
    parser.add_argument("board_path", help="Path to .kicad_pcb board file")
    parser.add_argument("-o", "--output", default="board_render.png", help="Output PNG path")
    parser.add_argument("--split-layers", action="store_true", help="Generate side-by-side per-layer panels")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    args = parser.parse_args()

    import pcbworld_pns_bridge as bridge

    b = bridge.PNSBridge()
    if not b.load_board(args.board_path):
        print(f"Error: failed to load board {args.board_path}", file=sys.stderr)
        sys.exit(1)

    geometry = b.get_board_geometry()
    pads = b.net_pads()

    if args.split_layers:
        render_board_layers_split(geometry, net_pads=pads, save_path=args.output, dpi=args.dpi, title=f"Board: {Path(args.board_path).name}")
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        render_board(geometry, net_pads=pads, ax=ax)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved board render to {args.output}")


if __name__ == "__main__":
    main()

