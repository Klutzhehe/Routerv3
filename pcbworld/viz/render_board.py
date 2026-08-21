"""Draws a pcbworld_pns_bridge BoardGeometry with matplotlib.

Takes whatever bridge.get_board_geometry() returns and draws board edge,
tracks, vias, and pads, colored by net. Duck-typed against the field names
bindings.cpp actually exposes (TrackSegment.x1/y1/x2/y2/width/layer/net,
ViaGeom.x/y/diameter/net, EdgeShape.shape_type/x1/y1/x2/y2, PadGeom.x/y/
size_x/size_y/net) rather than a specific class, so it works identically
against the real bridge object and against a plain namedtuple fixture with
matching field names -- which is how this is tested locally, since nothing
here touches pcbnew or the bridge itself.

Coordinates are nm (KiCad internal units, 1mm = 1e6 nm, same convention as
every env/script in this repo) -- converted to mm for the plot.
"""

from __future__ import annotations

import colorsys
import hashlib

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

MM = 1_000_000


def _net_color(net: str) -> tuple[float, float, float]:
    """Deterministic, visually-distinct-ish color per net name.

    A fixed palette (tab20 etc.) repeats after ~20 colors, which a 24+ net
    board (this repo's default synthetic boards) blows past immediately --
    two unrelated nets would render identically. Hashing the net name into
    a hue instead scales to arbitrary net counts without repeats, at some
    cost to how distinguishable any two SPECIFIC colors are; acceptable for
    a debug/inspection view, not a print-quality deliverable.
    """
    if not net:
        return (0.55, 0.55, 0.55)  # unnamed/no-net items: neutral gray
    digest = hashlib.sha1(net.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    return colorsys.hsv_to_rgb(hue, 0.65, 0.85)


def _expand_bounds(bounds: list[float] | None, x: float, y: float) -> list[float]:
    """Grows [min_x, max_x, min_y, max_y] to include (x, y). bounds=None on
    the first call starts it at that single point."""
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
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor="black", linewidth=0.8))
    else:
        # rect / segment / arc / polygon -- the bridge itself falls back to
        # a bounding box for arc/polygon (pns_bridge.cpp), so a rectangle
        # from (x1,y1) to (x2,y2) is the right primitive for everything
        # except circle.
        ax.add_patch(
            Rectangle(
                (min(x1, x2), min(y1, y2)),
                abs(x2 - x1),
                abs(y2 - y1),
                fill=False,
                edgecolor="black",
                linewidth=0.8,
            )
        )


def render_board(
    geometry,
    net_pads=None,
    ax: "plt.Axes | None" = None,
    title: str | None = None,
    unit: str = "mm",
):
    """Draws `geometry` (a BoardGeometry, real or fixture) onto `ax`.

    net_pads: optional list of NetPad-like objects (net, pad_name, x, y) --
      bridge.net_pads() output. GetBoardGeometry()'s own `pads` field may or
      may not be populated depending on bridge version; net_pads() is
      known-populated in every run so far this project, so it's accepted as
      a fallback/supplement pad source rather than assumed unnecessary.
    unit: "mm" (default) or "nm" for the axis scale.

    Returns the Axes drawn on, so the caller can plt.show()/plt.savefig()
    -- in a Jupyter/Colab cell, returning the Axes/Figure as the cell's
    last expression is enough to display it inline with no extra call.
    """
    mm = MM if unit == "mm" else 1.0
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Explicit bounding box, not matplotlib's autoscale. ax.plot() (used for
    # tracks below) autoscales the view automatically; ax.add_patch() (used
    # for board edges, vias, and pads) does NOT -- a real matplotlib
    # behavior, not a version quirk. A board state with real pads but zero
    # committed tracks and no board_edge shape (any not-yet-routed net,
    # first thing after LoadBoard(), or a net that got abandoned before
    # ever committing) previously rendered as a blank (0,1)x(0,1) plot even
    # though the pads were genuinely present in the data -- confirmed by
    # reproducing it locally with a synthetic BoardGeometry (real pads,
    # empty tracks/vias/board_edge) before this fix, from a live Colab run
    # whose first net's PNG showed exactly this. Tracking bounds explicitly
    # across every primitive, patches included, and setting xlim/ylim from
    # it directly sidesteps the whole add_patch-doesn't-autoscale issue
    # rather than depending on matplotlib's autoscale picking it up.
    bounds: list[float] | None = None

    for edge in geometry.board_edge:
        _draw_edge(ax, edge, mm)
        bounds = _expand_bounds(bounds, edge.x1 / mm, edge.y1 / mm)
        bounds = _expand_bounds(bounds, edge.x2 / mm, edge.y2 / mm)

    seen_nets: set[str] = set()
    for t in geometry.tracks:
        color = _net_color(t.net)
        seen_nets.add(t.net)
        ax.plot(
            [t.x1 / mm, t.x2 / mm],
            [t.y1 / mm, t.y2 / mm],
            color=color,
            linewidth=max(0.5, t.width / mm * 2),  # visually readable, not to scale
            solid_capstyle="round",
        )
        bounds = _expand_bounds(bounds, t.x1 / mm, t.y1 / mm)
        bounds = _expand_bounds(bounds, t.x2 / mm, t.y2 / mm)

    for v in geometry.vias:
        color = _net_color(v.net)
        seen_nets.add(v.net)
        ax.add_patch(Circle((v.x / mm, v.y / mm), v.diameter / mm / 2, color=color, alpha=0.85))
        ax.add_patch(Circle((v.x / mm, v.y / mm), v.drill / mm / 2, color="white"))
        r = v.diameter / mm / 2
        bounds = _expand_bounds(bounds, v.x / mm - r, v.y / mm - r)
        bounds = _expand_bounds(bounds, v.x / mm + r, v.y / mm + r)

    pad_source = list(geometry.pads) if geometry.pads else list(net_pads or [])
    for p in pad_source:
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
        # A margin proportional to the span, with an absolute floor so a
        # single point (one pad, nothing else) doesn't collapse to a
        # zero-size view.
        margin_x = max((max_x - min_x) * 0.05, 1.0)
        margin_y = max((max_y - min_y) * 0.05, 1.0)
        ax.set_xlim(min_x - margin_x, max_x + margin_x)
        ax.set_ylim(min_y - margin_y, max_y + margin_y)

    ax.set_aspect("equal")
    ax.invert_yaxis()  # KiCad's Y grows downward; matplotlib's grows upward
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    routed = sum(1 for n in seen_nets if n)
    ax.set_title(
        title
        or f"{len(geometry.tracks)} track segment(s), {len(geometry.vias)} via(s), "
        f"{routed} net(s) with copper"
    )
    return ax
