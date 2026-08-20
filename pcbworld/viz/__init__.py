"""Rendering utilities for pcbworld_pns_bridge's BoardGeometry.

Deliberately pure Python + matplotlib, no pcbnew dependency -- unlike
scripts/plot-style tooling that would need a second system-pcbnew process
(see docs/performance.md), this reads only get_board_geometry()'s already-
bound fields (TrackSegment/ViaGeom/PadGeom/EdgeShape, bindings.cpp), so it
can run directly inside the same kernel/process that already has
pcbworld_pns_bridge loaded -- no process-separation dance needed.
"""

from pcbworld.viz.render_board import render_board

__all__ = ["render_board"]
