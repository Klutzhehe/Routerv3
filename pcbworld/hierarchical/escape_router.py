"""Phase 0.5: Concurrent Pin Escape & Fanout Router for Dense Pad Clusters.

Implements simultaneous round-robin stepping to fan out all pins in dense clusters
(BGAs, high-density ICs, pin headers) simultaneously. Prevents outer pins from trapping
inner pins before the macro-routing phases begin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Any

from pcbworld.hierarchical.specs import PadInfo, PipelinePhase


@dataclass
class EscapeStub:
    """Escaped endpoint for a pad after concurrent fanout."""
    net_name: str
    pad_name: str
    orig_x: int  # in nm
    orig_y: int  # in nm
    escape_x: int  # in nm
    escape_y: int  # in nm
    layer: int
    step_count: int


class ConcurrentEscapeRouter:
    """Simultaneously steps out all pins in dense clusters to create clear breakout stubs."""

    def __init__(
        self,
        cluster_threshold_nm: int = 3_000_000,  # 3.0 mm clustering radius
        escape_distance_nm: int = 2_000_000,    # 2.0 mm escape length
        step_size_nm: int = 500_000,            # 0.5 mm per round-robin step
        min_cluster_size: int = 3,              # At least 3 nearby pads to trigger cluster escape
    ):
        self.cluster_threshold_nm = cluster_threshold_nm
        self.escape_distance_nm = escape_distance_nm
        self.step_size_nm = step_size_nm
        self.min_cluster_size = min_cluster_size

    def identify_clusters(self, pads: List[PadInfo]) -> List[List[PadInfo]]:
        """Groups pads into spatial density clusters using proximity thresholding."""
        clusters: List[List[PadInfo]] = []
        visited: Set[str] = set()

        for i, p1 in enumerate(pads):
            key1 = f"{p1.net}:{p1.pad_name}:{p1.x}:{p1.y}"
            if key1 in visited:
                continue

            cluster = [p1]
            visited.add(key1)

            for j, p2 in enumerate(pads):
                if i == j:
                    continue
                key2 = f"{p2.net}:{p2.pad_name}:{p2.x}:{p2.y}"
                if key2 in visited:
                    continue

                dist = math.hypot(p2.x - p1.x, p2.y - p1.y)
                if dist <= self.cluster_threshold_nm:
                    cluster.append(p2)
                    visited.add(key2)

            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)

        return clusters

    def route_cluster_escapes(
        self,
        cluster: List[PadInfo],
        target_lookup: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> List[EscapeStub]:
        """Simultaneously grows all pins in the cluster outward in round-robin steps."""
        if not cluster:
            return []

        # 1. Compute cluster centroid (center of mass)
        cx = sum(p.x for p in cluster) // len(cluster)
        cy = sum(p.y for p in cluster) // len(cluster)

        # 2. Compute radial outward unit vector for each pad
        fanout_dirs: Dict[str, Tuple[float, float]] = {}
        for p in cluster:
            key = f"{p.net}:{p.pad_name}"
            # Direction pointing away from cluster centroid
            dx = float(p.x - cx)
            dy = float(p.y - cy)
            length = math.hypot(dx, dy)

            if length > 0:
                fanout_dirs[key] = (dx / length, dy / length)
            else:
                # If pad is exactly at centroid, bias toward its net's target if known
                if target_lookup and p.net in target_lookup:
                    tx, ty = target_lookup[p.net]
                    tdx, tdy = float(tx - p.x), float(ty - p.y)
                    tlen = math.hypot(tdx, tdy)
                    fanout_dirs[key] = (tdx / tlen, tdy / tlen) if tlen > 0 else (1.0, 0.0)
                else:
                    fanout_dirs[key] = (1.0, 0.0)

        # 3. Concurrent stepping loop (round-robin expansion)
        total_steps = max(1, self.escape_distance_nm // self.step_size_nm)
        current_positions: Dict[str, Tuple[float, float]] = {
            f"{p.net}:{p.pad_name}": (float(p.x), float(p.y)) for p in cluster
        }

        for _ in range(total_steps):
            for p in cluster:
                key = f"{p.net}:{p.pad_name}"
                ux, uy = fanout_dirs[key]
                cur_x, cur_y = current_positions[key]
                next_x = cur_x + ux * self.step_size_nm
                next_y = cur_y + uy * self.step_size_nm
                current_positions[key] = (next_x, next_y)

        # 4. Generate final EscapeStubs
        stubs: List[EscapeStub] = []
        for p in cluster:
            key = f"{p.net}:{p.pad_name}"
            esc_x, esc_y = current_positions[key]
            stubs.append(
                EscapeStub(
                    net_name=p.net,
                    pad_name=p.pad_name,
                    orig_x=p.x,
                    orig_y=p.y,
                    escape_x=int(esc_x),
                    escape_y=int(esc_y),
                    layer=p.layer,
                    step_count=total_steps,
                )
            )

        return stubs

    def escape_all_dense_clusters(
        self,
        all_pads: List[PadInfo],
        target_lookup: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Dict[str, EscapeStub]:
        """Runs concurrent escape for all dense clusters on the board.
        Returns a mapping from 'net:pad_name' -> EscapeStub."""
        clusters = self.identify_clusters(all_pads)
        escape_map: Dict[str, EscapeStub] = {}

        for cluster in clusters:
            stubs = self.route_cluster_escapes(cluster, target_lookup)
            for stub in stubs:
                key = f"{stub.net_name}:{stub.pad_name}"
                escape_map[key] = stub

        return escape_map
