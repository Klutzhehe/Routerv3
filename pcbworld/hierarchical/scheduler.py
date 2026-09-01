"""Model A / Phase 0: Constraint & Schedule Manager.

Parses board netlist, identifies high-speed diff pairs, synchronous length-matched groups,
and bulk nets, and builds an optimal phased routing schedule.
"""

from __future__ import annotations

import re
import math
from typing import List, Dict, Tuple, Optional, Any

from pcbworld.hierarchical.specs import (
    NetTier,
    PadInfo,
    DiffPairSpec,
    LengthGroupSpec,
)


class ConstraintScheduler:
    """Classifies nets and constructs the multi-phase routing queue."""

    def __init__(
        self,
        default_diff_gap_nm: int = 150_000,
        default_diff_width_nm: int = 200_000,
        default_length_tol_nm: int = 250_000,
        num_layers: int = 2,
    ):
        self.default_diff_gap_nm = default_diff_gap_nm
        self.default_diff_width_nm = default_diff_width_nm
        self.default_length_tol_nm = default_length_tol_nm
        self.num_layers = num_layers

    def analyze_board(self, pads_input: List[Any]) -> Tuple[
        List[DiffPairSpec],
        List[LengthGroupSpec],
        List[str],  # sensitive analog nets
        List[str],  # bulk digital nets
        Dict[str, List[PadInfo]],  # net -> pads
    ]:
        """Analyzes all pads on the board and extracts structured net classes."""
        net_to_pads: Dict[str, List[PadInfo]] = {}

        for p in pads_input:
            net_name = getattr(p, "net", None) or (p.get("net") if isinstance(p, dict) else None)
            if not net_name:
                continue
            pad_name = getattr(p, "pad_name", "") or (p.get("pad_name", "") if isinstance(p, dict) else "")
            x = int(getattr(p, "x", 0) if not isinstance(p, dict) else p.get("x", 0))
            y = int(getattr(p, "y", 0) if not isinstance(p, dict) else p.get("y", 0))
            layer = int(getattr(p, "layer", 0) if not isinstance(p, dict) else p.get("layer", 0))

            pad_info = PadInfo(net=net_name, pad_name=pad_name, x=x, y=y, layer=layer)
            net_to_pads.setdefault(net_name, []).append(pad_info)

        # 1. Parse Differential Pairs
        diff_pairs: Dict[str, Dict[str, str]] = {}
        diff_pair_specs: List[DiffPairSpec] = []

        # 2. Parse Length Matched Groups
        length_groups: Dict[str, Dict[int, str]] = {}
        length_group_specs: List[LengthGroupSpec] = []

        # 3. Sensitive Analog & Bulk Digital
        sensitive_analog: List[str] = []
        bulk_digital: List[str] = []

        for net_name in net_to_pads.keys():
            if net_name.startswith("diffpair_"):
                # Format: diffpair_<i>_P or diffpair_<i>_N
                parts = net_name.rsplit("_", 1)
                if len(parts) == 2 and parts[1] in ("P", "N"):
                    base, leg = parts[0], parts[1]
                    diff_pairs.setdefault(base, {})[leg] = net_name
            elif net_name.startswith("lengthgrp_"):
                # Format: lengthgrp_<group>_<member>
                parts = net_name.split("_")
                if len(parts) >= 3:
                    group_id = parts[1]
                    try:
                        member_idx = int(parts[2])
                    except ValueError:
                        member_idx = 0
                    length_groups.setdefault(group_id, {})[member_idx] = net_name
            elif any(net_name.lower().startswith(prefix) for prefix in ("analog_", "rf_", "sensor_")):
                sensitive_analog.append(net_name)
            else:
                bulk_digital.append(net_name)

        # Build DiffPairSpec objects
        for pair_id, legs in sorted(diff_pairs.items()):
            p_net = legs.get("P")
            n_net = legs.get("N")
            if p_net and n_net:
                spec = DiffPairSpec(
                    pair_id=pair_id,
                    p_net=p_net,
                    n_net=n_net,
                    p_pads=net_to_pads.get(p_net, []),
                    n_pads=net_to_pads.get(n_net, []),
                    target_gap_nm=self.default_diff_gap_nm,
                    target_width_nm=self.default_diff_width_nm,
                    assigned_layer=0,
                )
                diff_pair_specs.append(spec)
            elif p_net:
                bulk_digital.append(p_net)
            elif n_net:
                bulk_digital.append(n_net)

        # Build LengthGroupSpec objects
        for group_id, members in sorted(length_groups.items()):
            # Sort member nets by member index
            sorted_members = [members[idx] for idx in sorted(members.keys())]
            ref_net = sorted_members[0] if sorted_members else None
            spec = LengthGroupSpec(
                group_id=group_id,
                member_nets=sorted_members,
                reference_net=ref_net,
                target_tolerance_nm=self.default_length_tol_nm,
                assigned_layer=0,
            )
            length_group_specs.append(spec)

        # Sort diff pairs by Manhattan distance between pad centers
        diff_pair_specs.sort(key=lambda dp: self._estimate_pair_span(dp))

        # Sort bulk nets by Manhattan distance (shortest first)
        bulk_digital.sort(key=lambda n: self._estimate_net_span(net_to_pads.get(n, [])))

        return diff_pair_specs, length_group_specs, sensitive_analog, bulk_digital, net_to_pads

    @staticmethod
    def _estimate_pair_span(dp: DiffPairSpec) -> float:
        if len(dp.p_pads) >= 2:
            p1, p2 = dp.p_pads[0], dp.p_pads[1]
            return math.hypot(p2.x - p1.x, p2.y - p1.y)
        return 0.0

    @staticmethod
    def _estimate_net_span(pads: List[PadInfo]) -> float:
        if len(pads) >= 2:
            return math.hypot(pads[1].x - pads[0].x, pads[1].y - pads[0].y)
        return 0.0
