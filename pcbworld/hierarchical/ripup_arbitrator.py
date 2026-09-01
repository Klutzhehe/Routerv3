"""Model E / Phase 3.5: Negotiated Congestion & Rip-Up Arbitrator.

Implements PathFinder-inspired negotiated congestion resolution. When a route fails,
identifies candidate lower-priority victim nets, executes bridge.rip_up(), increments
congestion costs, and re-queues nets for alternative detour routing.
"""

from __future__ import annotations

from typing import List, Dict, Set, Optional, Any
from collections import defaultdict

from pcbworld.hierarchical.specs import (
    NetTier,
    RouteResult,
    PipelinePhase,
)


class RipUpArbitrator:
    """Manages negotiated congestion and rip-up-and-reroute arbitration."""

    def __init__(self, bridge: Any, max_ripup_per_net: int = 3):
        self.bridge = bridge
        self.max_ripup_per_net = max_ripup_per_net
        self.ripup_history: Dict[str, int] = defaultdict(int)
        self.congestion_history: Dict[str, float] = defaultdict(float)

    def select_victim(
        self,
        failed_net: str,
        failed_tier: NetTier,
        committed_nets: Set[str],
        net_tiers: Dict[str, NetTier],
    ) -> Optional[str]:
        """Selects the best victim net to rip up to unblock failed_net."""
        # Never rip up higher-priority tiers to accommodate lower-priority tiers
        candidates = []
        for net in committed_nets:
            tier = net_tiers.get(net, NetTier.BULK_DIGITAL)
            # Only consider nets of equal or lower priority (higher tier value = lower priority)
            if tier.value >= failed_tier.value:
                if self.ripup_history[net] < self.max_ripup_per_net:
                    candidates.append((net, tier))

        if not candidates:
            return None

        # Sort candidates: prefer lower-priority (higher tier value) and least-ripped-up
        candidates.sort(key=lambda item: (-item[1].value, self.ripup_history[item[0]]))
        best_victim = candidates[0][0]
        return best_victim

    def execute_ripup(self, victim_net: str) -> bool:
        """Rips up the copper for victim_net via the bridge."""
        if hasattr(self.bridge, "rip_up"):
            success = self.bridge.rip_up(victim_net)
            self.ripup_history[victim_net] += 1
            self.congestion_history[victim_net] += 1.0
            return bool(success)
        return False
