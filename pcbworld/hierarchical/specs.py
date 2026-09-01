"""Data structures, enums, and specs for the Hierarchical PCB Routing System."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional, Any


class NetTier(Enum):
    """Functional classification tier for PCB nets."""
    DIFF_PAIR = 1           # High-speed serial (PCIe, 10GbE, USB3, SATA)
    LENGTH_GROUP = 2        # Synchronous parallel bus (DDR DQ/DQS, Clock, Address)
    SENSITIVE_ANALOG = 3    # Analog / RF / Sensor nets
    BULK_DIGITAL = 4        # General control, GPIO, I2C, SPI, Power/Reset


class PipelinePhase(Enum):
    """The 5 distinct execution phases of the hierarchical routing pipeline."""
    PHASE_0_ANALYSIS = 0          # Netlist classification & stackup planning
    PHASE_1_DIFF_PAIR = 1         # High-speed differential pair routing
    PHASE_2_BUS_RESERVATION = 2   # Synchronous bus baseline routing & meander reservation
    PHASE_3_BULK_ROUTING = 3      # Bulk single-ended routing around reserved zones
    PHASE_4_MEANDER_TUNING = 4    # Meander expansion & length tuning polish


@dataclass
class PadInfo:
    """Pad location and layer."""
    net: str
    pad_name: str
    x: int  # in nm
    y: int  # in nm
    layer: int


@dataclass
class DiffPairSpec:
    """Specification for a differential pair."""
    pair_id: str
    p_net: str
    n_net: str
    p_pads: List[PadInfo] = field(default_factory=list)
    n_pads: List[PadInfo] = field(default_factory=list)
    target_gap_nm: int = 150_000        # Default 0.15 mm
    target_width_nm: int = 200_000      # Default 0.20 mm
    max_skew_nm: int = 50_000           # Target max skew tolerance (0.05 mm)
    assigned_layer: int = 0             # Preferred routing layer


@dataclass
class LengthGroupSpec:
    """Specification for a synchronous length-matched bus group (e.g. DDR byte lane)."""
    group_id: str
    member_nets: List[str] = field(default_factory=list)
    reference_net: Optional[str] = None
    target_tolerance_nm: int = 250_000  # Default matching tolerance (0.25 mm)
    routed_lengths: Dict[str, int] = field(default_factory=dict)
    target_length_nm: int = 0
    assigned_layer: int = 0


@dataclass
class ReservationZone:
    """Spatial 2D bounding box or polygon reserved for future meander expansion.
    
    Non-critical bulk nets (Phase 3) treat active reservation zones as high-cost obstacles.
    """
    zone_id: str
    owner_net: str
    bbox_nm: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) in nm
    layer: int
    created_in_phase: PipelinePhase = PipelinePhase.PHASE_2_BUS_RESERVATION
    active: bool = True

    def contains_point(self, x: int, y: int, layer: int) -> bool:
        """Checks if a point (x, y) on the given layer falls inside the reservation zone."""
        if not self.active or layer != self.layer:
            return False
        x_min, y_min, x_max, y_max = self.bbox_nm
        return x_min <= x <= x_max and y_min <= y <= y_max

    def intersects_segment(self, x1: int, y1: int, x2: int, y2: int, layer: int) -> bool:
        """Fast AABB check if a line segment intersects the reservation bounding box."""
        if not self.active or layer != self.layer:
            return False
        x_min, y_min, x_max, y_max = self.bbox_nm
        # Check if segment bounding box overlaps zone bounding box
        seg_xmin, seg_xmax = min(x1, x2), max(x1, x2)
        seg_ymin, seg_ymax = min(y1, y2), max(y1, y2)
        return not (seg_xmax < x_min or seg_xmin > x_max or seg_ymax < y_min or seg_ymin > y_max)


@dataclass
class RouteResult:
    """Outcome of routing a specific net or net group."""
    net_name: str
    success: bool
    phase: PipelinePhase
    wirelength_nm: int = 0
    num_segments: int = 0
    num_vias: int = 0
    skew_nm: int = 0
    length_mismatch_nm: int = 0
    drc_clean: bool = True
    error_message: Optional[str] = None


@dataclass
class HierarchicalPipelineReport:
    """Summary report across all 5 phases of hierarchical routing."""
    total_nets: int = 0
    routed_nets: int = 0
    diff_pairs_routed: int = 0
    diff_pairs_total: int = 0
    length_groups_tuned: int = 0
    length_groups_total: int = 0
    bulk_nets_routed: int = 0
    bulk_nets_total: int = 0
    ripup_count: int = 0
    drc_violations: int = 0
    phase_results: Dict[PipelinePhase, List[RouteResult]] = field(default_factory=dict)
    all_clean: bool = False
