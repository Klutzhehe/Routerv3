"""CFP -- the Cost-Field Policy agent (docs/AI_ARCHITECTURE.md).

Two-tower relational policy for board-level routing: a netlist tower over
nets-as-nodes, a convolutional tower over the board raster, bidirectional
cross-attention between them, and three heads (which net to act on, what
cost field to route it under, and a value estimate).

Nothing here talks to pcbworld_pns_bridge -- this package is pure PyTorch
over the tensor contract in spec.py, so unlike pcbworld/env/* it is fully
runnable and testable off-Colab. The env that produces real CFPObservations
from a live board is not written yet; spec.make_dummy_observation() stands
in for it.
"""

from pcbworld.agents.cfp.model import CFPConfig, CFPNet
from pcbworld.agents.cfp.policy import CFPAction, CFPPolicy, CFPScore
from pcbworld.agents.cfp.spec import (
    CANVAS_CHANNELS,
    EDGE_TYPE,
    NET_FEATURES,
    NUM_CANVAS_CHANNELS,
    NUM_EDGE_TYPES,
    NUM_NET_FEATURES,
    CFPObservation,
    build_edge_type_matrix,
    make_dummy_observation,
)

__all__ = [
    "CANVAS_CHANNELS",
    "CFPConfig",
    "CFPNet",
    "CFPObservation",
    "CFPAction",
    "CFPPolicy",
    "CFPScore",
    "EDGE_TYPE",
    "NET_FEATURES",
    "NUM_CANVAS_CHANNELS",
    "NUM_EDGE_TYPES",
    "NUM_NET_FEATURES",
    "build_edge_type_matrix",
    "make_dummy_observation",
]
