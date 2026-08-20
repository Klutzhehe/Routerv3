"""The tensor contract between the routing env and the CFP policy.

This module is the single source of truth for what an observation *is*:
channel order, net-feature order, edge-type codes, action layout. The env
(not written yet) rasterizes a live KiCad board into exactly this; the
policy in model.py consumes exactly this. Keeping it in one importable
place means the rasterizer can be written and unit-tested later without
re-deriving any of these orderings from the model code.

Everything here is plain numpy/torch -- no pcbnew, no pcbworld_pns_bridge
-- so it imports and runs anywhere, which is the whole point (pcbworld/env/*
can only run inside the Colab flow; see ROADMAP.md).

Two conventions are inherited unchanged from
pcbworld/data/generate_board.py and must stay in sync with it:
  - net naming: "net_<i>", "diffpair_<i>_P"/"_N", "lengthgrp_<g>_<member>"
  - 2 copper layers, which is why CANVAS_CHANNELS names copper_l0/copper_l1
    explicitly rather than carrying a variable-length layer stack. Growing
    past 2 layers is a spec change here, not a config flag.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

# --------------------------------------------------------------------------
# Canvas channels -- the board raster, (C, H, W), values in [0, 1] unless
# noted. The env is responsible for the normalization; the model assumes
# these are already O(1) and does not normalize them itself.
# --------------------------------------------------------------------------
CANVAS_CHANNELS: tuple[str, ...] = (
    "copper_l0",         # committed track/via copper on layer 0
    "copper_l1",         # committed track/via copper on layer 1
    "pads_unrouted",     # pads belonging to nets not yet routed
    "pads_routed",       # pads belonging to nets already routed
    "board_mask",        # 1 inside the board outline, 0 outside
    "keepout",           # explicit keepouts + courtyard obstacles
    "pads_diff_pair",    # unrouted diff-pair pads only (corridor demand)
    "pads_length_group", # unrouted length-group pads only
    "meander_demand",    # per-net slack still owed, painted at its pads,
                         #   normalized by board diagonal -- this is what
                         #   lets the policy reserve space *before* the
                         #   region fills up
    "reserve_carry",     # the accumulated reserve plane the policy itself
                         #   emitted on earlier steps, fed back as input so
                         #   reservations are visible to later decisions
    "hpwl_congestion",   # bbox-density estimate over unrouted nets
    "vias",              # committed via density
)
NUM_CANVAS_CHANNELS = len(CANVAS_CHANNELS)
CANVAS_CHANNEL_INDEX = {name: i for i, name in enumerate(CANVAS_CHANNELS)}

# --------------------------------------------------------------------------
# Per-net features -- (N, F). Positions are board-normalized to [0, 1];
# lengths are normalized by the board diagonal so they stay comparable
# across board sizes.
# --------------------------------------------------------------------------
NET_FEATURES: tuple[str, ...] = (
    "centroid_x",
    "centroid_y",
    "bbox_w",
    "bbox_h",
    "hpwl",                  # half-perimeter wirelength / board diagonal
    "num_pads",              # / 8
    "is_routed",
    "is_plain",
    "is_diff_pair",
    "is_length_group",
    "is_diff_pair_driver",   # the "_P" leg -- the only one PNS routes
                             #   directly (see DiffPairRouteEnv: the "_N"
                             #   leg is found by name matching, never
                             #   driven), so the pointer head must never
                             #   select an "_N" leg. Enforced by
                             #   action_mask, but exposed as a feature too.
    "routed_length",         # actual routed length / board diagonal, 0 if
                             #   unrouted
    "target_length",         # length-group reference length, 0 if n/a
    "length_deficit",        # target - routed, signed. The tuning signal.
    "group_size",            # / 8, 0 for plain nets
    "track_width",           # / 1mm
)
NUM_NET_FEATURES = len(NET_FEATURES)
NET_FEATURE_INDEX = {name: i for i, name in enumerate(NET_FEATURES)}


class EDGE_TYPE:
    """Relational edge codes for the netlist tower's attention bias.

    Dense (N, N) rather than a sparse edge list: N is small (a D2-style
    board is tens of nets, MAX_NETS caps it at 256) so an N*N int64 matrix
    is cheaper than the bookkeeping a sparse format would need, and it
    keeps the whole model dependency-free (no torch_geometric).

    Constraint edges outrank geometric ones -- see build_edge_type_matrix.
    """

    NONE = 0            # unrelated
    SELF = 1            # i == j
    DIFF_PAIR = 2       # the two legs of one differential pair
    LENGTH_GROUP = 3    # members of one length-matched group (a clique)
    BBOX_CONFLICT = 4   # bounding boxes overlap -> competing for the same
                        #   region. This is the edge that gives the netlist
                        #   tower any sense of geometry at all.
    SPATIAL_KNN = 5     # k-nearest by centroid, for pairs no other edge
                        #   already covers


NUM_EDGE_TYPES = 6

# The pointer head emits 2 logits per net slot. Slot layout of the flat
# (2 * N,) action space and its mask:
ACTION_ROUTE = 0   # route this net/group now
ACTION_RIPUP = 1   # tear this already-routed net out
NUM_ACTION_KINDS = 2

MAX_NETS = 256


@dataclasses.dataclass
class CFPObservation:
    """One batched observation. All tensors share leading dim B.

    canvas:      (B, NUM_CANVAS_CHANNELS, H, W) float32
    net_feats:   (B, N, NUM_NET_FEATURES)       float32
    net_xy:      (B, N, 2)                      float32, board-normalized
                   centroid. Duplicated out of net_feats on purpose: the
                   cross-attention position bias needs it as geometry, not
                   as one more feature the encoder has to disentangle.
    net_mask:    (B, N)                         bool, True = real net slot
    edge_type:   (B, N, N)                      int64, EDGE_TYPE codes
    action_mask: (B, NUM_ACTION_KINDS, N)       bool, True = legal action

    N is padded to a fixed width per batch; net_mask marks the real slots.
    """

    canvas: torch.Tensor
    net_feats: torch.Tensor
    net_xy: torch.Tensor
    net_mask: torch.Tensor
    edge_type: torch.Tensor
    action_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.canvas.shape[0])

    @property
    def num_nets(self) -> int:
        return int(self.net_feats.shape[1])

    def to(self, device: Any) -> "CFPObservation":
        return CFPObservation(
            **{f.name: getattr(self, f.name).to(device) for f in dataclasses.fields(self)}
        )

    def validate(self) -> None:
        """Shape/dtype assertions. Cheap, but not free -- call it in tests
        and in the env's first few steps, not in the hot training loop."""
        b, n = self.batch_size, self.num_nets
        assert self.canvas.ndim == 4
        assert self.canvas.shape[1] == NUM_CANVAS_CHANNELS, (
            f"canvas has {self.canvas.shape[1]} channels, spec says "
            f"{NUM_CANVAS_CHANNELS} ({CANVAS_CHANNELS})"
        )
        assert self.net_feats.shape == (b, n, NUM_NET_FEATURES)
        assert self.net_xy.shape == (b, n, 2)
        assert self.net_mask.shape == (b, n)
        assert self.edge_type.shape == (b, n, n)
        assert self.action_mask.shape == (b, NUM_ACTION_KINDS, n)
        assert self.net_mask.dtype == torch.bool
        assert self.action_mask.dtype == torch.bool
        assert self.edge_type.dtype == torch.int64
        assert int(self.edge_type.max()) < NUM_EDGE_TYPES
        # A fully-masked action row would make the pointer categorical
        # undefined; the env must always leave at least one legal move.
        assert bool(self.action_mask.flatten(1).any(dim=-1).all()), (
            "some batch element has no legal action -- the env should have "
            "terminated that episode instead of emitting this observation"
        )
        # An action may never point at a padding slot.
        assert not bool((self.action_mask & ~self.net_mask.unsqueeze(1)).any()), (
            "action_mask marks a padded net slot legal"
        )


def parse_net_name(name: str) -> tuple[str, str | None, str | None]:
    """Splits generate_board.py's naming convention.

    Returns (kind, group_key, leg) where kind is
    "plain" | "diff_pair" | "length_group":
      "net_3"          -> ("plain",        None,          None)
      "diffpair_2_P"   -> ("diff_pair",    "diffpair_2",  "P")
      "lengthgrp_1_4"  -> ("length_group", "lengthgrp_1", "4")
    """
    if name.startswith("diffpair_"):
        base, leg = name.rsplit("_", 1)
        return "diff_pair", base, leg
    if name.startswith("lengthgrp_"):
        _, group_idx, member_idx = name.split("_")
        return "length_group", f"lengthgrp_{group_idx}", member_idx
    return "plain", None, None


def build_edge_type_matrix(
    net_names: list[str],
    centroids: np.ndarray,
    bboxes: np.ndarray,
    k_spatial: int = 8,
) -> np.ndarray:
    """Reference construction of the (N, N) edge-type matrix.

    centroids: (N, 2); bboxes: (N, 4) as (min_x, min_y, max_x, max_y), both
    in whatever consistent units the caller likes (only comparisons and
    distances are used, never absolute scale).

    Assignment is priority-ordered -- a pair that is both a diff pair and
    spatially near gets DIFF_PAIR, because the constraint is the more
    informative fact. Geometric edges only fill slots still NONE.
    """
    n = len(net_names)
    assert centroids.shape == (n, 2), f"centroids {centroids.shape} != ({n}, 2)"
    assert bboxes.shape == (n, 4), f"bboxes {bboxes.shape} != ({n}, 4)"

    edges = np.full((n, n), EDGE_TYPE.NONE, dtype=np.int64)

    groups: dict[str, list[int]] = {}
    for i, name in enumerate(net_names):
        _kind, group_key, _leg = parse_net_name(name)
        if group_key is not None:
            groups.setdefault(group_key, []).append(i)

    for group_key, members in groups.items():
        code = (
            EDGE_TYPE.DIFF_PAIR
            if group_key.startswith("diffpair_")
            else EDGE_TYPE.LENGTH_GROUP
        )
        for a in members:
            for b in members:
                if a != b:
                    edges[a, b] = code

    free = edges == EDGE_TYPE.NONE
    overlap = (
        (bboxes[:, None, 0] <= bboxes[None, :, 2])
        & (bboxes[None, :, 0] <= bboxes[:, None, 2])
        & (bboxes[:, None, 1] <= bboxes[None, :, 3])
        & (bboxes[None, :, 1] <= bboxes[:, None, 3])
    )
    edges[free & overlap] = EDGE_TYPE.BBOX_CONFLICT

    if n > 1:
        dist = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
        np.fill_diagonal(dist, np.inf)
        k = min(k_spatial, n - 1)
        neighbors = np.argpartition(dist, kth=k - 1, axis=-1)[:, :k]
        rows = np.repeat(np.arange(n), k)
        cols = neighbors.reshape(-1)
        knn_free = edges[rows, cols] == EDGE_TYPE.NONE
        edges[rows[knn_free], cols[knn_free]] = EDGE_TYPE.SPATIAL_KNN

    np.fill_diagonal(edges, EDGE_TYPE.SELF)
    return edges


def make_dummy_observation(
    batch_size: int = 2,
    num_nets: int = 12,
    canvas_size: int = 256,
    device: Any = "cpu",
    seed: int = 0,
) -> CFPObservation:
    """A structurally valid random observation, for shape/gradient smoke
    tests until the real rasterizing env exists. Values are noise -- this
    checks that tensors flow, nothing about routing behavior.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    b, n = batch_size, num_nets
    assert n >= 2, "dummy observations need at least 2 net slots"

    canvas = torch.rand(b, NUM_CANVAS_CHANNELS, canvas_size, canvas_size, generator=g)
    net_feats = torch.randn(b, n, NUM_NET_FEATURES, generator=g)
    net_xy = torch.rand(b, n, 2, generator=g)

    net_mask = torch.ones(b, n, dtype=torch.bool)
    if n > 2:  # exercise the padding path
        net_mask[:, -1] = False

    rng = np.random.default_rng(seed)
    names = ([f"net_{j}" for j in range(max(0, n - 4))] + [
        "diffpair_0_P",
        "diffpair_0_N",
        "lengthgrp_0_0",
        "lengthgrp_0_1",
    ])[:n]

    edge_type = torch.zeros(b, n, n, dtype=torch.int64)
    for i in range(b):
        centroids = rng.random((n, 2))
        bboxes = np.concatenate([centroids - 0.05, centroids + 0.05], axis=1)
        edge_type[i] = torch.from_numpy(build_edge_type_matrix(names, centroids, bboxes))

    action_mask = torch.zeros(b, NUM_ACTION_KINDS, n, dtype=torch.bool)
    action_mask[:, ACTION_ROUTE] = net_mask
    action_mask[:, ACTION_RIPUP, 0] = True  # pretend net 0 is already routed

    obs = CFPObservation(
        canvas=canvas,
        net_feats=net_feats,
        net_xy=net_xy,
        net_mask=net_mask,
        edge_type=edge_type,
        action_mask=action_mask,
    )
    obs.validate()
    return obs.to(device)
