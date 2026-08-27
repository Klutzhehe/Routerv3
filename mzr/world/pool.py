"""Board-solvability review.

There is **no solvability pre-filter** on training -- boards are generated fresh
and the policy is expected to route them. This module is the *review* step for
when it doesn't: given the eval seeds the policy failed on, run the
sequential + PathFinder expert on each and report whether a full routing even
exists.

    python -m mzr.world.pool --stage 0 --seeds 900042 900091

A "not solvable" verdict here means the expert could not connect every net --
which is a conservative signal (the expert is not optimal), but a board the
expert *can* fully route is definitely routable, so a policy failing that board
is a policy problem, not a board problem.
"""

from __future__ import annotations

import argparse

import torch

from mzr.world.expert import ExpertConfig, route_board
from mzr.world.generator import generate_board
from mzr.world.spec import BoardSpec

#: 6 PathFinder iterations settles contention on the small boards stages 0-3
#: use; more just costs time without changing the verdict.
_FILTER_CFG = ExpertConfig(iterations=6)


def _board_legs(board):
    legs = []
    for ni, net in enumerate(board.netlist.nets):
        for li, (src, dst) in enumerate(net.endpoints()):
            legs.append((ni, li, src, dst, net.width_class))
    return legs


def expert_route(spec: BoardSpec, gcfg, seed: int, device: str = "cpu"):
    """Run the expert on the board `seed` generates. Returns the ExpertResult
    plus the leg count, so callers can report `completed / total`."""
    board = generate_board(spec, gcfg, seed)
    legs = _board_legs(board)
    if not legs:
        return None, 0
    res = route_board(
        board.spec,
        torch.from_numpy(board.static),
        legs,
        _FILTER_CFG,
        negotiate=True,
        device=device,
    )
    return res, len(legs)


def is_solvable(spec: BoardSpec, gcfg, seed: int, device: str = "cpu") -> bool:
    """True iff the expert connects every leg of the board `seed` generates."""
    res, total = expert_route(spec, gcfg, seed, device)
    return res is not None and len(res.completed) == total


def main() -> int:
    from mzr.training.curriculum import STAGES

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    stage = STAGES[args.stage]
    spec, gcfg = stage.board_spec(), stage.generator
    print(f"stage {args.stage}: {stage.name}")
    solvable = 0
    for s in args.seeds:
        res, total = expert_route(spec, gcfg, s, args.device)
        if res is None:
            print(f"  seed {s}: NO NETS -- generator failure")
            continue
        done = len(res.completed)
        ok = done == total
        solvable += ok
        print(
            f"  seed {s}: expert routed {done}/{total} "
            f"({'SOLVABLE' if ok else 'expert could not fully route -- inspect this board'})"
        )
    print(f"{solvable}/{len(args.seeds)} solvable by the expert")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
