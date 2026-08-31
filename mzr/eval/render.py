"""Export routed boards as JSON, for looking at what the policy actually drew.

`docs/RL_PLAN.md` names debuggability as the reason RL was abandoned in this
repo once, and gives the answer directly: *render every failed episode; a
contact sheet of 100 failures shows the failure mode at a glance, a reward curve
never will.* This is that, for MZR.

It emits **JSON, not pixels**. A completion number says a board failed; only the
geometry says *why* -- whether the frontier walked into a pocket, whether it
never changed layer on a cross-layer net, whether it spent the episode orbiting
its own pad. Polylines survive being looked at in a browser, diffed between
checkpoints, and pasted into a report; a PNG does not.

    python -m mzr.eval.render --stage 0 \
        --ckpt /content/drive/MyDrive/mzr_ckpt/stage0/stage0_best.pt \
        --device cuda --boards 9 --out /content/stage0_routes.json

Size discipline: the occupancy planes go out as base64 bitmaps (one bit per
cell, ~400 bytes for a 48x48 layer) rather than nested lists, so a nine-board
export is tens of kilobytes and fits in a notebook cell's stdout.
"""

from __future__ import annotations

import argparse
import base64
import json

import numpy as np
import torch

from mzr.scripts.diagnose_stage0 import load_policy
from mzr.training.curriculum import EVAL_SEEDS, STAGES
from mzr.training.run import make_env
from mzr.world.engine import STATUS_DONE, STATUS_FAILED
from mzr.world.spec import NUM_ENDS


def _bitmap(plane: np.ndarray) -> str:
    """(H, W) bool -> base64 of the packed bits, row-major."""
    return base64.b64encode(np.packbits(plane.astype(np.uint8))).decode("ascii")


@torch.no_grad()
def export(policy, stage, device: str, seeds: list[int], deterministic: bool = True) -> dict:
    env = make_env(stage, batch=len(seeds), device=device, seed=0)
    obs = env.reset(seeds=seeds)

    # Keepouts are static, so capture them before any copper is laid --
    # afterwards `occ` is a mix of obstacle and route and the two cannot be
    # told apart by sign alone at a glance.
    occ0 = env.world.occ.clone().cpu().numpy()

    steps = 0
    while True:
        step = env.step(policy.act(obs, deterministic=deterministic)["action"])
        obs = step.obs
        steps += 1
        if step.done:
            break

    w = env.world
    L, H, W = w.shape
    occ = w.occ.cpu().numpy()
    rv = w.route_v.cpu().numpy()
    rn = w.route_n.cpu().numpy()
    pad = w.net_pad.cpu().numpy()
    status = w.net_status.cpu().numpy()
    valid = w.net_valid.cpu().numpy()
    vias = w.net_vias.cpu().numpy()
    comp = env.completion().cpu().numpy()

    boards = []
    for b, seed in enumerate(seeds):
        # One polyline per frontier. Frontier f belongs to net f // (2*NUM_ENDS);
        # each net grows from both pads, so a finished leg is two runs that meet
        # in the middle rather than one path from src to dst.
        traces = []
        for f in range(w.F):
            n = int(rn[b, f])
            if n < 2:
                continue
            pts = rv[b, f, :n].astype(int).tolist()
            traces.append({"net": f // (2 * NUM_ENDS), "frontier": f, "pts": pts})

        nets = []
        for n in range(w.cfg.max_nets):
            if not bool(valid[b, n]):
                continue
            nets.append({
                "id": n,
                "src": pad[b, n, 0, 0].astype(int).tolist(),
                "dst": pad[b, n, 0, 1].astype(int).tolist(),
                "status": ("done" if status[b, n] == STATUS_DONE
                           else "failed" if status[b, n] == STATUS_FAILED else "routing"),
                "vias": int(vias[b, n]),
            })

        boards.append({
            "seed": int(seed),
            "completion": float(comp[b]),
            "keepout": [_bitmap(occ0[b, l] < 0) for l in range(L)],
            "copper": [_bitmap(occ[b, l] > 0) for l in range(L)],
            "nets": nets,
            "traces": traces,
        })

    return {
        "stage": stage.name,
        "height": H, "width": W, "layers": L,
        "steps": steps,
        "deterministic": deterministic,
        "mean_completion": float(comp.mean()),
        "boards": boards,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="0", choices=sorted(STAGES))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--boards", type=int, default=9)
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="explicit seeds; default is the first --boards eval seeds")
    p.add_argument("--sampled", action="store_true", help="export a sampled rollout instead of argmax")
    p.add_argument("--out", default="routes.json")
    args = p.parse_args()

    torch.manual_seed(0)
    stage = STAGES[args.stage]
    seeds = args.seeds if args.seeds else EVAL_SEEDS[: args.boards]
    policy, _, _ = load_policy(args.ckpt, stage, args.device)
    blob = export(policy, stage, args.device, seeds, deterministic=not args.sampled)

    with open(args.out, "w") as f:
        json.dump(blob, f, separators=(",", ":"))
    import os
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1024:.1f} KB) "
          f"-- {len(seeds)} boards, mean completion {blob['mean_completion']:.3f}")
    for bd in blob["boards"]:
        st = ",".join(f"{n['status']}/{n['vias']}v" for n in bd["nets"])
        print(f"  seed {bd['seed']}: completion {bd['completion']:.2f}  [{st}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
