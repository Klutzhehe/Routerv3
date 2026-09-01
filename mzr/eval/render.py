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


#: Octant index of each (sign(dy), sign(dx)) heading, matching the engine's
#: direction table ordering closely enough for a bend histogram.
def _octant(dy: int, dx: int) -> int:
    import math

    return int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8


def _bends(pts: list[list[int]]) -> dict:
    """Histogram of direction changes along one polyline, in 45-degree octants.

    Reported because the corner reward (`RewardConfig.corner`) is only worth
    keeping if it actually moves this: fab practice replaces every 90-degree
    corner with two 45-degree bends, so `right_angle` falling while `soft`
    rises is the shape of success. Layer changes are skipped -- a via is not a
    corner.
    """
    hist = [0] * 5
    prev = None
    for a, b in zip(pts, pts[1:]):
        dy, dx = b[1] - a[1], b[2] - a[2]
        if dy == 0 and dx == 0:
            continue                      # a via: same cell, different layer
        o = _octant(dy, dx)
        if prev is not None:
            d = abs(o - prev)
            hist[min(d, 8 - d)] += 1
        prev = o
    return {"straight": hist[0], "soft": hist[1], "right_angle": sum(hist[2:])}


@torch.no_grad()
def export(policy, stage, device: str, seeds: list[int], deterministic: bool = True,
           *, copper_seeded: bool = False, geodesic_refresh: int = 8) -> dict:
    # The env must match the one the policy was TRAINED in. A copper-seeded
    # policy measured in a pad-targeted world is measuring a different game.
    env = make_env(stage, batch=len(seeds), device=device, seed=0,
                   copper_seeded=copper_seeded, geodesic_refresh=geodesic_refresh)
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
        # One polyline per frontier. Frontier f belongs to net
        # f // (max_legs * NUM_ENDS);
        # each net grows from both pads, so a finished leg is two runs that meet
        # in the middle rather than one path from src to dst.
        traces = []
        for f in range(w.F):
            n = int(rn[b, f])
            if n < 2:
                continue
            pts = rv[b, f, :n].astype(int).tolist()
            traces.append({"net": f // (w.cfg.max_legs * NUM_ENDS), "frontier": f,
                           "pts": pts, "bends": _bends(pts)})

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

        bends = {k: sum(t["bends"][k] for t in traces) for k in ("straight", "soft", "right_angle")}

        boards.append({
            "seed": int(seed),
            "bends": bends,
            "completion": float(comp[b]),
            "keepout": [_bitmap(occ0[b, l] < 0) for l in range(L)],
            "copper": [_bitmap(occ[b, l] > 0) for l in range(L)],
            "nets": nets,
            "traces": traces,
        })

    tot = {k: sum(b["bends"][k] for b in boards) for k in ("straight", "soft", "right_angle")}
    return {
        "bends": tot,
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
    p.add_argument("--copper-seeded", action="store_true")
    p.add_argument("--sampled", action="store_true", help="export a sampled rollout instead of argmax")
    p.add_argument("--out", default="routes.json")
    args = p.parse_args()

    torch.manual_seed(0)
    stage = STAGES[args.stage]
    seeds = args.seeds if args.seeds else EVAL_SEEDS[: args.boards]
    policy, _, _ = load_policy(args.ckpt, stage, args.device)
    blob = export(policy, stage, args.device, seeds, deterministic=not args.sampled,
                  copper_seeded=args.copper_seeded)

    with open(args.out, "w") as f:
        json.dump(blob, f, separators=(",", ":"))
    import os
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1024:.1f} KB) "
          f"-- {len(seeds)} boards, mean completion {blob['mean_completion']:.3f}")
    for bd in blob["boards"]:
        st = ",".join(f"{n['status']}/{n['vias']}v" for n in bd["nets"])
        bn = bd["bends"]
        print(f"  seed {bd['seed']}: completion {bd['completion']:.2f}  [{st}]  "
              f"bends straight {bn['straight']} 45deg {bn['soft']} >=90deg {bn['right_angle']}")
    t = blob["bends"]
    n = t["straight"] + t["soft"] + t["right_angle"]
    print(f"\ncorners over all boards: straight {t['straight']}, 45deg {t['soft']}, "
          f">=90deg {t['right_angle']}"
          + (f"  ({t['right_angle'] / n * 100:.1f}% right angles)" if n else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
