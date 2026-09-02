"""Held-out evaluation of a trained checkpoint, at whatever scale you ask for.

Training's own eval runs 64 boards on `EVAL_SEEDS[:64]` (900000-900063) and
uses that score to decide which checkpoint is `best`. So those seeds are **not
held out** for the checkpoint they selected -- re-scoring on them measures how
well the selection worked, not how well the policy generalises. This script
defaults to a disjoint range for that reason.

Run:
    python -m mzr.scripts.eval_stage --stage 0 \\
        --checkpoint /content/drive/MyDrive/mzr_ckpt/stage0/stage0_best.pt \\
        --n 1000 --batch 64

The architecture flags must match the run that produced the checkpoint -- the
checkpoint stores weights, not shapes, so a mismatch surfaces as a state-dict
error rather than a wrong answer, which is the failure mode you want.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from mzr.training.curriculum import EVAL_SEEDS, STAGES
from mzr.training.run import make_env
from mzr.models.policy import PriorPolicy

_SQ2 = 2.0 ** 0.5


def _octile(a, b) -> float:
    """Distance on an 8-connected grid: the shortest a trace could possibly be."""
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    lo, hi = min(dx, dy), max(dx, dy)
    return (hi - lo) + _SQ2 * lo


def _polyline_len(pts) -> float:
    return sum(_octile(p, q) for p, q in zip(pts, pts[1:]))


def copper_stats(world, completion) -> list[dict]:
    """Per board: how much copper each net actually cost, and whether it was
    drawn twice.

    Completion cannot see double-routing, and that is not a hypothesis --
    DESIGN_COPPER_SEEDED.md measured 20 of 32 nets routed twice at 1.94x
    copper while completion read a clean 1.000, because a net drawn twice IS
    connected. So any eval that reports only completion will certify that
    failure as success. This is the number that catches it.

    Each net grows from BOTH pads inward and the two frontiers are meant to
    meet in the middle, so a healthy leg is two runs of about half the
    pad-to-pad distance each. When both runs are most of the full distance,
    the frontiers mirror-routed past each other and both pad-snapped -- that
    is the double-route, and `both_long` counts it.
    """
    from mzr.world.spec import NUM_ENDS

    rv = world.route_v.cpu().numpy()
    rn = world.route_n.cpu().numpy()
    pad = world.net_pad.cpu().numpy()
    valid = world.net_valid.cpu().numpy()
    per_net_frontiers = world.cfg.max_legs * NUM_ENDS

    out = []
    for b in range(rv.shape[0]):
        runs: dict[int, list[float]] = {}
        for f in range(world.F):
            n = int(rn[b, f])
            if n < 2:
                continue
            runs.setdefault(f // per_net_frontiers, []).append(
                _polyline_len(rv[b, f, :n].astype(int).tolist())
            )

        copper = straight = 0.0
        doubled = 0
        for net in range(world.cfg.max_nets):
            if not bool(valid[b, net]):
                continue
            src = pad[b, net, 0, 0].astype(int).tolist()
            dst = pad[b, net, 0, 1].astype(int).tolist()
            sl = _octile(src, dst)
            lens = runs.get(net, [])
            copper += sum(lens)
            straight += sl
            # Both halves nearly the whole way = neither stopped at the meet.
            if sl > 0 and sum(1 for L in lens if L >= 0.70 * sl) >= 2:
                doubled += 1

        out.append({
            "copper": copper,
            "straight": straight,
            "ratio": (copper / straight) if straight > 0 else float("nan"),
            "doubled": doubled,
            "completion": float(completion[b]),
        })
    return out


@torch.no_grad()
def evaluate_seeds(policy, stage, device: str, seeds: list[int], batch: int,
                   *, copper_seeded: bool | None = None, geodesic_refresh: int | None = None,
                   deterministic: bool = True, progress: bool = True):
    """Completion per board, over `seeds`, in chunks of `batch`.

    Chunked because the env allocates per-board geodesic fields: one 1000-wide
    batch is a memory spike for no throughput gain, and on a GPU still busy
    with a training run it is how you get an OOM instead of a number.
    """
    policy.eval()
    completions: list[float] = []
    stats: list[dict] = []
    t0 = time.time()

    for i in range(0, len(seeds), batch):
        chunk = seeds[i:i + batch]
        env = make_env(stage, batch=len(chunk), device=device, seed=0,
                       copper_seeded=copper_seeded, geodesic_refresh=geodesic_refresh)
        obs = env.reset(seeds=chunk)
        while True:
            step = env.step(policy.act(obs, deterministic=deterministic)["action"])
            obs = step.obs
            if step.done:
                break
        comp = env.completion()
        completions.extend(comp.tolist())
        stats.extend(copper_stats(env.world, comp))
        del env
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if progress:
            done = len(completions)
            print(f"  {done:5d}/{len(seeds)}  running perfect "
                  f"{sum(1 for c in completions if c >= 0.999) / done:.4f}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)

    return completions, stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="0", choices=sorted(STAGES))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=1000, help="number of boards")
    p.add_argument("--seed-start", type=int, default=950_000,
                   help="disjoint from EVAL_SEEDS (900000-900127) on purpose")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--field-width", type=int, default=64)
    p.add_argument("--token-width", type=int, default=192)
    p.add_argument("--encoder-levels", type=int, default=2)
    p.add_argument("--token-depth", type=int, default=2)
    p.add_argument("--copper-seeded", action=argparse.BooleanOptionalAction, default=None,
                   help="unset = whatever the STAGE asks for")
    p.add_argument("--geodesic-refresh", type=int, default=None)
    p.add_argument("--sampled", action="store_true",
                   help="evaluate the sampled arm instead of argmax (the gate is argmax)")
    p.add_argument("--max-show", type=int, default=0,
                   help="cap the printed failure list; 0 = show every one")
    args = p.parse_args()

    stage = STAGES[args.stage]
    dev = args.device

    policy = PriorPolicy(
        num_layers=stage.layers,
        field_width=args.field_width,
        token_width=args.token_width,
        encoder_levels=args.encoder_levels,
        token_depth=args.token_depth,
    ).to(dev)

    blob = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    policy.load_state_dict(blob["policy"])
    print(f"checkpoint : {args.checkpoint}")
    print(f"  saved at update {blob.get('update')}, best-so-far {blob.get('best')}, "
          f"stage {blob.get('stage')}")

    seeds = list(range(args.seed_start, args.seed_start + args.n))
    overlap = sorted(set(seeds) & set(EVAL_SEEDS))
    print(f"seeds      : {seeds[0]}..{seeds[-1]}  ({len(seeds)} boards, "
          f"batch {args.batch}, {'sampled' if args.sampled else 'argmax'}, {dev})")
    if overlap:
        print(f"  WARNING: {len(overlap)} of these are training-eval seeds "
              f"({overlap[0]}..{overlap[-1]}) -- NOT held out")
    else:
        print("  disjoint from EVAL_SEEDS -- genuinely held out")
    print()

    comps, stats = evaluate_seeds(policy, stage, dev, seeds, args.batch,
                                  copper_seeded=args.copper_seeded,
                                  geodesic_refresh=args.geodesic_refresh,
                                  deterministic=not args.sampled)

    fails = [(s, c) for s, c in zip(seeds, comps) if c < 0.999]
    n_perfect = len(comps) - len(fails)

    print("\n" + "=" * 62)
    print(f"boards            : {len(comps)}")
    print(f"perfect (100%)    : {n_perfect}  ({n_perfect / len(comps):.4f})")
    print(f"mean completion   : {sum(comps) / len(comps):.4f}")
    print(f"failed            : {len(fails)}")

    # The number completion cannot see.
    ratios = [s["ratio"] for s in stats if s["ratio"] == s["ratio"]]
    doubled = sum(s["doubled"] for s in stats)
    n_boards_doubled = sum(1 for s in stats if s["doubled"])
    if ratios:
        ratios_sorted = sorted(ratios)
        print(f"copper ratio      : {sum(ratios) / len(ratios):.3f}x mean, "
              f"{ratios_sorted[len(ratios_sorted) // 2]:.3f}x median, "
              f"{ratios_sorted[-1]:.3f}x worst")
        print(f"double-routed     : {doubled} nets on {n_boards_doubled} boards "
              f"({n_boards_doubled / len(stats):.1%} of boards)")
        if n_boards_doubled:
            print("  ^ a net drawn twice still reads as COMPLETE. See "
                  "mzr/DESIGN_COPPER_SEEDED.md -- the fix is --copper-seeded, "
                  "not reward shaping (four attempts failed).")
    print("=" * 62)

    if fails:
        shown = fails if args.max_show == 0 else fails[:args.max_show]
        print(f"\nFAILED SEEDS ({len(shown)} shown of {len(fails)}):\n")
        print(f"{'seed':>10}  {'completion':>10}")
        print("-" * 24)
        for s, c in shown:
            print(f"{s:>10}  {c:>10.4f}")
        # A single line that can be pasted straight into the hand-review tool.
        head = " ".join(str(s) for s, _ in shown[:24])
        print(f"\nreview with:\n  python -m mzr.world.pool --stage {args.stage} --seeds {head}")
        if len(fails) > 24:
            print(f"  (first 24 of {len(fails)}; pool takes as many as you pass)")
    else:
        print("\nno failures.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
