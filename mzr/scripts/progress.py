"""Completion progress for a training run, read from its jsonl.

    python -m mzr.scripts.progress --dir /content/drive/MyDrive/mzr_ckpt/stage0_copper
    python -m mzr.scripts.progress --dir <run> --compare <other run>

The trainer's stdout is mostly heartbeat lines -- an eval only lands every
`--eval-every` updates -- so tailing the log between evals shows loss numbers
and no completion at all. This reads the jsonl the trainer appends to the
checkpoint directory, which has every eval in it, and answers the two questions
worth asking mid-run: *is completion going up*, and *how far from the gate*.

Reads the jsonl on Drive, so it works while the run is detached and needs
nothing from the training process.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

BAR = "█"
GATE_HITS = 3


def load(d: Path) -> tuple[list[dict], list[dict]]:
    """(all rows, eval rows) from a run directory or a jsonl path."""
    p = d if d.suffix == ".jsonl" else next(iter(sorted(d.glob("*.jsonl"))), None)
    if p is None or not p.exists():
        return [], []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                      # a torn final line while training writes
    return rows, [r for r in rows if "argmax_completion" in r]


def bar(x: float, width: int = 28) -> str:
    n = max(0, min(width, round(x * width)))
    return BAR * n + "·" * (width - n)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True, help="checkpoint dir, or a .jsonl path")
    p.add_argument("--compare", default=None, help="another run to score against")
    p.add_argument("--updates", type=int, default=600, help="planned total, for ETA")
    p.add_argument("--last", type=int, default=24, help="eval rows to show")
    args = p.parse_args()

    rows, evals = load(Path(args.dir))
    if not rows:
        print(f"no jsonl under {args.dir} yet -- the run has not written an update.")
        return 0

    cmp_by_update: dict[int, float] = {}
    if args.compare:
        _, ce = load(Path(args.compare))
        cmp_by_update = {r["update"]: r["argmax_completion"] for r in ce}

    done = rows[-1]["update"] + 1
    sec = sum(r.get("sec", 0.0) for r in rows[-50:]) / max(1, len(rows[-50:]))
    left = max(0, args.updates - done)

    print(f"run      {args.dir}")
    print(f"progress {done}/{args.updates} updates  ({done/max(1,args.updates)*100:.0f}%)  "
          f"{sec:.1f}s/update  ETA {left*sec/60:.0f} min")

    if not evals:
        nxt = min((r["update"] for r in rows if False), default=None)
        print(f"\nno eval yet -- the first one lands at the first --eval-every boundary.")
        print(f"latest: vloss {rows[-1].get('value_loss', 0):.1f}  "
              f"kl {rows[-1].get('approx_kl', 0):.3f}  ent {rows[-1].get('entropy', 0):.2f}")
        return 0

    best = max(r["argmax_completion"] for r in evals)
    hits = evals[-1].get("gate_hits", 0)
    print(f"best     {best:.3f} argmax   |   gate {hits}/{GATE_HITS} consecutive at 1.000\n")

    head = f"{'update':>7} {'argmax':>7} {'sampled':>8} {'':30}"
    print(head + ("  vs cmp" if cmp_by_update else ""))
    for r in evals[-args.last:]:
        u, a = r["update"], r["argmax_completion"]
        s = r.get("sampled_completion", a)
        mark = "  <- GATE" if r.get("gate_hits", 0) >= GATE_HITS else ""
        delta = ""
        if cmp_by_update:
            near = min(cmp_by_update, key=lambda k: abs(k - u), default=None)
            if near is not None and abs(near - u) <= 25:
                d = a - cmp_by_update[near]
                delta = f"  {d:+.3f}"
        print(f"{u:>7} {a:>7.3f} {s:>8.3f}  {bar(a)}{delta}{mark}")

    # Trend on the record, not on one noisy eval -- individual evals swing hard
    # for the first ~150 updates while `best` climbs monotonically, and judging
    # "stale vs improving" off the latest row triggers pointless restarts.
    first = evals[: max(1, len(evals) // 3)]
    last = evals[-max(1, len(evals) // 3):]
    fa = sum(r["argmax_completion"] for r in first) / len(first)
    la = sum(r["argmax_completion"] for r in last) / len(last)
    print(f"\ntrend    first third {fa:.3f} -> last third {la:.3f}  ({la - fa:+.3f})")
    fails = evals[-1].get("argmax_fail_seeds", [])
    if fails:
        print(f"failing  {len(fails)} seed(s) shown: {fails[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
