"""Live training dashboard, read from a run's jsonl.

    # in a Colab cell -- refreshes in place until you interrupt it
    from mzr.scripts.dashboard import live
    live('/content/drive/MyDrive/mzr_ckpt/stage1_replan',
         refs={'layer_hop': 0.8542, 'greedy': 0.6597, 'expert': 1.0})

    # or one static render / a terminal summary
    python -m mzr.scripts.dashboard --dir <run> --once

`scripts/progress.py` answers "is completion going up" as text. This is the
same data as a picture, plus the three series that text kept hiding:

* **return_mean** every update, not just on eval steps. Reward and completion
  are different objectives in this project -- a random policy once scored -330
  against greedy's -177 and still routed more nets -- so they are plotted
  together precisely so the gap is visible rather than assumed away.
* **dir_d0_frac**, the fraction of actions taken straight down the geodesic
  gradient. This is the number that decides stage 1: the field cannot see other
  nets' live copper, so yielding a channel *requires* leaving your own gradient.
  Every run before behaviour cloning sat pinned at d0 = 1.000 with 1 of 8
  directions in use, at completion ~0.83, and no reward weight moved it.
* **copper_median** and **right_angle_frac** against the stage's own gate
  thresholds, because completion alone has certified a policy that
  double-routed 46.5% of boards at 2.3x copper.

Reads the jsonl only. Nothing is imported from the training process and no GPU
is touched, so this is safe to run beside a live run -- or after it has died.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

#: Series pulled from every update row (cheap, dense).
_PER_UPDATE = ("return_mean", "value_loss", "approx_kl", "clip_frac")


def _find_jsonl(d: Path) -> Path | None:
    if d.suffix == ".jsonl":
        return d if d.exists() else None
    return next(iter(sorted(d.glob("*.jsonl"))), None)


def load(path: Path) -> tuple[list[dict], list[dict]]:
    """(all update rows, eval rows). Tolerates a torn final line -- the trainer
    may be mid-write."""
    if path is None or not path.exists():
        return [], []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows, [r for r in rows if "argmax_completion" in r]


def _stage_of(path: Path):
    """Recover the stage from the jsonl filename (`stage1.jsonl`), so the gate
    lines are the stage's real thresholds rather than guesses."""
    name = path.stem
    if not name.startswith("stage"):
        return None
    try:
        from mzr.training.curriculum import STAGES
        return STAGES.get(name[len("stage"):])
    except Exception:
        return None


def summarise(rows: list[dict], evs: list[dict], stage=None, refs=None) -> str:
    if not rows:
        return "no rows yet"
    refs = refs or {}
    out = [f"{len(rows)} updates, {len(evs)} evals"]
    sec = sum(r.get("sec", 0.0) for r in rows) / max(1, len(rows))
    out.append(f"{sec:.1f}s/update")
    if evs:
        e = evs[-1]
        out.append(
            f"u{e['update']}: argmax {e['argmax_completion']:.4f} "
            f"perfect {e['argmax_perfect']:.3f} | d0 {e['dir_d0_frac']:.3f} "
            f"({e['dir_distinct']}/8 dirs) | copper {e['copper_median']:.3f} "
            f"RA {e['right_angle_frac']:.3f} | return {e.get('return_mean', float('nan')):+.2f}"
        )
        best = max(x["argmax_completion"] for x in evs)
        out.append(f"best {best:.4f}")
        for k, v in refs.items():
            out.append(f"{k} {v:.4f}")
    return "  |  ".join(out)


def render(path: Path, refs: dict | None = None, ax_dpi: int = 100):
    """Draw the four panels. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt

    refs = refs or {}
    rows, evs = load(path)
    stage = _stage_of(path)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), dpi=ax_dpi)
    fig.suptitle(f"{path.parent.name} — {summarise(rows, evs, stage, refs)}", fontsize=9)

    u_all = [r["update"] for r in rows]
    u_ev = [r["update"] for r in evs]

    # -- completion ---------------------------------------------------------
    ax = axes[0][0]
    if evs:
        ax.plot(u_ev, [r["argmax_completion"] for r in evs], "o-", label="argmax", lw=2)
        if "sampled_completion" in evs[0]:
            ax.plot(u_ev, [r.get("sampled_completion", float("nan")) for r in evs],
                    ".--", label="sampled", alpha=0.6)
        ax.plot(u_ev, [r["argmax_perfect"] for r in evs], ".:", label="boards at 100%", alpha=0.7)
    for name, val in refs.items():
        ax.axhline(val, ls="--", lw=1, alpha=0.7, color="gray")
        ax.annotate(name, (0.01, val), xycoords=("axes fraction", "data"),
                    fontsize=7, va="bottom", color="gray")
    if stage is not None:
        ax.axhline(stage.gate[1], color="crimson", ls="-", lw=1.2, alpha=0.8)
        ax.annotate("GATE", (0.01, stage.gate[1]), xycoords=("axes fraction", "data"),
                    fontsize=7, va="bottom", color="crimson")
    ax.set_title("completion", fontsize=10); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # -- reward -------------------------------------------------------------
    ax = axes[0][1]
    if rows and "return_mean" in rows[0]:
        ax.plot(u_all, [r.get("return_mean", float("nan")) for r in rows], lw=1, alpha=0.8)
    ax.set_title("return_mean  (reward != completion — track both)", fontsize=10)
    ax.grid(alpha=0.25)

    # -- steering -----------------------------------------------------------
    ax = axes[1][0]
    if evs:
        ax.plot(u_ev, [r["dir_d0_frac"] for r in evs], "o-", lw=2, label="d0 fraction")
        ax.plot(u_ev, [r["dir_distinct"] / 8.0 for r in evs], ".--",
                alpha=0.7, label="distinct dirs / 8")
        if stage is not None and stage.max_d0_frac < 1.0:
            ax.axhline(stage.max_d0_frac, color="crimson", ls="--", lw=1)
            ax.annotate("max d0", (0.01, stage.max_d0_frac),
                        xycoords=("axes fraction", "data"), fontsize=7, color="crimson")
    ax.set_title("steering  (d0 = straight down the geodesic gradient)", fontsize=10)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # -- route quality ------------------------------------------------------
    ax = axes[1][1]
    if evs:
        ax.plot(u_ev, [r["copper_median"] for r in evs], "o-", lw=2, label="copper median")
        ax.plot(u_ev, [r["right_angle_frac"] for r in evs], "s-", lw=2, label="right-angle")
        if stage is not None:
            ax.axhline(stage.max_copper, color="tab:blue", ls="--", lw=1, alpha=0.7)
            ax.axhline(stage.max_right_angle, color="tab:orange", ls="--", lw=1, alpha=0.7)
    ax.set_title("route quality vs gate thresholds", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    for a in axes.ravel():
        a.set_xlabel("update", fontsize=8)
        a.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def live(run_dir, refs: dict | None = None, refresh: int = 30, minutes: int = 180):
    """Refresh the dashboard in place inside a notebook.

    Blocks the cell it runs in (not the training process, which is detached) --
    interrupt it whenever. `minutes` is a stop so an abandoned cell does not
    hold the kernel forever.
    """
    from IPython.display import clear_output, display
    import matplotlib.pyplot as plt

    path = _find_jsonl(Path(run_dir))
    if path is None:
        print(f"no jsonl under {run_dir} yet — is the run started?")
        return
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        fig = render(path, refs)
        clear_output(wait=True)
        display(fig)
        plt.close(fig)
        time.sleep(refresh)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True, help="checkpoint dir, or a .jsonl path")
    p.add_argument("--once", action="store_true", help="print a summary and exit")
    p.add_argument("--save", default=None, help="write the figure to this path")
    p.add_argument("--refresh", type=int, default=30)
    p.add_argument("--ref", action="append", default=[],
                   help="reference line, name=value (repeatable)")
    args = p.parse_args()

    refs = {}
    for r in args.ref:
        k, _, v = r.partition("=")
        try:
            refs[k] = float(v)
        except ValueError:
            pass

    path = _find_jsonl(Path(args.dir))
    if path is None:
        print(f"no jsonl under {args.dir}")
        return 1
    rows, evs = load(path)
    print(summarise(rows, evs, _stage_of(path), refs))
    if args.save:
        import matplotlib
        matplotlib.use("Agg")
        render(path, refs).savefig(args.save, bbox_inches="tight")
        print(f"wrote {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
