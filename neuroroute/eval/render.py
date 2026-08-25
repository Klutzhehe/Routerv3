"""Render routed boards to PNG, for looking at failures.

`docs/RL_PLAN.md` names debuggability as the reason RL was abandoned here once,
and gives the answer directly: *"Render every failed episode. A contact sheet of
100 failures shows the failure mode at a glance; a reward curve never will."*
This is that, for the lattice world.

The distinction that matters in the output is **which nets failed and where the
copper that blocked them is** -- not a pretty picture. So failed nets are drawn
as bright dashed lines between their pads, over the copper that actually got
laid, and the per-layer split is preserved because "why did this fail" on an
8-layer board is usually "everything piled onto layer 0".

matplotlib only. No KiCad, no `pcbnew`, works in a headless Colab cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neuroroute.world.engine import STATUS_DONE, STATUS_FAILED, BatchedRouterWorld


def _colourise(occ: np.ndarray) -> np.ndarray:
    """Occupancy plane -> RGB. Distinct hue per net, grey for keepout."""
    h, w = occ.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[...] = 0.06  # near-black substrate

    img[occ < 0] = np.array([0.30, 0.30, 0.34])  # keepout / board edge

    nets = occ[occ > 0]
    if nets.size:
        # Golden-ratio hue stepping keeps adjacent net ids visually distinct,
        # which matters because adjacent ids are often adjacent on the board.
        import colorsys

        for n in np.unique(nets):
            hue = ((int(n) * 0.61803398875) % 1.0)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
            img[occ == n] = np.array([r, g, b])
    return img


def render_board(
    world: BatchedRouterWorld,
    board_index: int,
    path: str | Path,
    title: str | None = None,
    max_layers: int = 8,
):
    """One board, one panel per copper layer, failed nets overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = board_index
    occ = world.occ[b].cpu().numpy()
    L = min(occ.shape[0], max_layers)
    comp = float(world.completion()[b])

    valid = world.net_valid[b].cpu().numpy()
    status = world.net_status[b].cpu().numpy()
    src = world.net_src[b].cpu().numpy()
    dst = world.net_dst[b].cpu().numpy()
    kind = world.net_kind[b].cpu().numpy()

    cols = min(L, 4)
    rows = (L + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.6 * rows), squeeze=False)
    fig.patch.set_facecolor("#0d1117")

    for l in range(rows * cols):
        ax = axes[l // cols][l % cols]
        ax.set_facecolor("#0d1117")
        ax.set_xticks([])
        ax.set_yticks([])
        if l >= L:
            ax.axis("off")
            continue
        ax.imshow(_colourise(occ[l]), interpolation="nearest", origin="upper")

        # Failed nets: dashed line pad-to-pad, drawn on the layers they touch.
        # This is the whole point of the render -- what did NOT get routed, and
        # what is sitting in the way.
        for n in range(len(valid)):
            if not valid[n] or status[n] != STATUS_FAILED:
                continue
            legs = 2 if kind[n] == 1 else 1
            for leg in range(legs):
                s, d = src[n, leg], dst[n, leg]
                if l not in (int(s[0]), int(d[0])):
                    continue
                ax.plot([s[2], d[2]], [s[1], d[1]], color="#ff3b30",
                        lw=1.1, ls="--", alpha=0.85, zorder=3)
                ax.scatter([s[2], d[2]], [s[1], d[1]], s=14, c="#ff3b30", zorder=4)

        n_here = int((occ[l] > 0).sum())
        ax.set_title(f"layer {l}   {n_here} cells", color="#c9d1d9", fontsize=9)

    n_fail = int(((status == STATUS_FAILED) & valid).sum())
    n_done = int(((status == STATUS_DONE) & valid).sum())
    head = title or f"board {b}"
    fig.suptitle(
        f"{head}   completion {comp:.1%}   routed {n_done}   failed {n_fail} (dashed red)",
        color="#e6edf3", fontsize=12,
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def contact_sheet(
    world: BatchedRouterWorld,
    path: str | Path,
    layer: int = 0,
    max_boards: int = 16,
):
    """One panel per board, worst completion first.

    Sorting by completion is deliberate: the interesting boards are the ones
    that failed, and a sheet in batch order buries them among the easy ones.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comp = world.completion().cpu().numpy()
    order = np.argsort(comp)[:max_boards]
    n = len(order)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows), squeeze=False)
    fig.patch.set_facecolor("#0d1117")
    occ = world.occ.cpu().numpy()

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.set_facecolor("#0d1117")
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue
        b = int(order[i])
        ax.imshow(_colourise(occ[b, min(layer, occ.shape[1] - 1)]),
                  interpolation="nearest", origin="upper")
        ax.set_title(f"board {b}   {comp[b]:.0%}", color="#c9d1d9", fontsize=9)

    fig.suptitle(f"worst {n} boards, layer {layer}", color="#e6edf3", fontsize=12)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def learning_curves(history: list[dict], path: str | Path):
    """Reward, completion, losses and rejected-action rate over updates.

    **Completion is plotted alongside reward on purpose.** This repo has a
    measured case of a policy scoring worse reward while completing more nets
    (`docs/RL_PLAN.md`), so a reward curve alone can move the wrong way and
    look like progress.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return None
    x = [h.get("update", i) for i, h in enumerate(history)]

    def series(key):
        return [h.get(key, float("nan")) for h in history]

    panels = [
        ("completion", ["completion"], "completion rate"),
        ("reward", ["reward"], "mean episode reward"),
        ("losses", ["policy_loss", "value_loss"], "PPO losses"),
        ("health", ["entropy", "clip_frac", "rejected_action_rate"], "policy health"),
        ("forecast", ["forecast", "forecast_mae", "baseline_mae"], "forecaster"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.5 * len(panels)), sharex=True)
    fig.patch.set_facecolor("#0d1117")

    for ax, (_name, keys, title) in zip(axes, panels):
        ax.set_facecolor("#161b22")
        for k in keys:
            y = series(k)
            if all(v != v for v in y):  # all NaN -- never logged
                continue
            ax.plot(x, y, label=k, lw=1.4)
        ax.set_title(title, color="#c9d1d9", fontsize=10, loc="left")
        ax.tick_params(colors="#8b949e")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, facecolor="#0d1117", labelcolor="#c9d1d9")
    axes[-1].set_xlabel("update", color="#8b949e")

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
