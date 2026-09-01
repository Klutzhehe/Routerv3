"""Contact sheet of routed boards, drawn inline in a Colab cell.

    from mzr.eval.plot import contact_sheet
    contact_sheet(ckpt='/content/drive/MyDrive/mzr_ckpt/stage0/stage0_best.pt')

matplotlib only -- no KiCad, no `pcbnew`, works headless. The point is not a
pretty picture: it is that a completion number tells you a board failed while
only the copper tells you *why* -- a frontier that walked into a pocket, a
cross-layer net that never changed layer, a trace that turned a right angle
where two 45s would have done.

Colour follows fab drawing convention rather than taste: the top layer warm,
the bottom layer cool, keepouts hatched, vias ringed. Right-angle corners are
ringed too, because `RewardConfig.corner` exists to drive that count down and a
picture that hides them cannot show whether it worked.
"""

from __future__ import annotations

import numpy as np

F_CU = "#EA580C"     # top copper, warm
B_CU = "#0891B2"     # bottom copper, cool
KEEPOUT = "#94A3B8"
PAD = "#0F172A"
CORNER = "#DC2626"
SUBSTRATE = "#F1F5F9"
LAYER_C = [F_CU, B_CU]


def _unpack(b64: str, H: int, W: int) -> np.ndarray:
    import base64

    bits = np.unpackbits(np.frombuffer(base64.b64decode(b64), dtype=np.uint8))
    return bits[: H * W].reshape(H, W).astype(bool)


def _octant(dy: int, dx: int) -> int:
    import math

    return int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8


def draw_board(ax, bd, H, W, layers, show_corners=True):
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)          # image convention: y grows downward
    ax.set_aspect("equal")
    ax.set_facecolor(SUBSTRATE)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#CBD5E1")

    ko = _unpack(bd["keepout"][0], H, W)
    ys, xs = np.nonzero(ko)
    ax.scatter(xs, ys, s=1.2, c=KEEPOUT, marker="s", linewidths=0, alpha=.55)

    for t in bd["traces"]:
        pts = t["pts"]
        prev_o = None
        for a, b in zip(pts, pts[1:]):
            if a[1] == b[1] and a[2] == b[2]:        # via: same cell, new layer
                ax.plot(a[2], a[1], "o", mfc="none", mec=PAD, ms=7, mew=1.4)
                prev_o = None                         # a via is not a corner
                continue
            ax.plot([a[2], b[2]], [a[1], b[1]], "-", color=LAYER_C[b[0]],
                    lw=2.0, solid_capstyle="round")
            o = _octant(b[1] - a[1], b[2] - a[2])
            if show_corners and prev_o is not None:
                d = abs(o - prev_o)
                if min(d, 8 - d) >= 2:                # 90 degrees or sharper
                    ax.plot(a[2], a[1], "o", mfc="none", mec=CORNER, ms=9, mew=1.6)
            prev_o = o

    for n in bd["nets"]:
        for q in (n["src"], n["dst"]):
            ax.plot(q[2], q[1], "s", color=PAD, ms=6)
            ax.plot(q[2], q[1], "s", color=LAYER_C[q[0]], ms=3)

    bn = bd.get("bends", {})
    n0 = bd["nets"][0] if bd["nets"] else {"vias": 0}
    bits = [f"seed {bd['seed']}", f"{bd['completion']*100:.0f}%"]
    if n0.get("vias"):
        bits.append(f"{n0['vias']}v")
    if bn.get("right_angle"):
        bits.append(f"{bn['right_angle']}x90deg")
    ax.set_title("  ".join(bits), fontsize=9, family="monospace",
                 color="#DC2626" if bd["completion"] < 0.999 else "#0F172A")


def contact_sheet(blob=None, *, ckpt=None, stage="0", device="cpu", boards=9,
                  seeds=None, sampled=False, cols=3, save=None, show_corners=True,
                  copper_seeded=False, geodesic_refresh=16):
    """Draw a grid of routed boards. Pass `blob` from `mzr.eval.render.export`,
    or a `ckpt` path to route the boards here.

    **`copper_seeded` must match the run that produced the checkpoint.**
    `export()` warns that "a copper-seeded policy measured in a pad-targeted
    world is measuring a different game", and this function used to ignore
    that and take the default -- so a copper-seeded checkpoint was drawn
    growing TWO frontiers per net, in the very world copper-seeding exists to
    replace. The picture then showed boards completing at 100% that the
    matching eval had scored 0.000, because the two runs were not the same
    experiment. A contact sheet whose numbers contradict the eval is worse
    than no contact sheet.

    `geodesic_refresh` defaults to 16 to match `training/run.py`'s default
    rather than `export()`'s 8, for the same reason.
    """
    import matplotlib.pyplot as plt

    if blob is None:
        import torch

        from mzr.eval.render import export
        from mzr.scripts.diagnose_stage0 import load_policy
        from mzr.training.curriculum import EVAL_SEEDS, STAGES

        torch.manual_seed(0)
        st = STAGES[stage]
        sd = seeds if seeds else EVAL_SEEDS[:boards]
        policy, _, _ = load_policy(ckpt, st, device)
        blob = export(policy, st, device, sd, deterministic=not sampled,
                      copper_seeded=copper_seeded, geodesic_refresh=geodesic_refresh)

    bs = blob["boards"]
    rows = (len(bs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(bs):]:
        ax.axis("off")
    for ax, bd in zip(axes, bs):
        draw_board(ax, bd, blob["height"], blob["width"], blob["layers"], show_corners)

    t = blob.get("bends", {})
    tot = sum(t.values()) or 1
    fig.suptitle(
        f"{blob['stage']}  --  argmax completion {blob['mean_completion']:.3f}, "
        f"{blob['steps']} macro-steps   |   bends: {t.get('straight',0)} straight, "
        f"{t.get('soft',0)} at 45deg, {t.get('right_angle',0)} at >=90deg "
        f"({t.get('right_angle',0)/tot*100:.0f}%)",
        fontsize=10, family="monospace", y=.99,
    )
    fig.text(.5, .005, f"{F_CU} F.Cu   {B_CU} B.Cu   o via   o right-angle corner",
             ha="center", fontsize=8, family="monospace", color="#64748B")
    fig.tight_layout(rect=[0, .02, 1, .97])
    if save:
        fig.savefig(save, dpi=140, bbox_inches="tight")
        print("wrote", save)
    return fig
