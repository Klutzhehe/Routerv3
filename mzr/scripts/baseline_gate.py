"""Run the NON-LEARNED baselines against a stage's own gate.

The standing rule this enforces (`mzr/DESIGN.md` section 7.1):

* If a **parameter-free heuristic clears the gate**, the stage is not testing
  what it claims to test.
* If a parameter-free heuristic **cannot** clear it, neither can a policy whose
  untrained argmax is exactly `baselines.greedy` -- so the stage is unpassable
  for a substrate reason, and training it is burning GPU on a bug.

Stage 0 sat under its gate for many sessions because nobody ran this. Measured
at the time, on the first 48 held-out seeds: `greedy` scored 0.7500 completion
at 2.00x copper with 24 of 48 boards double-routed and 40.4% right angles --
all four numbers produced with no policy involved.

    python -m mzr.scripts.baseline_gate --stage 0
"""

from __future__ import annotations

import argparse

import torch

from mzr.eval.quality import route_quality
from mzr.training.curriculum import EVAL_SEEDS, STAGES
from mzr.world import baselines


@torch.no_grad()
def baseline_report(stage_name: str, n_boards: int = 48, device: str = "cpu") -> dict:
    # Imported here: training.run imports eval.quality, which imports
    # eval.render, which imports diagnose_stage0 -> training.run.
    from mzr.training.run import make_env

    stage = STAGES[stage_name]
    seeds = EVAL_SEEDS[:n_boards]
    rows = {}
    for name, fn in baselines.BASELINES.items():
        env = make_env(stage, batch=len(seeds), device=device, seed=0)
        env.reset(seeds)
        baselines.rollout(env.world, fn, max_steps=stage.max_macro_steps)
        st = env.world.board_stats()
        q = route_quality(env.world)
        comp = float(st["completion"].mean())
        passes = (
            comp >= stage.gate[1]
            and q["copper_median"] <= stage.max_copper
            and q["right_angle_frac"] <= stage.max_right_angle
            and q["doubled"] == 0
        )
        rows[name] = {
            "completion": comp,
            "copper_median": q["copper_median"],
            "copper_mean": q["copper_mean"],
            "doubled": q["doubled"],
            "right_angle_frac": q["right_angle_frac"],
            "vias": float(st["vias"].mean()),
            "clears_gate": passes,
        }
    return rows


def print_report(stage_name: str, rows: dict) -> None:
    stage = STAGES[stage_name]
    print(f"\n== baseline preflight: stage {stage_name} ({stage.name}) ==")
    print(f"   gate: completion {stage.gate[1]:.2f}, copper_median <= {stage.max_copper}, "
          f"right_angle <= {stage.max_right_angle}, 0 double-routed")
    for name, r in rows.items():
        print(f"   {name:10s} completion {r['completion']:.4f}  copper {r['copper_median']:.3f} "
              f"(mean {r['copper_mean']:.3f})  doubled {r['doubled']:3d}  "
              f"right-angle {r['right_angle_frac']:.3f}  vias {r['vias']:.2f}  "
              f"-> {'CLEARS GATE' if r['clears_gate'] else 'below gate'}")

    best = max(r["completion"] for r in rows.values())
    if best < stage.gate[1]:
        print(f"   !! no baseline reaches the gate (best {best:.4f}). A policy whose "
              f"untrained argmax IS `greedy` starts below it too -- check the substrate "
              f"before spending GPU. See mzr/DESIGN.md section 7.1.")
    elif any(r["clears_gate"] for r in rows.values()):
        print("   note: a parameter-free baseline clears this gate outright. That is "
              "correct for stage 0 (the policy starts AT the bar and must not regress); "
              "on a later stage it means the gate is not measuring the new mechanism.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="0", choices=sorted(STAGES))
    p.add_argument("--boards", type=int, default=48)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    print_report(args.stage, baseline_report(args.stage, args.boards, args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
