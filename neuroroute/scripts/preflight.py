"""One command that proves the machine can train, and reports why if it cannot.

    python -m neuroroute.scripts.preflight

Run this **before** any training run, and paste the whole output back. It is
designed around how this project is operated: whoever drives Colab reports real
output rather than diagnosing (`AGENTS.md`), so the diagnosis has to already be
in the output.

Seven checks, cheapest and most load-bearing first, each independent so a later
failure does not hide an earlier pass:

1. environment      -- torch, CUDA, GPU, commit hash, whether the tree is dirty
2. imports          -- every module, so a syntax error surfaces here not at hour 3
3. geometry         -- lattice ops against a brute-force reference
4. environment sim  -- invariants re-derived from the occupancy grid
5. refine phase     -- length tuning actually changes length, reversibly
6. KiCad DRC gate   -- real DRC_ENGINE on exported boards (skipped if no kicad-cli)
7. training smoke   -- a few real updates on this device, forward AND backward

Exit code 0 means: start training. Anything else means the number that comes
out of training would not have meant anything.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"\n>>> {name}: {'PASS' if ok else 'FAIL'}" + (f" -- {detail}" if detail else ""),
          flush=True)


def header(n: int, title: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(f"[{n}/7] {title}", flush=True)
    print("=" * 78, flush=True)


def run_module(mod: str, argv: list[str] | None = None) -> tuple[bool, str]:
    """Run a check module in a subprocess so a hard crash is contained.

    A segfault or an OOM inside one check must not take the rest of preflight
    with it -- the point is to collect every signal in one pass.
    """
    cmd = [sys.executable, "-m", mod, *(argv or [])]
    t = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t
    print(proc.stdout[-6000:], flush=True)
    if proc.stderr.strip():
        print("--- stderr ---", flush=True)
        print(proc.stderr[-3000:], flush=True)
    return proc.returncode == 0, f"{dt:.1f}s, exit {proc.returncode}"


def main() -> int:
    p = argparse.ArgumentParser(description="NeuroRoute preflight")
    p.add_argument("--device", default=None, help="cuda / cpu; auto-detected by default")
    p.add_argument("--skip-drc", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--out", default="preflight_out")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  NEUROROUTE PREFLIGHT")
    print("  Paste this entire output back. Every check is independent.")
    print("=" * 78, flush=True)

    # -- 1. environment ------------------------------------------------------
    header(1, "environment")
    try:
        from neuroroute.training.telemetry import environment_report

        env = environment_report()
        for k, v in env.items():
            print(f"  {k:<18} {v}")
        device = args.device or ("cuda" if env["cuda_available"] else "cpu")
        print(f"  {'device chosen':<18} {device}")
        if env.get("git_dirty"):
            print("  !! working tree is DIRTY -- running code is not the commit above")
        if not env["cuda_available"]:
            print("  !! no GPU. Training will work but be far slower; in Colab set")
            print("     Runtime > Change runtime type > GPU.")
        record("environment", True, f"{env.get('gpu', 'CPU')} / torch {env['torch']}")
    except Exception:
        traceback.print_exc()
        record("environment", False, "could not import neuroroute.training.telemetry")
        return 1

    # -- 2. imports ----------------------------------------------------------
    header(2, "imports")
    mods = [
        "neuroroute.world.spec", "neuroroute.world.geometry",
        "neuroroute.world.generator", "neuroroute.world.engine",
        "neuroroute.env.observation", "neuroroute.env.rewards",
        "neuroroute.env.route_env", "neuroroute.env.baselines",
        "neuroroute.models.encoder", "neuroroute.models.forecaster",
        "neuroroute.models.policy", "neuroroute.training.ppo",
        "neuroroute.training.curriculum", "neuroroute.training.telemetry",
        "neuroroute.training.run", "neuroroute.eval.kicad_export",
        "neuroroute.eval.render",
    ]
    bad = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as exc:
            bad.append((m, repr(exc)))
    for m, e in bad:
        print(f"  FAILED {m}: {e}")
    print(f"  {len(mods) - len(bad)}/{len(mods)} modules import cleanly")
    record("imports", not bad, f"{len(mods) - len(bad)}/{len(mods)}")
    if bad:
        return 1

    # -- 3-5. correctness ----------------------------------------------------
    header(3, "lattice geometry vs brute force")
    ok, d = run_module("neuroroute.scripts.verify_geometry")
    record("geometry", ok, d)

    header(4, "environment invariants (connectivity re-derived by flood fill)")
    ok, d = run_module("neuroroute.scripts.verify_env")
    record("environment sim", ok, d)

    header(5, "refine phase (length tuning)")
    ok, d = run_module("neuroroute.scripts.verify_refine")
    record("refine phase", ok, d)

    # -- 6. the sim-to-real gate --------------------------------------------
    header(6, "KiCad DRC gate -- does real KiCad agree the copper is legal?")
    if args.skip_drc:
        record("kicad drc", True, "skipped by flag")
    elif shutil.which("kicad-cli") is None:
        print("  kicad-cli not found. Install it and re-run:")
        print("      !apt-get -qq install -y kicad")
        print("  This is the one check that catches 'training against a fiction',")
        print("  so do not skip it permanently.")
        record("kicad drc", False, "kicad-cli not installed")
    else:
        print(f"  kicad-cli: {shutil.which('kicad-cli')}")
        subprocess.run(["kicad-cli", "version"], check=False)
        ok, d = run_module(
            "neuroroute.scripts.validate_kicad",
            ["--boards", "4", "--nets", "30", "--layers", "8", "--size", "80",
             "--heads", "5", "--steps", "80", "--wide-frac", "0.3", "--keepouts", "2",
             "--out", str(out / "drc")],
        )
        record("kicad drc", ok, d)

    # -- 7. training smoke ---------------------------------------------------
    header(7, "training smoke -- real forward AND backward on this device")
    if args.skip_train:
        record("training smoke", True, "skipped by flag")
    else:
        ok, d = run_module(
            "neuroroute.training.run",
            ["--stage", "0", "--device", device, "--batch", "4", "--heads", "2",
             "--width", "16", "--layers", "2", "--size", "64", "--rollout", "8",
             "--updates", "4", "--eval-every", "2", "--checkpoint-every", "3",
             "--render-every", "2", "--drc-every", "0",
             "--checkpoint-dir", str(out / "smoke")],
        )
        record("training smoke", ok, d)
        smoke = out / "smoke"
        for f in ("train_log.jsonl", "latest.pt", "curves.png", "console.log"):
            print(f"  {'OK ' if (smoke / f).exists() else 'MISSING'} {smoke / f}")

    # -- summary -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  PREFLIGHT SUMMARY")
    print("=" * 78)
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<20} {detail}")
    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 78)
    if failed:
        print(f"  {len(failed)} check(s) FAILED: {', '.join(failed)}")
        print("  Report this whole output. Do not start a long training run.")
        return 1
    print("  All checks passed. Safe to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
