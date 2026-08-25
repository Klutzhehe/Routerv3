"""The sim-to-real gate: does KiCad agree that our copper is legal?

**Run this before training anything.** The whole architecture rests on one
claim -- that a lattice whose pitch is `min_track_width + min_clearance` makes
cell occupancy equivalent to a clearance check, so anything the fast engine
accepts is DRC-clean by construction. If that reasoning is wrong at corners,
pad edges, or 45-degree segments, every number produced by training on this
lattice is measuring a fiction, and the sooner that is known the better. It is
an afternoon's question, and `neuroroute/DESIGN.md` section 11 deliberately
puts it before step 3.

It needs no trained policy: route boards with the non-learned baseline, export
them, and hand them to KiCad's own `DRC_ENGINE`.

    python -m neuroroute.scripts.validate_kicad --boards 8 --out /tmp/nr_drc

Requires `kicad-cli` on PATH (ships with KiCad 7+). On Colab that is
`apt-get install -y kicad` -- the full source build the PNS bridge needs is
**not** required here, which is the point: this check is cheap enough to run
on every change.

The two failure modes it separates:

* `kicad-cli` cannot parse the file  -> the exporter is wrong (most likely the
  layer ordinals; see `eval/kicad_export.py::_layer_ordinal`).
* it parses and reports violations   -> the *lattice model* is wrong, and the
  fix is a wider pitch or a wider dilation rule, not a bug hunt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from neuroroute.env.baselines import layer_hop_action
from neuroroute.env.route_env import EnvConfig, NeuroRouteEnv
from neuroroute.eval.kicad_export import export_board
from neuroroute.world.engine import WorldConfig
from neuroroute.world.generator import GeneratorConfig
from neuroroute.world.spec import BoardSpec, LayerStack


def run_drc(pcb: Path, report: Path) -> tuple[bool, dict]:
    """Run `kicad-cli pcb drc` and parse its JSON report."""
    cmd = [
        "kicad-cli", "pcb", "drc",
        "--format", "json",
        "--severity-error", "--severity-warning",
        "--exit-code-violations",
        "-o", str(report),
        str(pcb),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not report.exists():
        return False, {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}
    try:
        return True, json.loads(report.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, {"parse_error": str(exc), "stdout": proc.stdout, "stderr": proc.stderr}


#: Violations that mean **the copper is illegal**. These are the ones the
#: lattice model claims cannot happen, so any of them is a real refutation.
LEGALITY_CODES = {
    "clearance", "shorting_items", "copper_sliver", "track_width",
    "via_diameter", "annular_width", "hole_clearance", "hole_near_hole",
    "drill_out_of_range", "track_angle", "creepage", "starved_thermal",
    "copper_edge_clearance", "silk_over_copper",
}
#: Violations that mean **the board is incomplete**, which is a statement about
#: the routing policy, not about whether the geometry model is sound. The
#: non-learned baseline routes roughly a third of nets, so these are expected
#: and are reported separately rather than counted against the model.
COMPLETENESS_CODES = {"unconnected_items", "track_dangling"}
#: Library/parity bookkeeping. Nothing to do with copper: KiCad is saying it
#: cannot find a footprint library named "neuroroute", which is true and
#: irrelevant.
BOOKKEEPING_CODES = {"lib_footprint_issues", "footprint", "footprint_type_mismatch"}


def summarise(report: dict) -> dict[tuple[str, str], int]:
    """Violation counts keyed by (error code, KiCad's own severity).

    Severity is taken from KiCad rather than assigned here. `copper_sliver`,
    for instance, is a manufacturability *warning* -- a thin fragment of
    copper, not a short and not a clearance failure -- and treating it as a
    hard legality failure would condemn a model KiCad itself considers
    fabricable.
    """
    counts: dict[tuple[str, str], int] = {}
    for key in ("violations", "unconnected_items", "schematic_parity"):
        for item in report.get(key, []) or []:
            code = item.get("type") or item.get("code") or key
            sev = item.get("severity", "error")
            counts[(code, sev)] = counts.get((code, sev), 0) + 1
    return counts


def classify(counts: dict[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    """Split a violation tally into legality / advisory / completeness / bookkeeping.

    Only **errors** that are about copper legality count against the lattice
    model. Warnings are reported alongside, because a rising sliver count is
    still worth knowing about even though it does not refute the model.
    """
    out = {"legality": {}, "advisory": {}, "completeness": {}, "bookkeeping": {}, "unknown": {}}
    for (code, sev), n in counts.items():
        if code in COMPLETENESS_CODES:
            bucket = "completeness"
        elif code in BOOKKEEPING_CODES:
            bucket = "bookkeeping"
        elif sev != "error":
            bucket = "advisory"
        elif code in LEGALITY_CODES:
            bucket = "legality"
        else:
            # An unrecognised *error* counts against the model. Erring the
            # other way would let a real violation hide behind a name this
            # list has not seen yet.
            bucket = "unknown"
        out[bucket][f"{code}({sev})" if bucket == "advisory" else code] = (
            out[bucket].get(f"{code}({sev})" if bucket == "advisory" else code, 0) + n
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Export routed boards and DRC them with real KiCad")
    p.add_argument("--boards", type=int, default=8)
    p.add_argument("--nets", type=int, default=40)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--seed-base", type=int, default=500000)
    p.add_argument("--wide-frac", type=float, default=0.0,
                   help="fraction of nets on a non-minimum track width -- stresses the "
                        "lateral dilation rule, which is the part of the lattice model "
                        "most likely to be too tight")
    p.add_argument("--keepouts", type=int, default=0)
    p.add_argument("--pours", type=int, default=0)
    p.add_argument("--out", default="drc_out")
    p.add_argument("--export-only", action="store_true", help="skip DRC; just write the files")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spec = BoardSpec(
        height_cells=args.size, width_cells=args.size,
        layers=LayerStack(num_layers=args.layers),
    )
    env = NeuroRouteEnv(
        EnvConfig(
            spec=spec,
            world=WorldConfig(
                batch_size=args.boards, max_heads=args.heads,
                max_nets=max(64, args.nets + 8), max_steps_per_net=args.steps, device="cpu",
            ),
            generator=GeneratorConfig(
                num_nets=args.nets, num_components=8,
                wide_net_frac=args.wide_frac, num_keepouts=args.keepouts,
                num_pours=args.pours,
            ),
            max_episode_steps=args.steps * 6,
        )
    )

    print("=" * 72)
    print("NeuroRoute -> KiCad sim-to-real DRC check")
    print(f"  lattice pitch {spec.pitch_mm:.3f} mm  =  {spec.rules.min_track_width} mm track "
          f"+ {spec.rules.min_clearance} mm clearance")
    print(f"  {args.boards} boards, {args.nets} nets, {args.layers} layers, "
          f"{args.size}x{args.size} cells ({args.size * spec.pitch_mm:.1f} mm square)")
    print("=" * 72, flush=True)

    seeds = list(range(args.seed_base, args.seed_base + args.boards))
    obs = env.reset(seeds)
    for _ in range(env.cfg.max_episode_steps):
        obs, rew, done, info = env.step(layer_hop_action(obs))
        if bool(done.all()):
            break

    completion = env.world.completion()
    print(f"routed with the non-learned baseline: mean completion {float(completion.mean()):.1%}\n")

    routed_nets = 0
    for b in range(args.boards):
        stats = export_board(env.world, b, out / f"board_{seeds[b]}.kicad_pcb")
        done = int(((env.world.net_status[b] == 2) & env.world.net_valid[b]).sum())
        routed_nets += done
        print(f"  board {seeds[b]}: {stats.nets} nets ({done} routed), {stats.tracks} tracks, "
              f"{stats.vias} vias, {stats.pads} pads -> board_{seeds[b]}.kicad_pcb")

    if args.export_only:
        print("\n--export-only: stopping before DRC.")
        return 0

    if shutil.which("kicad-cli") is None:
        print("\nkicad-cli not on PATH -- files were written but not checked.")
        print("Install KiCad (Colab: `apt-get install -y kicad`) and re-run.")
        return 2

    print("\nrunning KiCad DRC...\n", flush=True)
    totals = {"legality": {}, "advisory": {}, "completeness": {}, "bookkeeping": {}, "unknown": {}}
    parse_failures = 0
    for b in range(args.boards):
        pcb = out / f"board_{seeds[b]}.kicad_pcb"
        ok, report = run_drc(pcb, out / f"board_{seeds[b]}.drc.json")
        if not ok:
            parse_failures += 1
            print(f"  board {seeds[b]}: KiCad could not produce a report")
            for k, val in report.items():
                if val:
                    print(f"      {k}: {str(val)[:400]}")
            continue
        groups = classify(summarise(report))
        n_legal = sum(groups["legality"].values()) + sum(groups["unknown"].values())
        detail = groups["legality"] | groups["unknown"]
        print(
            f"  board {seeds[b]}: {n_legal} legality violations"
            + (f"  {detail}" if detail else "  CLEAN")
            + f"   (incomplete: {sum(groups['completeness'].values())})"
        )
        for g, d in groups.items():
            for k, val in d.items():
                totals[g][k] = totals[g].get(k, 0) + val

    print("\n" + "=" * 72)
    if parse_failures:
        print(f"EXPORTER PROBLEM: {parse_failures}/{args.boards} boards could not be read by KiCad.")
        print("Fix the writer before drawing any conclusion about the lattice model.")
        print("First suspect: layer ordinals -- see eval/kicad_export.py::_layer_ordinal.")
        return 1

    legality = sum(totals["legality"].values()) + sum(totals["unknown"].values())
    per_1k = 1000.0 * legality / max(1, routed_nets)
    print(f"SIM-TO-REAL GAP: {legality} legality violations over {routed_nets} "
          f"ROUTED nets = {per_1k:.1f} per 1000 nets")
    if legality == 0:
        print("\nThe lattice model is DRC-clean by construction, as designed.")
        print("Cell occupancy really is equivalent to a clearance check at this")
        print("pitch, so training against the fast engine is training against")
        print("something KiCad agrees with. Proceed to training.")
    else:
        print("\nLegality violations by code:")
        for k, val in sorted((totals["legality"] | totals["unknown"]).items(), key=lambda kv: -kv[1]):
            print(f"    {k}: {val}")
        print("\nThe lattice pitch or a dilation rule is too tight somewhere.")
        print("Widen it (DesignRules._radius_cells) and re-run -- do not train")
        print("against a model KiCad disagrees with.")

    inc = sum(totals["completeness"].values())
    book = sum(totals["bookkeeping"].values())
    adv = sum(totals["advisory"].values())
    print("\nreported but not counted against the model:")
    print(f"    manufacturability warnings:     {adv}   {totals['advisory'] or '{}'}")
    print(f"    incompleteness (unrouted nets): {inc}   {totals['completeness'] or '{}'}")
    print(f"    library bookkeeping:            {book}   {totals['bookkeeping'] or '{}'}")
    if adv:
        print("    (severity is KiCad's own, not a reclassification here -- a rising")
        print("     warning count is still worth watching even though it is not a short)")
    print("=" * 72)
    return 0 if legality == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
