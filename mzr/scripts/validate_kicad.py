"""The sim-to-real gate: does KiCad agree that our copper is legal?

**Run this before training anything.** `mzr/DESIGN.md` build order step 2, and
it is deliberately before step 3.

The whole architecture rests on one claim -- that a lattice whose pitch is
``min_track_width + min_clearance`` makes cell occupancy equivalent to a
clearance check, so anything the fast engine accepts is DRC-clean by
construction. If that reasoning is wrong at corners, pad edges, or 45-degree
segments, every number produced by training on this lattice is measuring a
fiction. It is an afternoon's question and it needs no trained policy: route
boards with a non-learned baseline, export them, and hand them to KiCad's own
`DRC_ENGINE`.

    python -m mzr.scripts.validate_kicad --out drc_out

**Run it across several board sizes, always.** NeuroRoute's diagonal
corner-guard bug -- a real 0.0828 mm clearance failure, exactly
``pitch/sqrt(2) - width`` -- was **clean on one board set and failing on
another**. A single passing configuration proves much less than it appears to,
which is why `--configs` sweeps four by default.

Requires `kicad-cli` on PATH (ships with KiCad 7+). On Colab that is
`apt-get install -y kicad`; the full source build the PNS bridge needs is
**not** required here, which is the point -- this check is cheap enough to run
on every change to geometry or design rules.

The two failure modes it separates:

* `kicad-cli` cannot parse the file -> the **exporter** is wrong (most likely
  the layer ordinals; see `eval/kicad_export.py::_layer_ordinal`).
* it parses and reports violations  -> the **lattice model** is wrong, and the
  fix is a wider pitch or a wider dilation rule, not a bug hunt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from mzr.eval.kicad_export import export_board
from mzr.world.baselines import BASELINES, rollout
from mzr.world.engine import SimultaneousRouterWorld, WorldConfig
from mzr.world.generator import GeneratorConfig, generate_board
from mzr.world.spec import BoardSpec, LayerStack, RipupRules


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
        return False, {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
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
#: the routing policy, not about whether the geometry model is sound. A
#: non-learned baseline leaves most nets unrouted, so these are expected and are
#: reported separately rather than counted against the model.
COMPLETENESS_CODES = {"unconnected_items", "track_dangling"}
#: Library/parity bookkeeping. Nothing to do with copper: KiCad is saying it
#: cannot find a footprint library named "mzr", which is true and irrelevant.
BOOKKEEPING_CODES = {"lib_footprint_issues", "footprint", "footprint_type_mismatch"}


def summarise(report: dict) -> dict[tuple[str, str], int]:
    """Violation counts keyed by (error code, KiCad's own severity).

    Severity comes from KiCad rather than being assigned here. `copper_sliver`,
    for instance, is a manufacturability *warning* -- a thin fragment of copper,
    not a short and not a clearance failure -- and treating it as a hard
    legality failure would condemn a model KiCad itself considers fabricable.
    """
    counts: dict[tuple[str, str], int] = {}
    for key in ("violations", "unconnected_items", "schematic_parity"):
        for item in report.get(key, []) or []:
            code = item.get("type") or item.get("code") or key
            sev = item.get("severity", "error")
            counts[(code, sev)] = counts.get((code, sev), 0) + 1
    return counts


def classify(counts: dict[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    """Split a tally into legality / advisory / completeness / bookkeeping.

    Only **errors** about copper legality count against the lattice model.
    Warnings are reported alongside, because a rising sliver count is worth
    knowing about even though it does not refute the model.
    """
    out: dict[str, dict[str, int]] = {
        "legality": {}, "advisory": {}, "completeness": {}, "bookkeeping": {}, "unknown": {}
    }
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
            # other way would let a real violation hide behind a name this list
            # has not seen yet.
            bucket = "unknown"
        label = f"{code}({sev})" if bucket == "advisory" else code
        out[bucket][label] = out[bucket].get(label, 0) + n
    return out


@dataclass
class Config:
    name: str
    boards: int
    nets: int
    layers: int
    size: int
    wide_frac: float
    keepouts: int


#: Four configurations, not one. The diagonal corner-guard bug that produced
#: real 0.0828 mm KiCad violations did **not** reproduce on the first board set;
#: a different board size and net count surfaced it immediately.
DEFAULT_CONFIGS = [
    Config("small-4L", boards=6, nets=24, layers=4, size=64, wide_frac=0.0, keepouts=0),
    Config("mid-8L", boards=4, nets=30, layers=8, size=80, wide_frac=0.30, keepouts=2),
    Config("wide-8L", boards=4, nets=40, layers=8, size=96, wide_frac=0.35, keepouts=2),
    Config("large-8L", boards=3, nets=50, layers=8, size=112, wide_frac=0.50, keepouts=4),
]


def route_config(cfg: Config, policy_name: str, seed0: int) -> SimultaneousRouterWorld:
    spec = BoardSpec(
        height_cells=cfg.size,
        width_cells=cfg.size,
        layers=LayerStack(num_layers=cfg.layers),
    )
    gcfg = GeneratorConfig(
        num_nets=cfg.nets,
        num_components=5,
        pin_pitch_cells=4,
        wide_net_frac=cfg.wide_frac,
        num_keepouts=cfg.keepouts,
    )
    boards = [generate_board(spec, gcfg, seed=seed0 + i) for i in range(cfg.boards)]
    wcfg = WorldConfig(
        batch_size=cfg.boards,
        max_nets=cfg.nets + 8,
        max_macro_steps=96,
        max_steps_per_frontier=96,
        # Rip-up off: this gate is about geometry, and a rip-up round midway
        # would only change *which* nets are routed, not whether their copper
        # is legal. One fewer moving part between a violation and its cause.
        ripup=RipupRules(interval=0),
    )
    world = SimultaneousRouterWorld(spec, wcfg)
    world.load(boards)
    rollout(world, BASELINES[policy_name])
    return world


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="drc_out", help="directory for .kicad_pcb and reports")
    p.add_argument("--policy", default="layer_hop", choices=sorted(BASELINES))
    p.add_argument("--seed", type=int, default=500000)
    p.add_argument("--configs", nargs="*", default=None, help="subset of config names")
    p.add_argument("--keep", action="store_true", help="keep exported files on success")
    args = p.parse_args()

    if shutil.which("kicad-cli") is None:
        print("kicad-cli not found on PATH.")
        print("  Linux/Colab : apt-get install -y kicad")
        print('  Windows     : export PATH="$PATH:/c/Program Files/KiCad/9.0/bin"')
        print("                (APPEND, never prepend -- its bundled python shadows the system one)")
        return 2

    ver = subprocess.run(["kicad-cli", "--version"], capture_output=True, text=True)
    print(f"kicad-cli {ver.stdout.strip() or ver.stderr.strip()}")

    configs = DEFAULT_CONFIGS
    if args.configs:
        wanted = set(args.configs)
        configs = [c for c in configs if c.name in wanted]
        if not configs:
            print(f"no configs matched {sorted(wanted)}")
            return 2

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    totals = {"legality": 0, "advisory": 0, "completeness": 0, "bookkeeping": 0, "unknown": 0}
    total_nets = total_tracks = total_vias = total_stubs = 0
    unparsable: list[str] = []
    per_config: list[tuple[str, int, int, float]] = []

    for cfg in configs:
        print(f"\n=== {cfg.name}: {cfg.boards} boards, {cfg.nets} nets, "
              f"{cfg.layers}L, {cfg.size}x{cfg.size}, {cfg.wide_frac:.0%} wide ===")
        world = route_config(cfg, args.policy, args.seed)
        completion = float(world.completion().mean())
        cfg_dir = out_root / cfg.name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        cfg_legality = 0
        routed_here = 0
        for b in range(cfg.boards):
            pcb = cfg_dir / f"board_{b:03d}.kicad_pcb"
            rep = cfg_dir / f"board_{b:03d}.drc.json"
            stats = export_board(world, b, pcb)
            total_tracks += stats.tracks
            total_vias += stats.vias
            total_stubs += stats.stub_legs

            n_done = int(
                (world.net_valid[b] & (world.net_status[b] == 1)).sum()
            )
            routed_here += n_done
            total_nets += n_done

            ok, report = run_drc(pcb, rep)
            if not ok:
                unparsable.append(f"{cfg.name}/board_{b:03d}")
                print(f"  board {b}: KICAD COULD NOT PARSE -- {report}")
                continue
            buckets = classify(summarise(report))
            for k, v in buckets.items():
                totals[k] += sum(v.values())
            cfg_legality += sum(buckets["legality"].values()) + sum(buckets["unknown"].values())
            detail = ", ".join(
                f"{k}={sum(v.values())}" for k, v in buckets.items() if v
            )
            print(f"  board {b}: {n_done} nets routed, {stats.tracks} tracks, "
                  f"{stats.vias} vias | {detail or 'clean'}")

        per_config.append((cfg.name, routed_here, cfg_legality, completion))

    print("\n" + "=" * 66)
    print(f"{'config':<12} {'nets routed':>12} {'completion':>11} {'legality viol.':>15}")
    for name, routed, viol, comp in per_config:
        print(f"{name:<12} {routed:>12} {comp:>10.1%} {viol:>15}")
    print("=" * 66)
    print(f"{'TOTAL':<12} {total_nets:>12} {'':>11} "
          f"{totals['legality'] + totals['unknown']:>15}")
    print(f"\ntracks {total_tracks}, vias {total_vias}, stub legs {total_stubs}")
    print(f"advisory (warnings)  : {totals['advisory']}")
    print(f"completeness         : {totals['completeness']}  "
          f"(unrouted nets + stubs -- a routing result, not a geometry one)")
    print(f"bookkeeping          : {totals['bookkeeping']}  (missing footprint library)")

    if unparsable:
        print(f"\nFAIL: kicad-cli could not parse {len(unparsable)} board(s): "
              f"{', '.join(unparsable[:5])}")
        print("      That is an EXPORTER bug, not a lattice bug -- start at "
              "eval/kicad_export.py::_layer_ordinal")
        return 1

    hard = totals["legality"] + totals["unknown"]
    if hard:
        print(f"\nFAIL: {hard} legality violations over {total_nets} routed nets.")
        print("      The LATTICE MODEL is wrong. Fix by widening the pitch or the")
        print("      dilation rule -- do not chase individual violations.")
        return 1

    print(f"\nPASS: sim-to-real gap is 0 legality violations over {total_nets} "
          f"routed nets, {len(configs)} configs.")
    if not args.keep:
        shutil.rmtree(out_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
