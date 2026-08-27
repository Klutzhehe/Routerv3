# MZR

A PCB router where **every net grows at once**. No net-by-net ordering, no
scheduler, no push-and-shove. Nets negotiate for room through a congestion
price, and (later) a learned latent model lets the router imagine how the board
plays out before committing copper.

> Architecture and the reasoning behind it: [`DESIGN.md`](DESIGN.md). Read that
> first — including §14, which states honest per-gate confidence rather than
> claiming this will work.

---

## The short version

| | NeuroRoute | MZR |
|---|---|---|
| Net coordination | `K=8` head slots + a scheduler that **never trained** | every net live from step 0 |
| Growth | one end per leg | **both pads inward**, meeting in the middle |
| Ordering | scheduled (badly) | **emergent** from a PathFinder congestion price |
| Episode length | grows with net count | **~constant** — measured 0.37× steps for 12× nets |
| Rip-up | a rare learned action | scheduled rounds; history persists so retreats mean something |
| Search | none | Sampled/Gumbel MuZero over joint frontier moves *(not built)* |

---

## Status

Tags follow the repo convention (`docs/ROUTER_CAPABILITIES.md`):
**[LIVE]** = measured with numbers, **[LOCAL]** = verified by a check script,
**[UNVERIFIED]** = written, never run.

| Piece | State |
|---|---|
| `world/` — lattice, macro-step, congestion price, rip-up | **built**, [LOCAL] |
| `eval/kicad_export.py` — lattice → `.kicad_pcb` | **built**, [LIVE] |
| `scripts/validate_kicad.py` — the sim-to-real DRC gate | **built**, [LIVE], **passing** |
| `world/baselines.py` — greedy / detour / layer_hop | **built**, [LIVE] |
| Sequential + PathFinder expert (stage-1 bar, BC source) | not built |
| `env/` — observation, reward, RL env | not built |
| `models/` — `h`/`g`/`f` | not built |
| `search/` — Sampled + Gumbel MuZero | not built |

**Nothing about learning exists yet.** No policy, no training, no search.

### The sim-to-real gate passes — **[LIVE]**

Everything rests on one claim: a lattice whose pitch is
`min_track_width + min_clearance` makes cell occupancy equivalent to a
clearance check, so anything the fast engine accepts is DRC-clean by
construction. Checked against **KiCad 9.0.2's own `DRC_ENGINE`** via
`kicad-cli pcb drc`, on boards routed by the non-learned `layer_hop` baseline:

```
config        boards  nets  layers  size      nets routed   legality violations
small-4L         6     24      4    64x64          53               0
mid-8L           4     30      8    80x80          34               0
wide-8L          4     40      8    96x96          30               0
large-8L         3     50      8   112x112         19               0
                                          TOTAL   136               0
```

4420 tracks, 91 vias, up to 50% wide traces. Four configurations deliberately:
NeuroRoute's diagonal corner-guard bug was **clean on one board set and failing
on another**, so a single passing run proves much less than it appears to.

**It did not pass first time, and the failure was informative.** The first run
returned **46 legality violations** — 30 `clearance`, 11 `shorting_items`,
5 `tracks_crossing`. The lattice was **innocent**: the *exporter* was drawing a
board that had never been routed.

Each leg's copper is two polylines, one per frontier. Joining them
unconditionally drew a phantom segment straight across the board whenever a leg
finished by reaching the far *pad* rather than by the two frontiers meeting —
connecting the far pad to the far end of the other half's stub, backed by no
copper at all. `_leg_runs` now tests contiguity instead of assuming it, and
emits two runs when the halves never touched.

This is the argument for running the gate rather than reasoning about it: the
error was in the direction that would have **condemned a correct geometry
model**.

### Local verification — **[LOCAL]**

`scripts/verify_world.py`, 21 checks, all passing. It re-derives answers from
the occupancy grid rather than trusting the engine's status flags:

* **every net marked done is connected end-to-end by flood fill**, including
  through vias — the check that caught NeuroRoute's simultaneous-write bug
* no pad is ever overwritten
* every polyline vertex sits on its own net's copper
* every exported run is contiguous, and every segment is backed by that net's
  copper
* a rejected frontier does not advance, and copper is never reassigned
* identical rollouts produce byte-identical boards
* contention fires on dense boards and is silent on one-net boards
* rip-up frees copper, restores pads, and **preserves historical price**

### Baselines to beat, held-out seeds — **[LIVE]**

| Board | greedy | detour | layer_hop |
|---|---|---|---|
| 8 nets, 4 layers, 64×64 | 26.6% | 20.3% | **64.1%** |
| 4 nets, 1 layer, 48×48 | 68.8% | — | — |

The greedy → layer_hop gap is entirely the via action: greedy never places one,
so it *cannot* finish a net whose pads sit on different layers.

`detour` (longest step) scoring **below** greedy is worth noting — long steps
overshoot and collide more on a congested board. NeuroRoute measured these two
as equal.

### The horizon claim holds — **[LOCAL]**

The load-bearing argument for MuZero being viable here (`DESIGN.md` §1) is that
simultaneous growth makes episode length independent of net count. Measured as
macro-steps to reach half of each config's eventual completion:

```
 2 nets -> 19 macro-steps
 8 nets -> 14
24 nets ->  7          12x the nets, 0.37x the steps
```

A sequential router would be ~12×. This is the number the whole design rests
on, and it is now measured rather than argued.

---

## Running it

### 1. The sim-to-real gate — do this on any change to geometry or design rules

```bash
python -m mzr.scripts.validate_kicad --out drc_out
```

Needs `kicad-cli` (ships with KiCad 7+; on Colab `apt-get install -y kicad`).
It does **not** need the compiled PNS bridge. Locally, append KiCad to PATH —
never prepend, its bundled python shadows the system one:

```bash
export PATH="$PATH:/c/Program Files/KiCad/9.0/bin"
```

### 2. Local verification — no GPU, no KiCad

```bash
python -m mzr.scripts.verify_world
```

---

## Layout

```
mzr/
  DESIGN.md              architecture, risks, honest confidence, references
  world/
    spec.py              design rules, lattice pitch, price/rip-up rules
    geometry.py          batched legality / stamping / raycast / geodesic  [ported]
    generator.py         procedural boards                                 [ported]
    price.py             PathFinder congestion price
    engine.py            SimultaneousRouterWorld: plan -> arbitrate -> commit
    baselines.py         greedy / detour / layer_hop; the BC expert lives here
  eval/
    kicad_export.py      routed lattice -> .kicad_pcb, no pcbnew import needed
  scripts/
    verify_world.py      local invariants, no GPU/KiCad
    validate_kicad.py    the sim-to-real DRC gate
```

---

## Things that will cost time if re-derived

Inherited from NeuroRoute, plus what this build added:

* **A leg's two frontier polylines are not always contiguous.** Test it, never
  assume it — see the gate result above.
* **`geometry.segment_claims` is not a record of stamped copper.** It is a
  deliberately conservative footprint for *checking* legality: it samples the
  line, adds diagonal corner guards, and dilates, so a one-cell diagonal at
  minimum width yields 27 cells — including cells *behind* the start that no
  move ever stamps. Using it to verify what was written reported 67 false
  violations on copper KiCad had already passed as clean.
* **A phantom join usually surfaces as a `via`, not a segment.** The stub tip
  and the far pad are typically on different layers — that is *why* the leg
  ended by pad-snap instead of meeting — so any export check that skips layer
  changes will wave the bug straight through. Ask for contiguity.
* **Every path that writes copper must apply the diagonal corner guards.** A
  45° trace passes *between* cells; a cell at perpendicular distance `1/√2` is
  only `0.4/√2 − 0.2 = 0.083 mm` from its copper. Two paths existed and
  disagreed → real 0.0828 mm KiCad violations, on some board sizes but not
  others.
* **Pad size and lattice reservation must derive from one number**
  (`DesignRules.pad_size`) → they drifted and produced 0.100 mm pad-to-track.
* **Never export unrouted nets' stub copper** → 478 `track_dangling`.
* **Two frontiers could write the same cell** in one batched step, one silently
  losing while its frontier advanced anyway. `step()` must be
  **plan → arbitrate → commit**.
* **Raycast is absolute; everything else is egocentric.** When the frames
  drifted, rejected-action rate was 86% while the baseline believed every move
  was safe.
* **The geodesic field is coarse and must be sampled bilinearly** — nearest
  makes it piecewise-constant, so the gradient between adjacent cells is zero
  and the descent direction is arbitrary.
* **`layer_hop`'s via threshold must be tiny.** The geodesic already charges
  `via_cost`, so a better layer is *already* worth the via. NeuroRoute's 0.15
  sat just above the 4-cell via cost in coarse units (4/32 = 0.125) and
  suppressed every legitimate hop — vias went to ~0 and 8-layer boards scored
  identically to single-layer ones.
* **A board with no nets scores 100%, not 0%** — otherwise a generator bug
  hides inside a training curve.
* **`snap_radius ≥ max(STEP_LENGTHS)/2`**, enforced in `WorldConfig`.
* **Colab's `kicad-cli` is 8.0.9, local is 9.0.2.** The exporter works on both.

And the failure mode this repo keeps hitting, in its own right: **a check that
passes vacuously.** Three of `verify_world.py`'s tests did, and were rewritten
— one saturated at its step cap, one measured board difficulty instead of the
thing it claimed to measure, and one passed the moment any frontier moved. Each
now carries a guard that fails if it stops exercising what it is named for.
