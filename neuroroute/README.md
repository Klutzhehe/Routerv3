# NeuroRoute

A pure-RL PCB router for **many nets, many layers**, with **learned**
differential pairs and **learned** length tuning. No LLM anywhere. No KiCad
push-and-shove solver called to do the hard part.

> Design rationale, and why the two existing threads in this repo cannot reach
> this target, is in [`DESIGN.md`](DESIGN.md). Read that first.

---

## The short version

| | RouterV3's PNS thread | RouterV3's raster thread | NeuroRoute |
|---|---|---|---|
| Layers | 1 (`switch_layer()` **0-for-32**) | 2, via action never worked | **8, working, DRC-verified** |
| Environment | 1 board/process, CPU, `nproc`=2 | 1 board/process, Python loop | **B boards x K heads, batched tensors** |
| Diff pairs | `MODE_ROUTE_DIFF_PAIR` (engine solver) | none | **learned `couple` action** |
| Length tuning | `MODE_TUNE_SINGLE` (engine solver) | none | **learned vertex-drag refine phase** |
| Track width / via size | fixed | fixed | **learned, from the rule table** |
| Lookahead | 4 negative results | 4 negative results | **dense spatial forecast fields** |

---

## Status

Everything below was run. Tags follow the repo convention
(`docs/ROUTER_CAPABILITIES.md`): **[LIVE]** = measured with numbers,
**[LOCAL]** = verified by a check script, **[UNVERIFIED]** = written, never run.

### The load-bearing claim is verified against real KiCad — **[LIVE]**

The architecture rests on one idea: a lattice whose pitch is
`min_track_width + min_clearance` makes *cell occupancy* equivalent to a
*clearance check*, so anything the fast engine accepts should be DRC-clean by
construction. Checked against **KiCad 9.0.2's own `DRC_ENGINE`**, via
`kicad-cli pcb drc`, on boards routed by the non-learned baseline:

```
config                                        routed nets   legality violations
4 boards, 30 nets, 8 layers,  80x80, 30% wide       34              0
6 boards, 40 nets, 8 layers,  96x96, 35% wide       57              0
6 boards, 24 nets, 4 layers,  64x64,  0% wide       61              0
5 boards, 50 nets, 8 layers, 112x112, 50% wide      40              0
                                            TOTAL  192              0
```

Four configurations, deliberately: the diagonal bug below was **clean on one
board set and failing on another**, so a single passing run proves less than it
appears to. Reported separately and not counted against the model: 2
`copper_sliver` (KiCad's own severity is *warning* -- a thin copper fragment,
not a short) and `unconnected_items` for nets the baseline never routed, which
is a routing result rather than a geometry one.

Getting there found **three** real bugs, all of which only a real DRC run
would have caught:

* **Pad-to-track clearance, actual 0.100 mm against a 0.200 mm rule.** The
  lattice reserved *one cell* for a pad while the exporter drew it a full
  0.4 mm wide. Fixed by deriving both from `DesignRules.pad_size`, so they
  cannot drift apart again.
* **`track_dangling` x478.** The exporter was emitting the stub routes of nets
  that never completed. Now only routed nets get copper; unrouted ones show up
  as `unconnected_items`, which is the honest way for a board to be incomplete.
* **Diagonal segments missing their corner guards, actual 0.0828 mm.** A
  45-degree trace passes *between* lattice cells, so any cell at perpendicular
  distance `1/sqrt(2)` is only `0.4/sqrt(2) - 0.2 = 0.083 mm` from its copper.
  `check_moves`/`stamp_moves` block those two cells at every step of a diagonal
  move -- but `check_segments`/`stamp_segments`, used by the **snap-to-pad**
  and by refine drags, did not. Another net legally occupied a guard cell and
  KiCad reported the pair at exactly the predicted 0.0828 mm. The two paths now
  share the rule.

  This one is the argument for running the gate rather than reasoning about it:
  it did **not** reproduce on the first board set. One configuration was clean;
  a different board size and net count surfaced it immediately.

### Geometry is exact — **[LOCAL]**

`verify_geometry.py` checks every lattice operation against an independently
written brute-force reference: all 128 (direction, step, width) move footprints,
1920 randomised legality checks, per-(direction, step) safety against
`check_moves`, stamp/erase round-trips, geodesic fields against hand-computed
distances, and via layer spans. **All exact.**

### The environment holds its invariants — **[LOCAL]**

`verify_env.py` re-derives the answers from the occupancy grid rather than
trusting status flags: no pad is ever overwritten, and **every net marked
"done" is connected end-to-end by flood fill** — including routes that change
layer through a via.

That check caught the worst bug in the build. With `K` heads acting in one
batched step, two heads both passed a legality test against the *pre-step*
occupancy and both wrote the same cell; one silently lost, its head advanced
anyway, and its route was left with a hole in it. `step()` is now
**plan -> arbitrate -> commit**, which makes that impossible rather than
unlikely.

### The untrained policy really does start at the greedy baseline — **[LIVE]**

The egocentric action frame plus near-zero actor init is supposed to mean an
untrained policy *is* the greedy router, so training starts at the baseline
rather than below it. Measured, on 8 layers:

| | completion | rejected actions |
|---|---|---|
| greedy (step=1) | 29.2% | 0.03% |
| **policy, deterministic, untrained** | **28.3%** | **0.03%** |

It did not hold on the first try, and both failures were the same shape as the
frame bug above -- a default that nobody had actually chosen:

* **`h_width.bias` was all zeros**, so near-zero weights decided the argmax
  arbitrarily and the untrained policy picked width class 1 (0.3 mm, *three*
  lattice cells) on **612 of 627 actions**. On a congested board almost
  everything collided: 88% rejected, 6.7% completion. Index 0 now means "the
  width this net requires", and widening is a decision rather than a default.
* **The layer head had no safety mask.** A through via must be free on all 8
  layers at once, so most are impossible on a populated board, and an untrained
  policy attempted one roughly half the time: **92.6% rejected actions**.
  Extending the same fixed suppression used for direction and step to the layer
  head, plus a stronger stay bias, brought that to 19.2% while still placing
  vias.

### Length tuning works mechanically — **[LOCAL]**

`verify_refine.py` exercises the refine-phase vertex drag on real routed
boards: drags are accepted, they change routed length by a usable amount
(up to 12.6 cells over 24 rounds), routes stay connected through them
(flood-fill verified), and a rejected drag restores the board **byte-identical**
-- which matters because the action erases copper *before* testing the new
geometry, so a bad restore would silently delete a working route.

No meander generator is involved. A meander is what alternating drags of
adjacent vertices look like, and that is expressible purely as a sequence of
ordinary actions.

### Baselines, on held-out seeds — **[LIVE]**

These are the numbers a trained policy has to beat.

| Board | greedy | detour | layer_hop (vias/board) |
|---|---|---|---|
| 1 net, empty, 2 layers | 75.0% | 75.0% | **87.5%** (0.2) |
| 20 nets, 2 layers | 28.1% | 28.1% | **42.5%** (8.4) |
| 60 nets, **8 layers** | 16.3% | 16.3% | **24.6%** (17.0) |

Measured *after* the diagonal corner-guard fix, which made every segment
strictly more conservative and cost a few points across the board — the
earlier, higher numbers were partly the illegal geometry the fix removed.

Multi-layer routing works end to end: **1 layer 27.3% -> 8 layers 32.8%**,
with connectivity verified *through* the vias. This is the capability KiCad's
PNS router is 0-for-32 on after three sessions of trying.

One honest caveat: the stage-0 gate in the curriculum is 98%, and the best
non-learned baseline reaches **87.5%** on that board. Stage 0 is one net on an
empty board, so the remaining 12.5% is not congestion — it is cross-layer nets
where the baseline's fixed via rule does not fire usefully. A trained policy
should clear it easily; if it does not, that is the plumbing signal stage 0
exists to give.

### Not yet run — **[UNVERIFIED]**

Everything about *learning*. The policy forward/backward pass, PPO update, and
checkpointing all execute, and the forecaster gate correctly reports "does NOT
beat baseline" on an untrained model — but no real training run has happened.
That is what Colab is for.

---

## Running it

### 1. The sim-to-real gate — do this first, on any change to the geometry

Needs `kicad-cli` (ships with KiCad 7+; on Colab `apt-get install -y kicad`).
It does **not** need the compiled PNS bridge.

```bash
python -m neuroroute.scripts.validate_kicad --boards 6 --nets 40 --layers 8 --wide-frac 0.35
```

### 2. Local verification — no GPU, no KiCad

```bash
python -m neuroroute.scripts.verify_geometry
```

```bash
python -m neuroroute.scripts.verify_env
```

```bash
python -m neuroroute.scripts.verify_refine
```

### 3. Training

```bash
python -m neuroroute.training.run --stage 0 --updates 100 --batch 8 --heads 4
```

Stage 0 is one net on an empty board. It should reach ~100%; anything less is
broken plumbing, not a hard problem. Then walk the curriculum:

```bash
python -m neuroroute.training.run --stage 3 --batch 16 --heads 8 --width 64 --updates 2000 --resume
```

`--stage 3` is the first 8-layer stage. Stages are in
[`training/curriculum.py`](training/curriculum.py); each one changes **only the
data**, never the model, which is what makes the jump to thousands of nets a
generalisation test rather than a rewrite.

---

## Layout

```
neuroroute/
  DESIGN.md            the architecture, and why the alternatives were rejected
  world/
    spec.py            design rules, the lattice pitch, action space constants
    geometry.py        batched legality / stamping / raycast / geodesic fields
    generator.py       procedural boards: components with pin arrays, pairs, groups
    engine.py          BatchedRouterWorld -- plan -> arbitrate -> commit
  env/
    observation.py     field / head / net tensors; exact local geometry
    rewards.py         potential-based shaping + the pair and length terms
    route_env.py       the batched RL environment
    baselines.py       greedy / detour / layer_hop -- the numbers to beat
  models/
    encoder.py         3-D field encoder, head crop, net transformer
    forecaster.py      FutureFieldPredictor + the gate that can kill it
    policy.py          factorised action heads, safety suppression, scheduler
  training/
    ppo.py             PPO with uint8-quantised observation storage
    curriculum.py      stages; forecaster supervision targets
    run.py             CLI entry point
  eval/
    kicad_export.py    routed lattice -> .kicad_pcb, no pcbnew import needed
  scripts/
    verify_geometry.py verify_env.py verify_refine.py validate_kicad.py
```

---

## Things that will cost time if re-derived

* **`Observation.head_pos` must be a clone, not a view.** `world.head_pos` is
  mutated in place by `step()`. An aliasing observation describes the world
  *after* the action it was used to choose, which made `policy.evaluate()`
  disagree with `policy.act()` on identical inputs — a permanently broken PPO
  importance ratio that would never have surfaced as a crash.
* **The raycast is in absolute directions; everything else is egocentric.** The
  observation rotates it once. When those two frames drifted apart, the
  rejected-action rate was **86%** while the baseline believed it was only
  taking moves the raycast had called safe. After the fix: **1.6%**.
* **The geodesic field is stored at 1/4 resolution and must be sampled
  bilinearly.** Nearest-neighbour indexing makes it piecewise-constant, so the
  gradient between adjacent lattice cells is exactly zero and the descent
  direction is arbitrary.
* **A board with no nets scores 100%, not 0%.** The generator used to give up
  quietly when component placement ran short of pins; scoring those boards 0%
  let a generator bug hide inside a training curve.
* **`layer_hop`'s via threshold must be small.** The geodesic already charges
  `via_cost` for a layer change, so a negative `geo_layer` *already* means
  "worth it after paying for the via". A threshold of 0.15 sat just above the
  4-cell via cost (4/32 = 0.125) and suppressed every legitimate hop — vias
  went to ~0 and 8-layer boards scored identically to single-layer ones.
* **Every path that writes copper must apply the diagonal corner guards.**
  There are two: the lattice-move path (`_move_cells`) and the arbitrary-segment
  path (`_segment_cells`, used by snap-to-pad and refine). They disagreed, and
  the result was a real 0.0828 mm clearance violation that only appeared on
  some board sizes.
* **An action head's `bias` is the thing that decides its untrained default.**
  With `gain=0.01` weights the bias is the whole signal, so a head left at
  zero bias picks arbitrarily -- and "arbitrarily" meant 3-cell-wide traces
  everywhere. Every head that has a sensible default now states it.
* **`snap_radius >= max(STEP_LENGTHS)/2`**, enforced in `WorldConfig`. A longer
  step jumps clean over the snap zone and the head orbits its target forever —
  a config bug that reads exactly like a learning failure. Inherited from
  `docs/HANDOVER.md`; the assertion is there so it cannot recur.
