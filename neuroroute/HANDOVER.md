# NeuroRoute — session handover

Written 2026-08-26, at the end of the session that built `neuroroute/` and ran
stage 0 twice on Colab. Everything here is committed and pushed to `main`.

**Read order for a fresh session:** this file → `neuroroute/DESIGN.md` (the
architecture and why alternatives were rejected) → `neuroroute/README.md`
(what is verified, with numbers) → `AGENTS.md` (the Claude/Antigravity split,
still governs).

---

## What this is

A pure-RL PCB router for **thousands of nets, 6–8 layers, learned differential
pairs, learned length tuning**, variable widths and via sizes. No LLM. No KiCad
push-and-shove solver doing the hard part (the user rejects that as cheating —
see memory `feedback_no_pns_routing_primitives`).

It is a **third thread**, not an extension of either existing one:

- `pcbworld/env/line_route_env.py` (PNS bridge) — cannot do 6–8 layers.
  `switch_layer()` is **0-for-32**. Its only working diff-pair/tune path is the
  engine's own solvers, which are ruled out.
- `pcbworld/environment.py` (raster) — works (99.90% single-net) but is
  single-head, single-net, 2 layers, fixed width, one board per process.

`docs/UNIFIED_RL_DESIGN.md` is **superseded** by `neuroroute/DESIGN.md`.

---

## Current state — verified

| Claim | Status |
|---|---|
| Lattice is DRC-legal | **[LIVE]** 0 legality violations / 192 routed nets, 4 configs, KiCad 9.0.2 locally and 8.0.9 on Colab |
| Geometry exact | **[LOCAL]** all lattice ops match a brute-force reference |
| Connectivity | **[LOCAL]** every "done" net verified by flood fill, including through vias |
| Multi-layer routing | **[LIVE]** 1 layer 27.3% → 8 layers 32.8%, vias placed and connected |
| Refine phase (length tuning) | **[LOCAL]** drags change length, preserve connectivity, restore exactly on reject |
| Untrained ≈ greedy | **[LIVE]** 28.3% vs 29.2%, both 0.03% rejected |
| Preflight | **[LIVE]** 7/7 PASS on Colab T4 |
| **Stage 0 trained** | **[LIVE]** see below — met the gate once, on the *old* code |

### Baselines (64 held-out boards, seeds 900000+)

| Board | greedy | detour | layer_hop |
|---|---|---|---|
| 1 net, empty, 2L (stage 0) | 73.4% | 73.4% | **95.3%** |
| 20 nets, 2L (stage 1) | 28.1% | 28.1% | **42.5%** |
| 60 nets, 8L (stage 3) | 15.8% | 15.8% | **26.5%** |

`layer_hop` = greedy + "place a via when another layer is closer". The gap
between greedy and layer_hop is entirely the via action.

---

## Training runs so far (both stage 0 only)

**Run A — commit `0a54ae9` (pre-fix), 16-board eval.** Reached **100.0% at
u275**, met the 98% gate. But 16 boards where `layer_hop` also scored 100% —
an easier, coarser set. Vias 0.0 → 0.2, arriving only at u150.

**Run B — commit `deaf9b6` (current), 64-board eval.** Best **90.6% at u150**,
ended 84.4%, never met the gate. Vias appear at **u25** (was u150), and the
policy beats greedy at **10/11 evals** (was 3/11).

Not directly comparable: run B's eval set is harder (`layer_hop` 95.3% vs
100%) and 4× better resolved.

---

## THE KEY UNRESOLVED FINDING — read this first

**During training the policy completes ~100% of nets. At eval it completes
73–90%.** From `train_log.jsonl`, run B: `completion` is 0.9375–1.0 on almost
every one of 300 updates, while every eval is 73.4–90.6%.

Training rolls out with **sampling**; eval uses **`deterministic=True`
(argmax)**. Same boards distribution, same policy.

**So the policy's mode is worse than its average.** The most likely mechanism,
and it is consistent with everything else observed:

- 4/16 (25%) of stage-0 boards have their two pads on **different layers** and
  are unroutable without a via. A via-less policy caps at exactly 75.0% —
  which is precisely where run A's eval sat for three consecutive evals.
- Sampled rollouts place vias (training log shows **0.125–2.875 vias/board**)
  and therefore finish those nets.
- Argmax rarely does (eval shows **0.0–0.2 vias/board**).

**The policy has learned that vias sometimes help, but not confidently enough
for the argmax to pick one.** It relies on stochasticity to stumble into the
via.

This was NOT diagnosed before the session ended. It is the single highest-value
thing to resolve.

### The immediately decisive experiment

Evaluate **both** deterministic and sampled on the same held-out boards and
report both. If sampled ≈ 100% and argmax ≈ 75%, the above is confirmed and
the problem is a mode/mean gap, not a capability gap. `evaluate()` in
`neuroroute/training/run.py` currently only runs `deterministic=True`.

---

## Other confirmed findings from run B

### The "rejected-action rate exploded" alarm is largely a metric artifact

`rejected_action_rate = rejected.sum() / active.sum()`. Stage 0 has **one net
per board**, so late in an episode almost every head is idle and the
denominator collapses to a handful. One stuck head then reads as 93.8%.

Evidence: the 93.8% / 96.9% spikes (u242, u263, u285, u287) all coincide with
`completion` 0.875–0.9375 and `reward ≈ 0` — i.e. one board's head flailing
while the other 15 are finished and idle.

Measured directly: an **untrained** policy on the same boards rejects only
**0.8%** of actions, and the only meaningful source is width>0 moves at ~3%.
There is no structural mask flaw of the size the alarm suggested.

**Do not chase this as a bug.** Fix the metric (denominator should be heads
that were active *and* attempted a move) before drawing conclusions.

### The forecaster works — and the gate is measuring the wrong thing

| | run A (old) | run B (new) |
|---|---|---|
| forecast correlation | +0.01 (noise) | **+0.57 → +0.90** |
| baseline correlation | +0.514 | +0.26 → +0.46 |
| policy corr beats baseline | 0/11 | **11/11** |
| gate verdict (MAE-based) | fails 11/11 | fails 9/11 |

After four prior negative results in this repo (`jepa/`,
`models/fast_lookahead.py`), the "look into the future" module is
discriminating spatial structure for the first time.

**Likely cause, unintended:** the chunked PPO update computes the forecast loss
over the whole chunk. The old code computed it on **one timestep per
minibatch** (`observation_at(idx[0])`), so the supervised head now gets ~8× the
gradient per update.

**The gate uses MAE, which on a sparse field rewards predicting the mean.**
By MAE the forecaster "loses"; by correlation it wins 11/11 (0.77 vs 0.36
average). `forecast_gate()` in `neuroroute/models/forecaster.py` should score
on correlation. As written, the gate can reject a module that works.

Also: the gate is computed on the **training** env state, not held-out. Should
move to eval boards.

### Detour ratio is high and not improving

Completed routes are **50–160% longer than straight-line** even on a
one-net empty board. Optimal 8-connected paths should be ~5–10% over Euclidean.
Something in the steering is mushy. Unexplained.

### Clip fraction is chronically high

0.2–0.5 typical, spiking to 0.65. Updates sit outside the PPO trust region far
too often. **Unaddressed** — lower LR and/or fewer epochs is the standard fix.

### GPU is still ~83% idle

2.7 / 15.6 GB on a T4. Chunking moved the update phase 68% → 54% of wall time,
which is real but modest. Both runs used `--batch 16`; the notebook recommends
`--batch 32 --ppo-chunk 8` and that has **never been tested**.

---

## Ranked next actions

1. **Run the deterministic-vs-sampled eval.** One change to `evaluate()`,
   decisive for the central open question. Do this before anything else.
2. **Fix the forecast gate to score on correlation**, and compute it on
   held-out boards. Justified purely by data already collected.
3. **Fix `rejected_action_rate`'s denominator** so the alarm stops firing on
   an artifact.
4. **Address convergence**: lower LR (clip fraction says updates are too big),
   and either run 600–900 updates or shape the entropy schedule to actually
   decay late.
5. **Then** move to stage 1 (20 nets, 2 layers) — the first stage where
   congestion, the scheduler and the forecaster are genuinely tested. Stage 0
   is a plumbing check and has largely served its purpose.
6. Diagnose the detour ratio.

Things designed but **not built**: a rip-up action head (the engine's
`world.ripup()` works and is tested, but the policy never emits one), and a
policy head for the refine phase (the drag action works and is verified, but
nothing chooses drags).

---

## Gotchas that will cost time if re-derived

- **`Observation.head_pos` must be a clone, not a view.** `world.head_pos` is
  mutated in place by `step()`; an aliasing observation made
  `policy.evaluate()` disagree with `policy.act()` and silently broke the PPO
  ratio.
- **Raycast is in absolute directions; everything else is egocentric.** The
  observation rotates it once. When the frames drifted, the rejected-action
  rate was **86%** while the baseline believed it was only taking safe moves.
  After the fix: 1.6%.
- **Every copper-writing path must apply the diagonal corner guards.** There
  are two (`_move_cells` and `_segment_cells`); they disagreed and produced
  real 0.0828 mm KiCad clearance violations — **clean on one board size,
  failing on another**. Always run `validate_kicad.py` across *several* board
  sizes.
- **Pad size and lattice reservation must derive from one number**
  (`DesignRules.pad_size`). They drifted and produced 0.100 mm pad-to-track
  violations.
- **An action head's `bias` decides its untrained default.** With `gain=0.01`
  weights the bias is the whole signal. A zero-bias width head picked 3-cell
  traces on 612/627 actions (88% rejected).
- **`h_layer.bias` must scale with layer count** (`log(3L)` gives P(stay)=0.75
  for any L). A fixed 4.0 put P(stay) at 0.97 on 2 layers and meant "never
  place a via" under argmax.
- **A board with no nets must score 100%, not 0%.** The generator used to give
  up quietly; scoring those 0% hid a generator bug inside a training curve.
- **`snap_radius >= max(STEP_LENGTHS)/2`** — enforced in `WorldConfig`.
- **Two heads could write the same cell** in one batched step. `step()` is now
  plan → arbitrate → commit. Do not reintroduce per-head check-and-write.
- **Colab's `kicad-cli` is 8.0.9, local is 9.0.2.** The exporter works on both.

---

## Files

```
neuroroute/
  DESIGN.md              architecture; why PNS and raster threads can't get there
  README.md              verified status, with numbers
  HANDOVER.md            this file
  ANTIGRAVITY_PROMPT.md  paste-into-Antigravity brief for Colab runs
  world/spec.py          design rules, lattice pitch, action space constants
  world/geometry.py      batched legality / stamping / raycast / geodesic
  world/generator.py     procedural boards (components with pin arrays)
  world/engine.py        BatchedRouterWorld: plan -> arbitrate -> commit; refine()
  env/observation.py     field / head / net tensors, exact local geometry
  env/rewards.py         potential shaping + pair and length terms
  env/route_env.py       the batched RL env
  env/baselines.py       greedy / detour / layer_hop
  models/encoder.py      3D field encoder, head crop, net transformer
  models/forecaster.py   FutureFieldPredictor + the gate
  models/policy.py       factorised heads, safety suppression, scheduler
  training/ppo.py        chunked PPO (stack_observations), optional AMP
  training/telemetry.py  JSONL + health checks + crash reports
  training/curriculum.py stages; forecaster targets
  training/run.py        CLI entry
  eval/kicad_export.py   lattice -> .kicad_pcb (no pcbnew import needed)
  eval/render.py         board renders with failed nets dashed red
  scripts/               preflight, verify_geometry, verify_env, verify_refine,
                         validate_kicad
```

## Running it

```bash
python -m neuroroute.scripts.preflight
```

```bash
python -m neuroroute.training.run --stage 1 --device cuda --batch 32 --heads 8 --width 48 --rollout 32 --ppo-chunk 8 --updates 1500 --eval-every 50 --eval-boards 64 --checkpoint-dir CKPT/stage1 --resume
```

Local KiCad for the DRC gate (append, never prepend — its bundled python
shadows the system one):

```bash
export PATH="$PATH:/c/Program Files/KiCad/9.0/bin"
```

## Operating model

`AGENTS.md` governs. Claude Code owns the logic and commits; **Antigravity runs
Colab and reports real output verbatim, and does not edit tracked source**.
`neuroroute/ANTIGRAVITY_PROMPT.md` is the brief to paste in — it is current
except that it does not yet mention the deterministic-vs-sampled question.
