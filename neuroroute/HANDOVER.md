# NeuroRoute — session handover

Written 2026-08-26, updated 2026-08-27 after stage 1's full 1500-update run
(60.8% best held-out completion, gate not met) and after making `evaluate()`
score both argmax and sampled on the same held-out boards to attack the
mode/mean question directly. Everything here is committed and pushed to
`main`.

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

## THE KEY UNRESOLVED FINDING — confirmed at init, not yet measured trained

**During training the policy completes ~100% of nets. At eval it completes
73–90% (stage 0) / capped at 60.8% (stage 1).** Training rolls out with
**sampling**; eval used only **`deterministic=True`** (argmax). Same board
distribution, same policy — so the gap is either the argmax mode being worse
than the sampled mean, or the eval boards being harder. Those were
unattributable with one number.

**`evaluate()` in `neuroroute/training/run.py` now scores both arms on the
same held-out seeds** (`policy/completion` = argmax, `policy_sampled/completion`
= sampled, `policy/sample_minus_argmax` = the gap). Not yet run on the trained
stage-1 checkpoint — that needs a GPU pass, queued for Antigravity via
`ANTIGRAVITY_PROMPT.md`. What ran locally (CPU, this session) was a **decisive
measurement on the untrained policy**, to have a reference the trained gap can
be read against rather than against zero:

| stage | boards | argmax completion | sampled completion | argmax vias | sampled vias |
|---|---|---|---|---|---|
| 0 (1 net, 2L) | 16 | **75.00%** | **93.75%** | 0.00 | 2.88 |
| 1 (20 nets, 2L) | 16 | **26.56%** | **37.19%** | 0.00 | 42.88 |

**Confirmed: the mode/mean gap is real and partly present before any
training.** Stage 0's untrained argmax number (75.00%) is exactly the plateau
run A's *trained* eval sat at for three consecutive evals — so at minimum,
part of that plateau could have been the untrained baseline showing through
rather than something training did. The mechanism, measured directly on stage
1 (`measure_failure_mode.py`, not committed — reproduce with the method
below): at init, argmax takes **direction 0 on 100% of steps** (straight down
the geodesic gradient), places **zero vias**, and cycles — 23,455 head-steps
over only 5,146 distinct cells, a **78.1% revisit rate** (sampled: 48.0%).
Nothing in the observation flags "I have been here before" — the dead-zone
channel keys off *rejection*, and argmax is rejected only 0.05% of the time,
so there is no rejection signal to trigger it. Sampling is the only thing that
breaks the loop and the only thing that proposes a via.

Ruled out as the cause of that cycling: a **stale geodesic field**. Heads'
distance-to-target field is computed once per net assignment and never
refreshed by default (`--geodesic-refresh 0`), so copper laid by other heads
afterward is invisible to it — a plausible culprit. Measured with
`--geodesic-refresh` at 0, 1, and 8: **byte-identical** completion, revisit
rate and head-step counts across all three. Expected in hindsight: near-zero
actor weights (`gain=0.01`) plus a dominant direction-0 bias means an
*untrained* argmax barely reads the field at all regardless of its staleness.
Whether refresh matters once the weights carry real signal (after training) is
still open — queued as a flag-only sweep in `ANTIGRAVITY_PROMPT.md`.

### What is still open

1. **Run the two-arm eval on the actual stage-1 checkpoint** (1500 updates,
   60.8% best argmax). If trained sampled ≈ trained argmax + ~11pt (the
   untrained reference), the gap never closed during training — same
   mechanism, untouched by learning. If it's much larger, training made the
   mode worse than the mean, which is a different, worse finding (relying on
   exploration noise to reach the goal). If trained argmax alone is
   materially above 26.6% with vias > 0, training *did* fix it and stage-1's
   remaining shortfall (60.8% vs the 75% gate) is a separate problem.
2. **The `--geodesic-refresh` sweep, on the trained checkpoint**, now that the
   untrained no-op is explained rather than just observed.

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

## Stage 1 has now run: 1500 updates, gate not met

Reported back from Colab, this session: **best held-out (argmax) completion
60.8% at update 1350**, against greedy 26.1%, detour 26.1%, layer_hop 39.2% —
decisively ahead of every baseline, but short of the 75% stage gate. Forecast
gate `BEATS baseline` on 29/29 evals (100%). This is the trained number the
two-arm eval above needs to be re-run against — it was still argmax-only when
this run happened.

## Ranked next actions

1. **Re-run eval on the stage-1 checkpoint with the new two-arm `evaluate()`**
   (code done, this session — see the finding above). One extra update on
   `--resume` triggers it; queued in `ANTIGRAVITY_PROMPT.md`. This is what
   decides whether stage 1's 60.8% ceiling is a mode/mean gap (fixable by
   changing how eval samples) or a real capability ceiling (needs more
   training or a different fix).
2. **Fix the forecast gate to score on correlation**, and compute it on
   held-out boards. Justified purely by data already collected.
3. **Fix `rejected_action_rate`'s denominator** so the alarm stops firing on
   an artifact.
4. **Address convergence**: lower LR (clip fraction says updates are too big),
   and either run more updates past 1500 or shape the entropy schedule to
   decay later — stage 1's rejected-action rate spiked repeatedly through
   training (0.04% → 4.39% at u550 → back down, never fully settling),
   consistent with clip fraction still being high late in the run.
5. Diagnose the detour ratio.

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
