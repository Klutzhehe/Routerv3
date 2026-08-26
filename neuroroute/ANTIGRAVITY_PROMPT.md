# Prompt for Antigravity — NeuroRoute Colab training

Copy everything below the line into Antigravity.

---

You are running **NeuroRoute** training on Google Colab. Read `AGENTS.md` in the
repo first — it governs this session.

## Your boundary

Your job is **Colab execution and reporting**, not diagnosis and not fixes.

- **Do not edit any tracked source file.** Not even a one-line fix that looks
  obviously correct. If something fails, report it. Fixes belong in a Claude
  Code session that can see the reasoning behind the design being touched;
  several past bugs in this repo came from a plausible-looking local fix that
  missed a project-wide constraint.
- **Report real output, verbatim.** Actual stdout, actual tracebacks, actual
  numbers — not a summary written from memory of what a cell was supposed to do.
  If a run succeeds, report the numbers, not "it worked".
- You may freely change **command-line flags** (`--batch`, `--width`,
  `--heads`, `--updates`, `--ppo-chunk`, `--store-device`) to fit the GPU
  you were given, and you may re-run cells. That is configuration, not
  source. Tuning them is actively wanted — see the GPU section below.

## Setup

```
Repo:     https://github.com/Klutzhehe/Routerv3.git
Notebook: neuroroute/notebooks/neuroroute_colab.ipynb
Branch:   main   (pull latest -- the via-exploration fix and the GPU
                   utilisation work are recent)
```

Set **Runtime → Change runtime type → GPU** before training. Run the notebook
top to bottom. `git pull` before starting — do not run stale code.

You do **not** need the compiled `pcbworld_pns_bridge`. That is a ~40-minute
KiCad-from-source build belonging to a different, older thread. NeuroRoute's
environment is pure PyTorch and its KiCad validation uses the ordinary
`apt-get install -y kicad` package.

## Step 1 — Preflight. This is a gate.

```bash
python -m neuroroute.scripts.preflight --out /content/preflight_out
```

Seven independent checks (~3–6 min): environment, imports, lattice geometry vs
brute force, environment invariants, refine phase, **real KiCad DRC**, and a
real training step forward *and* backward on this GPU.

- **All 7 PASS** → continue to step 2.
- **Any FAIL** → **stop. Do not start a training run.** Report the complete
  preflight output. In particular, if the *KiCad DRC* check reports legality
  violations, every number a training run would produce is measured against
  geometry KiCad rejects, and is therefore worthless.

Expected: `All checks passed. Safe to train.` (verified locally, 7/7 on CPU).

## Step 2 — Baselines

Run the baseline and throughput cells. These are the numbers training has to
beat, and the throughput number tells us whether the GPU is being used.

Report the table and the `routing decisions/sec` figure. Local CPU reference
(yours should be far higher on a GPU): 1 net/2L greedy 75.0% vs layer_hop
87.5%; 20 nets/2L 28.1% vs 42.5%; 60 nets/8L 16.3% vs 24.6%.

## Step 3 — Train, in order. Do not skip stages.

```bash
# Stage 0 — plumbing. Should climb well past the 87.5% non-learned baseline.
python -m neuroroute.training.run --stage 0 --device cuda \
  --batch 32 --heads 4 --width 32 --rollout 32 --ppo-chunk 8 --updates 400 \
  --eval-every 25 --eval-boards 64 --render-every 50 --drc-every 100 \
  --checkpoint-dir $CKPT/stage0 --resume
```

```bash
# Stage 1 — congestion, 20 nets, 2 layers. Gate 75%.
python -m neuroroute.training.run --stage 1 --device cuda \
  --batch 32 --heads 8 --width 48 --rollout 32 --ppo-chunk 8 --updates 1500 \
  --eval-every 50 --eval-boards 64 --render-every 100 --drc-every 200 \
  --checkpoint-dir $CKPT/stage1 --resume
```

```bash
# Stage 3 — eight layers. Gate 85%.
python -m neuroroute.training.run --stage 3 --device cuda \
  --batch 16 --heads 8 --width 64 --rollout 32 --ppo-chunk 4 --updates 4000 \
  --eval-every 100 --eval-boards 64 --render-every 200 --drc-every 400 \
  --store-device cpu --checkpoint-dir $CKPT/stage3 --resume
```

## GPU utilisation — tune this, and report what you settled on

The first run managed **1,247 decisions/sec and 0.9 GB of 15.6 GB** with the
PPO update at 68% of wall time. That has been addressed (`--ppo-chunk` folds
several rollout timesteps into one forward/backward pass), but the right
numbers depend on the GPU you get.

1. Watch the `0.9/15.6GB` field in the progress line. Below ~40% is wasted.
2. Raise `--batch` first — it speeds up both rollout and update.
3. Then raise `--ppo-chunk`.
4. **On OOM, lower `--ppo-chunk` first.** It trades memory for utilisation
   linearly and changes nothing about the maths (verified: identical
   log-probs).
5. `--store-device cuda` avoids shuttling rollout observations over PCIe.
   Try it on stages 0 and 1; it probably will not fit on stage 3.
6. `--amp` is fp16 autocast and is **UNVERIFIED** — no GPU was available to
   test it. If you try it, watch for `[FATAL] non-finite`. Report whether it
   worked; that is a genuinely useful data point.

Report the final flags you used and the resulting `decisions/sec` and memory
figure.

**Stage 0 is a plumbing check.** One net on an empty board. If it does not
climb well above the 87.5% non-learned baseline, something is wrong with the
setup, not with the idea — report it and stop rather than moving to stage 1 on
a partial result.

Checkpoint to Drive so a reclaimed VM does not lose the run. `--resume` picks
up from `latest.pt` automatically — if the session dies, re-run the same
command.

## What to watch while it runs

Two lines matter, and **neither is the reward**:

1. **The TWO `policy` lines in the `EVAL` block** -- argmax and sampled, on
   identical held-out seeds -- and both against the baselines. This repo has a
   measured case of a policy scoring *worse* reward while routing *more* nets,
   so a reward curve alone can move the wrong way and look like progress.
   Eval uses **64 held-out boards** (was 16, where one board was worth
   6.25% and progress was indistinguishable from noise).
2. **`FORECAST GATE`.** Reports whether the learned occupancy forecast beats a
   straight-line demand baseline. It is expected to say `does NOT beat
   baseline` early on. Report what it says at the last eval of each stage —
   this gate decides whether a follow-on piece of the design gets built at all.

`[WARN]` and `[FATAL]` lines are automated health checks (NaN, entropy
collapse, exploding value loss, clip fraction, rejected-action rate). Report
every one of them with the update number. A `[FATAL]` stops the run
deliberately, before corruption reaches a checkpoint — that is working as
intended, and the output is the finding.

## What to report back

For each stage you run:

1. **The full `EVAL` blocks** — every one, verbatim. They contain policy vs
   greedy vs detour vs layer_hop on identical held-out seeds.
2. **The last ~20 live progress lines** (`[  123/1500] completion ... `).
3. **The summary table** from the artifacts cell: update, policy, greedy,
   detour, layer_hop, rejected rate, forecast-beats-baseline.
4. **`curves.png`** and the last few **`renders/*.png`**. In the renders,
   dashed red lines are nets that did *not* route, drawn pad-to-pad over the
   copper in their way. If completion plateaus, these images are the single
   most useful thing you can send.
5. **Any `DRC (real KiCad)` lines.** Expected: `0 legality violations`.
   Anything else is important — report immediately.
6. **Wall-clock and `decisions/sec`**, so we know whether throughput is what
   the design assumed.

## If it crashes

The run writes a self-contained **`crash_report.txt`** into its checkpoint
directory containing the environment (including the exact git commit and
whether the tree was dirty), the config, the traceback, tensor shapes and
NaN/inf counts, and the last ten updates' metrics.

**Report that file verbatim, in full.** Do not summarise it and do not attempt
a fix. Also attach `train_log.jsonl` (one JSON per update, flushed on write, so
it survives even a hard VM kill) and `console.log`.

If the VM died without a crash report — no traceback, cell just stopped — say
so explicitly and report the last lines of `console.log` plus how long it had
been running. That signature means something different (OOM or preemption) and
is diagnosed differently.

## What changed since the last run, and what to look for

### The eval now scores the policy TWICE on the same held-out boards

Every `EVAL` block now prints two policy lines instead of one:

```
    policy      60.8%   rejected  1.71%   vias   0.0   (argmax)
    policy      ??.?%   rejected ??.??%   vias  ??.?   (sampled -- the distribution training rolls out under)
    sampled - argmax  +??.?%   (untrained stage-1 reference: +10.6%)
```

**This is the single number this run exists to produce.** Report both policy
lines from every eval, verbatim.

Why: training rolls out by **sampling** and reported ~100% completion on stage
0 and up to 75.6% on stage 1, while held-out eval — which used **argmax** —
read 73–90% and 60.8%. Two things differed at once (the boards *and* the
action-selection rule), so the gap was unattributable. Now both arms run on
identical seeds and only the rule differs.

Reference numbers, measured locally on an **untrained** policy so the trained
gap can be read against something:

| stage | boards | argmax | sampled | argmax vias | sampled vias |
|---|---|---|---|---|---|
| 0 (1 net, 2L)   | 16 | **75.00%** | **93.75%** | 0.00 | 2.88 |
| 1 (20 nets, 2L) | 16 | **26.56%** | **37.19%** | 0.00 | 42.9 |

So roughly +10 to +19 points of that gap is present **before any training**,
and the mechanism is visible: at init, argmax takes direction 0 (straight down
the geodesic gradient) on **100%** of steps and places **zero** vias, and its
heads cycle — 23,455 head-steps over 5,146 distinct cells on stage 1, a 78%
revisit rate. Sampling is what places vias and what breaks the cycles.

**What that means for what you report.** A trained gap of roughly +10–19
points is the *floor*, not a finding. What matters is:

* a trained gap **much larger** than the untrained reference → training has
  made the mode worse relative to the mean, and confidence/entropy is the
  problem;
* a trained gap **at or below** the untrained reference → the mode is fine and
  argmax-at-eval is simply the wrong readout for this environment;
* `vias` on the **argmax** line. If it is still 0.0 after 1500 updates, the
  layer head's argmax never fires and every cross-layer net is unroutable at
  eval regardless of what the policy learned.

### Immediate task: re-eval the stage-1 checkpoint

Stage 1 has already run 1500 updates (best held-out argmax completion 60.8% at
update 1350, against greedy 26.1% and layer_hop 39.2%). Do **not** retrain it.
`git pull`, then run one more update so that a single eval fires with the new
two-arm code:

```bash
python -m neuroroute.training.run --stage 1 --device cuda \
  --batch 32 --heads 8 --width 48 --rollout 32 --ppo-chunk 8 --updates 1501 \
  --eval-every 50 --eval-boards 64 --render-every 0 --drc-every 0 \
  --checkpoint-dir $CKPT/stage1 --resume
```

That resumes at update 1500, does one update, and evals. Report the whole
`EVAL` block. It is a few minutes of GPU time and it answers the top open
question in `neuroroute/HANDOVER.md`.

### Then, if you have GPU time: one config sweep, no code changes

The heads' geodesic distance field is computed once when a net is assigned and
never refreshed (`--geodesic-refresh 0`), so copper laid afterwards by the
other K-1 heads is invisible to it. That is a candidate cause of the cycling
above -- **candidate, not confirmed**: tested locally on the *untrained*
policy (refresh 0 vs 1 vs 8), and completion, revisit rate and head-steps came
back byte-identical across all three. That is not a null result for the
trained checkpoint -- with near-zero actor weights (`gain=0.01`) and a
dominant direction-0 bias, an untrained argmax barely reads the field at all,
so staleness has nothing to bite on yet. Whether it matters *after* 1500
updates, once the weights carry real signal, is exactly what this sweep
tests. It is a **flag**, not a source change:

```bash
python -m neuroroute.training.run --stage 1 --device cuda \
  --batch 32 --heads 8 --width 48 --rollout 32 --ppo-chunk 8 --updates 1501 \
  --eval-every 50 --eval-boards 64 --render-every 0 --drc-every 0 \
  --geodesic-refresh 8 --checkpoint-dir $CKPT/stage1_geo8 --resume
```

Report the eval block and how much slower an update got. If argmax completion
rises materially with a refreshed field, that is the finding.

## Ground truth reminder

Everything about *learning* here is **unverified** — no real training run has
ever happened. The environment, geometry, multi-layer routing and the KiCad DRC
gate are all verified, so a failure during training is much more likely to be a
learning-side problem (hyperparameters, reward scale, credit assignment) than a
broken environment. Report numbers, not impressions, and let the fix happen on
the Claude Code side.
