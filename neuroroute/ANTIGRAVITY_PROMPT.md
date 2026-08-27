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

**Stage 1 is not a fresh start any more** — a checkpoint already exists with
real progress. Use the resume command in "What changed since the last run"
below, not the from-scratch one here. The stage-1 command below is kept for
reference (what a genuinely fresh stage-1 run looks like) but is stale on
tuning: no `--lr 1.5e-4`, no `--amp`, no `--store-device cuda`, none of which
existed when it was written.

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
intended, and the output is the finding. With `--amp` on, a `[WARN] gradient
... is non-finite (GradScaler already skipped this step)` is expected and
routine (GradScaler's own safety mechanism, not a corruption) — only a
`[FATAL]` needs reporting as urgent; that WARN is fine to just include in the
normal report.

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

The two-arm eval question from earlier sessions is **answered** (both a
mode/mean gap and a real learning-rate instability were confirmed and fixed
along the way -- full story in `HANDOVER.md`). The current state: stage 1
peaked at **64.8% argmax completion at update 1550**, and has not exceeded
that through roughly update 2199 since. `--amp` is on and confirmed working
(~1.9x real speedup once past a one-time per-process warm-up cost) after
fixing one real bug in it (a forecaster loss running in unsafe fp16). None of
that needs re-running.

### The big one: the scheduler had never trained, on any run, ever

Traced it down while building something else: `policy.act()` computed a
log-prob for the net-scheduling decision every step, but nothing downstream
ever used it -- `h_schedule` received **zero gradient** on every stage-0 and
stage-1 run in this project's history, for a stage whose whole point is
*"congestion and ordering."* Fixed, along with wiring in a rip-up action
(`world.ripup()` already existed and was already verified; the policy simply
never emitted it) through the same fix, since both are the same kind of
missing piece. Full detail in `HANDOVER.md`.

**This has never been trained on. That is the main thing to find out now.**

### Step A -- resume stage 1 with the new code, on the SAME checkpoint

```bash
git pull origin main   # must be at commit 1b82baf or later
python -m neuroroute.training.run --stage 1 --device cuda \
  --batch 32 --heads 8 --width 48 --rollout 32 --ppo-chunk 32 \
  --store-device cuda --lr 1.5e-4 --amp \
  --updates 3000 --eval-every 50 --eval-boards 64 --render-every 100 --drc-every 0 \
  --checkpoint-dir $CKPT/stage1 --resume
```

(`--render-every 100` is intentionally back on this time -- see Step B.)

**Expect one `[WARN]` on resume, and it is not a problem:**

```
[WARN] checkpoint predates the current architecture -- missing [...], unexpected []...
```

This is the checkpoint's old weights loading correctly; the three new
parameters (the ripup head, its "do nothing" bias, and the new board-level
value function) start at their own fresh init, exactly as intended. Report
it once, then move on -- it is expected, not a fault to fix.

**What to watch that is new:**
* Every progress line now has `sched`, `ripup`, and `bv` fields. `bv` (the new
  board-level critic's loss) should move around like a real value function
  is learning -- if it sits pinned near one number for hundreds of updates,
  that is worth reporting, not just the headline completion number.
* Every `EVAL` block now has a `ripups` count next to `vias`, for both argmax
  and sampled. Report it every time -- it is the only way to see whether the
  policy is actually using the new capability, since the trained checkpoint
  has literally never had this option before.
* **The number that answers the open question**: does argmax completion move
  past 64.8% now? Report every `EVAL` block in full, verbatim, same as always.

### Step B -- once a handful of evals have landed: look at the renders

`--render-every 100` above already turns this on. After the run has been
going a while, pull the images:

```bash
ls -la $CKPT/stage1/renders/
```

Send back the **latest** `*_worst.png` and `*_sheet.png`. Nobody has looked
at an actual failed board this whole project -- every finding so far has come
from numbers. Dashed red lines in the render are nets that did not route,
drawn pad-to-pad over whatever is in their way; this is expected to be far
more informative than any more delta-of-numbers, and it's needed regardless
of how Step A comes out.

## Ground truth reminder

Everything about *learning* here is **unverified** — no real training run has
ever happened. The environment, geometry, multi-layer routing and the KiCad DRC
gate are all verified, so a failure during training is much more likely to be a
learning-side problem (hyperparameters, reward scale, credit assignment) than a
broken environment. Report numbers, not impressions, and let the fix happen on
the Claude Code side.
