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
  `--heads`, `--updates`) to fit the GPU you were given, and you may re-run
  cells. That is configuration, not source.

## Setup

```
Repo:     https://github.com/Klutzhehe/Routerv3.git
Notebook: neuroroute/notebooks/neuroroute_colab.ipynb
Branch:   main   (commit 1921bb6 or later)
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
# Stage 0 — plumbing. ~5 min. Should climb toward ~100%.
python -m neuroroute.training.run --stage 0 --device cuda \
  --batch 16 --heads 4 --width 32 --rollout 32 --updates 300 \
  --eval-every 25 --render-every 50 --drc-every 100 \
  --checkpoint-dir $CKPT/stage0 --resume
```

```bash
# Stage 1 — congestion, 20 nets, 2 layers. Gate 75%.
python -m neuroroute.training.run --stage 1 --device cuda \
  --batch 16 --heads 8 --width 48 --rollout 32 --updates 1500 \
  --eval-every 50 --render-every 100 --drc-every 200 \
  --checkpoint-dir $CKPT/stage1 --resume
```

```bash
# Stage 3 — eight layers. Gate 85%.
python -m neuroroute.training.run --stage 3 --device cuda \
  --batch 12 --heads 8 --width 64 --rollout 32 --updates 4000 \
  --eval-every 100 --render-every 200 --drc-every 400 \
  --store-device cpu --checkpoint-dir $CKPT/stage3 --resume
```

**Stage 0 is a plumbing check.** One net on an empty board. If it does not
climb well above the 87.5% non-learned baseline, something is wrong with the
setup, not with the idea — report it and stop rather than moving to stage 1 on
a partial result.

If you hit VRAM limits: reduce `--batch` first, then `--width`, then `--heads`.
Keep `--store-device cpu`. Report what you changed and why.

Checkpoint to Drive so a reclaimed VM does not lose the run. `--resume` picks
up from `latest.pt` automatically — if the session dies, re-run the same
command.

## What to watch while it runs

Two lines matter, and **neither is the reward**:

1. **`completion` vs the baselines in the `EVAL` block.** This repo has a
   measured case of a policy scoring *worse* reward while routing *more* nets,
   so a reward curve alone can move the wrong way and look like progress.
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

## Ground truth reminder

Everything about *learning* here is **unverified** — no real training run has
ever happened. The environment, geometry, multi-layer routing and the KiCad DRC
gate are all verified, so a failure during training is much more likely to be a
learning-side problem (hyperparameters, reward scale, credit assignment) than a
broken environment. Report numbers, not impressions, and let the fix happen on
the Claude Code side.
