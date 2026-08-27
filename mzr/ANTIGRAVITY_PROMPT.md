# Prompt for Antigravity — MZR Colab training

Copy everything below the line into Antigravity.

---

You are running **MZR** training on Google Colab. Read `AGENTS.md` in the repo
first — it governs this session.

## Your boundary

Your job is **Colab execution and reporting**, not diagnosis and not fixes.

- **Do not edit any tracked source file.** Not even a one-line fix that looks
  obviously correct. If something fails, report it verbatim. Fixes belong in a
  Claude Code session that can see the design reasoning; several past bugs in
  this repo came from a plausible local fix that missed a project-wide
  constraint.
- **Report real output, verbatim.** Actual stdout, actual tracebacks, actual
  numbers — never a summary from memory. If a run succeeds, paste the numbers,
  not "it worked".
- You may freely change **command-line flags** (`--batch`, `--rollout`,
  `--field-width`, `--token-width`, `--lr`, `--updates`, `--eval-every`,
  `--eval-boards`) to fit the GPU. That is configuration, not source. Tuning
  them is wanted.

## Setup

```
Repo:   https://github.com/Klutzhehe/Routerv3.git
Branch: main   (git pull before starting — do not run stale code)
```

Set **Runtime → Change runtime type → GPU**. This is pure PyTorch — you do
**not** need the compiled `pcbworld_pns_bridge` (that is a ~40-min
KiCad-from-source build for an older thread). The KiCad DRC gate uses the
ordinary `apt-get install -y kicad` package.

```bash
!git clone https://github.com/Klutzhehe/Routerv3.git && cd Routerv3
%cd Routerv3
!pip -q install torch numpy
!apt-get -qq install -y kicad
```

## Step 1 — Local verification. This is a gate. ~3–5 min, no GPU needed.

```bash
python -m mzr.scripts.verify_world
```

**32 checks. Every one must print `[PASS]` and the script must end with
`all checks passed`.** These re-derive answers from the occupancy grid rather
than trusting engine flags — the discipline that caught every real bug in the
predecessor. If any check fails, **stop and report the full output**; do not
train on top of a failing world.

## Step 2 — KiCad legality gate. Also a gate. ~5–10 min.

```bash
python -m mzr.scripts.validate_kicad --out /content/drc_out
```

Routes four board configs (4–8 layers, up to 50% wide traces) with the
non-learned baseline, exports real `.kicad_pcb`, and runs KiCad's own
`DRC_ENGINE`. **Expected: `PASS: sim-to-real gap is 0 legality violations`.**
Locally it passes at 0 over 136 routed nets.

- If `kicad-cli` cannot parse a file → exporter bug. Report it; do not train.
- If it parses and reports **legality** violations → the lattice model is
  wrong. Report the full per-config table; do not train.
- `completeness` and `bookkeeping` counts are expected and fine (unrouted nets
  and a missing footprint library — neither is about copper legality).

## Step 3 — Stage 0 training.

Single net, static keepouts + pads, 2 layers, 48×48. This is the plumbing
check: engine + observation + reward + PPO + eval + checkpointing end to end.
The single-net sub-problem was taken to 100%/1000 by the raster thread, so it
**is** achievable — do not read "stage 0 passed" as "the design works".

```bash
python -m mzr.training.run \
  --stage 0 --device cuda \
  --batch 32 --rollout 32 --updates 600 \
  --field-width 64 --token-width 192 \
  --lr 3e-4 --eval-every 25 --eval-boards 64 \
  --checkpoint-dir /content/drive/MyDrive/mzr_ckpt/stage0
```

Mount Drive first so checkpoints survive a VM death:
```python
from google.colab import drive; drive.mount('/content/drive')
```

### What to watch and report

Every eval line looks like:
```
u  25 | argmax 0.812 sampled 0.905 (perfect 0.75) | best 0.812 | hits 0 | kl 0.021 clip 0.28 ent 3.9 | 4.2s
```

- **`argmax` is the gated metric.** Gate is **≥ 0.99, sustained 3 consecutive
  evals** — the script prints `GATE CLEARED` when that happens.
- **`sampled` will lead `argmax`** early — that is the known mode/mean gap, not
  a bug. Report both. A policy that only works sampled is leaning on
  exploration noise.
- **`clip` (clip fraction)** was chronically high in the predecessor (0.3–0.6).
  If it sits above ~0.5 for many updates, try `--lr 1.5e-4`. Report what you
  changed.
- **`kl` (approx KL)** spiking > 0.2 means the update stepped outside the trust
  region — same fix (lower `--lr`).
- **Kill-number:** if `argmax` cannot reach **0.95 in 500 updates**, that is a
  geometry or reward bug, not a hard problem — stop and report the full
  `stage0.jsonl` and the last 10 eval lines.

### Report back

1. The **full step-1 and step-2 output**, verbatim (pass/fail + numbers).
2. Every **eval line**, verbatim — or attach `stage0.jsonl`.
3. Whether `GATE CLEARED` printed, and at which update.
4. `wall-clock per update` and `GPU memory` (`nvidia-smi`), so `--batch` /
   widths can be tuned for stage 1.
5. If it crashed: the **full traceback**, and whether `git status` was clean
   (`git rev-parse HEAD` + `git status --porcelain`).

## If step 3 clears the gate

Do **not** proceed to stage 1 on your own — stage 1 adds the congestion price,
rip-up, and BC-from-expert, and its gate is measured against the expert
baseline the run computes at startup. Report that stage 0 cleared and wait for
an updated prompt.

## Do not

- Edit tracked source (flags only).
- Summarise output from memory.
- Skip step 1 or step 2 to "save time" — they are each one afternoon's
  insurance against training on a fiction.
- Run the compiled PNS bridge build.
