# Prompt for Antigravity — MZR Colab training

Copy everything below the line into Antigravity.

---

You are running **MZR** stage-0 and stage-0v training on Google Colab. Read `AGENTS.md` in the repo — it governs this session.

## Your boundary
- **Do not edit any tracked source file.** Not even a one-line fix. If something fails, report it verbatim and stop. Fixes belong in a Claude Code session.
- **Report real output, verbatim** — actual stdout, actual tracebacks, actual numbers. Never a summary from memory.
- You may change **command-line flags only** (`--batch`, `--rollout`, `--field-width`, `--token-width`, `--lr`, `--updates`, `--eval-every`, `--eval-boards`) to fit the GPU. Tuning them is wanted. Do **not** touch `--bc-coef` (stays 0 — pure RL).

## Setup — run this cell

```bash
%cd /content
!rm -rf Routerv3 && git clone https://github.com/Klutzhehe/Routerv3.git
%cd /content/Routerv3
!git rev-parse HEAD && git status --porcelain
!pip -q install torch numpy
# kicad-cli ships with KiCad 7+. Colab is Ubuntu 22.04, whose universe repo
# has KiCad 6.0.2 -- which predates kicad-cli entirely, so plain
# `apt-get install kicad` installs pcbnew and friends and NO kicad-cli.
# The official PPA is the only route. ~36s with --no-install-recommends,
# which skips demos/footprints/symbols the DRC gate does not read.
!apt-get -qq install -y software-properties-common
!add-apt-repository --yes ppa:kicad/kicad-9.0-releases
!apt-get -qq update
!DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends kicad
!kicad-cli --version   # must print 9.x -- if it says 6.x or 'not found', stop
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 1 — Local verification. GATE. ~3–5 min, no GPU.

```bash
!cd /content/Routerv3 && python -m mzr.scripts.verify_world
```

**Every check must print `[PASS]` and the script must end with `all checks passed`.** If any check fails: stop, paste the full output, do not proceed.

## Step 2 — KiCad legality gate. GATE. ~5–10 min.

```bash
!cd /content/Routerv3 && python -m mzr.scripts.validate_kicad --out /content/drc_out
```

**Expected: `PASS: sim-to-real gap is 0 legality violations`** (locally: 0 over 136 routed nets, 4 configs).
- `kicad-cli` cannot parse a file → exporter bug. Paste it, stop.
- Parses but reports **legality** violations → lattice model is wrong. Paste the full per-config table, stop.
- `completeness` / `bookkeeping` counts are expected and fine — not about copper legality.
## Step 3 — Baseline preflight. GATE. ~2 min, no GPU.

New, and it is a gate for a reason: stage 0 sat under its gate for many sessions because nobody ran the **non-learned** baselines against it. `greedy` is the all-zero action — literally what an untrained policy emits — so if `greedy` cannot approach the gate, no policy starting there can either, and the bug is in the substrate, not the learner.

```bash
!cd /content/Routerv3 && python -m mzr.scripts.baseline_gate --stage 0
!cd /content/Routerv3 && python -m mzr.scripts.baseline_gate --stage 0v
```

**Expected** — measured locally on CPU over the same 48 held-out seeds. These must reproduce:

```
stage 0    greedy     completion 1.0000  copper 1.000  doubled 0  right-angle 0.000  -> CLEARS GATE
stage 0    layer_hop  completion 1.0000  copper 1.000  doubled 0  right-angle 0.000  -> CLEARS GATE
stage 0v   greedy     completion 0.7500  copper 1.045  doubled 0  right-angle 0.091  -> below gate
stage 0v   layer_hop  completion 1.0000  copper 1.024  doubled 0  right-angle 0.001  -> CLEARS GATE
```

If any of those differs materially, **stop and paste the table** — the substrate has regressed and training is pointless until it is fixed.

What the numbers mean: stage 0 is now genuinely solved by following the geodesic field, so the untrained policy already sits at the gate. That is the intended outcome — stage 0's job is now "the plumbing works end to end and the policy does not *regress* off the bar." The first rung that requires real learning is **0v**, where the entire `greedy` → `layer_hop` gap (0.75 → 1.00) is the via decision.

## Step 4 — Stage 0 training. Pure RL. Short.

Single net, **one layer**, static keepouts + pads, 48×48. Plumbing check: engine + observation + reward + PPO + eval + checkpointing end to end. **No behaviour cloning, no board pre-filter** — the policy routes whatever the generator produces. Expect `GATE CLEARED` almost immediately; what matters is that it holds for 3 consecutive evals and that copper / right-angle / doubled stay clean.

```bash
!cd /content/Routerv3 && python -m mzr.training.run \
  --stage 0 --device cuda \
  --batch 32 --rollout 32 --updates 200 \
  --field-width 64 --token-width 192 \
  --lr 3e-4 --eval-every 25 --eval-boards 64 \
  --checkpoint-dir /content/drive/MyDrive/mzr_ckpt/stage0
```

## Step 5 — Stage 0v training. The real first rung.

Two layers. Pads land on outer layers, so ~25% of boards cannot be routed without a via, and `greedy` scores exactly 0.75 because it never places one. This stage measures **via discovery alone**.

```bash
!cd /content/Routerv3 && python -m mzr.training.run \
  --stage 0v --device cuda \
  --batch 32 --rollout 32 --updates 1000 \
  --field-width 64 --token-width 192 \
  --lr 3e-4 --eval-every 25 --eval-boards 64 \
  --checkpoint-dir /content/drive/MyDrive/mzr_ckpt/stage0v
```

**Watch `via_frac` on the profile line.** If it sits near 0 while `argmax` sits near 0.75, the policy has settled into never-via. That is the known local optimum — a misplaced via is worth about −0.8 reward and a correct one about +0.2, so early gradients say "never via" — not a capacity problem. Report it; do not tune around it.

**KILL-NUMBER (stage 0v):** cannot reach 0.95 in 1000 updates → the via penalty is drowning discovery. Stop and report. The fix is a source change (`RewardConfig.via`, or seeding the layer head from `layer_hop`), and source changes belong in a Claude Code session.

### Reading the eval lines

```
u  25 | argmax 0.812 sampled 0.905 (perfect 0.75) | best 0.812 | hits 0 | kl 0.021 clip 0.28 ent 3.9 | 4.2s
       argmax < 100% on seeds: [900007, 900031, 900044]  (review with: python -m mzr.world.pool --stage 0 --seeds 900007 900031 900044)
```

- **Gate is `argmax` completion == 1.000, sustained 3 consecutive evals.** The script prints `GATE CLEARED` when it happens.
- **`perfect`** = fraction of eval boards at 100%. This is the number that has to reach 1.00.
- **The failing-seed line matters.** If the run stalls a few points short, paste that seed list — it goes to a hand review (`mzr.world.pool` checks whether the expert can route each; a board the expert routes but the policy can't is a policy problem).
- **`sampled` will lead `argmax`** early — known mode/mean gap, not a bug. Report both.
- **`clip`** above ~0.5 for many updates → try `--lr 1.5e-4`. **`kl`** spiking > 0.2 → same fix. Report what you changed.
- **KILL-NUMBER (stage 0):** if `argmax` cannot reach **0.95 in 500 updates**, that is a geometry or reward bug, not a hard problem — stop and paste the last 10 eval lines + `stage0.jsonl`.

### Report back
1. Full step-1, step-2 and step-3 output, verbatim -- the baseline table especially.
2. Every eval line from steps 4 and 5, verbatim — or attach the `.jsonl` from each checkpoint dir.
3. Whether `GATE CLEARED` printed for each stage, at which update; and the **final failing-seed list** if it did not.
4. Wall-clock per update, and `!nvidia-smi` (memory), so batch/widths can be tuned for stage 1.
5. If it crashed: full traceback + `git rev-parse HEAD` + `git status --porcelain`.

## Do not
- Proceed to stage 1 on your own. Report stage 0 and 0v, then wait for an updated prompt.
- Edit tracked source. Summarise from memory. Skip steps 1-3. Change `--bc-coef`. Build the compiled PNS bridge (not needed — pure PyTorch).
