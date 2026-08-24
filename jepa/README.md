# JEPA-style fast lookahead (in progress)

Goal: replace the expensive real-environment simulation inside
`lookahead_select_action` (`models/router_policy.py`) -- which solved stage 2
to 100% (1000/1000, seeds 9000-9999) but costs ~top_k*horizon extra real env
steps and full-network forward passes per real decision -- with a cheap
learned predictor operating in latent space, so lookahead-quality decisions
approach plain-argmax speed.

**Isolation, deliberate**: everything in this folder is new. Nothing in
`pcbworld/`, `models/router_policy.py`'s existing functions, `render_episode.py`,
or `training/evaluation.py`'s existing (non-JEPA) paths has been touched. If
this doesn't pan out, delete `jepa/` and the proven, working `--lookahead`
path (100% on the full benchmark, just slow) is exactly as it was.

## Status

- [x] Predictor architecture + combined objective + collapse diagnostics
      implemented and smoke-tested (mechanical correctness only -- random
      init, tiny synthetic run, not a real training result).
- [x] Real data collection run: 1000 episodes, seeds 100000-100999, stage 2,
      v7 checkpoint -- 26,070 transitions (997/1000 completed).
- [x] First real training run (50 epochs, same data) -- diagnosed a real
      issue, described below, not yet re-validated after the fix.

### Known issue (found in the first real run, fixed, not yet re-validated)

The first real 50-epoch run showed `pred_loss` collapsing to ~0.0000 by
epoch 2-3 while `aux dist MAE` stayed flat at ~0.123-0.127 for all 50
epochs -- statistically indistinguishable from the "predict the dataset
mean" baseline (0.1237) the entire time. No outright representational
collapse (`z_hat_std` stayed non-zero, action-vs-state sensitivity ratio
stayed ~0.8-1.07), but this is still the hollow-victory failure mode
section 4 above exists to catch: the predictor satisfied the cosine
predictive loss almost for free (one router step barely moves the state, so
`delta -> 0` -- "predict no change" -- nearly solves it without learning
anything about the action), and the auxiliary anchor that was supposed to
catch that wasn't extracting any real signal either.

Root cause: the frozen encoder's `global_latent` (mean-pooled over 256
post-LayerNorm patch tokens) turned out to have surprisingly small scale
across the dataset (~0.01-0.02 std) -- averaging many roughly-independent
unit-scale token vectors shrinks the aggregate. `DynamicsPredictor` and
`DistanceHead` were plain `nn.Linear` stacks with PyTorch's default init,
which implicitly assumes ~unit-scale input -- the same class of bug this
repo already hit once before in `models/router_policy.py`'s policy head
(the `gain=0.01` vs `0.1` story). Compounding it: `ActionEncoder`'s
freshly-initialized embeddings sit at the default ~unit scale, ~70x larger
than `z_t` -- concatenated together, the action channel likely dominated
the predictor's first layer.

Fix applied (`dynamics_model.py`): both modules now `LayerNorm`-normalize
their input before their MLPs; `DynamicsPredictor` additionally applies a
learnable `delta_scale` (initialized small) so the residual itself starts
close to `z_t`'s own natural scale rather than being swamped by an
internally-normalized (~unit-scale) delta from the first step of training.
This does NOT require re-collecting data -- only re-running
`train_dynamics.py` on the existing shards. **Not yet re-validated against
a real run** -- next step is exactly that.

- [ ] Real training run + collapse diagnostic actually checked against real
      data, WITH the input-normalization fix above.
- [ ] Fast action-selector (`jepa_lookahead_select_action`) -- **not built
      yet, deliberately**. Building a selector against an unvalidated
      predictor would mean debugging two unknowns (does the predictor work?
      does the selector wire it up right?) at once. Comes after the
      diagnostics below are checked against a real run.
- [ ] Validation: speed (approach plain-argmax) and correctness (known hard
      seeds + full 1000-board benchmark) as two SEPARATE checks, same
      evidence-first discipline as the rest of this project.

## Design decisions

### 1. Architecture: MuZero-style dynamics network, JEPA/BYOL-family anti-collapse

Neither literal I-JEPA nor V-JEPA fits directly -- neither is
action-conditioned, and this problem is "predict next state given current
state AND a specific chosen action." Instead: an action-conditioned latent
dynamics predictor `g(z_t, a_t) -> z_hat_{t+1}` (MuZero's dynamics network
shape), trained with the JEPA-family anti-collapse technique (EMA target +
stop-gradient) since that mechanism is architecture-agnostic -- see
`dynamics_model.py`.

### 2. Data pipeline: frozen encoder, logged embeddings -- not raw pixels, not an EMA-trained encoder

This is the one place this implementation deviates from a literal "EMA
target encoder": storing raw `(10, 256, 256)` float32 observations for a
dynamics dataset is prohibitive (~2.6MB each; a useful dataset needs tens of
thousands of transitions, i.e. hundreds of GB). The resolution: **the
PCBEncoder that produces the embeddings is frozen** (the already-trained
one from the stage-2 checkpoint) and `jepa/collect_transitions.py` stores
only its ~256-d output per observation -- a >1000x storage reduction that
makes the whole approach affordable on Colab Drive.

Consequences of this choice, stated plainly:
- There is no online encoder left to fine-tune or collapse in
  `train_dynamics.py` -- only the new `DynamicsPredictor` and `DistanceHead`
  are trained. The "target encoder" is just "the same frozen encoder,
  applied to the real next observation" -- trivially already
  stop-gradient'd (nothing in it has `requires_grad`), so there's no EMA
  schedule to run because there's nothing drifting to track.
- This does NOT remove the collapse risk the combined objective is meant to
  guard against -- it relocates it. A frozen, already-good target embedding
  space can't collapse, but the **predictor** mapping `(z_t, a_t) -> z_hat`
  can still learn a shortcut (e.g. "always predict close to the batch mean
  next-embedding", which can look like a good cosine match if nearby-in-time
  states already sit close together in this frozen space) that ignores the
  action entirely. That's exactly the failure mode the auxiliary distance
  head and the action-sensitivity probe below are built to catch.
- If validation later shows the frozen representation is too coarse (e.g.
  the predictor can't beat the naive baselines no matter what), the natural
  next iteration is making the encoder trainable with a real EMA schedule --
  at that point raw observations (or a smarter partial-channel encoding,
  since several of the 10 channels are static per net-attempt) would need to
  be logged instead. Isolated in its own folder specifically so that's a
  contained change, not a rewrite.

### 3. Combined objective (required, not optional)

- **Predictive loss** (`predictive_loss` in `dynamics_model.py`): BYOL-style
  `2 - 2*cosine_similarity(z_hat, z_target)`. Cosine, not raw MSE, so
  shrinking every embedding's norm toward zero can't trivially lower the
  loss.
- **Auxiliary anchor** (`DistanceHead`): decodes the *predicted* `z_hat` into
  the real, verifiable normalized geodesic distance-to-target (same
  normalization `environment.py`'s Channel 7 uses:
  `clip(raw_geodesic / hypot(256,256), 0, 1)`). Applied to `z_hat`
  specifically (not to `z_target`) -- a collapsed/constant `z_hat` cannot
  also satisfy this per-sample-varying real target, which is what makes
  collapse self-defeating instead of merely discouraged.

### 4. Collapse diagnostics (computed every epoch, before/alongside the main loss -- not after training "looks done")

Three checks in `train_dynamics.py`'s `compute_diagnostics`, all on the VAL
split:
1. **Embedding std** (`z_hat_std` vs `z_target_std`) -- the classic
   BYOL/DINO collapse check. Near-zero `z_hat_std` is an unambiguous red
   flag.
2. **Aux MAE vs two naive baselines** -- "predict the training-set mean
   distance" and "predict no change from the current distance" (the harder,
   more meaningful floor in this domain, since one router step is small
   relative to the whole board). The model MUST beat both by a real margin,
   not just match them -- matching means it learned nothing beyond a boring
   default.
3. **Action-sensitivity probe** -- for a batch of held-out states, evaluate
   the predictor under EVERY possible action and measure the spread across
   actions, then compare that to the spread across DIFFERENT states. A
   predictor that quietly learned to ignore its action input can still show
   healthy dataset-level embedding variance (from varying `z_t` alone) while
   this ratio is near zero -- this is the check that specifically catches
   that failure mode, which (1) and (2) alone would not.

Every diagnostic print includes an explicit `***` flag when a threshold looks
bad. Read these before trusting any run -- "loss went down" is not sufficient
evidence in this project (see the session's stated history of this exact
pitfall), and a collapsed predictor can drive the predictive loss down while
being useless.

### 5. Train/val split

By whole EPISODE, not by individual transition (`episode_split`) --
consecutive steps within one episode are highly correlated, so a
transition-level split would leak near-duplicates across train/val.

### 6. Seed hygiene

`collect_transitions.py` defaults to seed block starting at 100000,
deliberately disjoint from the canonical eval block (9000-9999) and the
known-hard-seed list. Training the fast selector on the exact boards it will
later be validated against would make that validation meaningless.

## Files

- `dynamics_model.py` -- `DynamicsPredictor` (residual MLP,
  `z_hat = z_t + f(z_t, action_embedding)`), `ActionEncoder` (embeds the
  action by its known dir/dist/layer/via factored structure, not a flat
  one-hot), `DistanceHead`, `predictive_loss`.
- `collect_transitions.py` -- rolls out a trained checkpoint (frozen encoder
  + policy), behavior policy = mostly greedy top-1 with an epsilon chance of
  a non-greedy top-k candidate (so the dataset covers exactly what the fast
  selector will query at inference, not just the on-policy trajectory), logs
  `(episode_idx, z_t, action, z_next, dist_t, dist_next, done, completed,
  failed)` to `.npz` shards.
- `train_dynamics.py` -- loads shards, trains `DynamicsPredictor` +
  `DistanceHead` with the combined loss, prints/saves diagnostics every
  epoch, saves `jepa_dynamics_latest.pt` / `jepa_dynamics_best.pt`.

## Running this on Colab (antigravity)

```bash
# 1. Collect transitions using the proven stage-2 v7 checkpoint (~1000 episodes,
#    disjoint seed block, matches the eval's max_net_restarts=2 setting).
python -m jepa.collect_transitions \
  --checkpoint /content/drive/MyDrive/pcb_ai_router/checkpoints_stage2_v7/single_net_router_latest.pt \
  --stage 2 --num-episodes 1000 --max-net-restarts 2 \
  --output-dir /content/drive/MyDrive/pcb_ai_router/jepa_data

# 2. Train the dynamics predictor + distance head.
python -m jepa.train_dynamics \
  --data-dir /content/drive/MyDrive/pcb_ai_router/jepa_data \
  --stage 2 --epochs 50 \
  --checkpoint-dir /content/drive/MyDrive/pcb_ai_router/jepa_checkpoints
```

Report back (verbatim, same discipline as every other run in this project):
the full per-epoch diagnostics block for the last few epochs, and whether any
`***` flags fired. Do not report just the final loss numbers -- the whole
point of section 4 above is that the loss alone doesn't tell us whether this
worked.

## Explicitly not done yet

Per the agreed ordering: build the fast action-selector
(`jepa_lookahead_select_action`) and wire a new `--jepa-lookahead` flag into
`render_episode.py` / `evaluate_policy` (alongside, not replacing,
`--lookahead`) only AFTER a real training run's diagnostics have been
checked. A fast-but-wrong predictor is worse than the current slow-but-proven
lookahead, not better -- no point building the selector around a predictor
that hasn't been shown to actually predict anything yet.
