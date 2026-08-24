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
- [x] First real training run (50 epochs, same data) -- diagnosed a
      flatlined auxiliary loss, described below.
- [x] Applied an input-normalization fix (LayerNorm + learnable
      `delta_scale`) and re-ran -- **the fix did not work**. `aux dist MAE`
      was bit-for-bit unchanged (still tied to the mean baseline), and
      action-vs-state sensitivity actually got worse. Scale mismatch was not
      the (sole) bottleneck. Kept in the codebase since it's harmless and
      may still matter once the real issue is fixed, but it did not resolve
      this on its own.
- [x] Built `probe_distance_from_embedding.py` to isolate the real question
      (see below) -- mechanically smoke-tested, not yet run against real
      data.

### Known issue: the auxiliary distance anchor isn't learning anything

Two full real 50-epoch runs (before and after the input-normalization fix)
both showed the same pattern: `pred_loss` collapses to ~0.0000 by epoch 2-3
(consistent with the predictor finding the trivial "predict no change"
solution to the cosine loss, since one router step barely moves the real
state -- normalize()'d cosine similarity doesn't care about scale, so the
LayerNorm fix was never going to touch this half), while `aux dist MAE`
stays flat at ~0.123-0.127 the ENTIRE time, statistically indistinguishable
from "predict the dataset mean" (0.1237) in both runs. No outright
representational collapse in the classic sense (`z_hat_std` non-zero,
action-vs-state sensitivity non-zero) -- but this is still exactly the
hollow-victory failure mode section 4 exists to catch: the main loss looks
solved while the auxiliary anchor that's supposed to verify real learning
is happening extracts nothing.

Importantly, `dist_next` clearly correlates strongly with `dist_t` in the
real data (the "no-change" baseline -- predict `dist_next = dist_t` using
the REAL known current distance, not a decoded one -- gets MAE 0.0173, ~7x
better than the mean baseline). So there IS plenty of real signal in the
target; the model just isn't extracting any of it from the embedding.

Working hypothesis now: `PCBRouterNet`'s policy/value heads were CO-TRAINED
with the encoder end-to-end over thousands of PPO updates, so whatever form
the encoder represents distance-relevant information in only has to be
usable by heads trained jointly with it. There's no guarantee that same
representation is easily decodable by a FRESH head trained in isolation for
a few dozen epochs on ~20k examples -- especially since mean-pooling over
256 patch tokens could dilute inherently local/spatial facts (where's the
head, where's the target) that a global average doesn't obviously preserve.

`probe_distance_from_embedding.py` tests this directly and in isolation:
given `z_t` alone (no predictor, no action, no "predict the NEXT state"),
can (a) a closed-form ridge linear regression, or (b) a small MLP, decode
`dist_t` (the CURRENT, already-known distance at that same timestep) at all?
If NEITHER beats the naive mean baseline, that's strong evidence the issue
is the embedding space itself, not the predictor architecture -- pointing at
switching the auxiliary anchor to match the ALREADY-TRAINED `value_head`'s
output instead (co-trained with this exact encoder, so proven decodable
from it) rather than a from-scratch geodesic-distance regression. If either
probe DOES beat the baseline, the problem is specific to the predictor/
`z_hat` pathway and needs a different fix there.

**Result, run against the real 26,070-transition dataset**: neither probe
beat the baseline. Ridge MAE 0.1261 vs baseline 0.1258 (essentially no
signal recovered -- the tiny gap is within noise); the MLP's 0.1252 is a
~0.5% relative improvement, also not meaningfully different from baseline.
Confirms this is a property of the embedding space, not the predictor
architecture, input scale, or a data-collection artifact -- distance-to-
target genuinely is not decodable from `z_t` by an independently-trained
head, linear or shallow-nonlinear, even given the full real dataset.

Before committing to the value_head-anchor idea the probe itself suggested,
`inspect_value_head.py` checks it cheaply first (same
evidence-before-architecture-change discipline, since "already trained"
does not automatically mean "decodes the specific thing we want a proxy
for" -- exactly the assumption that just failed for distance). It loads the
ORIGINAL checkpoint's `value_head` and applies it directly to the already-
logged `z_t` vectors -- no new environment rollouts needed. Reports
`value_head(z_t)`'s own std (is IT also degenerate?) and its Pearson
correlation with `dist_t` (expected notably negative if value tracks
progress-to-target).

**Result, run against the real checkpoint + full dataset**: `value_head(z_t)`
ranges from 0.2014 to 0.2088 across all 26,070 timesteps -- a spread of
0.0074 on a base of ~0.205, i.e. under 2% relative variation, across boards
with wildly different obstacle layouts, head positions, and progress through
the episode. Correlation with distance: 0.02, essentially zero. This is
functionally a constant, even though the script's own hardcoded
`std < 1e-4` collapse threshold didn't fire on it (0.0037 std) -- that
threshold was too strict for what's clearly a degenerate result in any
practical sense, worth noting so this printed verdict isn't over-trusted
literally.

**This is now a THIRD independent negative result** (linear probe, MLP
probe, and the CO-TRAINED value_head itself) all failing to extract any
meaningful state-dependent signal from `z_t` as logged. This is bigger than
"picked the wrong auxiliary anchor" -- it suggests the globally mean-pooled
`global_latent`, as extracted and logged, may not carry a usable
state-differentiating scalar signal at all, for ANY downstream head,
including one trained end-to-end with the encoder via thousands of PPO
updates. Yet the POLICY (also fed only this same pooled vector) clearly
DOES differentiate states well enough to solve stage 2 to 100% -- so the
embedding isn't information-free in general, just seemingly not in a form
any of these three scalar-decoding attempts could extract.

**This is a genuine decision point, not a code problem to keep patching.**
Live options, not yet decided:
  1. One more cheap, decisive test before deciding further: try decoding
     something simpler and more direct than geodesic distance (e.g. raw
     head (x,y) position from Channel 3, or straight-line Euclidean
     distance rather than the obstacle-aware geodesic field) -- this
     disentangles "is ANY positional information surviving the pool" from
     "is specifically the nonlinear geodesic-distance function of position
     unrecoverable."
  2. A bigger redesign: use the encoder's per-token `encoded_tokens`
     (256 spatial patches, pre-pooling) instead of the pooled
     `global_latent` -- pooling is exactly what's suspected of destroying
     the local/spatial signal. Meaningfully larger scope (more storage,
     more architecture work) than anything built so far.
  3. Per the original plan's own stated exit criterion: if this doesn't pan
     out, delete `jepa/` and stay on the proven, working `--lookahead` path,
     moving on to stage 3 instead.

**Decision: option 2, the per-token redesign.** Before rewriting the whole
data pipeline around it, `probe_token_features.py` validates the hypothesis
cheaply first (same discipline as every other check above) -- rather than
store all 256 patch tokens (256x the pooled vector's storage, and a much
bigger predictor input than needed), it extracts just the ONE token whose
16x16-downsampled patch covers the head's current position, and the one
covering the target pad's position (both known exactly at collection time,
no learning needed to find them), and runs the same ridge/MLP probe against
`pooled` (control), `head_token`, `target_token`, and `head_token +
target_token` concatenated.

**Smoke test result (random-init checkpoint, 277 timesteps -- mechanical
sanity check only, not the real verdict)**: `head_token` alone crushed the
baseline via ridge regression (0.0308 vs baseline 0.1408) -- even with
UNTRAINED weights. This makes mechanistic sense: the geodesic distance field
is already one of the 10 input CHANNELS (Channel 7), so the patch centered
on the head's own position has near-direct local access to the distance-to-
go value baked into the raw pixel input at that exact spot, independent of
training. `target_token` alone did not help (expected -- target position
alone can't determine distance without also knowing where the head is,
which is a different image region entirely). Strong signal the redesign
direction is mechanically sound; real validation still needs the actual
trained checkpoint and real-scale data.

**First real run (200 episodes, plain deterministic policy) came back
negative** -- all four representations tied to baseline (0.1204-0.1209).
But the collection used plain deterministic action selection, which this
project's history documents as prone to oscillation-trap failures in under
20 steps; this run averaged only ~16.5 steps/episode, suspiciously matching
that signature. A dataset dominated by short, stuck episodes under-samples
the "closer to target" end of the distance range and could suppress ANY
representation's apparent decodability regardless of whether per-token
features are useful -- a real confound, not just a guess, so this result
was not trustworthy as-is.

**Fixed**: `probe_token_features.py` now uses the SAME exploring top-k
behavior policy `collect_transitions.py` uses (proven to reach 997/1000
completions on this checkpoint), and reports completion rate + avg
steps/episode directly so this can be checked going forward instead of
inferred after the fact. Re-smoke-tested (random-init checkpoint): same
strong `head_token` result as before (ridge 0.0373 vs baseline 0.1184).

- [ ] Run the FIXED `probe_token_features.py` against the real v7
      checkpoint (fresh rollouts with the exploring policy, not the
      existing shards -- this needs per-token features that were never
      logged before) and confirm `head_token` (or `head_token +
      target_token`) clearly beats the pooled control AND that the
      completion rate is reasonable this time, not just this smoke test.
- [ ] If confirmed: rewrite `collect_transitions.py` to log `head_token`/
      `target_token` (not the pooled `global_latent`) and rebuild
      `dynamics_model.py`/`train_dynamics.py` around that representation.
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
