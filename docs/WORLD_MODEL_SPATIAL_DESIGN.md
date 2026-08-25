# Spatially-Aware World Model: Design

## Problem

Stage 2 is solved (100%/1000, see `checkpoints_stage2_v7/`) but only with
`lookahead_select_action` (`models/router_policy.py:169`) wrapped around the
network at eval time -- real env-simulation, ~top_k*horizon extra cost per
decision. The plain deterministic policy (`select_deterministic_action`)
gets caught in stable 2-cycles at tight multi-obstacle corners: the
locally-best action at A leads to B, and B's own locally-best action leads
back to A. The docstring on `lookahead_select_action` documents this
directly, measured on seeds 9148/9251 under an earlier checkpoint.

**This is not a missing-information problem.** The observation the network
already receives (`pcbworld/environment.py:869` `_build_observation`)
includes, at full 256x256 resolution:

- Channel 7: obstacle-aware geodesic distance-to-target field (not
  Euclidean -- routes around obstacles already, see
  `compute_geodesic_distance_field`)
- Channel 9: rejection feedback (where the last move was rejected + how
  stuck the net is) and permanent dead-zone markers from prior failed
  attempts at this net
- Channels 0/1: copper and obstacles, i.e. exactly what a corner even is

And the action frame is already egocentric: `dir_idx=0` means "toward the
target, routed around obstacles" (`_bearing_vector`, `environment.py:452`),
not a fixed board-absolute compass direction. The hard engineering problem
line-route-env's design doc worried about (relearning direction per board
pose) is already solved.

**The actual bottleneck is `PCBEncoder.forward`** (`models/pcb_encoder.py:91`):
the 256x256 input becomes a 16x16 grid of patch tokens (each token = a
16x16 pixel region), and then `encoded_tokens.mean(dim=1)` collapses ALL 256
tokens into one 512-vector *before* the policy head ever sees anything.
Whatever fine-grained "which of my 12 directions is physically open one
step from here" signal channels 1/7/9 carry gets averaged together with the
rest of the board. A decision that is inherently local (what's next to the
head right now) is being made from a representation that is structurally
global. This is the same shape of problem that made 2-cycles possible in
the first place: the encoder has no way to represent "I am boxed in on 3
sides" distinctly from "the board has some obstacles somewhere."

Two prior efforts (see memory: `project_jepa_fast_lookahead`,
`project_fast_lookahead_distance_predictor`) tried to *decode* distance-to-
target from this pooled embedding and failed 4 independent times. That is a
different question from this design -- those efforts asked "can a learned
embedding reconstruct a scalar," this design instead stops throwing the
local signal away before the policy head sees it. `analytic_lookahead.py`
already proved the underlying geometry is cheap to compute directly from
the raster/env state without any decoding step; this design brings that
same directness into the network's own forward pass instead of only using
it as an external eval-time patch.

## Proposed changes (Tier 2 scope)

Three additions to `PCBEncoder` / `PCBRouterNet`, all backward-compatible
with the existing 10-channel observation and 96-action space -- no changes
to `pcbworld/environment.py`, reward, or action decoding.

### 1. Head-local token attention (replaces mean-pool as the sole signal)

Currently: `global_latent = encoded_tokens.mean(dim=1)` -- one fixed,
content-independent pooling for every board and every head position.

New: keep the mean-pool (global board context is still useful -- board
edges, overall congestion) but ADD a second vector from a learned query
that cross-attends only to the 3x3 neighborhood of patch tokens around the
head's own grid cell (head_x//16, head_y//16, clamped to [0,15]).

**Corrected from the original draft of this doc**: rather than threading a
new `(B, 2)` head-position argument through `PCBEncoder.forward` (and
therefore through every one of its ~10 external callers --
`training/train.py`, `training/evaluation.py`, `scripts/render_episode.py`,
`scripts/train_ai_router.py`, `models/analytic_lookahead.py`,
`models/fast_lookahead.py`, several `jepa/*` scripts), `PCBEncoder.forward`
recovers head position itself: Channel 3 of the observation is already a
Gaussian spot peaked exactly at `(head_x, head_y)` (`environment.py`'s
`_build_observation`), so `argmax` over the flattened channel-3 map gives
the exact grid coordinate, no new argument needed. This keeps every
external caller's code identical -- only `PCBEncoder`'s and
`PCBRouterNet`'s internals change. Verified exactly against the env's own
ground-truth head position in `scripts/verify_spatial_encoder.py`.

```
local_tokens = encoded_tokens[:, neighborhood_indices(head_cell)]   # (B, 9, 512)
local_query = self.local_query.expand(B, 1, 512)                    # learned param
local_latent, _ = self.local_attn(local_query, local_tokens, local_tokens)  # (B, 1, 512)
```

`global_latent = concat([mean_pooled, local_latent.squeeze(1)])` -> now
1024-dim into the policy/value heads (widen their first Linear
accordingly).

### 2. Explicit raycast/freespace sensor (auxiliary vector input)

**Corrected from the original draft of this doc**: `dir_idx` has **8**
distinct values, not 12. Traced by hand through `decode_action`
(`environment.py:210`): for the 96-action space, `action = dir_idx*12 +
dist_idx*4 + layer_change*2 + via_flag`, and `dir_idx = action // 12` ranges
over `96/12 = 8` values (45 degrees apart, `_bearing_vector`: `angle =
bearing + dir_idx*pi/4`, 8*45=360). The `*12` is the number of action
indices sharing one `dir_idx` (`dist_idx`(3) x `layer_change`(2) x
`via_flag`(2) = 12) -- confirmed independently by `router_policy.py`'s own
init code, which tilts `bias[0:12]` as "all actions where dir_idx==0". The
24-action space (`dir_idx*3 + dist_idx`) also has 8 dir_idx values
(`24/3=8`). So the raycast sensor is an **8-vector**, one entry per
`dir_idx`.

For each of the 8 `dir_idx` bearings, cast a ray from `(head_x, head_y)`
through channels 0+1 of the *raw* observation tensor (not the downsampled
tokens), and record the distance in cells to first collision, capped at
`max(DIST_STEPS)=8`. Output: `(B, 8)` vector, normalized to [0,1] by the
cap.

**Bearing reference, also corrected**: `_bearing_vector`'s "toward target"
direction is normally `state.smoothed_descent_dir` -- an EMA smoothed
*across steps* (`_smoothed_descent_dir`), which is per-net history not
recoverable from a single observation frame. The raycast instead uses the
*raw* (unsmoothed) spatial gradient of Channel 7 (the geodesic field) at
the head's position -- same math as `_geo_descent_dir`, minus the temporal
EMA. This can differ from the true action bearing by a few degrees, which
does not matter at 45-degree bucket resolution; `scripts/verify_spatial_encoder.py`
checks the raycast against brute-force pixel lookups using this same
raw-gradient basis (an internal consistency check of the tensor math, not a
claim of bit-exactness to the env's smoothed action frame).

This is deterministic geometry computed directly from the same obs tensor
already being built -- no environment change, no simulation, no learned
decoding. It is the network-input equivalent of what `analytic_lookahead.py`
does at decision time, so the policy no longer has to hope the pooled
embedding encodes it -- the answer is just handed to the policy/value heads
directly, concatenated alongside the rest of the latent below.

**Beyond the original scope of this doc**: implemented as a direct
`raycast_to_logit_bias: Linear(8, 8)` path in `PCBRouterNet`, added straight
onto the 8 dir_idx blocks of `action_logits` -- not only concatenated into
`policy_head`'s input. This means the "avoid a blocked direction" behavior
is structurally present from checkpoint step 0 (initialized so a lower
raycast reading directly lowers its dir_idx's logits), rather than purely
depending on the rest of the network learning that association from a
1000+ dim concatenated vector. See the confidence assessment this repo's
implementation plan recorded for why this is the highest-confidence piece
of the whole design: it isn't a "can a network learn to represent this"
bet (the shape of bet that failed 4 times previously, see memory
`project_jepa_fast_lookahead` / `project_fast_lookahead_distance_predictor`)
-- the geometry is computed fresh every forward pass, not decoded from a
trained representation.

Implementation lands in `models/pcb_encoder.py` as a pure function operating
on the input tensor (vectorized numpy or torch, no gradient needed --
detach it), called once per forward pass.

### 3. Local-crop CNN stream (native-resolution near-field detail)

The 16x16 token grid's 16px/token resolution is coarse for exact corner
geometry -- two obstacles 3px apart on the raster are invisible past that
downsampling. Add a small second CNN branch that takes a fixed-size crop
(e.g. 48x48 pixels, all 10 channels) centered on the head from the *raw*
observation, at native resolution, padded with zeros/obstacle-value at
board edges:

```
Stage 1: 48x48 -> 24x24   Conv2d(10, 32, k3, s2, p1) + GroupNorm + ReLU
Stage 2: 24x24 -> 12x12   Conv2d(32, 64, k3, s2, p1) + GroupNorm + ReLU
Stage 3: 12x12 -> 6x6     Conv2d(64, 128, k3, s2, p1) + GroupNorm + ReLU
Global avg pool -> 128-dim local_crop_latent
```

Concatenated alongside the other two signals. This is the piece that
actually resolves fine corner geometry the tokenized path structurally
cannot -- everything else in this design either re-weights or directly
reads already-downsampled information; this reads pixels.

### Combined head input

```
policy/value input = concat([
    mean_pooled_global,   # d_model  -- whole-board context (unchanged)
    local_attn_latent,    # d_model  -- coarse local (16px/token) attention
    raycast_vector,       # 8        -- exact local freespace, non-learned
    local_crop_latent,    # 128      -- fine local (native-res) CNN
])   # 2*d_model + 8 + 128 total, replaces the current d_model-dim pcb_latent
```

Width is derived from `d_model` (`models/pcb_encoder.py`'s
`combined_latent_dim(d_model)`), not hardcoded -- every script in this repo
that constructs `PCBRouterNet` uses `d_model=256` (so the real width is
`2*256+8+128=648`), except the class's own unused-in-practice default of
512 (`648` -> `1160`); `scripts/verify_spatial_encoder.py` checks both.
Widen `policy_head` and `value_head`'s first `Linear` to this computed
width; everything downstream (action_dim, value scalar) is unchanged. Also
add a small separate `raycast_to_logit_bias: Linear(8, 8)` path added
directly onto `action_logits` (not only concatenated into the shared
latent) -- see the raycast section above and the confidence assessment for
why.

## What does NOT change

- `pcbworld/environment.py` -- observation channels, reward, action
  decoding, geodesic field computation: untouched.
- Action space: still 24 or 96 discrete actions depending on stage
  (`enable_layer_via`), still egocentric dir_idx (8 values either way).
- `NetSelectorHead`: untouched (separate concern, multi-net ordering).
- `analytic_lookahead.py` / `lookahead_select_action`: kept as fallbacks.
  This design's goal is for the plain deterministic policy to stop needing
  them for corner cases, not to remove them -- they stay as the safety net
  exactly as `--lookahead` already is documented to be, regardless of
  outcome (see memory: `project_analytic_lookahead`).

## Checkpoint / training implications

The policy/value heads' input dimension changes, so
`checkpoints_stage2_v7/single_net_router_latest.pt` cannot be loaded into
the new architecture directly -- this requires training a fresh checkpoint,
not fine-tuning the existing one. Same trainer
(`training/train.py`'s `train_single_net_policy`, invoked via
`scripts/train_ai_router.py`), same PPO hyperparameters, same reward -- only
the network forward pass changes. **Correction**: an earlier draft of this
doc said "same curriculum entry point (`scripts/train_curriculum.py`)" --
that file is currently mid-repurposing (uncommitted) for an unrelated,
deliberately parked vector/line-segment routing effort (see memory
`project_pcb_router_workflow`) and is not the raster-track's trainer at
all; this design does not touch it. Expect a comparable or slightly
higher per-step cost than the current network (local-crop CNN + attention
add real but small FLOPs versus the transformer already in place; raycast
is cheap vectorized geometry, not a network) -- much smaller than
`lookahead_select_action`'s top_k*horizon env-simulation cost.

## Validation plan (same discipline as `analytic_lookahead`)

1. **Local, no GPU needed, DONE**: `scripts/verify_spatial_encoder.py`
   asserts (a) head-position recovery (argmax of Channel 3) exactly matches
   the env's own ground-truth head position, (b) the vectorized raycast
   sensor's per-direction distances exactly match an independently-written
   brute-force Python pixel-walk along the same 8 bearings, and (c) the
   local-crop extraction is pixel-exact at board edges/corners (head near
   x=0/255 or y=0/255, tested explicitly, not left to chance from random
   episodes) -- same discipline as `scripts/verify_analytic_lookahead.py`'s
   11,804/11,804 check, plus a full `PCBRouterNet` forward-pass shape/
   finiteness check for both `d_model=256` (what every script actually
   uses) and `d_model=512`. **Result: 12,000/12,000 decisions matched
   exactly** (150 episodes x up to 80 steps), all three geometry checks,
   plus the 8 explicit edge/corner positions and all 3 forward-pass
   configs -- zero mismatches. Also confirmed the existing
   `tests/test_grid_router.py` suite
   (4 tests, predates this change, uses a third `d_model=128` config) still
   passes unmodified.
2. **Colab retrain, DONE (2026-08-25)**: fresh stage-2 run (30k steps,
   Tesla T4, checkpoint `checkpoints_stage2_v8_spatial`), plain
   `select_deterministic_action` only, no lookahead wrapper. Reached 100%
   on the rolling training window by ~step 5000 -- notably fast, consistent
   with the raycast->logit-bias path being structurally present from init
   rather than something the network had to discover from scratch (see
   the confidence assessment above).
3. **Eval, DONE**: full 1000-board benchmark (seeds 9000-9999). **Plain
   deterministic argmax alone: 99.90% (999/1000)** -- this is the win this
   design targeted: the OLD architecture needed `--lookahead`
   (~16x-slower-per-step real env-simulation) to close the same gap up to
   checkpoint-v7's 100%/1000. Single remaining failure: seed 9764, one of
   the 10 known hard seeds (9648, 9681, 9764, 9779, 9148, 9251, 9091, 9390,
   9535, 9901) -- 9/10 of those are now solved without any wrapper.
   `--analytic-lookahead` on this SAME checkpoint scored 99.60% (996/1000,
   worse) failing on a disjoint seed set (9111, 9170, 9450, 9677) -- an
   unexpected interaction between analytic_lookahead's replay and this
   policy's own action ranking, not yet root-caused (see memory
   `project_spatial_world_model` / `project_analytic_lookahead`). Not
   urgent since plain alone already beats what used to require any
   wrapper, but means `--analytic-lookahead` should not be assumed
   better-or-equal by default on this checkpoint the way it was on v7.
   `--lookahead` (the original, proven-but-slow wrapper) was not
   separately re-run against this checkpoint yet.

## Addendum: per-(direction, distance) collision reduction

Added after the Colab result above, in response to a follow-up ask: even
though plain deterministic hits 99.90%/1000, the eval trace for the one
remaining failure (seed 9764, via `render_episode.py --verbose`) showed
many REJECTED-collision steps even on nets that eventually succeed --
e.g. the same position rejecting three different actions in a row before
finding a legal one. That is a **granularity** problem in the raycast fix
above, not a missing-information one: `raycast_to_logit_bias` discriminates
only by `dir_idx` (8 values), so it applies the identical bias to all 3
`dist_idx` choices (step sizes 2/4/8) within a direction. A direction that
is clear for 2 cells but blocked at 8 reads as "fine" at the direction
level while still colliding at the specific distance an action actually
tries -- exactly the observed failure signature.

Fix: `PCBEncoder._raycast_sensor` already computes, internally, whether
each of the 8 sampled cells out to `RAYCAST_MAX_STEPS` is blocked, before
collapsing that into the single per-direction `raycast` scalar. It now
ALSO returns `dist_safe`: a `(B, 8, len(DIST_STEPS))` boolean -- for each
direction AND each of the environment's actual 3 step distances, whether a
hop of exactly that length lands before the first blocked cell. This is
concatenated into the combined latent (24 more floats: `DIST_SAFETY_DIM =
RAYCAST_NUM_DIRS * len(DIST_STEPS)`) AND, more importantly, added as a
second direct bias in `PCBRouterNet` -- `dist_safe`'s flattened
`(dir_idx, dist_idx)` ordering lines up exactly with both the 24-action
space's `dir_idx*3 + dist_idx` encoding and (via `repeat_interleave` over
the 4 layer/via combos) the 96-action space's `dir_idx*12 + dist_idx*4 +
...` encoding, so no remapping is needed.

Unlike `raycast_to_logit_bias` (a learned `Linear`, deliberately given a
strong-but-adjustable init), this new bias uses a **fixed, non-learned**
constant (`DIST_SAFETY_SUPPRESSION = 8.0`, in `pcb_encoder.py`): 0 for a
safe `(dir_idx, dist_idx)`, `-8.0` for one the raycast proves will collide.
This project has direct prior history (see `router_policy.py`'s
policy_head-init comment) of a learned bias becoming negligible once the
weight-driven logits it competed with grew large during training -- a
fixed additive term added at the very end of `forward()`, after
`policy_head`, cannot be trained away the same way. If every option at a
fully boxed-in cell is unsafe, the constant is added uniformly across all
of them, which changes nothing (a uniform shift never changes
softmax/argmax) -- it only discriminates when it has real information to
add, so it cannot force a bad choice in a genuinely no-good-options cell.

**Scope note, same honesty as the raycast bearing caveat above**: this
targets *self-inflicted collisions* (repeatedly proposing an action the
policy's own geometry already proves illegal) -- a different failure class
from seed 9764's actual root cause (a policy that needs to see 4 steps
ahead to find a non-obvious corridor, which only `--lookahead`'s real
multi-step simulation currently provides; see the Colab result above and
memory `project_spatial_world_model`). This addendum should reduce wasted
collision-retry steps and may improve wall-clock/steps-per-net efficiency,
but is not expected to be what closes seed 9764 specifically -- that
remains `--lookahead`'s job, a structurally different (external,
decision-time search) mechanism, not something built into the model's
forward pass at all.

**Verification**: `scripts/verify_spatial_encoder.py` extended with a
fourth check (`dist_safe` against an independently-written brute-force
per-(direction,distance) reference). **Result: 12,000/12,000 decisions
matched exactly** across all four checks (150 episodes x up to 80 steps),
same as the original spatial-encoder validation. Requires a fresh retrain
(new bias mechanism, not loadable into `checkpoints_stage2_v8_spatial`) to
measure the actual collision-rate effect on Colab.

## File structure (new / modified)

```
models/
  pcb_encoder.py       MODIFIED: add local-attn pool, raycast fn, crop CNN
  router_policy.py      MODIFIED: widen policy_head/value_head input dim
scripts/
  verify_spatial_encoder.py   NEW: local geometry validation, no GPU
docs/
  WORLD_MODEL_SPATIAL_DESIGN.md   THIS FILE
```

## Risks

- **Retrain cost**: this is a from-scratch retrain on Colab, not a
  fine-tune -- full curriculum time again, not a quick patch.
- **Params/compute increase**: local-crop CNN + attention add real
  parameters; if Colab's 2-vCPU-class budget matters for the *other*
  parked thread (`docs/UNIFIED_RL_DESIGN.md`'s CPU-only design), note this
  spatial world model targets the GPU-trained raster pipeline
  (`pcbworld/environment.py` + `models/router_policy.py`), not that
  CPU-only vector approach -- the two are unrelated, per
  `project_pcb_router_workflow`'s note that the vector-based thread is
  parked separately.
- **Might not fully close the gap alone**: if corner-stuck cases need more
  than 1-step local reasoning (e.g. a dead-end 3 cells deep only visible
  past the 48x48 crop), the raycast/crop give the network the INPUT to
  learn a better local policy, but training still has to actually learn to
  use it -- validation step 2/3 is what confirms this, not the design
  alone.
