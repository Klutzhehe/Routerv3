# NeuroRoute — a pure-RL router for many-net, many-layer PCBs

Target: **thousands of nets, 6–8 layers, learned differential pairs, learned
length tuning**, variable trace widths, variable via sizes, copper pours.

Verification tags follow the repo convention (`docs/ROUTER_CAPABILITIES.md`):

- **[LIVE]** — measured against a real run, with numbers.
- **[LOCAL]** — verified by a local check script, no GPU / no KiCad.
- **[UNVERIFIED]** — written, never run.

Everything in this document is **[UNVERIFIED]** unless tagged otherwise.
Local verification scripts (`neuroroute/scripts/verify_*.py`) upgrade
specific claims to **[LOCAL]**; Colab upgrades them to **[LIVE]**.

---

## 0. Why the existing two threads cannot get there

Both existing directions are good work that dead-ends short of this target,
for reasons that are already measured in this repo — not opinions.

### The PNS-bridge thread (`pcbworld/env/line_route_env.py`)

| Requirement | Reality |
|---|---|
| 6–8 layers | `switch_layer()` is **0-for-32** [LIVE]; `toggle_via_placement()` places a via but a mid-route `fix(force_finish=False)` is rejected outright, so the route never reaches the far layer. Three sessions have gone here. |
| Thousands of nets | CPU-only, `nproc`=2. 0.86 ms/net median is fine for 24 nets; a 2000-net board is ~10⁵–10⁶ sequential PNS calls per episode, single-threaded, with no batching path (`PNS::ROUTER` has none, anywhere). |
| Learned diff pairs / tuning | The only working path through PNS is `MODE_ROUTE_DIFF_PAIR` / `MODE_TUNE_SINGLE` — the engine's own solvers. **You have ruled these out**, and the ruling is right: a policy that calls them has learned *where to invoke a solver*, not how to route a pair. |

`docs/UNIFIED_RL_DESIGN.md` is built on those two primitives (§6, stages 4–5)
and on `switch_layer()` staying closed. **It is superseded by this document.**

### The raster-grid thread (`pcbworld/environment.py`)

Genuinely works — 99.90%/1000 boards on single-net stage 2 with plain argmax
[LIVE]. But it is architecturally single-head, single-net-at-a-time, 2 layers,
fixed track width, fixed via size, no diff pairs, no length targets, no pours,
and one Python-loop environment per process. The parts of it that are *right*
are carried forward here, explicitly, in §6.

### The four failed lookahead attempts

`jepa/`, `models/fast_lookahead.py`, and two probe rounds all asked the same
question: **can a scalar (distance-to-target) be decoded from a globally
mean-pooled board embedding?** Four independent negative results. The
diagnosis in `docs/WORLD_MODEL_SPATIAL_DESIGN.md` is correct and is the single
most useful finding in the repo: `encoded_tokens.mean(dim=1)` destroys the
spatial structure *before* anything downstream can use it.

**This design does not retry that bet.** See §5 for what "look into the
future" means here instead, and why it is a structurally different question.

---

## 1. The three ideas the architecture rests on

**1. A gridded routing lattice, not a rasterised image.** **[LIVE -- verified
against KiCad 9.0.2, see section 12.]**
The repeated objection to rasters in this repo — "256 px over 50 mm is
0.195 mm/px and clearance is 0.2 mm, the legality margin is sub-pixel"
(`docs/RL_PLAN.md`) — is an objection to *rasterising continuous geometry*.
It does not apply to a **routing lattice**, where one cell *is* one routing
track slot and the pitch is **defined** as `min_width + min_clearance`. On a
50 mm board with 0.2 mm traces and 0.2 mm clearance that is 0.4 mm pitch =
125 tracks per direction — a real, physically meaningful number, not an
approximation of one. Legality becomes cell occupancy, which is exact within
the lattice model, and *by construction* satisfies the clearance rule that
defined the pitch. Wide traces occupy several adjacent cells laterally; that
is what the width class means.

This is what gridded commercial routers do, and it is the only representation
that is simultaneously (a) exact enough to be legal, (b) dense enough to
convolve over, and (c) cheap enough to batch on a GPU.

**2. The environment must live on the GPU, batched over boards.**
Every stuck thread in this repo shares one root cause: **sample starvation**.
One CPU environment at ~0.03–0.7 ms/step, 2 workers, means ~10⁶ steps takes
hours and a 2000-net board never completes an episode at all. `NeuroRoute`'s
world is a set of torch tensors and its `step()` is pure tensor algebra — no
Python loop over nets, no loop over cells. `B=64` boards × `K=8` simultaneously
active nets = **512 routing decisions per forward pass**. That is a
~500× throughput change, and it is the actual unlock. KiCad becomes the
*validator* (§7), which is what it is uniquely good at.

**3. The scaling story is architectural, not a bigger board.**
Nothing in the model has a board-size-dependent parameter. The field encoder
is fully convolutional; net attention is local (bucketed by G-cell); the action
frame is egocentric. **Train on 128×128×8 lattices with 50–300 nets; run
inference on 1024×1024×8 with 3000.** If that generalisation does not hold, it
is visible immediately as a held-out-scale eval number, not after a month.

---

## 2. World model (the environment)

### State

All tensors, batch dim `B` over independent boards.

| Tensor | Shape | Meaning |
|---|---|---|
| `occ` | `(B, L, H, W)` int16 | Occupancy. `0` = free, `n>0` = owned by net `n-1`, `-1` = keepout/board-edge/pour |
| `pad_mask` | `(B, L, H, W)` int16 | Pad cells, same net encoding. Obstacles from step 0 — the thing that made `net_0` fail on an empty board [LIVE] |
| `nets` | `(B, N, F)` float | Net table: src/dst cell + layer, kind, width class, required length, pair partner, group id, status |
| `heads` | `(B, K, 6)` int | The `K` currently-active routing heads: net idx, x, y, layer, steps, phase |
| `routes` | `(B, N, V, 3)` int16 | Per-net polyline vertices (padded). The refine phase edits these |

`L` is a real parameter (2 → 8, curriculum). Vias are first-class: a via at
`(x,y)` spanning layers `[a,b]` marks `occ[b_i, a:b+1, y±r, x±r]`, so blind /
buried / through vias are all the same operation with different spans.

### Actions — factorised, not a product space

Six heads sampled jointly; log-probs sum. A flat product space would be
`8 × 4 × 9 × 4 × 4 × 2 ≈ 9216` actions, which is unlearnable; factorised it is
`8 + 4 + 9 + 4 + 4 + 2 = 31` logits.

| Head | Values | Notes |
|---|---|---|
| `direction` | 8 | **Egocentric**: index 0 = down the geodesic gradient toward the target. Carried over from `pcbworld/environment.py` — proven [LIVE] |
| `step` | {1,2,4,8} cells | Long steps are the difference between 150-step and 30-step nets |
| `layer` | stay / go-to-layer-`ℓ` (L options) | A layer change *is* a via. No separate "place via" action |
| `via_class` | 4 | Via diameter/drill from the design-rule table. Only read when `layer ≠ stay` |
| `width_class` | 4 | Track width from the design-rule table |
| `couple` | 2 | Diff-pair only: keep the two legs locked, or split them |

Plus two **board-level** heads that run once per env step, before the per-head
actions:

| Head | Output | Purpose |
|---|---|---|
| `scheduler` | pointer over pending nets | Which `K` nets are active. Learned ordering, not a heuristic — at 2000 nets, ordering is most of the problem |
| `ripup` | pointer over routed nets ∪ {none} | Remove a completed net's copper and return it to the pending pool |

`ripup` is the density lever the repo already identified as necessary once
layer changes closed. Here layers *and* rip-up are both available.

### Termination

A head terminates when it reaches its target pad cell (auto-snap, no learned
stop action — the `snap_radius ≥ step/2` lesson from `docs/HANDOVER.md`
applies and is enforced in `engine.py`), when it exhausts its step budget, or
when the scheduler deactivates it.

---

## 3. Differential pairs, learned

**Not** `MODE_ROUTE_DIFF_PAIR`. A pair is one net token with **two heads** and
a `couple` action bit:

- **Coupled** (`couple=1`): the sampled `direction`/`step` drives the pair's
  centreline; the two legs are placed at `± gap/2` perpendicular to travel.
  The env checks both legs for legality and rejects the move if *either*
  collides. Both legs advance or neither does.
- **Split** (`couple=0`): the legs are routed as two independent heads for
  that step — which is what you need to get around a via field or a pin
  escape, and is exactly the behaviour a solver-call cannot express.

The reward makes coupling *worth* choosing rather than mandatory:

```
r_pair = − w_gap  · mean|actual_gap − nominal_gap| / nominal_gap
         − w_skew · |len_P − len_N| / L_ref
         − w_split · (fraction of pair length routed uncoupled)
```

So the policy has to learn **where coupling is affordable and where it is
worth breaking** — the actual engineering skill — instead of learning where to
call a function that always couples. `w_split` is small: splitting is a real
technique, not a failure.

---

## 4. Length tuning, learned

**Not** `MODE_TUNE_SINGLE`. Once a net is topologically connected it enters
the **refine phase**, a second MDP over the *same* board state:

```
action = (vertex_index_bucket, perpendicular_offset δ ∈ {−A..+A}, subdivide?)
```

The route is a polyline. The action **drags one vertex sideways** — literally
what a human does in the KiCad editor. The env auto-subdivides any segment
longer than `s_max` so vertices are always available to drag. Legality is
checked the same way as the connect phase; an illegal drag is rejected.

```
r_refine = Φ(|len − target|) shaping   +  terminal bonus inside tolerance
           − w_len · |len − target| / target
           − w_dr  · DRC violations introduced
```

A meander is not a primitive here. **A meander is what an optimal policy of
this MDP looks like** — alternating drags of adjacent vertices, with amplitude
and spacing chosen by the policy. That is the strongest form of "the policy
learned it" available, and it is dense-reward and short-horizon, so it is
actually trainable.

The refine phase is not only for length. The same action set does wirelength
reduction, diff-pair gap repair, and clearance-margin improvement — one MDP,
three jobs.

> **Honesty note.** Vertex-drag is a *geometric* action, not a routing
> primitive, but it is a design choice worth naming: it is the smallest action
> that can change route length without re-solving connectivity. The stricter
> alternative — force length matching to emerge from the connect phase alone
> with a length term in the reward — is available (`--no-refine-phase`) and is
> a legitimate ablation. It will be much harder to train and I expect it to
> lose; if it wins, that is a real finding and the refine phase should go.

---

## 5. "Looking into the future" — what it actually means here

This is the part the four prior negative results are about, so it needs to be
precise about *what* is being predicted and *in what shape*.

### What failed, and why it was the wrong question

| Attempt | Predicted | From | Result |
|---|---|---|---|
| `jepa/` ×3 | distance-to-target (scalar) | mean-pooled board embedding | negative |
| `models/fast_lookahead.py` | distance-to-target (scalar) | mean-pooled board embedding | negative |

A scalar target gives one gradient per sample. A pooled input has already
destroyed the spatial structure the scalar depends on. The bet was that a
bottleneck would *preserve* information it was never trained to preserve.

### What is predicted instead: dense spatial fields

`FutureFieldPredictor` consumes the encoder's **spatial** latent
`z ∈ R^{D×L×H×W}` — never a pooled vector — and emits three fields at the
same resolution:

| Field | Target | Loss | Why it is decodable |
|---|---|---|---|
| `Ô_final` | occupancy of every cell when the board is **finished** | BCE | `L·H·W` targets per sample instead of 1. Free labels: the terminal board state of the rollout you already ran |
| `Ĉ_contend` | how many still-unrouted nets will want to cross this cell | Poisson NLL | Ground truth countable from the finished board |
| `Ĵ_jam` | probability a net needing this cell will **fail** | BCE | Labelled from actual failures in the rollout |

Three properties make this a different bet from the four failures:

1. **The output is spatially aligned with the input.** It is a segmentation
   problem. No pooling anywhere on the path.
2. **The supervision is dense** — ~10⁶ labelled cells per rollout, not 1.
3. **The labels are free and on-policy.** Every episode that finishes emits
   its own ground truth. No separate data collection step (which is where
   `jepa/collect_transitions.py` spent its time).

### How the policy consumes it

The three fields are concatenated as extra channels into the policy head's
spatial input, **gradient-detached** so the RL objective cannot corrupt the
forecaster and vice versa. The geometry head then sees, per candidate cell,
*"this will be contended later"* and *"a net that needs this will fail"* — at
decision time, for free, in one forward pass.

This is the thing the repo has been missing. Every failure in the
`docs/HANDOVER.md` open questions — *15/24 nets unreachable*, *first nets block
later nets*, *straight-line is not the best completion strategy* — is one
failure: **a greedy router with no model of future demand.** The forecaster is
a model of future demand.

### Stage 2 (gated): value-equivalent latent rollout

Once the forecaster is measurably working, add a latent transition model
`z_{t+1} = f(z_t, a_t)` — but trained by **value equivalence** (predict future
reward, value, and action legality `k` steps out), never by reconstruction.
This gives a shallow MuZero-style imagined search at decision time and would
replace `models/analytic_lookahead.py`'s real-env replay (which measured
*worse* than plain argmax on v8_spatial [LIVE], root cause still open).

**Explicitly gated.** If `Ô_final` does not beat a straight-line-demand
baseline on held-out boards, this stage does not start. That gate exists so
this design cannot become negative result #5 by momentum.

---

## 6. What is carried forward from the existing repo

Not a rewrite for its own sake — these are the parts with measured wins.

| Carried forward | Why | Tag |
|---|---|---|
| **Non-learned raycast → fixed logit suppression** | `DIST_SAFETY_SUPPRESSION`: Rejected-Action Rate 1.51% → 0.40%. Geometry computed fresh every forward pass; cannot be trained away | [LIVE] |
| Extended here to **(direction × step × layer)** | The v11 lesson was that per-direction granularity is too coarse. With a layer action the same argument extends one more axis | — |
| **Egocentric action frame** (dir 0 = toward target) | Removes board-pose generalisation entirely | [LIVE] |
| **Near-zero actor init** | An untrained policy ≈ the greedy baseline, so training starts *at* the baseline, not below it | [LIVE] |
| **Geodesic (obstacle-aware) distance field**, not Euclidean | `compute_geodesic_distance_field`; re-implemented as a batched GPU relaxation | [LIVE] |
| **Potential-based shaping** `Φ = −geodesic/L` | Policy-invariant, so it cannot bias the optimum | [LIVE] |
| **Rejection-feedback / dead-zone channels** | Within-net feedback that a policy can react to now, not via the reward gradient over many episodes | [LIVE] |
| **Track completion rate, not reward** | Measured: random scored −330 reward vs greedy's −177 and still completed *more* nets | [LIVE] |
| **Render every failed episode** | A contact sheet of failures shows the failure mode; a reward curve never will | [LIVE] |

And the mistakes recorded as constraints, which the new engine avoids by
construction: `snap_radius ≥ step/2`; pad obstacles need `size_x/size_y`;
never call the full-board DRC per step.

---

## 7. KiCad's role — validator, not environment

The bridge stays, with a narrower and more defensible job:

| Job | Call | When |
|---|---|---|
| **Ingest** a real board → `BoardSpec` | `net_pads`, `get_board_geometry`, `get_design_rules` | Once per board, subprocess |
| **Export** a routed lattice → `.kicad_pcb` | grid → mm, emit tracks + vias | End of episode |
| **Validate** | `run_drc()` — KiCad's real `DRC_ENGINE` | Eval only, never per step (267 ms, 73% of engine time [LIVE]) |

The number that matters is the **sim-to-real gap**: DRC violations per 1000
nets on boards the fast engine declared clean. The lattice pitch is *defined*
as `width + clearance`, so the gap should be ~0 by construction; measuring it
is how we find out where that reasoning is wrong. It is the first eval to run
and it needs no trained policy at all — route a board with the greedy baseline
and DRC it.

Hard constraints from `ROADMAP.md` still apply in full: bridge and system
`pcbnew` never share a process; subprocess for anything needing `pcbnew`.

---

## 8. Network

```
                      occ / pads / pours / demand / forecast       nets (N tokens)
                                  │                                     │
                    ┌─────────────▼─────────────┐            ┌──────────▼──────────┐
                    │  FieldEncoder (3D U-Net)  │            │  NetEncoder (MLP)   │
                    │  (1,3,3) convs in-plane   │            └──────────┬──────────┘
                    │  (3,1,1) convs cross-layer│                       │
                    │  axial attention over L   │◄──── cross-attn ──────┤
                    └─────────────┬─────────────┘                       │
                        z: (B,D,L,H,W)                                  │
              ┌───────────────────┼───────────────────┐                 │
              ▼                   ▼                   ▼                 ▼
    FutureFieldPredictor    head-local gather    global pool      scheduler head
    Ô_final Ĉ_contend Ĵ_jam   (K heads)          (board ctx)      ripup head
              │                   │                   │
              └────── detach ─────┴───────────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │  ActionHeads (factorised)   │  + raycast logit suppression
                    │  dir/step/layer/via/width/  │    (fixed, non-learned)
                    │  couple  +  V(s)            │
                    └─────────────────────────────┘
```

Why 3D and not 2D-per-layer: a via is a **cross-layer** event and layer choice
is most of the 8-layer problem. `(3,1,1)` convs plus axial attention over `L`
(cheap — `L ≤ 8`) let the encoder represent "layer 3 is congested here but
layer 5 is open two cells over," which is precisely the decision a via is.

Size: ~8–15 M parameters at `D=96`, `L=8`, `H=W=128`. GPU-resident with the
environment, which is the whole point — no CPU↔GPU round trip per step.

---

## 9. Curriculum

Auto-advance on held-out completion rate, not on reward.

| Stage | Board | What is new | Gate |
|---|---|---|---|
| 0 | 1 net, empty, 2 layers | plumbing | ~100% or the plumbing is wrong |
| 1 | 20 nets, 2 layers | congestion, ordering | > greedy baseline |
| 2 | 20 nets, **8 layers** | vias, layer choice | > stage 1 completion |
| 3 | 200 nets, 8 layers | scale, scheduler | > 95% |
| 4 | + variable widths / via classes | rule-aware geometry | ≤ 1% width violations |
| 5 | + **diff pairs** | `couple` head, gap/skew reward | > 80% pairs inside gap tolerance |
| 6 | + **length groups** | refine phase | > 80% groups inside tolerance |
| 7 | + copper pours | pour-as-obstacle, pour-as-terminal | no regression |
| 8 | 1000–3000 nets, held-out scale | generalisation | the actual target |

Stage 3 → 8 is the generalisation claim (§1.3) and is the highest-risk step.
It is deliberately reachable early: nothing between stages 3 and 8 changes the
model, only the data.

---

## 10. Risks, stated up front

| Risk | Why it might bite | Signal that it is biting | Mitigation |
|---|---|---|---|
| **Lattice ≠ KiCad legality** | Pitch reasoning could be wrong at corners, pad edges, or 45° | Sim-to-real DRC gap > 0 | Measured in §7 *first*, before any training. Fix by widening the pitch or the dilation rule |
| **Forecaster is negative result #5** | Same family of bet, different shape | `Ô_final` fails to beat straight-line demand | Hard gate before stage 2. Policy still trains without it |
| **Scheduler + geometry is too much at once** | Two learned pointer problems | Stage 3 plateaus | Fall back to shortest-first ordering (proven), keep geometry learned |
| **Scale generalisation fails** | 128→1024 is 64× the cells | Stage 8 collapses vs stage 3 | Tile inference into overlapping windows; the encoder is convolutional so this is legal |
| **Reward hacking on refine** | Drag vertices to hit length while adding DRC risk | Length good, DRC bad | DRC term is in the refine reward, and real KiCad DRC is the eval |
| **Diff-pair split abuse** | `couple=0` everywhere avoids the hard case | Split fraction → 1.0 | `w_split` penalty + report split fraction as a first-class metric |
| **Colab session death** | Known | — | Checkpoint every N updates to Drive (non-negotiable, per `RL_PLAN`) |

---

## 11. Build order

1. `world/` — lattice engine + generator. **Verify locally**, no GPU, no KiCad:
   occupancy/clearance/via/raycast exactness against brute force, the same
   discipline as `scripts/verify_spatial_encoder.py`'s 12,000/12,000.
2. `eval/kicad_bridge.py` — export + DRC. **Measure the sim-to-real gap with
   the greedy baseline before training anything.**
3. `models/` + `training/ppo.py` — stage 0–1 on Colab.
4. Forecaster aux loss; run its gate.
5. Stages 2–3 (layers, scale).
6. Diff pairs, then refine phase / length groups.
7. Pours, held-out scale eval.

Step 2 is deliberately before step 3. If the lattice is not legal, nothing
trained on it matters, and that is a one-afternoon question.


---

## 12. What was measured while building this

Added after the build. Tags as in the header: **[LIVE]** = real numbers,
**[LOCAL]** = a check script, **[UNVERIFIED]** = written, never run.

### The lattice claim survived contact with KiCad -- **[LIVE]**

Section 1's first idea is the one everything else rests on, and it is now
checked against **KiCad 9.0.2's own `DRC_ENGINE`** rather than argued for:

```
4 configs: 64x64 to 112x112 cells, 4 and 8 layers, 0-50% wide traces, 0-4 keepouts
SIM-TO-REAL GAP: 0 legality violations over 192 routed nets, all four configs
```

Four configurations rather than one, because the diagonal defect below was
clean on the first board set and failing on the second.

It did **not** pass first time, and the failure was informative. The first run
returned 479 clearance violations, every one of them pad-to-track and every one
of them reporting *actual 0.1000 mm* against a 0.2 mm rule. Cause: the lattice
reserved **one cell** for a pad while the exporter drew it a full 0.4 mm wide,
so a trace in the neighbouring cell sat 0.4 - 0.2 - 0.1 = 0.1 mm from the pad
edge. The pitch reasoning was right; the *pad* was the thing that had never
been given a size. Both now derive from `DesignRules.pad_size`, so they cannot
drift apart again.

This is exactly why section 11 puts the DRC gate before training. Had it run
after, every completion number up to that point would have been measured
against copper KiCad rejects.

### Multi-layer routing works -- **[LIVE]**

The capability `switch_layer()` is **0-for-32** on after three sessions:

| layers | completion | vias/board |
|---|---|---|
| 1 | 27.3% | 0.0 |
| 2 | 32.8% | 7.2 |
| 8 | 32.8% | 7.2 |

Connectivity is re-derived by flood fill **through the vias**, not taken from
a status flag: 42/42 legs verified.

2, 4 and 8 layers scoring identically is a property of the *baseline*, which
hops to the nearest better layer and so rarely uses more than two. Whether more
layers keep paying is a question for the learned layer head, and it is exactly
what stage 3 measures.

### Baselines to beat, held-out seeds -- **[LIVE]**

| Board | greedy | detour | layer_hop |
|---|---|---|---|
| 1 net, empty, 2 layers | 75.0% | 75.0% | **87.5%** |
| 20 nets, 2 layers | 28.1% | 28.1% | **42.5%** |
| 60 nets, 8 layers | 16.3% | 16.3% | **24.6%** |

Measured after the corner-guard fix, which made every segment strictly more
conservative; the earlier, higher numbers included the illegal geometry it
removed. The `greedy` -> `layer_hop` gap is precisely the nets whose pads sit
on different layers: `greedy` never places a via and therefore *cannot*
finish them.

### The refine phase is built and mechanically verified -- **[LOCAL]**

Section 4 is implemented (`BatchedRouterWorld.refine`) and checked by
`scripts/verify_refine.py`: drags change routed length, routes stay connected
through them (flood fill, not bookkeeping), rejections restore the board
byte-identically, and repeated alternating drags accumulate length -- a meander,
from ordinary actions, with no generator. What is **not** yet built is the
policy head that chooses those drags; the action set is proven, the learning on
top of it is not.

### Bugs worth not re-deriving

1. **Simultaneous heads could write the same cell.** With `K` heads acting in
   one batched step, two heads both passed a legality check against the
   pre-step occupancy and both wrote; one silently lost and its head advanced
   anyway, leaving a route with a hole in it. Found by the flood-fill check
   (30/31 legs connected), never by a reward curve. `step()` is now
   plan -> arbitrate -> commit.
2. **Two direction frames drifted apart.** The raycast reports absolute
   directions; moves resolve egocentrically. The observation was handing the
   policy an unrotated safety mask. Symptom: an **86% rejected-action rate**
   from a baseline that believed it was only taking moves the raycast had
   called safe. After the rotation: **1.6%**.
3. **Boards with zero nets scored 0%.** The generator gave up quietly when
   component placement ran short of pins, and `completion()` scored the result
   0% rather than "vacuously complete" -- so a generator bug was hiding inside
   what looked like a routing metric.
4. **Two copper-writing paths disagreed about diagonal clearance.** Lattice
   moves guard the two cells a 45-degree trace passes between; the
   arbitrary-segment path used by snap-to-pad and refine did not. Real KiCad
   clearance failures at 0.0828 mm, on some board sizes but not others.
5. **Untrained action defaults nobody had chosen.** A zero-bias width head
   picked 3-cell-wide traces on 612 of 627 actions (88% rejected); an
   unsuppressed layer head attempted mostly-impossible through vias about half
   the time (92.6% rejected). Both destroyed the "untrained policy == greedy
   baseline" property that the egocentric frame exists to provide.

They share a shape: **a metric that looked plausible while the thing under
it was broken.** That is the argument for `verify_env.py` re-deriving its
answers from the occupancy grid instead of trusting the engine's own flags.

### Still unverified

Everything about *learning*. The policy's forward and backward passes, the PPO
update, checkpointing and the forecaster gate all execute, and the gate
correctly reports "does NOT beat baseline" on an untrained model -- but no real
training run has happened. Section 10's risk table stands unchanged for every
row except the first.
