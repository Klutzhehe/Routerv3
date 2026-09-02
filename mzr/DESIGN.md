# MZR — simultaneous-growth PCB routing with Sampled/Gumbel MuZero

Target: **thousands of nets, 6–8 layers, learned differential pairs, learned
length tuning**, variable trace widths, variable via sizes, copper pours — with
every net growing **at the same time** rather than one after another, and a
learned latent model that lets the router *imagine* how the board plays out
before committing copper.

Verification tags follow the repo convention (`docs/ROUTER_CAPABILITIES.md`):

- **[LIVE]** — measured against a real run, with numbers.
- **[LOCAL]** — verified by a local check script, no GPU / no KiCad.
- **[UNVERIFIED]** — written, never run.

**Everything in this document is [UNVERIFIED] unless tagged otherwise.**
Claims tagged [LIVE] or [LOCAL] are inherited from `neuroroute/` and carry
their original evidence; they are re-verified in this package before being
relied on.

---

## 0. What this replaces, and what it keeps

This is a **clean rewrite** of the routing agent. It supersedes
`neuroroute/DESIGN.md`, which in turn superseded `docs/UNIFIED_RL_DESIGN.md`.

It does **not** discard `neuroroute/`'s substrate work. The lattice geometry,
the KiCad bridge, and the board generator are **ported with their verification
scripts intact** (§10). Re-deriving those from zero would mean re-discovering
the same three DRC bugs (§11) for no upside.

### Why a rewrite rather than an extension

`neuroroute/` plateaued at **60–65% completion** against a 75% gate, across
many Colab sessions [LIVE]. Every failure symptom in its handover — *"first
nets block later nets"*, *"15/24 nets unreachable"*, *"straight-line is not the
best completion strategy"* — is one disease: **it routes greedily, net by
net.** `K=8` heads were chosen by a scheduler that received **zero gradient on
every run in that project's history** [LIVE], so in practice early nets
committed copper and later nets had to survive whatever was left. No net ever
yielded. No net had a mechanism to act on "another net needs this channel more
than I do."

That is not a hyperparameter problem. It is the coordination model, and it is
load-bearing enough to be worth rebuilding around.

### The user constraint that governs everything

Three tools, all **learned**: routing, differential pairs, length tuning. KiCad
PNS's own solvers (`MODE_ROUTE_DIFF_PAIR`, `MODE_TUNE_SINGLE`) are **ruled
out** — a policy that calls them has learned *where to invoke a solver*, not
how to route. KiCad's role is ingest, export, and DRC validation only (§9).

Specifically ruled out for the same reason: **push-and-shove**. If the engine
shoves existing copper aside to make room, the policy never has to learn to
leave room. Nothing bails the agent out here; spacing has to be learned.

---

## 1. The reframe that makes MuZero viable

MuZero on a **sequential** router is a bad bet, and it is worth stating why so
nobody re-proposes it later:

```
sequential episode length  ≈  Σ (all net lengths)  ≈  10,000+ steps
```

MuZero's latent dynamics model degrades over long rollouts. It shines on
short-horizon problems. At 10,000 steps it has nothing to offer.

But **simultaneous growth collapses the horizon**. If every net's frontier
advances together in macro-steps:

```
macro-episode length  ≈  max_net_length / mean_step  ≈  120 / 2.5  ≈  ~48 steps
```

— and that number is **independent of net count**. 20 nets or 3000 nets, the
episode is ~48 macro-steps deep. This is the structural insight the whole
design rests on:

> **Parallel growth turns routing from a ~10,000-step problem into a ~48-step
> problem, which is exactly the regime a learned latent model can serve.**

The two ideas reinforce each other. Simultaneous growth is what the router
needs to negotiate congestion in the open; short horizons are what MuZero needs
to work. Neither is an add-on to the other.

Each net grows from **both pads inward** (two frontiers, snapping when they
meet), which halves per-net depth again and matches how real routers rendezvous.

---

## 2. The MDP

### State

All tensors, batch dim `B` over independent boards.

| Tensor | Shape | Meaning |
|---|---|---|
| `occ` | `(B, L, H, W)` int16 | Occupancy. `0` free, `n>0` owned by net `n-1`, `-1` keepout / board-edge / pour |
| `pad_mask` | `(B, L, H, W)` int16 | Pad cells, same net encoding. Obstacles from step 0 |
| `price` | `(B, L, H, W)` float | **Congestion price** — present over-subscription + accumulated history (§3) |
| `nets` | `(B, N, F)` float | src/dst cell + layer, kind, width class, required length, pair partner, group id, status |
| `frontiers` | `(B, M, 7)` int | The `M` live frontiers: net idx, x, y, layer, which-end, steps, phase |
| `routes` | `(B, N, V, 3)` int16 | Per-net polyline vertices (padded). The refine phase edits these |

`L` is a real parameter (2 → 8, curriculum). A via at `(x,y)` spanning layers
`[a,b]` marks `occ[b_i, a:b+1, y±r, x±r]`, so blind / buried / through vias are
one operation with different spans.

### Macro-step

Every **live** frontier moves once per env step. Frontiers advance
*concurrently*; `step()` is **plan → arbitrate → commit** so two frontiers can
never claim the same cell (this exact bug cost `neuroroute/` a silent
route-with-a-hole, caught only by flood fill — see §11).

`M` swings from ~40 (20 nets × 2 ends) to ~6000 (3000 nets) and **shrinks
during an episode** as nets finish. Everything downstream must be
cardinality-agnostic; §5 is how.

### Per-frontier action (factored)

| Head | Values | Notes |
|---|---|---|
| `direction` | 8 | **Egocentric**: index 0 = down the geodesic gradient toward the target [LIVE] |
| `step` | {1, 2, 4} cells | Long steps are the difference between a 150-step and a 30-step net |
| `layer` | stay / go-to-layer-ℓ | A layer change *is* a via; no separate "place via" action |
| `via_class` | 4 | Read only when `layer ≠ stay` |
| `width_class` | 4 | Index 0 = "the width this net requires" (§11) |
| `couple` | 2 | Diff-pair only: keep the legs locked, or split |

≈ 32 logits per frontier. A flat product space would be unlearnable; factored
it is small and shared across all frontiers.

### Reward (dense, per macro-step)

```
r_t = Σ_frontiers [ γΦ(s_{t+1}) − Φ(s_t) ]      Φ = −geodesic_dist / L
      − w_price · Δ(total over-subscription)
      − w_step  · (number of live frontiers)      step cost
terminal:
      + w_done  · (nets connected)
      − w_fail  · (nets abandoned)
```

Potential-based shaping is policy-invariant, so it cannot bias the optimum
[LIVE]. **Track completion rate, never reward** — measured in this repo: a
random policy scored −330 reward vs greedy's −177 and still completed *more*
nets [LIVE].

`w_step · (live frontier count)` matters: it makes stalling expensive, which is
the main defence against the congestion term being gamed (§8, risk 4).

---

## 3. Congestion price — the negotiation substrate

Borrowed from **PathFinder** (McMurchie & Ebeling, FPGA '95 — see §15), which
solved exactly this problem for exactly this reason: no net gets priority, and
ordering becomes *emergent* rather than scheduled. Its predecessor (Nair 1987)
assigned *infinite* cost to over-capacity resources; PathFinder's contribution
was making the penalty **gradual**, so nets negotiate rather than hard-fail.
That gradualness is why it can be an observation channel a policy learns to
read, rather than a constraint.

Each cell carries a price:

```
price(c) = base(c) · (1 + h(c)) · (1 + p(c))

  p(c)  present cost   — how many frontiers currently contend for c
  h(c)  historical cost — accumulates every iteration c stays over-subscribed
```

`price` is an **observation channel**, not a hard constraint. The policy learns
to read it and respond — detour, change layer, or wait. Historical cost is what
breaks ties that present cost alone oscillates on: a cell that has been hot for
many iterations stays expensive even when momentarily free, so nets learn to
route around persistently-contended channels instead of thrashing through them.

### Rip-up and regrow

Every `T` macro-steps (default 8), frontiers sitting on the worst-priced cells
**retract** N cells while historical price stays elevated. Nothing commits
permanently early. This is the mechanical form of *"let all nets branch out
slowly so the AI can route things clearly"* — nets negotiate in the open across
several rounds rather than racing to claim copper on iteration 1.

Scheduled at first (a fixed rule), **learnable later**. Making it learned from
day one would reintroduce the pointer-over-nets credit-assignment problem that
never trained in `neuroroute/`.

---

## 4. The MuZero stack

Three networks, standard MuZero decomposition.

```
   board tensors ──► h ──►  z : (B, D, L, H, W)     spatial latent, NEVER pooled
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        f: π (per-frontier)  f: v      aux: per-cell legality / final occupancy
              │
   (z_t, a_t) ─► g ─► z_{t+1}, r̂_t      value-equivalent, never reconstruction
```

| Net | Signature | Notes |
|---|---|---|
| **h** (representation) | board → `z ∈ R^{D×L×H×W}` | 3-D encoder: `(1,3,3)` in-plane convs, `(3,1,1)` cross-layer convs, axial attention over `L` (cheap, `L ≤ 8`). Fully convolutional — no board-size parameter anywhere |
| **g** (dynamics) | `(z_t, a_t) → z_{t+1}, r̂_t` | Shallow 3-D conv-resnet, 3–4 blocks (deliberately shallow — wall-clock, §8 risk 2) |
| **f** (prediction) | `z → π, v` | π via the frontier transformer (§5); `v` pools |

### Where pooling is and is not allowed

This repo has **four negative results** on learned lookahead (`jepa/` ×3,
`models/fast_lookahead.py`). All four asked the same question: *can a geometric
scalar (distance-to-target) be decoded from a globally mean-pooled board
embedding?* Four independent nos. The diagnosis — `encoded_tokens.mean(dim=1)`
destroys the spatial structure before anything downstream can use it — is the
single most useful finding in the repo.

So, explicitly:

| Path | Pooled? | Why it is safe |
|---|---|---|
| policy prior π | **No** | Per-frontier, gathered from `z` at the frontier's own cell + local crop |
| per-cell legality / occupancy aux | **No** | Segmentation-shaped, spatially aligned with input |
| dynamics `g` | **No** | Conv-resnet, stays at `D×L×H×W` throughout |
| value `v`, reward `r̂` | **Yes** | Expected return from a pooled latent with TD targets is textbook MuZero, and is a *different animal* from decoding a geometric scalar with a supervised probe |

That last row is stated in full because the repo is scarred here and someone
will otherwise raise it as objection #5. The four failures were supervised
probes for a *specific geometric quantity*. A scalar value function trained by
bootstrapping is not that.

### Losses (unrolled 5 macro-steps)

| Loss | Target | Why |
|---|---|---|
| policy CE | Gumbel search visit counts | standard MuZero policy improvement |
| value TD(λ) | n-step bootstrapped return | — |
| reward MSE | observed macro-step reward | — |
| **per-cell legality / final-occupancy BCE** | the rollout's own terminal board | **~10⁶ labelled cells per rollout instead of 1** |

That last loss is the one that makes `g` trainable, and it is what the four
prior negatives lacked. Labels are **free and on-policy** — every episode emits
its own ground truth from the terminal board state. No separate collection
step, which is where `jepa/collect_transitions.py` spent all its time.

It also absorbs `neuroroute/`'s `FutureFieldPredictor`, which was *working*
(correlation **+0.90** vs a straight-line baseline's +0.46, beating it 11/11
evals [LIVE]) but was judged by an MAE gate that on a sparse field rewards
predicting the mean. **Score on correlation, on held-out boards.** The module
was right; its gate was wrong. Do not re-derive this.

---

## 5. Variable-cardinality actions — the hard structural problem

`M` live frontiers is not fixed: it varies by stage (40 → 6000) and shrinks
mid-episode. The action is therefore **set-structured**, and every component
must be cardinality-agnostic by construction.

```
each frontier ─► token
                  │
     ┌────────────┴────────────┐
     │  frontier transformer   │   LOCAL attention, bucketed by G-cell
     │  + cross-attn to z crop │   (full O(M²) is dead at M=6000)
     └────────────┬────────────┘
                  ▼
       shared MLP head ─► 32 logits    (same weights for every frontier)
```

- **Joint action** = one draw per frontier from the product of `M`
  conditionally-independent categoricals given `z`.
- **Sampled MuZero** (Hubert et al. 2021): the joint space is astronomical, so
  draw **K = 16–32** joint actions from the prior and search only over those.

**Do not invent this factorization — it already exists.** *Multiagent Gumbel
MuZero* (Hao et al., AAAI 2024) addresses precisely this: action spaces that
"increase exponentially with the number of agents," planned over with few
simulations. Start from its formulation and reference implementation
(`github.com/tjuHaoXiaotian/MA-MuZero`) rather than deriving a bespoke one.

**Local attention is not an optimisation, it is a hard requirement.** The
surveyed failure mode for learned routers is explicit: vanilla quadratic
attention becomes memory-prohibitive beyond **~10,000 candidate positions**,
which blocks scaling to large designs. At `M = 6000` frontiers, full attention
is dead on arrival.

### The conditional-independence assumption, and where it breaks

Two adjacent frontiers can independently choose the same channel. Three
defences, in order:

1. **`plan → arbitrate → commit`** resolves the hard conflict — the board can
   never be corrupted, only a move wasted.
2. **The search's value estimate learns to down-rank** joint actions that
   arbitrate badly, because they score worse.
3. **Fallback if 1+2 prove insufficient**: light autoregressive correction over
   *spatially-clustered frontiers only* (frontiers in the same G-cell bucket
   condition on each other's draw). Local, so it does not reintroduce O(M²).

### The action's route into `g`

`g` must consume a joint action of varying cardinality. It does so **as a
spatial field**: each frontier's chosen move is splatted onto its own cell
(direction/step/layer encoded in channels), and the resulting field is
FiLM-conditioned into the resnet. `g` stays convolutional and stays
cardinality-agnostic — no flattening, no padding to a max-`M`.

---

## 6. Search

**Gumbel MuZero**, chosen specifically because it is *designed* to work at low
simulation counts. Classic MuZero MCTS needs 50–800 sims/step; Gumbel gets
policy improvement guarantees at **8–16**, which is the difference between
fitting on a Colab T4 and not.

| Parameter | Default |
|---|---|
| Sampled joint actions `K` | 16–32 |
| Simulations per macro-step | 8–16 |
| Unroll depth | 5 macro-steps |
| Re-grounding | `h` re-applied every **real** macro-step (MuZero-Reanalyze) |

`g` therefore never needs 48-step accuracy — only **5**. This is a deliberate
and large reduction in what the dynamics model must be good at.

**Search is OFF for curriculum stages 0–2** (§7). Most of the curriculum costs
no search at all, and the prior policy alone is a complete working router
(§8, "graceful degradation").

---

## 7. Curriculum — one mechanism per rung, each with a kill-number

> **Revised 2026-09-02.** Stage 0 was stuck below its gate for many sessions.
> The cause was measured, and it was not the learner — see §7.1. Stage 0 is now
> split in two (`0` obstacles, `0v` vias) so each rung tests one skill, and the
> substrate defects that made the old rung unpassable are fixed.

Start with a single net learning to get around obstacles, and only widen once
that is genuinely solved. One sharpening, stated honestly: **single-net
de-risks the *substrate*, not the *architecture bet*.** A lone net has zero
congestion, one frontier, static obstacles, and trivial dynamics — so stage 0
exercises the engine, export, training loop and checkpointing, but **none** of
the four ranked risks in §8, which only switch on when frontiers compete. The
raster thread already took this exact sub-problem to 100%/1000 [LIVE], so it is
known-achievable; do not read "stage 0 passed" as "the design works."

Hence: climb to 2 / 5 / 20 live nets quickly, adding **exactly one mechanism
per rung**, so any regression has exactly one suspect.

| Stage | Board | New mechanism | Gate | **Kill-number** |
|---|---|---|---|---|
| **0** | 1 net, **1 layer**, static keepouts + pads | engine, export, training loop, checkpointing end-to-end. Pure "route around things" | **100%** + quality (copper ≤ 1.15, right-angle ≤ 15%, 0 double-routed) | can't reach 95% in 500 updates → geometry or reward bug, not a hard problem |
| **0v** | 1 net, **2 layers** | **the via.** Pads land on outer layers, so ~25% of boards cannot be routed without one | **100%** + quality | can't reach 95% in 1000 updates → the via penalty is drowning discovery |
| **0.5** | 1 net + **frozen random pre-routed copper** (dense) | policy reacts to dense static obstacles; raycast→logit suppression, local crop | ≥ **97%** | plateaus < 90% → spatial encoder inadequate; fix before adding agents |
| **1** | **2–5 live simultaneous nets**, price field ON, **search OFF** | simultaneous arbitration + congestion price. *Does the prior policy negotiate at all?* | beat **sequential + PathFinder negotiation** (not naive greedy) by ≥ **10 pts** | no gain in 2000 updates → **the simultaneous-growth premise is wrong**; stop and fall back to learned-sequential + negotiation |
| **2** | 5–20 live nets | add `h/g/f` + value-equivalence losses. **Search still OFF** (MuZero net used as a plain policy/value net) | **diagnostic, not a kill-switch** (see §8 correction): track `g`'s next-step error and its growth with unroll depth vs the analytic "replay the geodesic field forward" baseline | no kill-number — a model can fail an absolute fidelity test and still serve search. Proceed to stage 3 and let *that* gate decide |
| **3** | 5–20 live nets | **search ON** (Gumbel, 8–16 sims) | **search-value gate**: search beats prior-only completion by ≥ **5 pts** | no gain → **ship prior-only** (still a complete router) |
| **4** | 20 → 200 nets, 2L → 8L | scale + layer choice | no regression; held-out-scale eval | — |
| **5** | + variable widths / via classes | rule-aware geometry | ≤ 1% width violations | — |
| **6** | + **diff pairs** | `couple` head, gap/skew reward | > 80% pairs inside gap tolerance | — |
| **7** | + **length groups** | refine phase (§12) | > 80% groups inside tolerance | — |
| **8** | + pours, then 1000–3000 nets held-out scale | generalisation | the actual target | — |


### 7.1 Why stage 0 was unpassable — measured, not theorised

Four substrate defects, each verified on the first 48 held-out eval seeds with
**non-learned baselines**, so no policy is implicated. `layer_hop` is ~30 lines
in `world/baselines.py`; `greedy` is the all-zero action, i.e. exactly what an
untrained `PriorPolicy` emits.

| router (48 held-out seeds) | completion | copper (med) | double-routed | right-angle |
|---|---|---|---|---|
| `greedy`, old substrate | 0.7500 | 2.00 | 24 / 48 | 40.4% |
| `layer_hop`, old substrate | 1.0000 | 1.79 | 24 / 48 | 5.1% |
| `greedy`, **stage 0 now** | **1.0000** | **1.000** | **0** | **0.0%** |
| `layer_hop`, **stage 0v now** | **1.0000** | **1.024** | **0** | **0.1%** |

**(a) The geodesic field was Chebyshev, not octile.** The relaxation charged
1.0 for all eight neighbours. Measured on an empty 48×48 board, a (10, 10)
displacement read `10.00` where the copper laid is `14.14`. Two consequences:
shaping paid a diagonal step what it paid an orthogonal one for 1.41× the
copper — while `route_quality` and `RewardConfig.wirelength` score in octile,
so the reward and the metric disagreed by construction — and every path inside
an L∞ ball was equal-cost, making the field one large plateau whose `argmin`
in `bearing_from_field` broke ties arbitrarily. That is a staircase route.
Fixed in `geometry._relax_octile`; the measured right-angle rate of a pure
field-follower went **40.4% → 0.0%**.

**(b) `geodesic_downsample=4` hid the obstacles.** On a 48×48 board that is a
12×12 field, and a coarse cell counts as blocked only when *every* fine cell in
it is. Measured:

```
3x3  keepout ->  0 coarse cells blocked   (invisible)
6x6  keepout ->  0-1, depending on alignment
10x10 keepout -> 1-4
```

Stage 0 samples keepouts from [3, 10]. The field the whole obstacle-avoidance
story rests on could not see most of them. The memory argument for `ds=4` binds
at 128×128×8, not here; stages 0–3 now run `ds=1` (a few MB) and scale the
iteration cap with the grid.

**(c) Double-routing was caused by the environment, not the policy.**
Dual-ended growth toward *static pads* lets the two frontiers mirror around
each other, swap positions, and each complete the whole run — `completion`
reads 1.000 for a net drawn twice. `layer_hop`, with no parameters, did this on
24 of 48 boards at 1.79× copper. Four reward patches (`leg_progress`,
`tip_progress`, `leg_budget_frac`, `wirelength`×12) were built to price it out
and none removed it. Copper-seeded single-ended growth removes it by
construction: **copper 1.79 → 1.00, doubled 24 → 0.** Stages 0–3 now default to
it (`Stage.copper_seeded`).

**(d) A legal limit cycle was invisible to every stall detector, and was the
cheapest policy in the MDP.** `fr_stuck` counts *rejected* moves only, and a
frontier's own copper is passable to itself. Traced on seed 900006 under
`greedy`:

```
t=15 f0=[0,38,21] d=16.94    t=16 f0=[0,38,22] d=16.94
t=17 f0=[0,38,21] d=16.94    t=18 f0=[0,38,22] d=16.94   ... for 34 steps
```

`fr_stuck` was 0 throughout, so `max_stuck_steps` never fired and the net burned
its whole budget. Cost of vibrating: `step_cost` = 0.01 per step, with progress
alternating +1/−1 and cancelling exactly. `WorldConfig.max_idle_steps` now
retires a frontier that takes legal moves without ever beating its own best
cost-to-go.

**(e) The corner penalty was unobservable.** `world.fr_dir` was written by the
engine and read *only* by `RewardConfig.corner`; it was never in the
observation. The direction head is egocentric to the geodesic bearing, which
rotates every step, so holding a straight line needs both the previous heading
and the bearing — the policy had neither, and the term was pure gradient noise.
The previous heading now enters `build_observation` in the *same egocentric
frame as the action*.

**The standing rule this produced:** run `layer_hop` against a stage's gate
*before* training it. If a parameter-free heuristic clears the gate, the stage
is not testing what it claims to test; and if a parameter-free heuristic
*cannot* clear it, neither can a policy initialised at `greedy`.


### 7.2 Where analytic routing actually fails — and why 0/0v are not training stages

`python -m mzr.scripts.baseline_gate --stage S`, 48 held-out seeds, no learned
parameters anywhere. Completion against a gate of 1.00:

| stage | `greedy` | `layer_hop` | headroom |
|---|---|---|---|
| **0** | 1.0000 | **1.0000** | **none** |
| **0v** | 0.7500 | **1.0000** | **none** |
| **1** | 0.6597 | 0.8542 | **0.146** |
| 1m | 0.2847 | 0.4722 | 0.528 |
| 2 | 0.5208 | 0.8083 | 0.192 |
| 3 | 0.4115 | 0.6693 | 0.331 |

**Stages 0 and 0v contain nothing to learn.** A parameter-free heuristic scores
1.0000 on both, at copper 1.000/1.024 and right-angle 0.000/0.001. After 7.1
and the layer-logit geodesic skip, an *untrained* `PriorPolicy` reproduces
`layer_hop` exactly, so it starts at the gate:

```
UNTRAINED on 0v, 64 held-out seeds:
  argmax completion 1.0000  perfect 1.0000  via_frac 0.0156
  copper med 1.024  right-angle 0.002  doubled 0   quality PASS   0 failing seeds
```

Training these stages can therefore only *lose*, and it does. Stage 0v, with
the skip: clean gate hit at u49 (completion 1.0000, copper 1.024, RA 0.002),
then **regression at u74 to 0.8750 with 8 failing seeds**. The u49 numbers are
byte-identical to the untrained policy's — that "hit" was PPO wandering back to
its initialisation, and u74 was it wandering off again.

The eval is not at fault. `evaluate()` called four times on a frozen policy
returns identical numbers to six decimal places, and the reused module-global
env matches a freshly built one exactly. The swings are real policy change, at
`approx_kl` 0.001 and `clip_frac` 0.01 — which is what happens once a head is
near-deterministic (`ent_direction` 0.01): tiny logit changes flip the argmax
without moving KL, and the gate reads argmax.

Three reward configurations were tried against this before the cause was
understood, and the fact that they barely differ is itself the evidence that
reward was never the lever:

| run | change | RA u24 | u49 | u74 |
|---|---|---|---|---|
| A | `length_cost` 0 → 0.03 | 0.220 | 0.220 | 0.220 |
| B | + `corner` 0.08 → 0.25, `entropy_coef` ×5 | 0.200 | 0.220 | 0.220 |
| C | + `gamma` 0.99 → 0.999 | 0.220 | 0.220 | 0.000 |

`gamma` was a genuine mis-specification and is kept (at 0.99, with a terminal
payout of ~12, halving time-to-arrival was worth ~1.79 while the right angles
it caused cost ~0.34/board, so haste outpaid quality 5:1). But it did not make
stage 0 a learning problem, because nothing can.

**So: stages 0 and 0v are substrate regression tests, not training rungs.**
`verify_world` asserts the untrained argmax *is* `layer_hop`, and
`baseline_gate` asserts the analytic ceiling. Neither should be given GPU.
Training starts at **stage 1**, the first rung where the analytic router
genuinely fails: `layer_hop` 0.8542 against a gate of 1.0000, and the 14.6-point
gap is exactly the simultaneous-negotiation problem this design exists to
solve — the geodesic field cannot see other nets' live copper, so yielding a
channel *requires* leaving your own gradient, which is why `max_d0_frac` is
armed at 0.95 from stage 1 and disabled below it.

**One defect this surfaced and did not fix:** stage 1m shows `doubled` 83–86 of
144 nets at copper 2.4–4.0x under every baseline. Multi-pin nets still
double-route badly even under copper-seeded growth, so the trunk/spoke handling
of a branch point is wrong. 1m is off the main spine, but that number should not
be trusted until it is chased.

### 7.3 Stage 1's plateau: the negotiation substrate never ran (fixed)

Stage 1 sits at **exactly 123 of 144 legs (0.8542)** for `layer_hop`, and every
trained policy converges to it (best 0.8490). That number is invariant under:

| swept | values | completion |
|---|---|---|
| `geodesic_downsample` | 4, 2, 1 | 0.8542 |
| `geodesic_refresh` | 16, 8, 4, 2, 1 | 0.8542 |
| `max_idle_steps` | 12, 24, off | 0.8542 |
| `max_steps_per_frontier` | 96, 192 | 0.8542 |
| width dilation of the field | off, on | 0.8542 |
| corner-cutting rule in the field | off, on | 0.8542 |
| entropy coef / advantage / BC | many | 0.79 - 0.849 |

An invariant that stubborn is a structural fact, not a tuning surface.

**The failing legs are not stuck for any of the obvious reasons.** Measured on
the 21 that fail:

* **Not entombed.** A plain flood fill over free-or-own cells reaches the
  target from the frontier tip for **21 of 21**.
* **Not out of time.** The field reports a *finite* 13.0-cell path, with 74 of
  96 macro-steps unspent.
* **Not a termination artefact.** Disabling `max_idle_steps` entirely and
  doubling the step budget changes nothing.

**The cause: `ripup_round` never fires at stage 1.**

```python
k = int(math.floor(n_elig * r.fraction))
if k <= 0:
    return torch.zeros(...)          # <- always taken at stage 1
```

Stage 1 has 3 nets and `RipupRules.fraction = 0.25`, so
`k = floor(3 * 0.25) = 0`. Measured `ripups = 0.0` over 48 boards. The
congestion price is computed, handed to the policy as two observation channels,
and charged in the reward -- and **nothing ever acts on it**. Stage 1 has been
testing simultaneous *greedy* growth all along, and 123/144 is that ceiling.

The trap is specific to small net counts: stage 2 (5 nets) gives `k = 1` and
stage 3 (8 nets) `k = 2`, so both do rip up. Stage 1, the rung whose entire
purpose is *"does the prior policy negotiate at all?"*, is the one where the
negotiation mechanism is switched off by integer arithmetic.

**Forcing it on does not rescue it, and that matters too.** With
`fraction = 0.5` so `k = 1`:

| include_settled | interval | fraction | completion | ripups |
|---|---|---|---|---|
| False | 8 | 0.25 | 0.8542 (123/144) | **0.0** |
| False | 8 | 0.50 | 0.5625 (81/144) | 11.5 |
| False | 4 | 0.50 | 0.3819 (55/144) | 24.0 |
| True | 8 | 0.50 | 0.3472 (50/144) | 12.0 |
| True | 4 | 0.50 | 0.2292 (33/144) | 24.0 |

Whole-net rip-up is too destructive at this scale: a net ripped at step 40 of
96 needs ~25 steps to re-route and frequently does not get them. PathFinder
survives this because it runs many full iterations to convergence; one episode
does not. `include_settled` (added here so rip-up *can* reclaim a finished
net's copper, which it otherwise never can) makes it worse still, because
ripping a completed net destroys work that had already succeeded.

Section 3 of this document specifies the gentler mechanism and the
implementation diverged from it:

> Every `T` macro-steps, frontiers sitting on the worst-priced cells **retract
> N cells** while historical price stays elevated.

`RipupRules` instead says *"Whole nets are ripped, not partial frontier
retractions"*. The retraction form is the one the horizon can afford.

**Resolved.** `retract_round` implements the section 3 mechanism -- score live
frontiers by the congestion price under their tip, pull the worst back a couple
of vertices, leave historical price elevated -- and `retract_fraction` uses
`max(1, round(...))` so it cannot silently floor to zero. Measured on
`layer_hop`, 48 held-out boards, **no learning anywhere**:

| stage | rip-up never fired | with retraction | delta |
|---|---|---|---|
| **1** | 0.8542 (123/144) | **0.9444 (136/144)** | **+9.0 pts** |
| **2** | 0.7917 | **0.8500** | +5.8 |
| **3** | 0.6719 | **0.7995** | +12.8 |
| **1m** | 0.4722 | **0.5625** | +9.0 |

Route quality improves *alongside* completion rather than trading against it:
stage 1 right-angle 0.119 -> 0.084 (0.034 at `retract_steps=2`), stage 3
0.173 -> 0.063. Stage 1's kill-number is "can't clear 0.90 in 2000 updates";
the non-learned baseline now clears 0.90 with no training at all.

The shape of the sweep is the substantive finding, not the peak value:

```
retract_steps  interval  completion
      1            4     0.9444  (136/144)   <- default
      2            4     0.9375  (135/144)   RA 0.034
      2            2     0.6181  ( 89/144)
      3            2     0.6181  ( 89/144)
      8            8     0.6389  ( 92/144)
```

Gentle and frequent wins; retract too much or too often and it self-destructs
exactly as whole-net rip-up did. That is PathFinder's gradualness argument --
section 3's reason for a *gradual* penalty rather than Nair's infinite one --
reproduced on this problem, and it is why the whole-net form could never have
worked at a 96-step horizon.

**What this does not fix.** Stage 1m still shows `doubled = 82` of 144 at
copper 2.49x under every baseline: multi-pin branch points double-route
regardless of retraction, which is a separate defect in trunk/spoke handling.
And no stage reaches its 1.0 gate, so there is still real headroom for a policy
to earn -- which is now the point, because for the first time that headroom is
being measured against a substrate where negotiation actually runs.

### 7.4 Two field bugs fixed on the way, neither of which moved completion

Both are real and both are kept; recording them so they are not "rediscovered"
as candidate causes of the plateau.

**The field planned through gaps the trace cannot enter.**
`engine._refresh_net_geo` / `_refresh_geodesic` built `blocked` per *cell*,
while `check_moves` / `move_claims` test the width-dilated footprint.
`expert.py::_dilate` had already fixed exactly this on the planner side, with
its docstring recording the cost -- "every one of 24 legs planned successfully
and only 46% of them stamped". The engine's own field never got the same
treatment. Fixed in `geo.dilate_blocked`.

**The field cut corners the mover forbids.** A 45-degree move reserves corner
guards beside itself, so a diagonal that clips an obstacle corner is illegal --
but the relaxation propagated diagonally regardless. Caught at a stuck frontier:

```
 dir  (dy,dx)   free cells   field 1 ahead
   1   (1, 1)            0        3.414    <- best, and ILLEGAL
   0   (0, 1)            1        3.828
   2   (1, 0)            1        4.414
```

The only descending direction was forbidden and every legal one went uphill --
a local minimum manufactured by field/mover disagreement, in a field that is
supposed to have none. `_relax_octile` now takes the blocked mask and permits a
diagonal only when both flanking orthogonals are free.

**Two diagnostics reported during this investigation were wrong**, and the
corrections are the reason 7.3 landed where it did. The first called
`route_world_board_live` *after* the net was already marked FAILED, so the
planner skipped that net and returned no path -- misread as entombment. The
second let the planner route through the net's own copper. The flood-fill test
in 7.3 is the one to trust: it uses no planner and no dilation.

### Expert demonstrations are not optional — add them from stage 1

**PRIMAL** (Sartoretti et al. 2019) is the closest published analogue to what
stage 1 attempts: a *fully decentralised* multi-agent path-finding policy,
parameter-shared, "copied onto any number of agents," scaling to **1024
agents**. It is the single strongest piece of evidence that §5's
cardinality-agnostic shared-weights design works.

And its authors did **not** get there with pure RL. The paper's own account of
what made it work is "demonstrations of an expert MAPF planner during training,
as well as careful reward shaping." The surveyed failure mode for RL routers
agrees: sparse-reward exploration "may prematurely converge to suboptimal
policies."

`neuroroute/` was pure RL and plateaued at 60–65%. So:

- **Bootstrap the prior policy by behaviour cloning from an expert** — A* /
  PathFinder-negotiated routes on the *same* generated boards. Labels are free;
  the expert already exists as a baseline.
- **Blend, then anneal**: BC loss alongside the RL objective, decayed as
  completion rises. Do not drop it to zero early.
- This compounds with §4: a stronger prior is *also* what makes the search work
  (see risk 1), so the same intervention buys down two risks at once.

### What "routing perfectly" means, operationally

> **Held-out `argmax` completion at the gate threshold, sustained across 3
> consecutive evals, with the known-hard seeds in the eval set.**

Not reward. Not a single spike. That definition is bought with two scars:

- `neuroroute/` stage 0 **met its gate once (100% at u275) and then
  regressed** [LIVE].
- There is a **real mode/mean gap**: training rolls out *sampled*, eval scored
  *argmax*, and sampled beat argmax by ~11–19 pts **even on an untrained
  policy** [LIVE]. Report both arms every eval; a policy that only works
  sampled is relying on exploration noise to reach the goal.

Known-hard seeds to keep in every eval set: **9648, 9681, 9764, 9779, 9148,
9251, 9091, 9390, 9535, 9901**.

---

## 8. Risk register — and what actually reduces each

### The biggest lever: graceful degradation

**The prior policy is a complete router at every stage.** Search is *upside*,
never load-bearing.

If the dynamics-fidelity gate (stage 2) or the search-value gate (stage 3)
fails, what ships is:

> simultaneous-growth prior policy + congestion price field + rip-up/regrow

— a **learned PathFinder**. That is novel, defensible, and directly serves the
stated goal of leaving space for future nets. The architecture cannot hard-fail
into nothing; the worst case is a working router that is less ambitious than
the best case.

This is the single most important risk-reduction decision in the document, and
it is why stages 2 and 3 are ordered the way they are: the MuZero machinery is
added to a system that **already works without it**.

### Ranked risks

| # | Risk | Why it might bite | Signal | Mitigations |
|---|---|---|---|---|
| **1** | **`g` fidelity over 5 macro-steps** | On a combinatorial board the future is genuinely multi-modal; occupancy prediction blurs | `r̂` error grows with unroll depth; **stage-3 search-value gate** fails | dense per-cell BCE (~10⁶ labels/rollout); free on-policy labels; unroll only 5 and re-ground with `h` every real step; **`g` starts training at stage 2 where boards are nearly deterministic** and gets harder on the same schedule the policy does; **strong BC-bootstrapped prior** (see §7) — this is the real mitigation, see below |
| **2** | **Wall-clock** | 16 sims × 48 macro-steps × B boards = thousands of `g` calls/episode on a T4 | steps/sec collapses; GPU idle | Gumbel (8–16 sims, not 50–800); **search OFF for stages 0–2**; batch all boards × all sims into one `g` call; keep `g` shallow (3–4 blocks); Reanalyze for sample efficiency; **instrument steps/sec from day one** — a slow stage is a stop-and-fix signal, not something to endure |
| **3** | **Simultaneous-growth premise is wrong** | Maybe negotiated congestion needs more than a price channel to beat sequential | stage 1 shows no gain over greedy-sequential | it is tested at **stage 1**, cheaply, on 2–5 nets, before any MuZero machinery exists. Kill-number is explicit |
| **4** | **Congestion reward hacking** | Frontiers park off-target to dodge price | completion flat while price term improves; frontiers stall | `w_step · (live frontier count)` makes stalling expensive; per-frontier geodesic-progress potential must dominate `w_price`; report **stall fraction** as a first-class metric |
| **5** | **Scale generalisation 20 → 3000** | 64× the cells; attention and encoder must not care | held-out-scale eval collapses | fully-convolutional encoder + **local bucketed attention** (no board-size parameters anywhere); **eval at held-out scale every stage** (train 5, eval 10) so a gap shows at stage 1, not stage 8; tile inference into overlapping windows if needed |
| **6** | **Joint-action independence breaks in dense regions** | Adjacent frontiers grab the same channel | rejected/arbitrated-away move rate climbs with density | plan→arbitrate→commit (correctness); search down-ranks bad joint draws; local autoregressive correction held in reserve (§5) |
| **7** | **Lattice ≠ KiCad legality** | Pitch reasoning wrong at corners, pad edges, 45° | sim-to-real DRC gap > 0 | ported verified geometry; **DRC gate before any training**, across ≥ 4 board sizes (§11 — the diagonal bug was clean on one size and failing on another) |
| **8** | **Colab session death** | Known | — | checkpoint every N updates to Drive, non-negotiable |

### An important correction to how risk 1 was originally framed

The first draft of this design gated search behind an **absolute
dynamics-fidelity** test: `g` must predict the next macro-step better than an
analytic baseline, or search never turns on. Published analysis of MuZero says
that gate is **the wrong shape**.

*What model does MuZero learn?* (He, Oliehoek et al., 2023/ECAI 2024) finds
MuZero's learned model is **not in fact value-equivalent** — it "struggles to
generalize when evaluating unseen policies," and is not even accurate enough to
correctly evaluate its own data-collection policy. By a strict fidelity
standard, MuZero's model *fails* — and MuZero works anyway.

Their explanation is the useful part: MuZero's "incorporation of the policy
prior in MCTS alleviates this problem, which **biases the search towards
actions where the model is more accurate**." The model does not need to be
globally correct. It needs to be locally correct *in the region the prior
already likes*, and the prior is what keeps search inside that region.

Three consequences, all adopted:

1. **The binding gate is stage 3 (search beats prior), not stage 2 (model beats
   analytic).** Model fidelity is a *diagnostic* to watch, not a kill-switch —
   an accurate-enough-where-it-matters model can fail an absolute test and still
   deliver. Keeping the original gate as a hard stop would have killed a
   working configuration.
2. **Prior quality is a first-class ingredient of search quality**, not just a
   fallback. This is the second, independent reason for the BC bootstrap in §7.
3. **Never widen the search to low-prior actions to "explore more."** That
   walks directly into the region where `g` is least accurate. Keep Gumbel's
   sampling tied to the prior.

### Process-level risk reduction

- **Port validated geometry + KiCad bridge *with* their verify scripts.** Do
  not re-derive §11.
- **KiCad legality gate before any training.** One afternoon. If the lattice is
  not legal, every completion number afterward is measured against copper KiCad
  rejects.
- **`--render-every` on from the very first run.** `neuroroute/` ran an entire
  project without anyone looking at a single failed board — every finding came
  from numbers. A contact sheet of failures shows the failure mode; a reward
  curve never will.
- **Every metric's denominator gets checked before it is believed.**
  `neuroroute/`'s "rejected-action rate exploded to 93.8%" alarm was a
  small-denominator artifact (one net per board, most heads idle) [LIVE].

---

## 9. KiCad's role — validator, not environment

| Job | Call | When |
|---|---|---|
| **Ingest** a real board → `BoardSpec` | `net_pads`, `get_board_geometry`, `get_design_rules` | Once per board, subprocess |
| **Export** a routed lattice → `.kicad_pcb` | grid → mm, emit tracks + vias | End of episode |
| **Validate** | `run_drc()` — KiCad's real `DRC_ENGINE` | Eval only, **never per step** (267 ms, 73% of engine time [LIVE]) |

The number that matters is the **sim-to-real gap**: DRC violations per 1000
nets on boards the fast engine declared clean. The lattice pitch is *defined*
as `min_track_width + min_clearance`, so the gap should be ~0 by construction;
measuring it is how we find out where that reasoning is wrong.

`neuroroute/` measured **0 legality violations over 192 routed nets across 4
configs** against KiCad 9.0.2 [LIVE]. That result is inherited with the ported
geometry and **re-run in this package before training**.

Hard constraints from `ROADMAP.md` still apply: the bridge and system `pcbnew`
never share a process; subprocess for anything needing `pcbnew`.

---

## 10. Ported vs new

### Ported (with verification scripts — do not rewrite)

| From `neuroroute/` | Why |
|---|---|
| `world/geometry.py` — batched legality / stamping / raycast / geodesic | DRC-verified [LIVE], exact vs brute force [LOCAL] |
| `world/generator.py` — procedural boards | works; carries the "no-nets board scores 100%" fix |
| `eval/kicad_export.py`, `scripts/validate_kicad.py` | the sim-to-real gate |
| `scripts/verify_geometry.py`, `verify_env.py` | they re-derive answers from occupancy rather than trusting flags |
| Non-learned **raycast → fixed logit suppression** | Rejected-Action Rate 1.51% → 0.40% [LIVE]; cannot be trained away |
| **Egocentric action frame** (dir 0 = toward target) | removes board-pose generalisation entirely [LIVE] |
| **Near-zero actor init** | untrained policy ≈ greedy baseline, so training starts *at* the baseline [LIVE] |
| **Geodesic (obstacle-aware) distance field** | [LIVE] |
| **Potential-based shaping** `Φ = −geodesic/L` | policy-invariant [LIVE] |

### New

`world/` (simultaneous-frontier engine, macro-step, congestion price),
`models/` (`h`/`g`/`f`, frontier transformer), `search/` (Sampled + Gumbel
MuZero), `training/` (value-equivalence losses, Reanalyze, curriculum + gates).

---

## 11. Things that will cost time if re-derived

Inherited verbatim from `neuroroute/README.md` and `HANDOVER.md` because every
one of them cost a debugging session:

- **Every path that writes copper must apply the diagonal corner guards.** A
  45° trace passes *between* lattice cells; any cell at perpendicular distance
  `1/√2` is only `0.4/√2 − 0.2 = 0.083 mm` from its copper. Two paths existed
  (`_move_cells`, `_segment_cells`) and disagreed → real **0.0828 mm** KiCad
  violations. **Clean on one board size, failing on another** — always validate
  across several.
- **Pad size and lattice reservation must derive from one number**
  (`DesignRules.pad_size`). They drifted → **0.100 mm** pad-to-track violations.
- **Never export unrouted nets' stub copper** → 478 `track_dangling`.
- **Two frontiers could write the same cell** in one batched step, and one
  silently lost while its frontier advanced anyway — a route with a hole in it,
  found only by flood fill. `step()` must be **plan → arbitrate → commit**.
- **Raycast is in absolute directions; everything else is egocentric.** When
  the frames drifted, rejected-action rate was **86%** while the baseline
  believed it was only taking safe moves. After the fix: **1.6%**.
- **The geodesic field is stored at ¼ resolution and must be sampled
  bilinearly.** Nearest-neighbour makes it piecewise-constant, so the gradient
  between adjacent cells is exactly zero and the descent direction is arbitrary.
- **An action head's `bias` decides its untrained default.** With `gain=0.01`
  weights the bias is the whole signal. A zero-bias width head picked 3-cell
  traces on **612/627** actions (88% rejected). `h_layer.bias` must scale with
  layer count (`log(3L)` gives P(stay)=0.75 for any `L`).
- **`snap_radius ≥ max(STEP_LENGTHS)/2`** — a longer step jumps clean over the
  snap zone and the frontier orbits its target forever, which reads exactly
  like a learning failure.
- **A board with no nets scores 100%, not 0%** — otherwise a generator bug
  hides inside a training curve.
- **Observation tensors must be clones, not views**, of anything `step()`
  mutates in place — an aliasing observation silently breaks the PPO/policy
  ratio without ever crashing.
- **Colab's `kicad-cli` is 8.0.9, local is 9.0.2.** Both work.
- **AMP**: cast forecaster/aux outputs to fp32 explicitly. A NaN aux loss
  poisons the *whole shared backward graph*, so the symptom appears on an
  unrelated head. A `GradScaler`-declined step is routine, not fatal — but a
  non-finite **parameter** stays fatal.

---

## 12. The three tools, in this frame

| Tool | Implementation | Not |
|---|---|---|
| **Route** | Frontier growth (§2), searched (§6). The policy decides every bend | not PNS `MODE_ROUTE_SINGLE`, not push-and-shove |
| **Diff pair** | One net, **two coupled frontiers**, `couple` bit. Coupled: sampled direction/step drives the centreline, legs placed at `± gap/2` perpendicular, **both** legs legality-checked, both advance or neither. Split: two independent frontiers for that step. Reward penalises gap error, skew, and split fraction — `w_split` small, because splitting is a real technique, not a failure | not `MODE_ROUTE_DIFF_PAIR` |
| **Length tune** | Refine phase: a separate short-horizon MDP over the *same* board. Action = `(vertex_index, perpendicular offset δ, subdivide?)` — literally dragging a vertex in the editor. **Amortized, no search** (dense-reward, ~10 steps; search buys little). A meander is what an optimal drag policy *looks like* | not `MODE_TUNE_SINGLE`, no meander generator |

The refine action set is already built and mechanically verified in
`neuroroute/` [LOCAL]: drags change length, routes stay connected through them
(flood fill), rejections restore the board byte-identically, and repeated
alternating drags accumulate length — **a meander, from ordinary actions, with
no generator**. What was never built is the policy that chooses drags.

The same refine MDP also does wirelength reduction, diff-pair gap repair, and
clearance-margin improvement — one MDP, four jobs.

---

## 13. Build order

1. **`world/`** — simultaneous-frontier lattice engine, macro-step, congestion
   price. Verify locally vs brute force (no GPU, no KiCad).
2. **KiCad legality gate** — greedy route → export → real DRC, **≥ 4 board
   sizes**. Before any training.
3. **Expert baselines + demonstration recorder** — sequential A* and
   sequential + PathFinder negotiation, on the generated board pool. These are
   simultaneously stage 1's *baseline to beat* and the *BC demonstration
   source* (§7). One piece of work, two jobs.
4. **Prior policy, search OFF** — curriculum stages 0 → 0.5 → 1, with the BC
   loss blended and annealed. Get a working *simultaneous* router first.
5. **Add `h`/`g`/`f` + value-equivalence losses** — stage 2. Watch model
   fidelity as a diagnostic; do not gate on it.
6. **Search ON** (Gumbel / MA-Gumbel-MuZero) — stage 3. Run the search-value
   gate. **This is the first point at which the MuZero bet is tested.**
7. Scale → layers (stage 4) → widths/via classes (5).
8. **Diff pairs** (6) → **refine phase / length tuning** (7).
9. Pours, held-out-scale eval at 1000–3000 nets (8).

The ordering is load-bearing. Step 2 precedes any training because if the
lattice is not legal, nothing trained on it matters. Step 4 precedes step 6
because if simultaneous growth does not beat sequential-plus-negotiation, then
MuZero is being added to the wrong foundation — and the cheapest way to learn
that costs no search machinery at all.

---

## 14. Confidence, stated honestly

Per-gate, before any code is written. These are priors to be updated against
real numbers, not predictions to defend.

| Gate | Confidence | Basis |
|---|---|---|
| Stage 0 — single net, obstacles | **~95%** | this repo already did it: 100% / 1000 boards [LIVE] |
| Stage 0.5 — dense static copper | **~85%** | same machinery + ported raycast suppression |
| **Stage 1 — simultaneous beats sequential+negotiation** | **~50%** | the core novel bet. No published precedent either way (§15) |
| Stage 3 — search beats prior policy | **~35–40%** | conditional on stage 1; MCTS-on-routing precedent is toy-scale only |
| Full vision: diff pairs + length tuning at 1000–3000 nets | **~10–15%** | every stage multiplies; nobody has published this |
| **Something better than `neuroroute/`'s 65% ships** | **~75–80%** | graceful degradation (§8) — the fallback is a well-trodden design |

The honest summary: **the fallback is likely, the full vision is a genuine
research bet.** That asymmetry is deliberate and is the reason §8's
graceful-degradation ordering exists. Anyone reading this doc should expect to
ship a learned negotiated-congestion router and treat working MuZero search as
upside.

Two facts that should keep expectations calibrated:

- The 2022 methodological survey (§15) reports **no RL router deployed at
  industrial scale**, and classical solvers persisting precisely because ML has
  not offered a reliable alternative. Routing is harder for ML than placement.
- The only published MCTS-on-circuit-routing work (He et al. 2020) is
  **single-layer, randomly generated, and sequential**. This design is well past
  the edge of what has been demonstrated.

---

## 15. References

Prior art this design leans on, and what each one actually licenses.

### Search and planning

- **Hubert et al., "Learning and Planning in Complex Action Spaces" (Sampled
  MuZero), ICML 2021** — arXiv:2104.06303. Planning over *sampled* action
  subsets, with principled policy evaluation/improvement. Reported to scale
  "gracefully down to small numbers of samples." → licenses §5's K=16–32.
- **Danihelka et al., "Policy improvement by planning with Gumbel" (Gumbel
  MuZero), ICLR 2022** — Gumbel top-k + sequential halving; guarantees policy
  improvement **without visiting all root actions**, and matches MuZero with far
  fewer simulations. → licenses §6's 8–16 sims, the T4 budget.
- **Hao et al., "Multiagent Gumbel MuZero," AAAI 2024** —
  `github.com/tjuHaoXiaotian/MA-MuZero`. Combinatorial action spaces that grow
  exponentially in agent count; up to an order-of-magnitude fewer environment
  interactions than model-free, "when planning with much fewer simulation
  budgets." → **this is the algorithm for §5/§6. Do not reinvent it.**
- **He, Oliehoek et al., "What model does MuZero learn?" 2023 / ECAI 2024** —
  arXiv:2306.00840. MuZero's model is *not* value-equivalent and fails on unseen
  policies; planning works because the **policy prior biases search toward
  where the model is accurate**. → **reshaped §8's risk 1 and deleted a
  mis-specified gate.** The most useful reference here.

### Multi-agent coordination

- **Sartoretti et al., "PRIMAL," IEEE RA-L 2019** — arXiv:1809.03531. Decentralised
  MAPF; one parameter-shared policy "copied onto any number of agents," scaling
  to **1024 agents**. → licenses §5's cardinality-agnostic design *and* §8's
  scale-generalisation claim. Crucially, it needed **expert demonstrations +
  careful reward shaping**, not pure RL → §7's BC bootstrap.
- **PRIMAL2 (lifelong), 2020** — arXiv:2010.08184.

### Routing (EDA)

- **McMurchie & Ebeling, "PathFinder," FPGA 1995** — negotiated congestion:
  `cost = b(n) · (1+h(n)) · (1+p(n))`. Signals "negotiate for a resource and
  thereby determine which signal needs the resource most." Gradual penalty,
  unlike Nair (1987)'s infinite cost. → §3 outright.
- **He et al., "Circuit Routing Using MCTS and Deep Neural Networks," 2020** —
  arXiv:2006.13607. Solves instances sequential A* and Lee's algorithm cannot,
  and beats vanilla MCTS — but on **randomly generated single-layer** circuits,
  modelled **sequentially**. → the only direct precedent, and it is small.
- **Liao et al., "A Deep RL Approach for Global Routing," 2019** —
  arXiv:1906.08809.
- **Cheng et al., "Towards ML for Placement and Routing in Chip Design," 2022** —
  arXiv:2202.13564. Survey. Routing less mature than placement; no
  industrial-scale RL routing; classical solvers persist. → §14's calibration.
- **"Transformer-based RL for Net Ordering in Detailed Routing," IJCAI 2025** —
  learned ordering is an active, unsolved problem in its own right; this design
  tries to *dissolve* it via negotiation rather than learn it.
- **Lin et al.** — asynchronous actor-critic routing **millions of nets** with
  policy distillation, beating a SOTA detailed router. → evidence learned
  routing *can* reach scale; distillation is the mechanism to copy if stage 8
  stalls.
- **"Accelerating Detailed Routing Convergence through Offline RL," 2025** —
  arXiv:2512.03594. CQL selects rip-up-and-reroute cost weights per iteration;
  5% fewer iterations (up to 31%), 1.56× median speedup on **unseen** ISPD19
  designs. → direct precedent for **learning the congestion-price schedule**
  (§3), which this design currently fixes by hand. A cheap, well-evidenced
  upgrade once §3 is stable.

### The field's verdict on concurrent routing — and why it does not sink §1

Surveys of multi-net global routing are blunt: concurrent approaches "attempt
to handle numerous nets simultaneously but are typically **too expensive** to
be applied on today's large designs," which is why sequential + rip-up-reroute
won in practice.

That objection is aimed at **exact** concurrent optimisation — multicommodity
flow, min-cost network flow, ILP — where cost grows with problem size at
solve time. This design's macro-step is a **neural network forward pass** whose
cost is amortised at training time and roughly constant per step at inference.
The historical objection is about a different mechanism.

**But it is still a warning, not a dismissal**: the field has repeatedly tried
concurrency and retreated to sequential-plus-negotiation. That is exactly why
stage 1's baseline is **sequential + PathFinder negotiation** rather than naive
greedy, and why its kill-number falls back to learned-sequential. If
simultaneous growth cannot beat the thing the field actually settled on, the
premise deserves to lose.

---

## 16. Operating model

`AGENTS.md` governs, unchanged. Claude Code owns the logic and commits;
**Antigravity runs Colab and reports real output verbatim, and does not edit
tracked source.**

Never claim a training or eval result without it coming back through that loop
— this repo's history has stale and fabricated-sounding reports the user has
explicitly warned about. Verify `git log -1` matches what a report claims.
