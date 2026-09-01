# Full-board routing: what it would actually take

Written 2026-09-01, after a day of stage-0 work. The goal stated by the project
owner is **routing a whole board with thousands of nets**. This is the honest
assessment of the distance to that, what the literature says, and the ordered
plan — including the parts I think are unlikely to work.

## 1. Where we actually are

Stage 0 is **one net** on a 48x48 two-layer board. It is a plumbing check, and
it took a full day and six experiments to get right. What that day produced:

* Three real bugs found and fixed (entombed pads, wedged frontiers, a blind
  critic), each of which moved the number a lot.
* An architecture change (copper-seeded fields) that removed double-routing
  entirely: 48/64 nets routed twice -> 0/64, copper 2.00x -> 1.16x.
* Four negative results, three of them reward-shaping attempts that failed for
  one shared reason: they tried to make a behaviour unprofitable while the
  geodesic field was actively instructing it.

The distance from "one net" to "thousands of nets" is not one more push. It is
a research programme, and the sections below try to say honestly which parts are
engineering and which are open questions.

## 2. What scales, and what does not

Sorted by how they behave as net count `N` grows.

### Scales fine (measured or structural)

| | why |
|---|---|
| **episode horizon** | Measured: 19 / 14 / 7 macro-steps to settle for 2 / 8 / 24 nets. 12x the nets, 0.37x the steps. This is the load-bearing claim of the whole design and it **holds**. |
| **field encoder** | Cost is area x layers, independent of `N`. A 128x128x8 board costs the same with 10 nets or 10,000. |
| **policy parameters** | Cardinality-agnostic by construction: every frontier is a token through shared weights. A netlist of 5 or 5000 is the same weights. |
| **geodesic memory** | After copper-seeding, one field per *net* rather than `2(k-1)` per net. At 2000 nets, 8 layers, 128x128 coarse: ~32 MB/board, ~1 GB at batch 32. Affordable. |

### Does not scale (in priority order)

**1. Credit assignment — the only blocker that gets *worse* with size.**
MAPPO broadcasts one board advantage to every frontier. Per-frontier
signal-to-noise falls as `1/N`: two frontiers share one scalar at one net,
~2000 share it at a thousand nets. Everything else on this list is a constant
factor; this one is a scaling law working against us.
**Status: fixed, `--per-frontier-adv`** (see section 3).

**2. `O(F^2)` frontier attention.** At 2000 nets `F ~= 4000`, so ~16M attention
pairs per board per step. `DESIGN.md` already anticipates this ("bucketed local
attention switches on at stage 4"). Standard fix, not yet built.

**3. The completion cliff.** The most uncomfortable number in the repo, from
`verify_world`'s own design-claims check:

    final completion {2 nets: 0.88, 8 nets: 0.31, 24 nets: 0.19}

The horizon scales beautifully and *completion falls off a cliff*. That is the
non-learned baseline, not the policy — but it is the shape of the difficulty
curve, and stage 3 is only 8 nets. **Whether a learned policy bends this curve
is the central open question of the project**, and nothing measured so far
answers it.

**4. Pure RL plateau.** NeuroRoute was pure RL and plateaued at 60-65%. PRIMAL
got 1024 agents working with *one shared policy* — the closest precedent to this
design — but needed **expert demonstrations**, not pure RL. This repo has the
expert and the BC loss already wired; only the demonstration *recorder* is
missing.

## 3. The credit-assignment fix (done)

`GPAE`, Generalized Per-Agent Advantage Estimation, arXiv 2603.02654. The paper
states MAPPO's assumption in exactly the form this codebase implemented it:

    MAPPO:  A_i(s,a) = A_global(s,a)  for all i
    GPAE:   EQ_i := E_{a_i ~ pi_i}[ Q(s, a_i, a_-i) ]

It averages over agent `i`'s own action while keeping the others fixed, so it
isolates one agent's contribution *without* COMA's marginalisation over joint
actions. That distinction is what makes it usable here: COMA's own authors call
it "difficult to apply with more than a few agents".

Reported: **+6% wall-clock over GAE**, cost **independent of agent count**,
unbiased at `lambda=1`. On `5m_vs_6m`, **93.7%** win rate against MAPPO's
**3.1%**.

**The signal was already in this codebase and was being discarded.**
`env/rewards.py::step_reward` returns a per-frontier `(B, F)` reward, and
`training/run.py::collect` did `step.reward.sum(dim=1)` one line before PPO saw
it. Implemented as `--per-frontier-adv`: a per-frontier value head, per-frontier
GAE, advantages normalised over live frontiers only, with the genuinely joint
board terms (failure penalty, terminal completion) shared across live frontiers.

Measured on stage 1 (4.16 live frontiers/board): **within-board advantage spread
1.2464 against an overall std of 1.4281**. About half the available learning
signal was being destroyed by the sum, and it was the half that identifies which
frontier was responsible.

## 4. The plan, ordered by evidence value

Each step is chosen so that a *negative* result is informative and cheap.

**A. Per-frontier advantage on stages 0-1.** Done, needs measuring. If it does
not improve stage 1, the credit-assignment theory is wrong and the rest of this
plan needs revisiting. *Cost: two runs.*

**B. BC demonstration recorder.** The expert (`world/expert.py`) already emits
byte-identical engine actions -- verified 410/410 path steps are single engine
actions -- and `ppo.py` already has the BC loss. Only the recorder is missing.
This is PRIMAL's lesson and NeuroRoute's plateau is the local evidence for it.
*Cost: a day. Highest value-per-effort item after A.*

**C. Stages 1-3 with A+B.** This is the real verdict on the project's core bet
(simultaneous growth beating sequential-plus-negotiation), currently self-rated
~50%. **Do not skip to scale before this.**

**D. Bucketed local attention.** Required before stage 4. Mechanical.

**E. The scaling curve — the decisive measurement.** Train one policy, then
measure completion against net count: 8, 24, 64, 128, 256. The *shape* of that
curve decides feasibility, and it can be measured long before thousands of nets.
If completion is still falling steeply at 64, a flat policy will not reach
thousands and the answer is hierarchy (section 5).

## 5. If the flat policy does not scale

The literature's consistent answer to RL routing at scale is **hierarchical
decomposition**: local agents owning regions, a global agent coordinating
between them, with GNNs for the graph structure. Precedent at scale exists but
is worth reading precisely:

* attention-based REINFORCE: detailed routing on *small benchmarks* with
  thousands of nets
* asynchronous actor-critic: **routing order** for millions of nets — an
  ordering decision, not the geometry
* GANGR (2026): GAN-learned net-interference patterns for **batch formation**

**No published system does full geometric routing of thousands of nets
end-to-end with RL.** The ones operating at that net count are making
*ordering* or *batching* decisions around a classical router. That is the honest
state of the art, and it should temper the target.

## 6. Honest probabilities

Stated before the experiments, so they can be scored later.

| outcome | estimate |
|---|---|
| stage 1 clears its gate with A+B | ~55% |
| stage 3 (8 nets) clears | ~35% |
| useful **assistive** router: routes most of a real board, flags the rest | ~60% |
| full autonomous board, thousands of nets, DRC-clean | **~5-10%** |

The last row is low and I do not want to dress it up. The binding constraints
are the completion cliff (section 2.3) and the absence of any precedent for
end-to-end geometric RL routing at that scale (section 5). Compute is a distant
third: 8.3s/update on a T4 for **one net** on 48x48.

## 7. The reframe worth considering

Commercial autorouters are **assistive**. They route what they can and hand back
the rest; no one ships a button that finishes a board unattended. "Route 90%
cleanly and flag the remainder" is a genuinely valuable product and it is
perhaps 6x more likely to land than full autonomy.

That reframe also changes what to optimise. A policy producing **1.16x copper
with zero double-routing** is a far better foundation for handing work to a human
than one that connects everything at 2.00x. Today's architecture work was not a
detour from shipping — it is the part that makes the output usable.

The full-autonomy goal is worth *aiming* at, because the work it forces
(credit assignment, scaling curve, hierarchy) is the same work the assistive
version needs. It should not be the thing the project is *judged* by until the
scaling curve in step E is measured.
