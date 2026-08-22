# RL plan — line-geometry policy

Supersedes `docs/AI_ARCHITECTURE.md` (CFP) as the live direction, and ends the
LLM-agent pivot recorded in `ROADMAP.md`. Read `docs/ROUTER_CAPABILITIES.md`
first for what the engine can actually do and what is measured vs assumed.

## Decisions taken (recorded so they are not relitigated)

| Decision | Choice | Rationale |
|---|---|---|
| Action space | **1-D heading**, fixed step | Smallest thing that can possibly learn. Upgrade path defined below. |
| Timeline | **Open-ended** | Full curriculum through diff pairs, length tuning, rip-up, held-out eval. |
| `switch_layer()` | **Debug before training** | 0-for-32 live. Two-layer routing raises the completion ceiling enough to be worth a session first. |
| Net ordering | **Not learned** — fixed heuristic | Removes a combinatorial dimension. Revisit only on plateau. |
| Board representation | **Line segments, not raster** | See below. |

## Why the LLM path ends here

First real Qwen3-4B run: 2/3 nets, ~9–62 s per decision, one net at 933 s for
15 steps. The context-overflow and repetition-collapse bugs were real and are
fixed, but they were symptoms. ~10 s × ~20 decisions × 24 nets is over an hour
per board for a decision a 200k-parameter network makes in ~0.1 ms.

The LLM stays in the repo as a **debugging oracle**, not as the router: when
the policy fails a board, the same board can be handed to `RoutingAgent` and
its reasoning read. That was the original reason for the pivot away from RL and
it is preserved rather than discarded.

## Representation: the board is already lines

`get_board_geometry()` returns track segments with exact endpoints. Everything
else on the board reduces to segments too. Three reasons this beats a raster:

1. **No rasterizer to write.** CFP's build order had it as step 2; never started.
2. **It removes the expensive part.** Measured on a T4, the canvas encoder was
   **71% of a forward pass** — dense 3×3 convs over a mostly-empty binary
   raster. 32 segments through two attention layers is ~100× cheaper and runs
   on CPU.
3. **A raster cannot see the constraint that decides legality.** 256 px over a
   50 mm board is **0.195 mm/px**. Clearance is **0.2 mm**. One pixel is one
   clearance — the legality margin is sub-pixel. Segment endpoints are exact.

**Unrouted nets are lines too.** A pending net is a straight pad→pad segment.
Feeding routed copper as solid lines and pending nets as *ghost* lines, split
by a type flag, gives the policy visibility of where future nets need to go —
which is most of what CFP's "reserve plane" existed to provide — for one
one-hot bit.

### Observation spec

Local frame: **origin at the routing head, +x pointing at the target pad**,
lengths scaled by `L = 10 mm`. Rotating or translating the whole board leaves
this representation unchanged, so the policy generalizes across board poses
without ever seeing them. With no training data and RL only, that is the
single largest free win available.

```
GLOBAL (8 floats)
  0  dist_to_target / L
  1  log1p(dist_to_target / L)
  2  steps_remaining / max_steps
  3  detour_ratio = routed_length / straight_line_dist   (1.0 = ideal)
  4  head_layer                     0 or 1
  5  head_collides                  0 or 1
  6  target_layer                   0 or 1
  7  length_slack / L               0 until stage 5 (length tuning)

SEGMENTS (K = 32, 11 floats each, + a (K,) validity mask)
  0-3   x1, y1, x2, y2   local frame, / L
  4     width / L
  5-8   kind one-hot: track | pad | board_edge | pending_net_ghost
  9     same_net         0 or 1
  10    same_layer       0 or 1
```

Two details that are bug sources if skipped:

- **Canonicalize endpoint order** (sort by x then y after transform) so a
  segment's vector does not depend on the arbitrary endpoint order the bridge
  happened to return.
- **Select K by point-to-segment distance from the head**, not by endpoint
  distance — a long segment passing close by matters more than a distant
  segment with a nearby endpoint.

Pads become degenerate segments (centre→centre, `width = max(size_x, size_y)`)
so the encoder has exactly one item type. Free augmentation: mirroring y
doubles the data at zero cost.

### Policy network (~35k parameters)

```
per-segment MLP   11 → 64 → 64          (shared weights)
masked max-pool + masked mean-pool  →  128
global MLP        8 → 64
concat 192 → 128 → 128
  actor  → 1 mean + 1 learned log_std
  critic → 1
```

Small enough to train on CPU beside the env workers — no GPU contention, and
the bottleneck stays where it belongs, on PNS.

### Action

`a ∈ [-1, 1]` → turn angle in `[-90°, +90°]` relative to the target direction,
then advance a fixed 1 mm. A turtle.

Because the frame points at the target, **a mean-0 untrained policy walks
straight at the pad** — training starts at the greedy baseline instead of below
it. That is the property CFP wanted from its zero-init flat field, obtained by
choosing coordinates well rather than by architecture.

Termination is automatic: within snap radius → `fix()`. No learned stop action.

**Upgrade path, in order, only when the previous one plateaus:** (1) add step
length as a second dim; (2) add a discrete via/layer-hop head, *gated on
Gate A below*; (3) add a rip-up action for dense boards.

### Reward

```
r_t = γΦ(s_{t+1}) − Φ(s_t)          Φ(s) = −dist_to_target(s) / L
      − 0.02                        step cost
      − 0.5 · head_collides         the only real per-step failure signal
terminal
      + 10   fix() succeeded
      − 5    timeout / abandoned
      − 2.0 · max(0, detour_ratio − 1)
per episode (multi-net stages only)
      − w_drc · drc_error_count     run_drc() is a full-board check: once, never per step
```

The collision term is load-bearing and rests on an assumption Gate B measures.

## Stage 0 — two gates, before any trainer is written

### Gate A: why `switch_layer()` is 0-for-32 — BLOCKED, then unblocked

**First run segfaulted before answering anything.** Trial 1 completed
(`switch_layer(2)` rejected); trial 2 crashed inside `start_route()`. Root
cause found and fixed: `PNS_BRIDGE::LoadBoard()` tore its old world down in
the wrong order — freeing the `BOARD` while the old interface and router
still pointed at it, then freeing the interface while the old router still
held it, so `~ROUTER()`'s `ClearWorld()` ran against two freed objects. A
use-after-free that corrupts the heap, returns `true`, and crashes at a
distance. The diagnostic was the first thing in this repo ever to call
`LoadBoard()` twice on one bridge; the envs `reset()` between episodes and
every other script loads exactly one board.

Two changes came out of it, and **both need one rebuild**:
- `pns_bridge.cpp` destroys router → interface → settings → board before
  installing the new one (the same order `~PNS_BRIDGE()` already had, which
  is why the destructor was always safe and only this path was not).
- The diagnostic no longer depends on that path: **one `load_board()` per
  board** instead of per trial, trials separated by `reset()`, every row
  printed and flushed as it completes, and the THT board loaded last — so a
  future crash costs one trial rather than the entire run.

Known before either run: rejections are **position-independent** (15/15 at a
straight-line midpoint in open board space, nowhere near a pad), which rules
out local contention.

Implemented as `scripts/diagnose_layer_switch.py`.

| # | Hypothesis | Discriminating trial |
|---|---|---|
| 1 | Via size never configured | Run the identical route with (0.6/0.3), (0.4/0.2), and nothing set. Fix is committed (dc9164e) but **never re-run** — cheapest test, do it first. |
| 2 | `LINE_PLACER::SetLayer()` refuses unless the **start item spans the target layer** | `generate_board.py` pads are `PAD_ATTRIB_SMD`, F_Cu only, so an SMD-started route can never legally reach B_Cu. Tested via the generator's new `--pad-type tht`: same board, same trial, SMD-started vs THT-started. |
| 3 | Router state machine — legal only when IDLE | Call `switch_layer()` before `start_route()` at all, and before the first push, and compare. |
| 4 | Wrong primitive entirely | `toggle_via_placement()` → `push()` → `fix()`, then read `get_board_geometry().vias` and `get_head_geometry().layer`. This is how the GUI actually changes layer mid-route. **If this works, `switch_layer()` is not needed at all.** |

Plus two controls that kill whole classes of false conclusion:

- **Same-layer no-op** — switch to the layer the head is *already* on. H1 and
  H2 both predict this succeeds (no via needed; a pad trivially spans its own
  layer). A rejection means the call is refused structurally, before geometry
  is considered, and knocks out H1 and H2 together.
- **Layer-id sweep** — a wrong `PCB_LAYER_ID` and a structural refusal are
  the same `False`. **Measured against the local KiCad 9.0 install: `F_Cu` is
  0 but `B_Cu` is 2** — not the pre-9 `31`, and not the `1` that
  `fake_bridge.py`'s toggle informally assumes. Nothing in the tree records
  which value the historical 0-for-32 runs passed, since
  `measure_layer_hop_rescue.py` takes `--back-layer` as a required argument.
  The sweep tries each candidate and adopts whichever is accepted.

The script's verdict engine prints the conclusions, not just a table of
booleans — per `AGENTS.md` the Colab side reports output rather than
diagnosing, so the reasoning has to be *in* the output.

Every trial reads `get_head_geometry().layer` back afterwards — a `False`
return with a changed layer, or a `True` return with an unchanged one, is
itself a finding. Run on a **1-net board** so nothing can be blamed on
contention.

Hypothesis 2 is the leading one: it is the only one that explains uniform
position-independent failure *and* is consistent with 4 succeeding.

### Gate B: PASSED — measured, 24-net board, Colab

**The dense per-step signal is real, and the separation is total:**

| | collision fired | n |
|---|---|---|
| `fix()` **rejected** | **1.00** | 90 |
| `fix()` **accepted** | **0.00** | 9 |

Separation +1.00 across 99 attempts, with the signal first firing a mean of
**1.9 pushes before** `fix()` was called. The `--no-collision-trace` control
reproduced 9/24 direct successes with an identical failed-net list, so
probing mid-route does not perturb the router and the correlation stands.

**The 1-D heading action space and the per-step collision penalty are
viable as specified.** No redesign needed.

Per-call wall clock (ms), which changes the observation design:

| call | mean | median | max | share |
|---|---|---|---|---|
| `run_drc` | 267.4 | 267.4 | 267.4 | **73.2%** |
| `get_board_geometry` | 8.6 | **0.13** | 76.8 | 21.3% |
| `push` | 0.027 | 0.024 | 0.124 | 1.7% |
| `head_collides` | 0.004 | 0.003 | 0.022 | 0.2% |
| `get_head_obstacle` | 0.004 | 0.003 | 0.010 | 0.2% |
| `fix` / `start_route` | 0.007 / 0.009 | — | — | <0.5% |

`T_pns`: mean 4.03ms, **median 0.86ms**, p90 1.03ms, max 77.69ms. The mean is
dragged by a single 77ms outlier that coincides with `get_board_geometry`'s
76.8ms max — one cold call, not the steady state. Colab gave **`nproc` = 2**.

Three consequences:

1. **Do not call `get_board_geometry()` every step.** Committed copper only
   changes when a net *finishes*, so fetch the board once per net and rebuild
   only the head-relative part per step. That takes a step from ~0.17ms to
   ~0.035ms — the difference between the observation being the dominant cost
   and being free.
2. **`run_drc()` once per episode, never per step.** Already the design; the
   measurement makes it non-negotiable at 73% of total time.
3. **The script's own throughput verdict is stale.** It compares `T_pns`
   against CFP's 14M-parameter GPU numbers (0.766 ms/board). This plan's
   policy is ~35k parameters and CPU-resident, so the GPU-bound inequality no
   longer applies. At ~0.035ms/step the env is not the constraint; `nproc`=2
   caps parallelism at 1–2 workers, which is ample.

Baseline established at the same time: **9/24 direct straight-push
successes, 0/24 rescued** by the polyline and 1–2mm perpendicular detours.
Read that as a weak baseline, not as a verdict on single-layer routing — pads
are obstacles from the very first net (which is why `net_0` failed on an
otherwise-empty board), and a 1–2mm detour cannot clear a 1mm pad plus 0.2mm
clearance. An agent with free heading and 20+ steps has far more room.
**9/24 is the number to beat.**

<details>
<summary>What the instrumentation measures (original spec)</summary>

One run of `scripts/measure_waypoint_fidelity.py`, now instrumented to also
answer:

- **Per-call wall clock** for `push`, `fix`, `commit_routing`,
  `get_board_geometry`, `get_head_geometry`, `head_collides`, `run_drc`. Sets
  the worker count and every throughput estimate — all of which are currently
  guesses.
- **The credit-assignment question.** `push()` accepted 72/72 while `fix()`
  rejected ~67%: success is silent, failure is late. So log, per net, the full
  sequence of `head_collides()` / `get_head_obstacle()` and whether `fix()`
  eventually succeeded, then compare
  `P(collision fired before fix | fix failed)` against
  `P(collision fired before fix | fix succeeded)`.

  **A clear separation means the per-step reward is real and this plan works.
  No separation means one terminal bit must carry ~20 steps of credit, and the
  action granularity has to change before anything else is built.**

- **Waypoint fidelity under contention.** Deviation was ~0.7 µm on one
  unobstructed push. If PNS drags the head far from where it was told on a
  dense board, the action has no causal effect and nothing can learn.
  *Measured: final-endpoint deviation mean 0.0000mm, max 0.0000mm across all
  9 completed nets. Fidelity is exact; this risk is closed.*

</details>

## Trainer

PPO, extending `pcbworld/agents/ppo_baseline.py` (GAE and the clipped surrogate
already work; it has never seen a real reward signal).

- **1–2 worker processes**, one bridge each (Colab reports `nproc` = 2 —
  the original "8 workers" figure was written before that was known).
  Processes, never threads — hard constraint 2. At ~0.035ms per env step
  this is not a throughput problem.
- **Pre-generate a board pool** (~200 seeds) as a separate process *before*
  training. Workers cannot generate boards themselves: `generate_board.py`
  needs system `pcbnew`, which can never share a process with the bridge —
  hard constraint 1. Workers sample from the pool.
- **A worker switching boards depends on the `LoadBoard()` fix.** Calling
  `load_board()` twice on one `PNS_BRIDGE` was a use-after-free that
  segfaulted the process at a distance (it corrupted the heap, returned
  true, and crashed inside the *next* `start_route()`). Fixed in
  `pns_bridge.cpp` by tearing down router → interface → settings → board
  before installing the new board, but **not yet re-verified in Colab**.
  Until it is, a worker can only be trusted with one board per process —
  which would mean process-per-board rather than a sampled pool.
- Rollout 256 steps/worker, 4 epochs, minibatch 512, clip 0.2, γ 0.99,
  λ 0.95, entropy 0.01 decaying to 0.001.
- Observation normalization (running mean/std) on the global vector only;
  segment coordinates are already scaled.
- **Checkpoint to Drive every N updates.** Colab sessions die. Non-negotiable.

## Curriculum

Auto-advance at >80% success on the current stage.

| Stage | Board | Target |
|---|---|---|
| 1 | 1 net, empty | ~100%. If this does not happen within ~10 min of training, the plumbing is broken, not the idea. |
| 2 | 8 nets, sequential, shortest-first | Beat the measured 33% straight-line baseline. |
| 3 | 24 nets, dense | Beat stock KiCad (below) on completion % at equal or lower wirelength. |
| 4 | + differential pairs | Engine does the coupling (verified: 0.15 mm gap exact). Policy only picks where. |
| 5 | + length-matched groups | Engine does the meandering (verified: 30.0000 mm to target). "Matched" is a tolerance — a real tune left a 0.2505 mm residual. |
| 6 | Everything, + rip-up action | The action PCBWorld's paper structurally could not have. |

## Baselines and evaluation

Free, because `set_collision_mode()` already exposes stock behavior:

| Baseline | What it is |
|---|---|
| B0 | Single straight-line push per net — **measured, 33%** |
| B1 | `RM_WALKAROUND`, greedy straight line |
| B2 | `RM_SHOVE`, greedy straight line — KiCad's own interactive behavior |

Report completion %, total wirelength, via count, DRC errors, and wall-clock,
on **held-out board seeds never trained on**.

## Debuggability — the reason RL was abandoned once

That concern was correct and gets a real answer, not a shrug:

- **Render every failed episode** with `pcbworld/viz/render_board.py` (already
  written, no `pcbnew` dependency, works in-process). A contact sheet of 100
  failures shows the failure mode at a glance; a reward curve never will.
- **Log `get_head_obstacle()` per step** — net name, item kind, position. Same
  information the LLM produced in prose, structured and without hallucination.
- **Hand failing boards to the LLM agent** and read its reasoning.

A visual transcript instead of a textual one. For a geometry problem that is an
upgrade, not a regression.

## Parked, and why

| Thing | Why |
|---|---|
| `pcbworld/agents/cfp/` (14 M params) | Its raster cannot represent clearance (0.195 mm/px vs 0.2 mm). Rasterizer, A\* planner, and env were never built. |
| Qwen in the training loop | ~10⁵× too slow per decision. Retained as a debugging oracle. |
| Learned net ordering | Fixed heuristic instead. One combinatorial dimension removed. |
| Via/layer actions | Blocked on Gate A. |

## Risks

| Risk | Signal | Status |
|---|---|---|
| No dense signal (`head_collides` uninformative) | Gate B separation | **CLOSED** — +1.00 separation, n=99 |
| `T_pns` too slow | Gate B | **CLOSED** — median 0.86ms/net, ~0.035ms/step |
| Waypoint infidelity under contention | Gate B deviation | **CLOSED** — 0.0000mm mean and max |
| `load_board()` unsafe to call twice | Gate A segfault | Fixed in C++, **awaiting rebuild**. Blocks the sampled board pool until verified |
| Colab session death | — | Drive checkpointing every N updates |
| Only 2 vCPUs | `nproc` = 2 | Accepted — at ~0.035ms/step, 1–2 workers is ample |
| Stage 3 plateaus | Completion % flat vs B2 | Add step-length dim, then rip-up action, then reconsider learned net ordering |
