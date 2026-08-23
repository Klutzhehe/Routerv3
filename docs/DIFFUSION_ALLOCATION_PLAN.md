# Diffusion allocation field over CFP's towers, feeding the line-geometry executor

Answers one question: give the agent real board-wide "spatial imagination" —
grid, GNN, transformer, diffusion — without repeating either of the two
*measured* failures already on record in this repo.

## Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Global representation | **Grid, but coarse and never load-bearing for legality** | Fixes the 0.195mm/px-vs-0.2mm-clearance failure in `docs/AI_ARCHITECTURE.md` by construction: the grid only ever biases a choice, PNS always makes the final call in real nm. |
| Global generator | **Conditional diffusion**, not CFP's direct-regression field head | Multi-net allocation has many equally-valid solutions (net A left, net B right vs. the reverse). A regression head is mean-seeking and averages modes into a useless blur; a diffusion model represents the distribution and samples one coherent solution. |
| Netlist relations | **Reuse `pcbworld/agents/cfp/modules.py` wholesale** | `RelationalSelfAttention` (typed edges: same-net, diff-pair, length-group) + `CrossBlock` (canvas ↔ nets) is already built, already tested, already the GNN+transformer piece being asked for. Do not rebuild it. |
| Local execution | **Keep `LineRouteEnv` exactly as validated** | It is the only piece of any grid/vector design in this repo actually measured against the real PNS bridge (Gate A/B passed, 9/24-net baseline). Nothing here replaces it — the diffusion field becomes a handful of extra input features, not a new action space. |
| Legality / geometry | **100% PNS, always in nm, never grid-snapped** | The grid's job is "which corridor", never "which micron". This is the one rule that makes the resolution decision above safe. |
| Training data for diffusion | **Classical solver labels, generated offline** | Sidesteps the exact failure mode from the last training run: an RL agent trying to learn global allocation *and* local geometry at once from a sparse, badly-scaled reward. Supervise the global part; RL only the local part. |

## Why not a bigger from-scratch grid CNN (`pcbworld/environment.py`'s path)

That env is architecturally the CFP idea minus everything CFP got right: no
netlist tower, no coarse/fine resolution split, one shared trunk feeding both
actor and critic with a 1000-vs-1 reward scale mismatch. The 0% success /
exploding value-loss run is a training bug (diagnosed earlier this session),
not evidence against grids — but finishing it would still walk straight into
CFP's two *already-measured* problems the moment it worked well enough to
need real precision or real board sizes. Building on CFP instead of that file
skips both.

## Architecture

```
                    board state (obstacles, committed copper, pads)
                    + pending net list (class, width, priority,
                      diff-pair/length-group membership)
                              │
              ┌───────────────┴────────────────┐
              │                                 │
       CanvasEncoder                      NetEncoder
       (CFP's, coarse-biased —            (CFP's RelationalSelfAttention,
        see resolution policy below)       typed edges: same-net,
              │                             diff-pair, length-group)
              │                                 │
              └───────────CrossBlock────────────┘
                    (canvas ↔ nets, ×2 rounds — CFP's fusion, unchanged)
                              │
              ┌───────────────┼────────────────────┐
              │               │                    │
      net-priority head   DIFFUSION field head   value head
      (replaces the       (denoises a per-net     (unchanged)
       fixed net_order     coarse corridor field,
       heuristic)          FiLM-conditioned on
                            the net embedding —
                            same conditioning
                            mechanism CFP already
                            has, new denoising
                            objective)
                              │
                    coarse allocation field
                    (~0.5–1.0 mm / cell)
                              │
              fan-sampled at N candidate headings
              around the current bearing
                              │
                    extra scalar features appended
                    to line_obs.py's observation vector
                              │
                     LineRouteEnv's ~35k-param
                     turtle-walk policy (unchanged
                     action space, unchanged reward
                     shaping, unchanged PNS legality)
                              │
                    push() / fix() — real nm, real DRC
```

## Resolution policy — the load-bearing decision

Two grids, two jobs, never confused:

| Grid | Cell size | Job | Precision needed |
|---|---|---|---|
| Diffusion allocation field | ~0.5–1.0 mm | "Which corridor should this net lean toward" | Corridor-scale. Wrong by half a cell costs nothing — it only nudges a heading choice that PNS still validates. |
| Actual routing geometry | exact nm | "Is this specific move legal" | Sub-micron. Handled entirely by `push()`/`head_collides()`/`fix()`, exactly as today. |

The field is *read*, never *written into*. There is no step anywhere that
converts a grid cell into committed copper. This is the direct fix for
`docs/AI_ARCHITECTURE.md`'s measured objection ("0.195mm/px cannot represent
0.2mm clearance") — that objection only applies if the raster *is* the
geometry, which it structurally cannot be here.

Canvas compute follows the same logic that already halved CFP's FLOPs:
`canvas_blocks_per_stage` stays coarse-biased (residual capacity at 32×32 and
16×16, near-zero at the finer stages), because corridor-level allocation
never needed 256×256 resolution in the first place — that resolution existed
in CFP only because the field head was, at the time, still being asked to
imply exact geometry.

## Diffusion field generator

**Why diffusion over CFP's original regression head:** the field head never
got a real training signal — the rasterizer, A* planner, and env it needed
were never built, so it trained on nothing. Separately and more
fundamentally: for N pending nets sharing a board, "who gets which side of
the channel" has multiple equally valid answers. A regression head trained
to minimize MSE against many valid solved boards learns the *average* of the
modes, which is a path through the wall between them — useless. A diffusion
model conditioned the same way (FiLM on the net embedding from the fusion
tower, exactly as CFP's `field head` already is) can sample one coherent mode
instead of averaging all of them.

**Training data — generated offline, no RL, no PNS bridge:**

1. `board_generator.generate_random_board()` — already exists, produces N
   random multi-net boards with obstacles.
2. Solve each board with a classical sequential router: for each net in
   priority order, relax `pcbworld/env/geodesic.py`'s `GeodesicField` (already
   an obstacle-aware wavefront cost-to-go field, already unit-tested against
   hand-computed geometry) against the *current* occupancy, take its
   `descent_direction`-traced path as that net's corridor, mark it occupied,
   move to the next net. This is `LineRouteEnv`'s existing per-net field,
   just run in a loop with shared occupancy instead of once per net in
   isolation — no new solver to write, only a multi-net wrapper around one
   that already exists and is tested.
3. Rasterize each solved net's corridor into the coarse (0.5–1.0mm) grid as
   the diffusion target.
4. This produces effectively unlimited (board, pending-nets, target-field)
   triples at zero RL sample cost and zero PNS bridge calls — directly
   avoiding the sample-inefficiency that sank the last training run, where
   global allocation was being asked to emerge from a sparse terminal reward
   at the same time as local geometry.

**Loss:** standard conditional diffusion (predict-noise or v-prediction) over
the coarse field, conditioned via the existing `FiLM` module on
`(canvas embedding, net embedding)` from the fusion tower.

## Local executor changes (small, additive)

`line_obs.py`'s observation vector gains one block:

```
FIELD READOUT (M = 6 headings, fan around current bearing, 1 float each)
  sample the diffusion field at step_size * k ahead, k = 1..M headings
  spanning ±90° — "how much does the allocation plan favor this direction"
```

Net ordering: the fixed shortest-first heuristic (`RL_PLAN.md`'s "not
learned, revisit only on plateau") is replaced by the net-priority head's
output, which now has real information to work with (diff-pair/length-group
relations, board-wide congestion) that the flat heuristic never had. This is
the one place CFP's *pointer head* concept survives unchanged.

Nothing else about `LineRouteEnv` changes: same 1-D heading action, same
potential-based shaping, same collision/DRC handling entirely through the
bridge. The measured 9/24 baseline and Gate A/B results stay valid — this is
strictly additive context, not a new agent.

## Training phases

| Phase | What | Needs PNS bridge? | Needs RL? |
|---|---|---|---|
| 0 | Generate solved-board dataset (classical sequential geodesic solver) | No | No |
| 1 | Supervised diffusion pretraining on Phase 0 labels | No | No |
| 2 | `LineRouteEnv` PPO fine-tune with frozen diffusion field as extra features + learned net ordering | Yes | Yes (same trainer as `RL_PLAN.md`) |
| 3 (optional) | Unfreeze the fusion tower, let the RL reward correct cases where the classical label was suboptimal against what PNS actually allows | Yes | Yes |

Phases 0–1 are the part that directly fixes what went wrong in the failed
run: global spatial reasoning is learned from cheap, abundant, exactly-labeled
data instead of from a miscalibrated sparse RL reward. Phase 2 only has to
learn the much smaller problem of *using* a field that already points
somewhere sensible — closer to the "walk straight at the target, learn when
to deviate" starting point `RL_PLAN.md` already established, now with
deviations that account for other nets instead of only the current one's
obstacles.

## File plan

```
pcbworld/diffusion/
  labels.py       # Phase 0: multi-net sequential solve using geodesic.py,
                   # shared occupancy, rip-up-on-conflict retry
  dataset.py       # board -> (coarse canvas, net features/edges, target field)
  unet.py          # small conditional denoiser; FiLM hooks reuse
                   # pcbworld.agents.cfp.modules.FiLM directly
  sample.py        # DDIM-style sampling at inference: canvas+nets -> field

pcbworld/agents/cfp/
  (unchanged) modules.py, spec.py — reused as the shared canvas/net/fusion
  towers for both the diffusion head and (if Phase 3 happens) the value head

pcbworld/env/line_obs.py
  + field readout block (M=6 floats), config flag to make it optional so
    LineRouteEnv keeps working with the field absent (falls back to today's
    behavior — no regression risk to the validated baseline)

pcbworld/env/line_route_env.py
  + optional `allocation_field` passed into `_observe()`
  + net ordering delegated to the priority head's output when provided,
    else the existing heuristic (unchanged default)

training/train_diffusion.py   # Phase 1, supervised, no env
training/train.py             # Phase 2/3, extends the existing PPO loop
```

## Curriculum

Same shape as `RL_PLAN.md`, gated by whether the field is present:

| Stage | Board | Target |
|---|---|---|
| 1 | 1 net, empty, no field | Reconfirm today's ~100% (regression guard) |
| 2 | 8 nets, sequential, field ON | Beat stage 2's existing straight-line baseline |
| 3 | 24 nets, dense, field ON | Beat stock KiCad completion % — this is the stage the field exists for |
| 4+ | diff pairs, length-matched groups | Field's reserve-plane concept (from CFP) carries over unchanged: an extra plane that only adds cost for *later* nets |

## Risks

| Risk | Mitigation |
|---|---|
| Classical sequential solver (Phase 0 labels) is itself a weak baseline | It only has to be *better than uniform/no signal* for the diffusion model to be worth conditioning on — it doesn't need to be optimal. Phase 3 lets RL correct residual suboptimality. |
| Diffusion sampling latency at inference (many nets per episode) | Field is per-*board*, not per-*step* — sample once per net (or once per episode if all nets are known upfront), same amortization already used for `GeodesicField` and `_refresh_static_segments()`. |
| Coarse field still leaks precision assumptions somewhere | The one invariant to test for: no code path may read the field and write to `bridge.push()`/`fix()` without an intervening PNS legality check. Keep this as an explicit review item before Phase 2 lands. |
| Two-tower network (14M params) reintroduces GPU-bound training | Already measured and mitigated in `AI_ARCHITECTURE.md` (`canvas_blocks_per_stage`, fp16 @ batch≥32, pipelined trainer). Same numbers apply — this reuses that network, not a new one. |
