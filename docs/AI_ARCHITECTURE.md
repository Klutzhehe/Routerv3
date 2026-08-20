# CFP — the Cost-Field Policy

The agent does not learn to draw traces. It learns to paint the cost map the
classical router walks on.

PNS is already a strong *local* geometer: it shoves, it couples differential
pairs, it inserts meanders, it enforces DRC. It is weak at exactly one thing,
**global space allocation** — which net gets the channel, where to leave a hole
for a meander, how wide a corridor to keep clear. That is the only thing the
network learns.

## One RL step = one net

Not one 200 µm push. `SimpleRouteEnv`'s per-step `(dx, dy)` action is superseded
by this; an episode becomes 20–200 decisions instead of 80 micro-moves per net.

```
board raster ──► CFPNet ──► ┌─ pointer head: which net/group, route or rip up
 (B,12,256,256)             └─ field head:   (B, L+1, 16, 16) Gaussian
                                       │
                              bilinear upsample
                                       │
                            deterministic A* (src pad → dst pad,
                            cost = field + occupancy + reserve + via penalty)
                                       │
                                waypoints (5–20 pts)
                                       │
                        bridge.push() × N  →  bridge.fix()
                        (MODE_ROUTE_SINGLE / DIFF_PAIR / TUNE_SINGLE)
                                       │
                            new board state, reward
```

**A zero field reproduces stock PNS.** A\* on a flat field is a near-shortest
path, which is roughly what the interactive router does anyway. The agent starts
at the classical baseline rather than below it — the network is initialized (small-gain
orthogonal heads, zero-init FiLM) so an untrained policy emits a flat field.

## Why this handles diff pairs and length tuning

- **Differential pairs** need a *corridor*, not a path. A cost field is the natural
  representation of a corridor: a low-cost valley two traces wide. PNS's diff-pair
  placer does the coupling; the agent only has to leave it room.
- **Length tuning** is a *space reservation* problem. Every classical router fails
  it the same way — by the time you know you need 4 mm of meander, the area is
  full. The last field plane is a **reserve plane**: it does not affect the current
  net's path, it is added to the cost every later net pays. That is the only
  mechanism in the architecture that can say "keep this area empty", and the
  `meander_demand` input channel tells the policy how much to reserve.
- **Rip-up and reroute**, which dense boards require, is one extra token in the
  pointer softmax. Not a separate module, not a hierarchy.

## Network — two towers, cross-attentive

Pure-graph loses the board; pure-CNN loses the netlist. Both towers, fused
bidirectionally:

| Tower | Input | Job |
|---|---|---|
| Netlist | `(B, N, 16)` net features + `(B, N, N)` typed edges | relational structure: diff-pair partners, length-group cliques, bbox contention |
| Canvas | `(B, 12, 256, 256)` raster | geometry: congestion, corridors, occupancy |

Fusion runs both directions, twice: nets read the canvas (attention biased by
`-α·distance²`, so a net looks near its own pads), then the canvas reads the nets,
then nets re-mix relationally. `canvas → nets` is the load-bearing half — it is
what lets a cell know *which* nets want it and of what type.

The field head is **FiLM-conditioned on the net the pointer just selected**, so
sampling is autoregressive within a step: pick the net, then draw that net's field.
An unconditioned field would be a single global map and could not express "route
*this* net around the region those three diff pairs need."

**14.0 M parameters** at the default config. The binding constraint is env
throughput, not model capacity. Any change that doesn't push GPU time above env
time is free; anything that does has to earn it. A 300 M-param transformer
trained on 10⁷ samples would be strictly worse than this.

### Measured cost (Tesla T4, fp32, first pass)

| batch | ms/batch | ms/board |
|---|---|---|
| 8 | 22.1 | 2.76 |
| 32 | 77.3 | 2.41 |
| 128 | 303.9 | 2.38 |

Flat in ms/board from batch 8 upward => compute-saturated, not latency-bound.
Attribution: the canvas encoder was **71% of a forward pass**, 8.3 GFLOPs/board,
of which stage 0 (@128x128) and stage 1 (@64x64) were **68%** -- dense 3x3 convs
over a mostly-empty binary raster. The transformer towers are nearly free.

Fix applied: `canvas_blocks_per_stage` became per-stage and defaults to
`(0, 0, 1, 2)`, moving residual capacity to 32x32 and 16x16 where a block costs
16-64x less. **2.04x fewer FLOPs, and parameter count rises** 13.1M -> 14.0M --
a relocation, not a cut.

Remaining levers, in order of value:

| lever | factor | cost |
|---|---|---|
| per-stage blocks `(0,0,1,2)` | 2.0x | done |
| canvas 256 -> 128 px | 4.0x | env-side; 0.39 mm/px on a 50 mm board. The A\* planner grid is independent, so this only coarsens *allocation* context |
| fp16 autocast (T4 tensor cores) | ~2-3x | `--amp`, free |
| `torch.compile` | ~1.2-1.5x | compile time |

### The metric to use

**ms/batch, not ms/board.** Env workers route concurrently, so one rollout round
costs the env a single net-route (T_pns) regardless of worker count, while it
costs the GPU one whole batched forward. Dividing by batch size flatters the model
by exactly the factor the workers already provided. The bet holds iff

    ms/batch at batch=num_workers  <<  T_pns for ONE net

**T_pns has never been measured.** The "single-digit ms per net" figure that
originally motivated this section was an estimate with nothing behind it. Measure
it alongside the waypoint-fidelity test -- both need the bridge built, so they are
one Colab run.

### Deliberately not built

An autoregressive decoder emitting trace coordinates token-by-token. It discards
the geometric prior and spends capacity relearning shove and coupling, which PNS
gives correct and free. The bet is: **network does allocation, PNS does geometry.**
A coordinate decoder breaks that bet.

## Reward

```
+ w_done  · net_completed
- w_len   · wirelength_nm
- w_via   · vias
- w_drc   · drc_violations
- w_rip   · is_ripup                             # anti-thrash
- w_match · max(0, |Δlength| - tolerance)        # at group completion
- w_coup  · uncoupled_fraction                   # at diff-pair completion
+ potential shaping Φ = fraction_of_nets_connected   # Ng et al., policy-invariant
terminal: + large bonus for 100% routed and DRC-clean
```

## Curriculum

`generate_board.py` already parameterizes all of it. Auto-advance a stage at
> 80 % success:

1. plain nets, low density
2. plain nets, high density (forces rip-up)
3. \+ differential pairs
4. \+ length-matched groups
5. both, at target density

## Build order

| # | Piece | Status |
|---|---|---|
| 0 | **Waypoint-fidelity test** + measure T_pns | **gates everything — see below** |
| 1 | `pcbworld/agents/cfp/` — spec, model, policy | **done**, 22 tests green, T4-verified |
| 2 | Board rasterizer (KiCad board → `CFPObservation`) | not started, Python |
| 3 | Coarse field → A\* → waypoints planner | not started, Python, ~200 lines |
| 4 | Waypoint follower (waypoints → `push()` sequence) | not started, thin |
| 5 | `rip_up(net)` on the bridge | not started, small C++ |
| 6 | `BoardRouteEnv` (one step = one net) | not started, supersedes `SimpleRouteEnv` |
| 7 | PPO trainer (vectorized, multi-process) | not started |

No C++ cost-function surgery. Injecting the field directly into PNS's cost model
was considered and rejected — PNS has no pluggable cost grid, and the waypoint
indirection gets ~90 % of the control for ~5 % of the work.

## The risk that could kill it

**Waypoint fidelity.** If PNS's shove engine routinely drags the head away from the
requested waypoints, the agent's action has no causal effect on the outcome and
nothing learns.

Test this before writing any more RL code: hand-author ~20 waypoint sequences on a
dense board, and measure how far the resulting copper deviates from the requested
polyline. Median deviation under ~2× track pitch means the architecture is sound.
If it isn't, the fix is `RM_MARK_OBSTACLES` mode (already implemented, see
`PNS_BRIDGE::SetCollisionMode`) so `push()` is a pure validator and the agent
handles avoidance through the field — which is strictly more learnable anyway.

## Training-config gotcha

The field is `num_field_planes × field_size²` = 768 Gaussian dimensions at the
default config, so its entropy is ~700 nats while the pointer categorical's is
~2. `CFPScore` therefore returns the two entropies **separately**: one PPO entropy
coefficient cannot serve both, and the field term's gradient only reaches
`field_log_std` — summing them yields a std regularizer wearing an
exploration-bonus costume. Use two coefficients.
