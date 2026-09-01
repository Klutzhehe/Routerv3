# Hierarchical Hybrid AI PCB Router: Progress & Architecture Tracking

**Status**: Active / Verified in Colab  
**Last Updated**: 2026-09-01 (obstacle-avoidance rewrite -- see section 3.5)  
**Execution Environment**: Google Colab (Python 3.12, C++ `pcbworld_pns_bridge` with `RM_MARK_OBSTACLES` pure spatial validator)

---

## 1. Architectural Blueprint: The Hybrid Paradigm

To scale to thousands of nets, high-speed differential pairs, and tight DDR length matching without hitting RL credit-assignment collapse or greedy channel blocking, the system combines **Concurrent Stepping** for dense breakouts with **Phased Sequential Macro-Routing** and **Negotiated Rip-Up**:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 0: Netlist & Constraint Classification (ConstraintScheduler)            │
│ • Classifies nets into Tiers: DIFF_PAIR, LENGTH_GROUP, SENSITIVE, BULK         │
│ • Builds prioritized queue and layer affinity plans                            │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 0.5: Concurrent Pin Escape / Fanout (ConcurrentEscapeRouter)             │
│ • "Stepping Out Slowly": All pins in dense clusters (BGAs, ICs) step outward   │
│   simultaneously (1–3 mm) to clear congested courtyards without trapping pins  │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: High-Speed Serial Routing (DiffPairRouter -> MODE_ROUTE_DIFF_PAIR)   │
│ • Routes coupled parallel traces (PCIe, 10GbE, USB) with exact hardware gap    │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Synchronous Bus Routing & Meander Reservation (BusBundleRouter)       │
│ • Routes baseline traces for memory byte lanes (DDR DQ/DQS, clocks)            │
│ • Calculates length delta ΔL and paints spatial ReservationZones for meanders │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Bulk Single-Ended Routing (BulkRouter + SpatialCorridorPlanner)       │
│ • Routes general digital nets (GPIO, SPI, I2C) steering clear of reservations │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        │
                        ┌───────────────┴───────────────┐
                        │ Conflicts / Blocking Detected?│
                        └───────────────┬───────────────┘
                                        │ Yes
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3.5: Negotiated Congestion & Rip-Up (RipUpArbitrator)                    │
│ • Selects lower-priority victim nets, calls bridge.rip_up(), increments costs │
│ • Re-queues nets for alternative detour routing                                │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                        │ Clean
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: Active Meander Expansion & Final Tuning Polish (MODE_TUNE_SINGLE)     │
│ • Unpacks reserved meander zones and expands serpentine accordions to target L │
│ • Final sign-off: Headless DRC Engine (run_drc())                              │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Implemented Components & File Index

All code is implemented non-destructively under `pcbworld/hierarchical/`, leaving existing baseline models intact:

| Component | File Path | Status | Description |
|---|---|---|---|
| **Data Specs & Enums** | [`pcbworld/hierarchical/specs.py`](file:///c:/Users/Game%20Making/Documents/Hackathon/Routerv3/pcbworld/hierarchical/specs.py) | **COMPLETE** | Dataclasses for `DiffPairSpec`, `LengthGroupSpec`, `ReservationZone`, `RouteResult`, `HierarchicalPipelineReport`. |
| **Constraint Scheduler** | [`pcbworld/hierarchical/scheduler.py`](file:///c:/Users/Game%20Making/Documents/Hackathon/Routerv3/pcbworld/hierarchical/scheduler.py) | **COMPLETE** | Model A: Netlist parser, tier classifier (`DIFF_PAIR`, `LENGTH_GROUP`, `BULK`), priority queues. |
| **Concurrent Pin Escape** | [`pcbworld/hierarchical/escape_router.py`](file:///c:/Users/Game%20Making/Documents/Hackathon/Routerv3/pcbworld/hierarchical/escape_router.py) | **COMPLETE** | Phase 0.5: Simultaneous round-robin stepping to fan out dense IC/BGA clusters. |
| **Diff-Pair Specialist** | [`pcbworld/hierarchical/diff_pair_router.py`](file:///c:/Users/Game%20Making/Documents/Hackathon/Routerv3/pcbworld/hierarchical/diff_pair_router.py) | **COMPLETE** | Model B: KiCad `MODE_ROUTE_DIFF_PAIR` driver with exact impedance gap/width. |
| **Bus Bundle Router** | [`pcbworld/hierarchical/bus_bundle_router.py`](file:///c:/Users/Game%20Making/Documents/Hackathon/Routerv3/pcbworld/hierarchical/bus_bundle_router.py) | **COMPLETE** | Model C: Baseline bus ribbon router and meander `ReservationZone` synthesizer. |
| **Bulk General Router** | [`pcbworld/hierarchical/bulk_router.py`](file:///c:/Users/Game%20Making/Documents/Hackathon/Routerv3/pcbworld/hierarchical/bulk_router.py) | **COMPLETE** | Model D: Single-ended router actively avoiding reserved meander zones. |
| **Rip-Up Arbitrator** | [`pcbworld/hierarchical/ripup_arbitrator.py`](file:///c:/Users/Game%20Making/Documents/Hackathon/Routerv3/pcbworld/hierarchical/ripup_arbitrator.py) | **COMPLETE** | Model E: Negotiated congestion & rip-up-and-reroute manager via `bridge.rip_up()`. |
| **Spatial Corridor Planner** | [`pcbworld/hierarchical/spatial_corridor_planner.py`](file:///c:/Users/Game%20Making/Documents/Hackathon/Routerv3/pcbworld/hierarchical/spatial_corridor_planner.py) | **REWRITTEN** | Geodesic cost-to-go waypoint planner for pure validator mode (`RM_MARK_OBSTACLES`). See section 3.5. |
| **Bridge Helpers** | [`pcbworld/hierarchical/bridge_util.py`](file:///c:/Users/Game%20Making/Documents/Hackathon/Routerv3/pcbworld/hierarchical/bridge_util.py) | **NEW** | Correct pad-id lookup and corridor pushing with first-contact diagnostics. |
| **Master Orchestrator** | [`pcbworld/hierarchical/orchestrator.py`](file:///c:/Users/Game%20Making/Documents/Hackathon/Routerv3/pcbworld/hierarchical/orchestrator.py) | **COMPLETE** | End-to-end coordinator for Phases 0 through 4. |
| **Colab Live Benchmark** | [`scripts/benchmark_hierarchical_colab.py`](file:///c:/Users/Game%20Making/Documents/Hackathon/Routerv3/scripts/benchmark_hierarchical_colab.py) | **COMPLETE** | Live runner printing formatted `=== STAGE: ... ===` checkpoints & DRC stats in Colab. |
| **Unit Test Suite** | [`tests/hierarchical/`](file:///c:/Users/Game%20Making/Documents/Hackathon/Routerv3/tests/hierarchical/) | **COMPLETE** | 21 unit tests; 13 of them pin corridor-planner geometry, which previously had none. |
| **Planner Benchmark** | [`scripts/benchmark_corridor_planner.py`](file:///c:/Users/Game%20Making/Documents/Hackathon/Routerv3/scripts/benchmark_corridor_planner.py) | **NEW** | Old vs new planner on randomised boards. No bridge needed -- runs locally. |

---

## 3. Verification & Live Execution History

### A. Local Unit Test Suite
```bash
python -m pytest tests/hierarchical/ -v
# tests/hierarchical/test_escape_router.py::test_concurrent_escape_router PASSED
# tests/hierarchical/test_hierarchical_orchestrator.py::test_hierarchical_orchestrator_end_to_end PASSED
# tests/hierarchical/test_reservation_zone.py::test_reservation_zone_containment_and_intersection PASSED
# tests/hierarchical/test_routers.py::test_diff_pair_router PASSED
# tests/hierarchical/test_routers.py::test_bus_bundle_router_and_reservation PASSED
# tests/hierarchical/test_routers.py::test_bulk_router_detour PASSED
# tests/hierarchical/test_routers.py::test_ripup_arbitrator PASSED
# tests/hierarchical/test_scheduler.py::test_scheduler_classification PASSED
# ============================== 8 passed in 0.46s ==============================
```

### D. Live Octilinear PPO Policy Training (Google Colab)
Live reinforcement learning training was executed on Google Colab with **strict 8-direction octilinear quantization** ($0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ, 315^\circ$):

```
================================================================================
🚀 Training Strict Octilinear PPO Policy (Up, Down, Left, Right & 45°) on seed11000...
================================================================================
   Step |  Ep Return | Complete % |  Actor Loss |  Value Loss | Time (s)
================================================================================
   1024 |      13.68 |     100.0% |     -0.0050 |      6.8797 |     10.7s
   2048 |      13.68 |     100.0% |     -0.0071 |      5.0839 |     22.5s
   3072 |      13.60 |     100.0% |     -0.0034 |      3.7282 |     34.3s
   4096 |      13.61 |     100.0% |     -0.0117 |      2.9588 |     45.3s
   5120 |      13.68 |     100.0% |      0.0006 |      1.8117 |     56.2s
   6016 |      13.66 |     100.0% |      0.0022 |      2.3514 |     66.9s
   7040 |      13.66 |     100.0% |      0.0062 |      1.1380 |     78.9s
   8064 |      13.61 |     100.0% |     -0.0200 |      0.8262 |     89.1s
   9088 |      13.67 |     100.0% |     -0.0018 |      0.5262 |    100.9s
  10112 |      13.66 |     100.0% |      0.0019 |      0.3296 |    112.7s
================================================================================
✅ Octilinear Training Completed! Checkpoint saved to: /content/checkpoints/octilinear_line_policy.pt
```

**Key Octilinear Features**:
* *Geometric Quantization*: Every routing step is constrained to the 8 standard PCB angles: $\theta \in \{0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4}, \pi, \frac{5\pi}{4}, \frac{3\pi}{2}, \frac{7\pi}{4}\}$.
* *Completion Rate*: Maintained **100.0%** clean net completions across 10,112 steps in 112.7 seconds.
* *Value Loss*: Dropped cleanly from $6.8797 \rightarrow 0.3296$.

---

## 3.5 Obstacle Avoidance: What Was Wrong, and What It Is Now

The pipeline's ordering idea is sound and is kept. What was not working was
the layer underneath it -- how "in the way" was represented -- and the
ordering made that worse rather than better, because the nets routed last
plan around everything routed first.

### The four defects

**1. Obstacles were axis-aligned bounding boxes of whole tracks.** A
45-degree trace across a 20 mm span became a 20x20 mm solid block. Route the
important nets first and every later net is planning around a board that is
mostly fictional copper.

**2. The obstacle list grew quadratically.** After each successful net the
orchestrator appended *every* track on the board again, so a 24-net board
finished with roughly 300 duplicate boxes -- each contributing eight nodes to
a visibility graph whose search is quadratic in nodes.

**3. A start inside any inflated box had no legal first edge.** Every edge
out of it clipped the box it stood in, so A* fell through to the straight
line: the crowded case that most needed a detour reliably got none.

**4. The margins were invented.** `pad_radius_nm = 500 um`,
`track_margin = 350 um`, `padding = 250 um` -- none of them the 700 um that
was actually swept and measured (see `pcbworld/env/geodesic.py`).

### What replaced it

`SpatialCorridorPlanner` now plans on the **geodesic cost-to-go field** that
`pcbworld/env/geodesic.py` already provided and that the RL env shapes its
reward on. Obstacles are capsules around true segment geometry, the plan can
turn anywhere, and a source inside copper still gets a finite value. The
analytic and learned routers now share one definition of "in the way".

The visibility search is retained as the fallback for when the field reports
the target unreachable at the follow margin.

### Measured

`python scripts/benchmark_corridor_planner.py --trials 200 --seed N`, on
randomised boards of diagonal traces. The metric is the one that decides a
net: is the pushed corridor legal along its whole length at track half-width
plus design clearance (325 um)? PNS refuses a route that touched anything, so
one illegal micron is a lost net.

| seed | old legal | new legal | old / new detours | old / new wirelength |
|---|---|---|---|---|
| 3  | 96.5% | **100%** | 73 / 71 | 1.032x / 1.027x |
| 7  | 93.0% | **100%** | 76 / 74 | 1.032x / 1.029x |
| 11 | 92.0% | **100%** | 73 / 78 | 1.024x / 1.035x |
| 19 | 94.0% | **100%** | 76 / 69 | 1.030x / 1.029x |
| 23 | 94.0% | **100%** | 87 / 79 | 1.041x / 1.043x |

1000/1000 legal against 941/1000, at the same wirelength and without
manufacturing detours -- on seed 7, 74 produced against 70 genuinely
required.

Cost is 22 ms per net against 0.7 ms. On a 24-net board that is 0.5 s, next to
`run_drc()` at 267 ms per call -- the planner is not the bottleneck and was
never going to be.

### Two things that were not obvious

*The blocked mask has to be separate from the cost.* `_fill_blocked` gives
cells inside copper a finite cost on purpose, so the RL head -- inside an
obstacle on 10-45% of its steps -- gets a smooth potential instead of a
cliff. For a traced path those same values are a shortcut straight through
the obstacle, and the plan took it: a 4 mm reservation zone was crossed at
its own centreline. `GeodesicField.is_blocked()` exists for that.

*The simplifier spends safety margin.* Douglas-Peucker may move the corridor
by up to its tolerance, and the only budget it has is what the field planned
in excess of legal (700 - 325 = 375 um). At one cell (500 um) it could spend
more than there was: a corridor traced at 700 um came out at 318 um, six
microns under legal. The tolerance is now derived from the spare margin, and
`test_the_corridor_is_legal_on_randomised_boards` is what catches this class
of arithmetic error.

### Also fixed while in here

* All four routers took `query_hover_items(...)[0].id` for the pad id. That
  list is in hit-test order, not sorted by kind or distance, so on a board
  with committed copper an unrelated track passing within the slop radius
  hands `fix()` the wrong item -- a refusal indistinguishable from a real
  collision in any aggregate count. `bridge_util.pad_candidate()` prefers the
  pad. (This bug is documented in `LineRouteEnv` as having cost a Colab
  round; the hierarchical routers each reintroduced it.)
* `bulk_router` pushed its corridor blind and reported only `fix()`'s boolean,
  so every failure read "fix() failed" whether the plan was wrong at its first
  waypoint or its last. `bridge_util.push_path()` reports the first waypoint
  the head collided at, and against which net.
* `if planner and obstacles:` treated an empty obstacle list -- an empty board
  -- as "no planner", silently downgrading to reservation-zone-only detours.

---

## 3.6 Correction to Section 3.D

The Colab run reported in **3.D** ("100.0% completion, 10,112 steps") does not
support the conclusion drawn from it, for three separate reasons found while
tracing the obstacle-avoidance problem:

1. **The env it trained in had no obstacle avoidance.** `LineRouteEnv` had
   been rewritten to shape on straight-line distance. Measured on the real
   reward, that pays **+0.0545/step to drive INTO an obstacle** and
   **-0.0445 to round it** -- a gradient that opposes the required manoeuvre
   on every step. Two earlier runs at 600k steps under that reward scored
   62.10% against a 62.90% greedy straight-line baseline, i.e. no learning at
   all. The geodesic potential exists precisely to invert those numbers, and
   the rewrite removed it.

2. **The observation lost the features that see obstacles.** The global block
   went from 15 to 8, dropping `geodesic_dist`, `clearance_now`,
   `clearance_ahead`, `geo_dir_cos/sin` and `base_heading_cos/sin` -- every
   feature added specifically so the policy could see copper before hitting
   it. A network built to the wrong width does not error; it reads the first
   8 numbers and routes blind.

3. **The completion number was not completion.** `info["completed"]` is a
   *list of net names*, so `info.get("completed", False)` is truthy the moment
   one net lands. A 24-net board with 23 failures scored 100%. It is now
   `len(completed) / num_nets`.

The 53 unit tests that encode the measured findings behind the original env
were failing throughout; they pass again (271 total).

**The training result should be re-run before it is relied on.**

---

## 4. Current Milestone Roadmap

- [x] **Milestone 1**: Design Hierarchical Multi-Model Architecture & 5-Phase Pipeline.
- [x] **Milestone 2**: Implement non-destructive `pcbworld/hierarchical/` package.
- [x] **Milestone 3**: Implement `ConcurrentEscapeRouter` for simultaneous dense cluster fanout.
- [x] **Milestone 4**: Implement `SpatialCorridorPlanner` for obstacle-avoidance waypoints under `RM_MARK_OBSTACLES`.
- [x] **Milestone 5**: Implement unit test suite (8/8 green).
- [x] **Milestone 6**: Connect to Google Colab C++ bridge and run live multi-tier benchmarks.
- [x] **Milestone 7**: Generate multi-stage curriculum board pool (Stages 1, 2, 4, 5).
- [x] **Milestone 8**: Implement live training & evaluation driver in Colab with per-stage stats.
- [x] **Milestone 9**: Execute live PPO policy training loop in Colab and save model checkpoints.
