# Unified RL Design: Pure RL PCB Router with Diff Pairs & Length Tuning

## Executive Summary

This design specifies a **pure reinforcement learning system** for PCB routing that:
- Uses **line-segment geometry** (not rasters) — exact, clearance-accurate, 100× cheaper
- Learns **one net at a time** with a **1-D heading action** (turtle graphics)
- Leverages **KiCad PNS primitives** for diff-pair coupling & meander insertion
- Handles **20–200 nets** via curriculum from empty → dense → diff-pairs → length-matched
- Runs on **CPU** (2 vCPUs in Colab) with **PPO**, ~35k parameter policy

---

## 1. Core Architecture Decisions (Locked)

| Decision | Choice | Rationale |
|---|---|---|
| **Representation** | Line segments from `get_board_geometry()` | No rasterizer; 0.195 mm/px < 0.2 mm clearance; 71% FLOP savings |
| **Action Space** | 1-D heading ∈ [-90°, +90°] relative to target, fixed 1 mm step | Smallest learnable; mean-0 policy walks straight at target (greedy baseline) |
| **Observation Frame** | Local: origin at head, +x toward target, scaled by L=10mm | Pose-invariant; generalizes across boards without training data |
| **Layer Changing** | **Disabled** (Gate A: 0-for-32) | Single-layer only; rip-up is the lever for density |
| **Net Ordering** | Fixed heuristic (shortest-first) | Removes combinatorial dimension; revisit only on plateau |
| **Diff Pairs / Length Tuning** | Engine primitives (`MODE_ROUTE_DIFF_PAIR`, `MODE_TUNE_SINGLE`) | PNS does coupling/meanders correctly; policy only picks *where* |
| **Rip-up Action** | Add at Stage 6 only | Structural capability PCBWorld paper lacked |

---

## 2. Observation Specification (Exact)

### Global Vector (8 floats)
```
0  dist_to_target / L
1  log1p(dist_to_target / L)
2  steps_remaining / max_steps
3  detour_ratio = routed_length / straight_line_dist   (1.0 = ideal)
4  head_layer                     0 or 1
5  head_collides                  0 or 1
6  target_layer                   0 or 1
7  length_slack / L               0 until Stage 5 (length tuning)
```

### Segment Array (K=32 segments, 11 floats each + validity mask)
Selected by **point-to-segment distance from head** (not endpoint distance):
```
0-3   x1, y1, x2, y2          local frame, / L
4     width / L
5-8   kind one-hot: [track, pad, board_edge, pending_net_ghost]
9     same_net                 0 or 1
10    same_layer               0 or 1
```

**Critical implementation details:**
- Canonicalize endpoint order: sort by x then y after transform
- Pads → degenerate segments (centre→centre, width = max(size_x, size_y))
- Free augmentation: mirror y → doubles data at zero cost
- `get_board_geometry()` fetched **once per net** (copper only changes at net finish)

---

## 3. Policy Network (~35k Parameters)

```
per-segment MLP    11 → 64 → 64          (shared weights)
masked max-pool + masked mean-pool  →  128
global MLP         8  → 64
concat 192 → 128 → 128
  actor  → 1 mean + 1 learned log_std   (Gaussian over [-1, 1])
  critic → 1
```

**Properties:**
- CPU-resident, no GPU contention
- ~0.1 ms forward pass
- Untrained mean-0 policy = straight-line router (greedy baseline)

---

## 4. Action & Dynamics

```
a ∈ [-1, 1]  →  turn angle ∈ [-90°, +90°] relative to target direction
advance fixed 1 mm
```

**Termination:** Automatic snap when within snap radius → `fix()` succeeds. No learned stop action.

**Upgrade path (only when previous plateaus):**
1. Add step length as 2nd action dimension
2. Add discrete via/layer-hop head (gated on Gate A reopening)
3. Add rip-up action for dense boards

---

## 5. Reward Function

### Per-Step (Potential-Based Shaping + Penalties)
```
r_t = γ Φ(s_{t+1}) − Φ(s_t)           Φ(s) = −dist_to_target(s) / L
      − 0.02                          step cost
      − 0.5 · head_collides           ONLY real per-step failure signal
```

### Terminal
```
+ 10    fix() succeeded
− 5     timeout / abandoned
− 2.0 · max(0, detour_ratio − 1)
```

### Per-Episode (Multi-Net Stages Only)
```
− w_drc · drc_error_count      run_drc() once per episode (267ms, 73% of time)
```

**Key insight:** `head_collides()` separates 100% (fix rejected) vs 0% (fix accepted) — fires mean 1.9 pushes before fix. This dense signal makes the 1-D action viable.

---

## 6. Environment Implementation

### Base: `LineRouteEnv` (Single-Net, Single-Layer)
- Wraps `pcbworld_pns_bridge` directly
- Observations built from `get_board_geometry()` + `get_head_geometry()` + `head_collides()`
- Action: continuous scalar → heading → push 1mm
- Auto-termination on snap + fix
- `get_board_geometry()` cached per net

### Extended: `PCBRouteEnv` (Multi-Net Sequential)
- Sequences nets via fixed heuristic (shortest Manhattan distance first)
- Episode = all nets on board
- Terminal reward includes DRC count
- Supports Stage 2 (8 nets) → Stage 3 (24 nets dense)

### Primitive-Aware: `DiffPairTuneEnv` (Stages 4–5)
- Parses net names: `diffpair_<i>_P/_N`, `lengthgrp_<g>_<member>`
- Leg sequence per ROADMAP.md:
  - Plain nets → `MODE_ROUTE_SINGLE`
  - Diff pairs → one `MODE_ROUTE_DIFF_PAIR` leg (driven by P net)
  - Length groups → reference net direct, then each member: direct + `MODE_TUNE_SINGLE`
- Observation adds 3-dim leg-kind one-hot (direct/diff_pair/tune)
- Tune legs: target length read from reference's **actual** routed length via `get_board_geometry()`
- Reward adds length-mismatch penalty at tune completion

---

## 7. Training Configuration

### PPO Hyperparameters
```
1–2 worker processes (Colab nproc=2), one bridge each
Pre-generate board pool (~200 seeds) as separate process
Rollout: 256 steps/worker
4 epochs, minibatch 512, clip 0.2
γ=0.99, λ=0.95
entropy 0.01 → 0.001 decay
Observation normalization: running mean/std on global vector only
Checkpoint to Drive every N updates
```

### Curriculum (Auto-Advance at >80% Success)

| Stage | Board | Target |
|---|---|---|
| 1 | 1 net, empty | ~100% (plumbing check) |
| 2 | 8 nets, sequential, shortest-first | Beat 33% straight-line baseline |
| 3 | 24 nets, dense | Beat stock KiCad (B2) on completion % at equal/lower wirelength |
| 4 | + diff pairs | Engine does coupling; policy picks corridor |
| 5 | + length-matched groups | Engine does meanders; "matched" = tolerance (0.25mm residual) |
| 6 | Everything + rip-up action | Action PCBWorld paper could not have |

### Baselines (Free via `set_collision_mode()`)
| Baseline | Description |
|---|---|
| B0 | Single straight-line push per net (measured 33%) |
| B1 | `RM_WALKAROUND`, greedy straight line |
| B2 | `RM_SHOVE`, greedy straight line — KiCad's interactive behavior |

**Eval:** Completion %, total wirelength, via count, DRC errors, wall-clock on held-out seeds.

---

## 8. Debuggability (Critical — Why RL Was Abandoned Once)

| Tool | Purpose |
|---|---|
| `pcbworld/viz/render_board.py` | Render every failed episode (contact sheet of 100 failures = failure mode at a glance) |
| `get_head_obstacle()` per step | Net name, item kind, position — structured, no hallucination |
| LLM Agent (`RoutingAgent`) | Hand failing boards; read reasoning as oracle |

---

## 9. Implementation Roadmap (Code to Write)

### Phase 1: Core Line-Route Env (Week 1)
- [ ] `pcbworld/env/line_route_env.py` — single-net, 1-D heading, cached geometry
- [ ] `scripts/verify_line_env.py` — smoke test against real bridge
- [ ] Verify Gate B numbers reproduce (collision separation, waypoint fidelity)

### Phase 2: PPO Trainer (Week 1–2)
- [ ] Extend `pcbworld/agents/ppo_baseline.py` for continuous action (Gaussian policy)
- [ ] Add observation normalization, reward scaling, Drive checkpointing
- [ ] `scripts/train_line_policy.py` — curriculum driver

### Phase 3: Multi-Net & Curriculum (Week 2)
- [ ] `PCBRouteEnv` multi-net sequencing
- [ ] Board pool generator (pre-gen 200 seeds via `generate_board.py` subprocess)
- [ ] Stage 1→2→3 auto-advance logic

### Phase 4: Diff-Pair & Length-Tuning Primitives (Week 3)
- [ ] `DiffPairTuneEnv` with leg sequencing
- [ ] Integrate `MODE_ROUTE_DIFF_PAIR`, `MODE_TUNE_SINGLE`
- [ ] Length-mismatch reward at tune completion

### Phase 5: Rip-Up Action (Week 4, if Stage 3 plateaus)
- [ ] Add `rip_up(net)` to bridge (C++ binding exists)
- [ ] Discrete rip-up action head
- [ ] Stage 6 curriculum

---

## 10. Risk Register (from RL_PLAN)

| Risk | Signal | Status |
|---|---|---|
| No dense signal | Gate B separation | **CLOSED** — +1.00, n=99 |
| `T_pns` too slow | Gate B timing | **CLOSED** — median 0.86ms/net, ~0.035ms/step |
| Waypoint infidelity | Gate B deviation | **CLOSED** — 0.0000mm mean/max |
| `load_board()` unsafe | Gate A segfault | **FIXED in C++**, awaiting Colab rebuild |
| Colab session death | — | Drive checkpointing every N updates |
| Only 2 vCPUs | `nproc`=2 | Accepted — ample at 0.035ms/step |
| Stage 3 plateaus | Completion % vs B2 | Add step-length dim → rip-up → learned ordering |

---

## 11. File Structure (New / Modified)

```
pcbworld/
  env/
    line_route_env.py          # NEW: core 1-D heading env
    pcb_route_env.py           # MODIFIED: multi-net sequencing
    diff_pair_tune_env.py      # NEW: primitive-aware env (stages 4-5)
  agents/
    ppo_continuous.py          # NEW: Gaussian PPO for 1-D action
    ppo_baseline.py            # EXISTING: discrete baseline (keep)
training/
  train_line_policy.py         # NEW: curriculum trainer
  replay_buffer.py             # EXISTING
  reward_scaling.py            # EXISTING
scripts/
  verify_line_env.py           # NEW: smoke test
  generate_board_pool.py       # NEW: pre-generate 200 boards
  train_curriculum.py          # MODIFIED: orchestrates stages
docs/
  UNIFIED_RL_DESIGN.md         # THIS FILE
```

---

## 12. Success Criteria

| Milestone | Metric | Threshold |
|---|---|---|
| Stage 1 | Single-net completion | >95% |
| Stage 2 | 8-net completion | >50% (beats 33% straight-line) |
| Stage 3 | 24-net dense completion | >B2 (KiCad shove) at ≤ wirelength |
| Stage 4 | Diff-pair routing | >80% pairs coupled at 0.15mm gap |
| Stage 5 | Length matching | >80% groups within 0.5mm tolerance |
| Stage 6 | Full board + rip-up | >90% completion on held-out seeds |

---

## 13. Why This Will Work (Where CFP Failed)

| CFP Failure Mode | This Design |
|---|---|
| Raster encoder = 71% FLOPs, sub-pixel clearance | Line segments: exact, zero rasterizer, 100× cheaper |
| 14M params, GPU-bound, T_pns unknown | 35k params, CPU, T_pns measured (0.035ms/step) |
| Autoregressive field decoder | Direct heading → push → PNS handles geometry |
| No per-step signal (fix rejects late) | `head_collides` separates 100%/0% at 1.9 pushes pre-fix |
| Learned net ordering = combinatorial | Fixed heuristic; only add if plateau |
| Layer actions from day 1 | Disabled (Gate A); rip-up is the density lever |

---

*This design supersedes `docs/AI_ARCHITECTURE.md` (CFP) and the LLM-agent pivot in `ROADMAP.md`. The live plan is `docs/RL_PLAN.md` — this document expands it into an implementable specification.*