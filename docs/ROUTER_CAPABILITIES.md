# What the router can do today

Paste this into a fresh session (or another model) to bring it up to speed
without reading the whole repo. Every claim below carries a verification tag:

- **[LIVE]** — confirmed against the real compiled bridge in Colab, with numbers.
- **[LOCAL]** — exercised only against `tests/fake_bridge.py` (Python control
  flow proven, real router behavior not observed).
- **[UNVERIFIED]** — written, never run for real.
- **[BROKEN]** — run against the real bridge and failed, every time so far.

---

## The one-paragraph version

`pcbworld_pns_bridge` is a pybind11 module that drives **KiCad 9's real
push-and-shove router (`PNS::ROUTER`) headlessly from Python**. It can load a
`.kicad_pcb`, enumerate every net and pad, start a route at a pad, walk the
routing head to arbitrary coordinates, read back the exact copper the router
produced (including whether the head is currently colliding and with *what*),
commit that copper to the board, run KiCad's full-board DRC over it, rip an
individual net back out, and reset the board to bare. It does this in four
routing modes: single-net, differential-pair, length-tune-single, and
length-tune-diff-pair-skew. The engine is the real thing — not a
reimplementation, not a simulator — so anything it accepts is geometry KiCad
itself would produce and DRC-check.

---

## Capabilities, by area

### Board I/O and inspection

| Call | What it does | Status |
|---|---|---|
| `load_board(path)` | Loads a `.kicad_pcb`, syncs it into the PNS world, calls `LoadSettings()` | **[LIVE]** |
| `save_board(path)` | Writes the board back to disk | **[LIVE]** — reloaded in a separate process and confirmed real tracks landed |
| `net_names()` / `net_pads()` | Every net; every pad's net, pad name, position, layer | **[LIVE]** |
| `query_hover_items(x, y, layer, slop)` | Items near a point, as router-usable candidate ids | **[LIVE]** — ids stable within a loaded board, invalidated by `load_board()` |
| `get_board_geometry()` | **The whole board as geometry**: track segments (x1,y1,x2,y2,width,layer,net,is_arc), vias, pads, zones, courtyards, board edge | **[LIVE]** — returned 27 real tracks across 4 nets on a routed board |
| `get_design_rules()` | Track width, via diameter/drill, clearance, and the *minimum* of each | **[LIVE]** — 0.2mm track / 0.6mm via / 0.2mm clearance on generated boards |
| `run_drc()` | KiCad's real `DRC_ENGINE` over the whole board, returning violations with error code, message, severity, position | **[LIVE]** — caught an injected 0.05mm clearance violation as `DRCE_CLEARANCE`, zero false positives on a clean control board |

### Routing

| Call | What it does | Status |
|---|---|---|
| `start_route(x, y, item_id, layer)` | Begins a route at a pad | **[LIVE]** |
| `push(x, y)` | `ROUTER::Move()` — walks the head toward a point | **[LIVE]**, with a caveat below |
| `fix(x, y, item_id, force_finish, force_commit)` | `ROUTER::FixRoute()` — locks the route in | **[LIVE]**, rejects far more than `push()` does |
| `commit_routing()` | Persists fixed copper onto the `BOARD` | **[LIVE]** |
| `stop_routing()` | Drops an uncommitted route | **[LIVE]** |
| `rip_up(net)` | Removes an already-committed net's copper | **[LIVE]** — confirmed by geometry readback, not just a return value |
| `reset()` | Strips all tracks/vias/arcs, keeps footprints | **[LIVE]** |
| `switch_layer()` / `toggle_via_placement()` | Layer hop via a via | **[BROKEN]** — 0 successes in 32 real attempts (15/15 and 17/17 rejections) |

### Head state — the feedback channel

| Call | What it does | Status |
|---|---|---|
| `get_head_geometry()` | Live head: active, segments, vias, end position, layer, accumulated length | **[LIVE]** — deviation from the requested point was ~0.7 µm on an unobstructed push |
| `head_collides()` | Whether the in-progress head touches other copper | **[LIVE]** — fixed to stop misreporting self-touch |
| `get_head_obstacle()` | **Which net and item kind** the head is colliding with, and where | **[LIVE]** |

### Modes and parameters

| Call | What it does | Status |
|---|---|---|
| `set_mode(MODE_ROUTE_SINGLE)` | Ordinary single-net routing | **[LIVE]** |
| `set_mode(MODE_ROUTE_DIFF_PAIR)` | Coupled pair routing; PNS finds the `_N` leg by net-name matching | **[LIVE]** — routed a USB pair at exactly the configured 0.15mm edge-to-edge gap, both legs 20.2692mm |
| `set_mode(MODE_TUNE_SINGLE)` | Meander insertion to hit a target length | **[LIVE]** — a 20mm track tuned to a 30mm target came out at **30.0000mm**, 21 segments |
| `set_mode(MODE_TUNE_DIFF_PAIR_SKEW)` | Skew matching within a pair | **[LIVE]** — a deliberate 4mm dogleg tuned to **0.0000mm** skew |
| `set_collision_mode(RM_MARK_OBSTACLES / RM_SHOVE / RM_WALKAROUND)` | Whether `push()` is a pure validator or actually shoves other nets aside | **[LIVE]** — both behaviors confirmed distinct. `load_board()` defaults to `RM_MARK_OBSTACLES` |
| `set_track_width` / `set_via_diameter` / `set_via_drill` / `set_diff_pair_gap` / `set_diff_pair_width` / `set_diff_pair_via_gap` / `set_target_length` / `set_meander_max_amplitude` / `set_meander_spacing` | Design parameters, applied before a routing call | **[LIVE]** for the diff-pair and tuning ones |

### Boards it can be pointed at

`pcbworld/data/generate_board.py` **[LIVE, via system KiCad]** emits synthetic
2-layer boards, 50×50mm by default, all pads SMD on `F_Cu`, positions
rejection-sampled to a 3mm minimum spacing:

- `--num-nets N` — plain two-terminal nets (`net_<i>`)
- `--num-diff-pairs N` — pairs (`diffpair_<i>_P` / `_N`) with legs 1.0mm apart
- `--num-length-matched-groups N --length-matched-group-size K` — groups
  (`lengthgrp_<g>_<member>`) to be routed to matching length

Net *kind* is recovered purely from the name convention — there is no separate
metadata channel.

### Python layers already built on top

| Layer | What it is | Status |
|---|---|---|
| `pcbworld/agent/tools.py` — `RouterTools` | A **validated action surface**: every call checks bounds / step length / design rules *before* touching the bridge, then reads head state back *after*, and returns a structured `ToolResult` with a machine-readable error code (`STEP_TOO_LONG`, `HEAD_COLLIDES`, `HEAD_DEVIATED`, `VIOLATES_DESIGN_RULE`, …) rather than a bare bool | **[LIVE]** via the Qwen runs |
| `pcbworld/agent/loop.py` + `backends.py` | LLM agent loop (Qwen3-4B, quantized, local) driving `RouterTools` | **[LIVE]** — see the performance note below |
| `pcbworld/env/simple_route_env.py` | Gym env, one net per episode, `(dx, dy)` action, potential-based shaping | **[LIVE]** — 8-step episode reached and snapped to target, reward +22.8 |
| `pcbworld/env/pcb_route_env.py` | Gym env, multi-net, 4-dim action, per-net DRC | **[LIVE]** |
| `pcbworld/env/diff_pair_route_env.py` | Gym env, leg-sequenced across plain / diff-pair / tune primitives | **[LIVE]** — 4-leg sequence executed in order, 27 tracks persisted |
| `pcbworld/agents/ppo_baseline.py` | Plain-PyTorch PPO, MLP actor-critic | **[LOCAL]** — has never seen a real reward signal |
| `pcbworld/agents/cfp/` | 14M-param two-tower cost-field policy | **[LOCAL]** — benchmarked on a T4; its env, rasterizer, and planner were never built |
| `pcbworld/viz/render_board.py` | Renders `get_board_geometry()` to a PNG with matplotlib, no `pcbnew` needed | **[LOCAL]**, 10 tests |

---

## What is *not* true, and the numbers behind it

These matter more than the capability list — they are the constraints any plan
has to be built around.

1. **`push()` almost never says no.** Across 72 net-attempts in three Colab
   runs, `push()` rejected **zero** times. `fix()` then rejected **~67%** of
   those same routes. Success is silent and failure is late: the useful
   per-step signal is `head_collides()` / `get_head_obstacle()`, **not**
   `push()`'s return value.
2. **~33% of nets route on a naive straight line** (7/24, 8/24, 8/24 across
   three runs on a 24-net board). A retry ladder (polyline, then four
   perpendicular detours) was written to handle the rest but has **never been
   run against the real bridge**.
3. **Layer hopping does not work.** `switch_layer()` is 0-for-32 against the
   real bridge. Whatever the cause (via sizing was suspected, never
   confirmed), **the system is effectively single-layer today.**
4. **`T_pns` — the wall-clock cost of one net-route — has never been
   measured.** `scripts/measure_waypoint_fidelity.py` measures it; that run
   has not happened. Every throughput number in `docs/AI_ARCHITECTURE.md` is
   built on an estimate, not an observation.
5. **Length tuning leaves a residual.** A tune leg closed to within 0.2505mm
   of its reference, not to zero — real meander-granularity behavior. Treat
   "matched" as a tolerance, not an equality.
6. **The LLM agent works but is far too slow to be the product.** First real
   run: **2/3 nets routed**, at **~9–62 seconds per step**, with one net
   costing **933s for 15 steps**. Later runs hit context-window overflow
   (fixed) and repetition collapse (fixed). Even perfectly healthy, ~10s per
   decision × ~20 decisions × 24 nets is over an hour per board.
7. **Colab is the only proven build path** — KiCad 9.0.x compiled from source
   with the bridge wired in as an extra CMake target. The bridge is CPU-only.

---

## Hard constraints (violating these corrupts or crashes)

1. **`pcbworld_pns_bridge` and the system `pcbnew` module can never be
   imported in the same process.** Both define KiCad's one-per-process
   globals. Anything needing system `pcbnew` (board generation, independent
   verification) runs as a genuinely separate subprocess.
2. **Parallelize with processes, not threads.** No GPU path exists for PNS.
   One OS process per environment instance.
3. **`LoadSettings()` before any routing call** — `PNS::ROUTER`'s constructor
   leaves `m_settings` null and the first routing call dereferences it
   unguarded. `LoadBoard()` already does this.
4. **`query_hover_items()` returns items in hit-test order, not sorted.**
   Filter by `kind == 'pad'`; taking `candidates[0]` silently resolves to a
   passing track on a dense board.
5. **`fix()` needs `force_finish=True, force_commit=True`** to snap to a
   target pad.
