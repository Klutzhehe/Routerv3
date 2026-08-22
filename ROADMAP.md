# RouterV3 Roadmap

This doc exists so a fresh conversation (no memory of how this was built)
can pick up the project without re-deriving hard-won lessons or re-reading
the entire commit history. Read this before writing code, especially
anything touching `pcbworld/engine/cpp/`.

## Where things stand

**Proven, working, verified end-to-end**: `pcbworld/engine/cpp/` compiles
into `pcbworld_pns_bridge`, a pybind11 module that drives KiCad's real
`PNS::ROUTER` headlessly — no GUI, no `kicad-python`/IPC API (which
doesn't expose routing at all — checked directly, it's static board CRUD
only). Confirmed in Colab: load a board → find pads via
`query_hover_items` → `start_route` → `push` → `fix` → `commit_routing` →
`save_board` → then, in a **separate Python process**, reload with the
plain system `pcbnew` module and independently confirm a real track exists
on disk. That last step matters — it's not just "the code ran without
throwing," it's proof KiCad's actual push-and-shove engine produced a real,
persisted track.

Getting here took a long debugging arc — full technical trail (every
symbol, every crash, why each fix is correct) is in
[`docs/engine_access.md`](docs/engine_access.md). Don't re-derive any of
this; it's already been through the hard part.

## Read before touching engine code: hard constraints

These aren't style preferences — violating them causes silent corruption
or hard crashes that are expensive to re-diagnose. All discovered the hard
way; see `docs/engine_access.md` / `docs/performance.md` for the full
"why" on each.

1. **`pcbworld_pns_bridge` and the system `pcbnew` module must never be
   imported in the same process.** Both statically link overlapping chunks
   of KiCad's own C++ code and both define KiCad's "exactly one instance
   per process" globals (`Kiface()`, `GFootprintTable`). Loading both
   crashes the process, no Python-catchable exception. Anything needing
   system `pcbnew` (board generation, independent verification) must run
   as a genuinely separate subprocess — see `scripts/make_toy_board.py`
   and `scripts/verify_routed_board.py` for the reference pattern. This
   applies to the RL env too: never let a worker process import both.
2. **The bridge is CPU-only; parallelize via multiprocessing, not
   threads.** No GPU path exists for `PNS::ROUTER` (in KiCad or anywhere).
   One OS process per environment instance. See `docs/performance.md`.
3. **The Colab notebook (`notebooks/00_setup.ipynb`) is the only proven
   build path.** It compiles `pcbworld_pns_bridge` as an extra CMake
   target inside a real KiCad 9.0.8 source checkout (`add_subdirectory`),
   linking against `pnsrouter`, `pcbcommon`, `connectivity`, `gal`,
   `common`, `scripting`, `pcbnew_kiface_objects`, `${PCBNEW_IO_LIBRARIES}`.
   (`pcbworld/engine/cpp/kicad_headless_mocks.cpp` used to stand in for
   GUI-tool-framework symbols before `pcbnew_kiface_objects` was linked for
   DRC support — deleted once that library made it redundant and
   conflicting; see `docs/engine_access.md`'s "Update, once DRC support
   needed `pcbnew_kiface_objects`" note for the full story if this ever
   needs to be split apart again.) Building outside this exact flow
   (different KiCad version, linking `pnsrouter` without the IO libraries,
   etc.) will very likely resurface some of the same missing-symbol/crash
   issues already solved. Drive caching (`$DRIVE_CACHE_TARBALL`) makes
   re-runs fast — don't skip
   step 2 (`git pull`) before rebuilding, or you'll rebuild against stale
   local code.
   **A restored cache can be stale in a way that isn't about code.**
   Hit for real: a cache built under a Python 3.12 Colab base image
   recorded a pip-installed `cmake` binary's absolute path
   (`/usr/local/lib/python3.12/dist-packages/cmake/data/bin/cmake`) inside
   the cached `build.ninja`'s own regeneration rule. A later session's
   Colab image had moved to Python 3.13 — a base-image change entirely
   outside this repo's control — that exact path no longer existed, and
   `ninja` failed outright (`subcommand failed`) before compiling
   anything, because `setup_env.sh`'s "skip reconfigure" check only ever
   tested whether `build/CMakeCache.txt` existed, never whether the
   toolchain paths it recorded were still valid in *this* runtime. Fixed
   in `setup_env.sh`: a `RESTORED_FROM_CACHE` flag, set only when a
   Drive tarball was actually restored *this run*, now forces a
   reconfigure regardless of whether `CMakeCache.txt` exists — the
   legitimate same-session-retry case (re-running after an earlier
   failure, same runtime, safe to skip) is unaffected, since that case
   never sets the flag.
   **Forcing that reconfigure surfaced a second Python issue, and the
   first attempt at fixing it made things worse in a real way — record
   kept honest, not silently corrected.** Symptom: `pybind11::module`
   included `/usr/include/python3.12` in its
   `INTERFACE_INCLUDE_DIRECTORIES`, a path absent on a Colab runtime whose
   `python3` resolves to 3.13. First attempt: `-DPYBIND11_FINDPYTHON=ON` +
   `-DPython3_EXECUTABLE=python3`, reasoning that pybind11 v2.9.2 (old
   enough to predate `PYBIND11_FINDPYTHON` defaulting on) was using
   CMake's legacy, deprecated `FindPythonLibs` and picking a mismatched
   Python. That flag genuinely fixed `pcbworld_bridge`'s *own*
   `find_package(pybind11 CONFIG REQUIRED)` call — confirmed, the next
   configure log showed it correctly resolving 3.13 — but it does nothing
   for KiCad's *own*, separate `find_package(PythonLibs)` calls elsewhere
   in KiCad's CMake tree (e.g. for `libkicommon`), which the flag never
   touches and which kept resolving to 3.12 regardless (confirmed by
   CMake's own `Python3_EXECUTABLE ... not used` warning). Net effect:
   pushed `pcbworld_pns_bridge`'s own module onto 3.13 headers while it
   still links against `libkicommon.so`, built against 3.12 — a real
   cross-version ABI mismatch the first fix *introduced*, not fixed, and
   the very next build stage caught it as a link error instead of a
   configure error (`libpython3.12.so`, needed by `libkicommon.so.9.0.8`,
   missing).
   **The actual root cause, once traced through both failures together:**
   every `find_package(PythonLibs)` call across KiCad's whole tree —
   including pybind11's own, unforced — has always and still consistently
   resolves to 3.12; that is *why* every build succeeded before this
   Colab image's base layer moved its default `python3` to 3.13. The
   resolution itself was never broken. What changed is that the image
   stopped shipping 3.12's headers/shared library by default, so the path
   CMake correctly expects points at nothing. The fix reverted the
   `PYBIND11_FINDPYTHON`/`Python3_EXECUTABLE` override entirely (fighting
   to redirect one target's Python version was the wrong lever) and
   instead added `python3.12-dev` to `setup_env.sh`'s apt dependencies —
   Ubuntu ships multiple `python3.X` packages side by side without
   conflict — so the *same* Python every `find_package` call in the tree
   already, consistently wants is simply real on disk again. Still
   unverified in Colab as of this writing.
4. **`ROUTER::LoadSettings()` must be called before any routing call.**
   `PNS::ROUTER`'s constructor leaves `m_settings = nullptr`; the first
   real routing call dereferences it unguarded. `PNS_BRIDGE::LoadBoard()`
   already does this (constructs a standalone `PNS::ROUTING_SETTINGS`) —
   don't remove it, and if you ever construct a second `PNS::ROUTER`
   instance somewhere else, remember to do the same.
5. **`query_hover_items` candidate ids are stable across calls within a
   loaded board**, but invalidated by `LoadBoard()`. Don't assume
   otherwise; don't reintroduce a per-call-cleared candidate cache (that
   was an earlier, real crash cause).

## How to run it

Open `notebooks/00_setup.ipynb` in Colab. Step 1 mounts Drive and sets up
config; step 2 pulls this repo (always run this first, even on a repeat
session); step 3 is one atomic build script (apt deps → restore-from-Drive
if available → clone KiCad source if needed → wire the bridge in →
configure → build → save back to Drive); step 4 imports the compiled
module; step 5 proves it works end-to-end on a toy board. Each step's
markdown explains what it does and why; the notebook fails loudly with
`=== STAGE: ... ===` checkpoints rather than silently.

## Immediate next steps (recommended order)

These are prerequisites for the "novel agent" work regardless of which
direction gets picked below — do these first.

1. **Headless DRC — fully Colab-verified.** Added `PNS_BRIDGE::RunDRC()`
   (`pcbworld/engine/cpp/pns_bridge.{h,cpp}`, bound as `PNSBridge.run_drc()`
   in `bindings.cpp`): constructs a `DRC_ENGINE` against the loaded board's
   own `BOARD_DESIGN_SETTINGS`, installs a violation-handler callback,
   `InitEngine()` with no rules file (KiCad's built-in defaults),
   `RunTests()`. Mirrors `qa/tests/pcbnew/drc/test_drc_copper_conn.cpp`'s
   harness pattern. Also added `pcbnew_kiface_objects` to the link line in
   `pcbworld/engine/cpp/CMakeLists.txt` — `drc/*.cpp` lives there, not in
   `pcbcommon`, and routing alone never needed it. Compiled and ran clean
   on the toy board in Colab (step 6 of `notebooks/00_setup.ipynb`), and is
   exercised per-leg inside `PCBRouteEnv`/`SimpleRouteEnv`. Also confirmed
   catching a real violation: injecting a track with 0.05mm actual
   clearance against another net (default rule 0.2mm) produced
   `DRCE_CLEARANCE` (error code 5) with the exact actual-vs-required
   clearance values, while a clean control board reported zero violations
   — no false positives, no crashes. DRC-based reward shaping in
   `PCBRouteEnv` can be trusted now.
2. **`reset()` + multi-net routing — Colab-verified.**
   `PNS_BRIDGE::Reset()` (walks `BOARD::Tracks()` — covers segments/vias/
   arcs — removing each via `BOARD::Remove()`, then `ClearWorld()`/
   `SyncWorld()` against the same `BOARD` instance, preserving footprint
   placement) and `PNS_BRIDGE::NetPads()` (enumerates every pad on the
   board so an agent/script can find route endpoints programmatically
   instead of guessing coordinates via `query_hover_items()`). Bound as
   `PNSBridge.reset()`/`PNSBridge.net_pads()`. Confirmed in Colab:
   `reset()` clears tracks and leaves footprints/pad count intact, and a
   fresh `start_route()`→`push()`→`fix()`→`commit_routing()` succeeds
   immediately afterward on the same board. `net_pads()` confirmed
   returning correct net name/pad name/position/layer for every pad.
3. **Synthetic board generator — implemented and locally verified.**
   `pcbworld/data/generate_board.py`: gridless 2-layer boards, N two-pad
   nets (default 5), positions rejection-sampled for minimum spacing.
   Deliberately simpler than the paper's full D2 spec (two-terminal nets
   only, not arbitrary fanout) — matches `scripts/make_toy_board.py`'s
   existing pattern rather than introducing new pcbnew API risk. Actually
   run against the local KiCad 9.0 install
   (`C:\Program Files\KiCad\9.0\bin\python.exe`) and round-tripped through
   `pcbnew.LoadBoard()` to confirm footprint/pad/net counts and positions
   — this one *is* verified, unlike the C++ pieces, since it only uses
   system `pcbnew` (no Colab/bridge dependency).
4. **The Gym environment — Colab-verified against real router behavior.**
   `pcbworld/env/pcb_route_env.py`: `PCBRouteEnv(gym.Env)`, one net routed
   per "leg", nets sequenced across an episode (`net_order` param —
   deliberately left as a caller choice, not decided here, since ROADMAP
   already flagged sequencing as its own design question). Action:
   `Box(4,)` = push delta (x, y) + fix/via thresholds. Reward:
   potential-based shaping on wirelength, via count, and `run_drc()`-
   reported error count (checked once per net-finish, not every step —
   `DRC_ENGINE` is a full-board check). Observation is a small 5-vector
   (delta-to-target, progress, via count, DRC errors) — deliberately not
   the paper's Fourier-feature geometry encoding; see "novel SOTA agent"
   below for where a richer encoder could go later. Locally verified with
   `tests/fake_bridge.py` (`tests/test_pcb_route_env.py`) for Python
   control flow, and now also run in Colab against a real multi-net board
   from `generate_board.py`, exercising net transitions, leg step budgets,
   via penalties, wirelength shaping, and per-leg DRC checks against the
   real bridge.
5. **Baseline PPO agent — implemented, training loop locally verified,
   never seen a real reward signal.** `pcbworld/agents/ppo_baseline.py`:
   plain-PyTorch single-process PPO (MLP actor-critic, GAE, clipped
   surrogate) — deliberately not matching the paper's Transformer policy,
   this is the "does the plumbing work" baseline ROADMAP asked for, not a
   paper-matching comparison yet. `tests/test_ppo_baseline.py` runs the
   full rollout-collect → GAE → update loop against the fake bridge and
   confirms finite losses/weights — real per-CPU-core multiprocessing
   (`docs/performance.md`) isn't implemented, single env only.

6. **Collision-response mode + a minimal single-net env — Colab-verified.**
   `PNS_BRIDGE::SetCollisionMode()`
   (`pcbworld/engine/cpp/pns_bridge.{h,cpp}`, bound as
   `PNSBridge.set_collision_mode()`) exposes `PNS::ROUTING_SETTINGS`'s own
   `PNS_MODE` (RM_MarkObstacles/RM_Shove/RM_Walkaround) -- previously
   unbound, and defaulting (via `ROUTING_SETTINGS`'s own constructor) to
   Shove, which means `push()` was letting PNS auto-move *other* traces to
   accommodate the agent's segment instead of just reporting a collision.
   `LoadBoard()` now sets `RM_MarkObstacles` explicitly so `push()` is a
   pure geometry/DRC validator during RL training; Shove/Walkaround are
   still available via `set_collision_mode()` for a classical-baseline
   comparison run. Confirmed in Colab: `RM_MARK_OBSTACLES` refuses a push
   through another net's track/locked geometry (pure validator, no
   auto-shove) and `RM_SHOVE` restores the old interactive-shove behavior.

   `pcbworld/env/simple_route_env.py` (`SimpleRouteEnv`) is a deliberately
   smaller sibling of `PCBRouteEnv`: one net per episode, one layer, a pure
   `(dx, dy)` action (no via/fix-threshold dims -- a net finishes
   automatically once the head is within snap radius and `fix()` confirms
   real connectivity), potential-based progress shaping instead of a raw
   distance-traveled penalty. Kept as a separate file so `pcb_route_env.py`
   stays available as-is. Verified locally against `tests/fake_bridge.py`
   (`tests/test_simple_route_env.py`), and now also Colab-verified against
   the real bridge on a synthetic board: an 8-step episode reached and
   snapped to the target pad, `terminated=True`/`truncated=False`, reward
   +22.8 total (non-exploding potential-based progress + step penalties +
   the +20 completion bonus) -- confirms the reward signal behaves
   sensibly against real router state, not just that it runs.

7. **Diff-pair routing + length tuning — fully Colab-verified, not yet
   used by any env.** `PNS_BRIDGE::SetMode()` (`PNS::ROUTER_MODE`) plus
   `SetDiffPairGap/Width/ViaGap()`, `SetTargetLength()`, and
   `SetMeanderMaxAmplitude/Spacing()` (`pcbworld/engine/cpp/pns_bridge.h`,
   committed in f654702) expose KiCad's native diff-pair and meander/
   length-tuning placers -- `MODE_ROUTE_DIFF_PAIR`, `MODE_TUNE_SINGLE`,
   `MODE_TUNE_DIFF_PAIR`, `MODE_TUNE_DIFF_PAIR_SKEW`. This was committed
   but never exercised by any env, test, or Colab run until now. Confirmed
   in Colab against real geometry (not just "didn't crash"):
   - `MODE_ROUTE_DIFF_PAIR`: routed `USB_P`/`USB_N` as coupled parallel
     tracks, edge-to-edge gap `0.1500 mm` exactly matching the configured
     `set_diff_pair_gap`, both legs `20.2692 mm`.
   - `MODE_TUNE_SINGLE`: a 20mm straight track tuned to a 30mm target
     produced a 21-segment meander with actual length `30.0000 mm` --
     zero delta from target.
   - `MODE_TUNE_DIFF_PAIR_SKEW`: a diff pair with a deliberately
     introduced 4mm dogleg (`USB_N` 24mm vs `USB_P` 20mm) tuned down to
     `0.0000 mm` skew (`USB_P` meandered out to match `USB_N`'s 24mm).
   - Regression control: standard single-net route + DRC on the toy board
     still passes after all of the above (independent-reload verify +
     0 DRC violations) -- none of this broke the existing proven path.

   **This is the key primitive set for the "dense board, diff pairs,
   length tuning" target end goal** -- see the new design section below.
   Not yet wired into `pcb_route_env.py`/`simple_route_env.py` (both
   hardcode `MODE_ROUTE_SINGLE`), and `generate_board.py` doesn't yet emit
   diff-pair or length-matched net groups to train against.

Items 2-6 were batched into one commit deliberately, so only one Colab
rebuild was needed to pick up all of it. That rebuild has now happened and
items 1-2, 4, and 6 are fully Colab-verified (see each item above for
specifics). Item 7 above was verified in a separate follow-up Colab
session. **Still open** from the original verification list:
- Item 5: whether `ppo_baseline.py` training against the real bridge shows
  *any* learning signal over episodes (even just "finishes nets more often
  over time"), as opposed to just "doesn't crash". This is genuinely
  untested against the real bridge.

## Pivot 2: back to RL, on line geometry — CURRENT DIRECTION

**Read [`docs/RL_PLAN.md`](docs/RL_PLAN.md) first; it is the live plan.**
[`docs/ROUTER_CAPABILITIES.md`](docs/ROUTER_CAPABILITIES.md) is the
paste-into-a-fresh-session summary of what the engine can actually do, with
every claim tagged LIVE / LOCAL / BROKEN.

The LLM direction below is now history too, for a measured reason rather
than a taste one: the first real Qwen3-4B run routed 2/3 nets at ~9-62
seconds *per decision*, one net costing 933s for 15 steps. The
context-overflow and repetition-collapse bugs found afterwards were real and
are fixed, but they were symptoms — ~10s x ~20 decisions x 24 nets is over
an hour per board. A ~200k-parameter policy makes the same decision in
~0.1ms. The LLM agent is kept as a **debugging oracle** (hand it a board the
policy failed and read its reasoning), which preserves the debuggability
argument that motivated the pivot away from RL in the first place.

What changes versus the abandoned CFP design: the board is fed to the policy
as **line segments** (`get_board_geometry()` already returns exactly that),
not a 256x256 raster. Three reasons, in ascending order of weight: no
rasterizer needs writing (CFP's build-order step 2, never started); the
canvas encoder was **71% of a measured forward pass** and disappears; and a
256px raster over a 50mm board is **0.195 mm/px against a 0.2 mm clearance**,
so the legality margin the router actually enforces is sub-pixel and the
raster structurally cannot represent it. Unrouted nets are lines too (a
straight pad-to-pad "ghost" segment), which is most of what CFP's reserve
plane existed to provide, for one one-hot bit.

**Two gates before any trainer is written** (both need one Colab run, and
both are now implemented and locally tested):

1. **Gate A — why `switch_layer()` is 0-for-32.**
   `scripts/diagnose_layer_switch.py` tests four hypotheses plus two
   controls in a single run, instead of the one-hypothesis-per-session pace
   that has already cost two rounds. Needs a THT board, so
   `generate_board.py` gained `--pad-type tht` (verified against the local
   KiCad 9.0 install: PTH attribute, layer set spanning F_Cu *and* B_Cu,
   0.5mm drill; the SMD default is unchanged at F_Cu-only, no drill).
   **Measured while building it: `pcbnew.B_Cu` is `2` on KiCad 9, not the
   pre-9 `31` and not the `1` that `fake_bridge.py`'s toggle informally
   assumes.** Nothing in the tree records which value the historical
   0-for-32 runs passed (`measure_layer_hop_rescue.py` takes `--back-layer`
   as a required argument), so a wrong constant is not yet excluded — the
   script's layer-id sweep settles it either way.
2. **Gate B — is there a dense per-step reward signal?**
   `scripts/measure_waypoint_fidelity.py` now also records per-call wall
   clock (a line-geometry observation calls `get_board_geometry()` every
   step, not once per net) and correlates `head_collides()` against whether
   `fix()` later accepted. `push()` accepted 72/72 while `fix()` rejected
   ~67%: if the collision signal separates the two populations, the planned
   1-D heading action space works; if it doesn't, one terminal bit has to
   carry ~20 steps of credit and the action granularity changes first.
   Run `--no-collision-trace` once as a control.

## Pivot: LLM routing agent, replacing the RL direction

**The RL direction below (CFP, `pcbworld/agents/cfp/`, the waypoint-RL env)
is abandoned, not just deprioritized.** Reason: debuggability, not
performance. A scalar reward cannot be read the way a tool-call transcript
can -- when a policy fails silently, there was no way to see *why* short of
staring at a reward curve. `pcbworld/agents/cfp/` and the waypoint-RL env
are parked on the `parked/waypoint-rl-env` branch, not deleted, in case pure
RL is revisited later. The rest of this section below (the "novel SOTA
agent" design) is superseded by the same reasoning and is being left in
place as history, not as a live plan.

**Current direction:** an LLM (Qwen3 small tier, local, quantized -- see
`pcbworld/agent/backends.py`) drives the router directly through a
validated tool surface (`pcbworld/agent/tools.py`'s `RouterTools`),
sequenced by `pcbworld/agent/loop.py`'s `RoutingAgent` over a
caller-supplied net priority order (no learned ordering -- that is a human
or LLM-strategy decision, not the loop's job). The tool layer's whole
design is answering one problem directly: `push()` is `ROUTER::Move()`, and
`scripts/measure_waypoint_fidelity.py` measured it accepting 72/72
net-attempts across three Colab runs while `fix()` then rejected ~67% of
those same routes -- silent success, not loud failure, is the failure mode
that matters here. Every tool validates before touching the bridge and
reads the router's actual state back afterward (deviation, collision)
rather than trusting a bare return value.

**New C++ bindings this depends on -- Colab-verified 2026** (build:
KiCad 9.0.9, `pcbworld_bridge` target, clean compile/link, no warnings
beyond a benign LTO serial-compilation notice):

```
=== 1. Loading Board: board.kicad_pcb ===
  [PASS] Board loaded successfully
=== 2. Testing get_design_rules() ===
  Track Width      : 0.2000 mm (200000 nm)
  Via Diameter     : 0.6000 mm (600000 nm)
  Via Drill        : 0.3000 mm (300000 nm)
  Clearance        : 0.2000 mm (200000 nm)
  Min Track Width  : 0.0000 mm (0 nm)
  Min Via Diameter : 0.5000 mm (500000 nm)
  Min Via Drill    : 0.3000 mm (300000 nm)
  Min Hole-to-Hole : 0.2500 mm (250000 nm)
=== 3. Idle head: active=False, collides=False (both expected) ===
=== 4. start_route(net_4) -> active=True, layer=0 ===
  push() to (39.40, 42.70) -> True
  head after push: end=(39.3993, 42.6990), layer=0, length=2.6585mm,
    2 segments, collides=False
=== 5. fix()+commit_routing(): 2 tracks committed for net_4 ===
=== 6. rip_up('net_4'): removed 2 items; get_board_geometry() confirms
  0 tracks remain for net_4 (live readback, not just the return value) ===
ALL 4 NEW BINDINGS VERIFIED SUCCESSFULLY AGAINST LIVE BRIDGE
```

Two things worth reading precisely out of that, not overclaiming:

- **`get_head_geometry()` deviation was ~0.7 microns** on this run (a
  single unobstructed push on a sparse 6-net board) -- essentially exact.
  That is the *easy* case; it does not yet show what deviation looks like
  under contention, which is what the tool layer's warning path exists for.
  A single push produced **2 head segments**, not 1 -- worth knowing when
  reading `HeadGeometry.segments` elsewhere, though `RouterTools` only
  reads `end_x`/`end_y`, not segment count, so this didn't affect anything
  downstream.
- **`min_track_width` reports 0.0000mm on a `generate_board.py` board.**
  Confirms directly what was previously only inferred: these synthetic
  boards carry no real track-width design-rule floor at all --
  `generate_board.py` sets no design rules, so `BOARD_DESIGN_SETTINGS`'s
  track-width minimum is whatever a bare board leaves it at, apparently
  zero. The via minimums (`0.5mm` diameter / `0.3mm` drill) *do* have real
  values, which is what makes `RouterTools.place_via()`/`switch_to_layer()`'s
  design-rule guard meaningful in practice, not just in theory. Also: the
  `600_000`/`300_000` nm via constants hardcoded across
  `pcb_route_env.py`/`diff_pair_route_env.py`/`measure_layer_hop_rescue.py`
  happen to sit exactly at this board's default-netclass via size and above
  its minimum -- so on THIS board they were never illegal. That does not
  resolve whether an illegal via size caused the historical
  `switch_layer()` rejections (381e1a7, dc9164e); it just means this
  particular run doesn't reproduce the conditions that would show it.

**Still open, not touched by this verification round:**
- `switch_layer()` itself was not re-exercised here -- this script tests
  only the four *new* bindings (`get_design_rules`, `get_head_geometry`,
  `head_collides`, `rip_up`); `switch_layer()` was bound in an earlier
  commit and still has never succeeded once against the real bridge in
  this repo (15/15 and 17/17 rejections, 381e1a7/dc9164e). Re-running
  `scripts/measure_layer_hop_rescue.py` is what would actually close that.
- `RouterTools` (the tool layer the LLM agent actually calls) has never
  run against the real bridge -- only the raw C++ bindings underneath it
  have. `place_via()`/`switch_to_layer()`/the deviation-warning path in
  `route_to()` are still unverified end to end.
- `T_pns` (wall-clock cost per net-route) is still unmeasured --
  `scripts/measure_waypoint_fidelity.py` has not been re-run this session.
- `QwenBackend` has never run against a real model -- no GPU involved in
  this verification round, CPU-only (the bridge).

## The "novel SOTA agent" — superseded, kept as history (see pivot above)

**Target end goal (user-specified):** route dense boards, including
differential pairs and length tuning -- not just the paper's plain
two-terminal single-net case. Design below picked with that goal in mind;
supersedes treating this as an open "pick one of these" list.

**Core decision: don't make the agent learn diff-pair coupling or meander
geometry from raw (dx, dy) pushes.** KiCad's own placers already do that
reliably (see item 7 above) -- reinventing it via RL would be slower to
learn and lower-quality than the engine's native geometry. Instead the
agent operates one level up:

- **Macro-action space, not per-pixel pushing.** Per net (or net group),
  the policy picks (a) a routing primitive -- direct single-net,
  diff-pair, length-tune-single, or length-tune-diff-pair-skew -- and (b)
  a short sequence of coarse waypoints (2-6 points), letting PNS's own
  push/shove handle the fine walking between them. This is the main lever
  for dense boards: it collapses episode length from dozens of pixel-level
  pushes per net to a handful of waypoint decisions, which is what makes
  boards with hundreds of nets tractable at all.
- **Two-stream encoder, matching the two-scale action space** (this
  replaces `ppo_baseline.py`'s current raw-5-vector MLP input):
  - *Graph encoder (net-level):* GNN or graph-transformer over a graph
    whose nodes are pads/nets and edges are ratsnest connectivity +
    diff-pair/length-group membership. Drives net-ordering, primitive
    selection, and which nets are paired/length-matched. This is the
    roadmap's earlier "graph-transformer" idea, now with a concrete job.
  - *Local raster encoder (waypoint-level):* small CNN over a patch
    pulled from `GetBoardGeometry()` centered on the current routing
    head -- separate channels per layer for copper, keepouts, courtyards,
    vias, edge-distance. Drives waypoint placement / obstacle avoidance.
  - Combine as one shared actor-critic trunk to start (concatenate both
    encoders' outputs), not a full hierarchical two-network split --
    simpler to get running, smaller diff from the existing PPO baseline.
    Only split into true macro/micro policies later if joint learning of
    net-ordering and waypoint placement doesn't work.
- **Net-ordering meta-policy is load-bearing**, not optional, once diff
  pairs and length matching are in play -- pairs must be identified and
  routed adjacently so a reference trace's length is known before tuning
  skew against it.
- **Rip-up-and-reroute as a standing macro-action, not just an episode
  reset.** The PCBWorld paper explicitly excluded track-removal from its
  action space (a stated limitation), and this bridge already has full
  `RemoveItem`/`Reset()` support -- an agent that can rip up its own bad
  decisions and retry is structurally impossible for their baseline, and
  becomes routine necessity (not just a nice-to-have) on dense boards with
  matched-length constraints, where a later net will often conflict with
  an earlier one's tuning.

**Concrete curriculum** (each stage trains/validates before moving on):
1. Single-net, sparse board -- done (`SimpleRouteEnv`, item 6).
2. Multi-net, dense board, no diff pairs -- in progress (`PCBRouteEnv`,
   item 4).
3. Diff-pair routing on sparse boards (`MODE_ROUTE_DIFF_PAIR`) -- primitive
   and `DiffPairRouteEnv`'s leg-sequencing both Colab-verified.
4. Length-tune single net (`MODE_TUNE_SINGLE`) -- primitive and env both
   Colab-verified (0.25mm residual gap on the one test case so far --
   real router behavior, see item 7).
5. Length-tune diff-pair skew (`MODE_TUNE_DIFF_PAIR_SKEW`, hardest) --
   primitive Colab-verified (item 7); not yet exercised by
   `DiffPairRouteEnv`, which only builds `MODE_TUNE_SINGLE` legs today.
6. Combine: dense board, mixed single/diff-pair/length-matched nets,
   rip-up-and-reroute enabled.

**Immediate blockers before policy work can start:**
- ~~`generate_board.py` only emits plain two-pad nets~~ **Done.** Extended
  to also emit diff-pair nets (`diffpair_<i>_P`/`_N`, legs offset
  perpendicular to the pair's direction at `diff_pair_pitch_mm`, default
  1.0mm to match item 7's verified Colab test) and length-matched groups
  (`lengthgrp_<g>_<member>`, plain two-terminal nets tagged by name for a
  caller to route to matching length). Net kind is recovered purely by
  name convention -- no separate metadata channel, same as everything
  else that consumes `NetPads()`. Locally verified against the system
  KiCad install (net/pad counts, exact 1.0mm P/N leg spacing, 0.3mm
  diff-pair pad size, no rejection-sampler failures on a dense
  6-diff-pair/3-length-group board) -- this is pure `pcbnew` scripting
  like the rest of the file, so no Colab/bridge dependency to verify it
  further.
- ~~`pcb_route_env.py`/`simple_route_env.py` both hardcode
  `MODE_ROUTE_SINGLE`~~ **Done, via a new env rather than editing those
  two.** `pcbworld/env/diff_pair_route_env.py` (`DiffPairRouteEnv`) parses
  `net_pads()` by `generate_board.py`'s name convention into a fixed
  sequence of "legs": plain nets route directly
  (`MODE_ROUTE_SINGLE`); each diff pair is one `MODE_ROUTE_DIFF_PAIR` leg
  driven by its P net (PNS finds N via net-name matching -- confirmed in
  Colab, item 7); each length-matched group's first member is the
  reference (routed directly), every other member gets a direct leg
  (straight baseline) immediately followed by a `MODE_TUNE_SINGLE` leg
  that re-opens that same track and tunes it toward the reference's
  *actual* routed length (read back via the newly-bound
  `get_board_geometry()` -- see below -- not assumed equal to whatever
  was passed to `set_target_length()`). Action/observation otherwise
  mirror `PCBRouteEnv` (4-dim push+fix+via action, 5-vector distance/
  progress/via/DRC observation) plus a 3-dim leg-kind one-hot
  (direct/diff_pair/tune), so the existing PPO baseline can be
  primitive-aware without the full two-stream encoder existing yet.
  `pcb_route_env.py`/`simple_route_env.py` are left as-is (plain-net-only
  envs stay useful as smaller/faster training targets and as a fallback).
  Locally verified against `tests/fake_bridge.py`
  (`tests/test_diff_pair_route_env.py`, 4 tests, all passing): a mixed
  board (1 plain net + 1 diff pair + 1 two-member length group) visits
  exactly the 5 expected legs in order, terminates correctly, and a tune
  leg's length-mismatch bookkeeping actually runs (not just "doesn't
  crash"). **Now also Colab-verified against the real bridge**, on a
  `generate_board.py` board with 1 diff pair + 1 two-member length group:
  leg sequence executed in the exact order `_build_legs()` specifies
  (`diff_pair:diffpair_0_P` -> `direct:lengthgrp_0_0` ->
  `direct:lengthgrp_0_1` -> `tune:lengthgrp_0_1`), episode terminated
  cleanly in 26 steps with bounded (non-exploding) reward, and
  `get_board_geometry()` afterward showed 27 real persisted tracks across
  all four nets. Length-tuning specifically: reference
  (`lengthgrp_0_0`) routed at 12.8713mm (2 segments), the real meander
  placer inserted 14 segments into `lengthgrp_0_1` and closed it to
  12.6208mm -- a 0.2505mm residual gap, not exact zero. That gap is real
  router behavior (meander segment granularity), not an env bug -- the
  target length it tuned toward was itself read back correctly from the
  reference's actual routed length (12.8713mm matches what
  `_start_next_leg()` would have passed to `set_target_length()`). Take
  this as a data point for reward-shaping / success-tolerance decisions
  later (e.g. treat a tune leg as "succeeded" within some mm tolerance,
  not exact-zero mismatch) rather than as a bug to fix.
  - Needed a bridge binding that didn't exist: `GetBoardGeometry()` was
    implemented in C++ (`pns_bridge.{h,cpp}`) but never exposed to Python
    -- added the full `bindings.cpp` pybind11 wiring for it
    (`TrackSegment`/`ViaGeom`/`PadGeom`/`ZoneGeom`/`FootprintBBox`/
    `EdgeShape`/`BoardGeometry`, plus `get_board_geometry()` on
    `PNSBridge`). Colab rebuild confirmed it compiles, links, and returns
    correct data (the 27-track/4-net inspection above came from it
    directly) -- no longer an open verification item.
- The two-stream encoder doesn't exist yet; `ppo_baseline.py`'s
  actor-critic is still a plain MLP over the 5-vector (now 8-vector, if
  trained against `DiffPairRouteEnv`).

## Parallel track: LLM Board Advisor plugin

`kicad_plugin/llm_advisor/` — a separate, working-today KiCad Action
Plugin (reads board state, asks an LLM, reports back via console print +
board comment). No compiling needed, uses KiCad's standard scripting API.
Built and believed correct but **never actually tested** — the Scripting
Console verification step in `kicad_plugin/README.md` was never run. If
picked back up, that's the first thing to do; the most likely soft spot is
the DRC-marker-count field in `board_summary.py`'s `_count_drc_markers`
(method name uncertainty across KiCad versions, already defensively
wrapped to degrade rather than crash). This track is independent of
everything else in this roadmap — it can't drive the interactive router
(that API isn't exposed via standard `pcbnew` scripting either), so it's
an advisor, not a step toward the autorouting goal.

## Repo map

```
pcbworld/
  engine/cpp/     # The proven bridge -- pns_bridge.{h,cpp}, bindings.cpp,
                  # CMakeLists.txt
  env/            # pcb_route_env.py -- Gym env, Python logic verified,
                  # real router behavior not yet observed
  agents/         # ppo_baseline.py -- PPO loop verified, no real reward
                  # signal observed yet
  data/           # generate_board.py -- synthetic board generator,
                  # verified against local KiCad install
kicad_plugin/     # LLM advisor Action Plugin, separate track, untested
notebooks/        # 00_setup.ipynb -- the only proven build/run path
scripts/          # make_toy_board.py, verify_routed_board.py -- both
                  # MUST run as separate processes from the bridge
docs/
  engine_access.md   # Full technical trail: why compile from source, every
                     # symbol/crash found and how each was fixed
  performance.md     # CPU-bound nature, multiprocessing requirement, the
                     # bridge/system-pcbnew process-isolation constraint
```
