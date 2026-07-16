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
   `common`, `scripting`, `${PCBNEW_IO_LIBRARIES}`, plus
   `pcbworld/engine/cpp/kicad_headless_mocks.cpp` (headless stand-ins for
   GUI-tool-framework symbols that leak in transitively — dead code from
   this bridge's perspective, safe, but required to link). Building
   outside this exact flow (different KiCad version, skipping the mocks,
   linking `pnsrouter` without the IO libraries, etc.) will very likely
   resurface some of the same missing-symbol/crash issues already solved.
   Drive caching (`$DRIVE_CACHE_TARBALL`) makes re-runs fast — don't skip
   step 2 (`git pull`) before rebuilding, or you'll rebuild against stale
   local code.
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

1. **Headless DRC — code written, not yet Colab-verified.** Added
   `PNS_BRIDGE::RunDRC()` (`pcbworld/engine/cpp/pns_bridge.{h,cpp}`,
   bound as `PNSBridge.run_drc()` in `bindings.cpp`): constructs a
   `DRC_ENGINE` against the loaded board's own `BOARD_DESIGN_SETTINGS`,
   installs a violation-handler callback, `InitEngine()` with no rules
   file (KiCad's built-in defaults), `RunTests()`. Mirrors
   `qa/tests/pcbnew/drc/test_drc_copper_conn.cpp`'s harness pattern.
   Also added `pcbnew_kiface_objects` to the link line in
   `pcbworld/engine/cpp/CMakeLists.txt` — `drc/*.cpp` lives there, not in
   `pcbcommon`, and routing alone never needed it. **This is genuinely
   untested** (no compiler has seen this code yet) — `notebooks/00_setup.ipynb`
   step 6 runs `b.run_drc()` on the toy board right after step 5's routing;
   expect the first Colab run to surface either link errors (most likely
   from `pcbnew_kiface_objects`, same "iterate from real linker output"
   process documented in `docs/engine_access.md`) or a `DRC_ENGINE` runtime
   issue (e.g. missing default rules because no `PROJECT`/`SETTINGS_MANAGER`
   was ever loaded — we only ever construct a bare `BOARD` via
   `PCB_IO_KICAD_SEXPR`). If it compiles and runs, next step is deliberately
   introducing a real violation (e.g. a too-tight clearance) on the toy
   board to confirm `RunDRC()` actually catches it, not just that it runs
   clean.
2. **`reset()` + multi-net routing — code written, not yet Colab-verified.**
   Added `PNS_BRIDGE::Reset()` (walks `BOARD::Tracks()` — covers segments/
   vias/arcs — removing each via `BOARD::Remove()`, then `ClearWorld()`/
   `SyncWorld()` against the same `BOARD` instance, preserving footprint
   placement) and `PNS_BRIDGE::NetPads()` (enumerates every pad on the
   board so an agent/script can find route endpoints programmatically
   instead of guessing coordinates via `query_hover_items()`). Bound as
   `PNSBridge.reset()`/`PNSBridge.net_pads()`. Only ever tested on one net
   on a toy board so far, still — this is new, uncompiled C++, same
   "expect iteration from real Colab output" caveat as everything else in
   this doc.
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
4. **The Gym environment — implemented, Python logic locally verified,
   real router behavior not yet observed.** `pcbworld/env/pcb_route_env.py`:
   `PCBRouteEnv(gym.Env)`, one net routed per "leg", nets sequenced across
   an episode (`net_order` param — deliberately left as a caller choice,
   not decided here, since ROADMAP already flagged sequencing as its own
   design question). Action: `Box(4,)` = push delta (x, y) + fix/via
   thresholds. Reward: potential-based shaping on wirelength, via count,
   and `run_drc()`-reported error count (checked once per net-finish, not
   every step — `DRC_ENGINE` is a full-board check). Observation is a
   small 5-vector (delta-to-target, progress, via count, DRC errors) —
   deliberately not the paper's Fourier-feature geometry encoding; see
   "novel SOTA agent" below for where a richer encoder could go later.
   Verified locally with `tests/fake_bridge.py` (a stand-in
   `pcbworld_pns_bridge` module) exercising `reset()`/`step()` control flow
   end to end (`tests/test_pcb_route_env.py`) — this catches Python bugs
   only, not whether the reward signal makes sense against real routing,
   which needs the actual bridge in Colab.
5. **Baseline PPO agent — implemented, training loop locally verified,
   never seen a real reward signal.** `pcbworld/agents/ppo_baseline.py`:
   plain-PyTorch single-process PPO (MLP actor-critic, GAE, clipped
   surrogate) — deliberately not matching the paper's Transformer policy,
   this is the "does the plumbing work" baseline ROADMAP asked for, not a
   paper-matching comparison yet. `tests/test_ppo_baseline.py` runs the
   full rollout-collect → GAE → update loop against the fake bridge and
   confirms finite losses/weights — real per-CPU-core multiprocessing
   (`docs/performance.md`) isn't implemented, single env only.

Items 2-5 above are batched into one commit deliberately, so only **one**
more Colab rebuild is needed to pick up all of it (per-item rebuilds were
the slow part previously). Item 2 is the only piece touching the compiled
bridge; items 3-5 are pure Python and need no recompile at all once item
2's build succeeds. First things to check once that Colab run happens:
does `b.reset()` actually leave footprints in place and let a second
`start_route()` succeed on the same board; does an episode of
`PCBRouteEnv` against a real `generate_board.py` board produce sane
(non-exploding) rewards; does `ppo_baseline.py` training against that env
show *any* learning signal (even just "manages to finish nets more often
over time") as opposed to just "doesn't crash".

## The "novel SOTA agent" — candidate directions

Deliberately not decided yet — the user wanted this deferred until the
engine wrapper was proven working, which it now is. Pick one (or combine)
once the plumbing above exists to actually train against.

- **Rip-up-and-reroute (leading candidate).** The PCBWorld paper
  explicitly lists this as a limitation they *couldn't* do — they excluded
  track-removal APIs entirely from their action space. This bridge already
  has full `RemoveItem` support (see `PNS_BRIDGE_IFACE`), so an agent that
  can rip up its own bad routing decisions and retry is something
  structurally impossible for their baseline. This is the most concrete,
  credible "beat the paper" angle — worth prioritizing first exploration
  here.
- **Graph-transformer policy.** The paper's RL wrapper flattens geometry
  into Fourier-feature sequences for a small Transformer. Routing state
  (nets, pads, obstacles, connectivity) is naturally graph-structured —
  a GNN/graph-transformer encoder is a plausible architectural improvement
  independent of the rip-up-reroute question.
- **Net-ordering meta-policy.** The order nets get routed in materially
  affects final quality (a well-known hard sub-problem in PCB routing) and
  isn't addressed by the paper's per-net action space. A meta-policy that
  sequences nets, with a lower-level policy (or even a classical router)
  executing each one, is a different axis of improvement than
  architecture changes to the per-step policy.
- Weaker/later ideas mentioned in passing: diffusion/flow-based generative
  routing, LLM-for-strategy + classical-for-execution hybrids,
  multi-agent-per-net RL with coordination. Not fleshed out — flag if
  picking one of these, since they need more design work than the three
  above before implementation starts.

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
                  # kicad_headless_mocks.cpp, CMakeLists.txt
  env/            # Empty -- next major piece to build
  agents/         # Empty
  data/           # Empty -- synthetic board generator goes here
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
