# Headless access to KiCad's PNS::ROUTER

This is the load-bearing technical question for the whole project. Findings
below are from directly inspecting the KiCad 9.0.8 source
(`gitlab.com/kicad/code/kicad`, tag `9.0.8`), not from the PCBWorld paper
(which names its dependency as `kicad-python 0.6.0` but does not open-source
its wrapper, and does not actually match what that package exposes).

## The public `kicad-python` IPC API is not enough

`kicad-python` (the official IPC client, PyPI `kicad-python`) talks to a
running KiCad instance over a protobuf/NNG socket. Its surface is static
board CRUD: read/add/remove tracks, vias, footprints, zones; run DRC; query
nets. There is no session concept for the interactive push-and-shove router
(no `start_route`, no incremental `push`, no PNS handle). Tools like
OrthoRoute that use this API implement their own routing algorithm and only
use IPC to read board state and write finished tracks back — they don't
touch the real engine. This is confirmed independently of the paper.

## The real thing: `PNS::ROUTER` needs no GUI at all

`PNS::ROUTER` (`pcbnew/router/pns_router.h`) is a plain C++ class. It only
*forward-declares* `KIGFX::VIEW` and never dereferences it unless a view was
explicitly attached — every display method in `PNS_KICAD_IFACE` early-returns
`if (!m_view) return;` (`pcbnew/router/pns_kicad_iface.cpp:974,1007,1043,1058`
etc). Rendering, `wx` event loops and `TOOL_MANAGER` are entirely optional.

Proof this works in practice: KiCad's own CI ships a **headless PNS test
harness** we can copy almost directly:

- `qa/tests/pcbnew/test_pns_basics.cpp` — a Boost.Test fixture that
  instantiates `PNS::ROUTER` directly with a minimal custom
  `PNS_KICAD_IFACE_BASE` subclass (`MOCK_PNS_KICAD_IFACE`), no board, no view:
  ```cpp
  m_router = new PNS::ROUTER;
  m_iface  = new MOCK_PNS_KICAD_IFACE( this );
  m_router->SetInterface( m_iface );
  ```
- `qa/tools/pns/pns_log_player.cpp` (`PNS_LOG_PLAYER::createRouter`,
  `ReplayLog`) — this is the closest thing to a reference implementation for
  what we're building. It loads a real `BOARD`, wires up the router, and
  **replays a full routing session as a sequence of engine calls**, exactly
  the "spoofed interactive events" pattern:
  ```cpp
  m_iface->SetBoard( m_board.get() );
  m_router->SetInterface( m_iface.get() );
  m_router->SyncWorld();
  m_router->SetMode( PNS::PNS_MODE_ROUTE_SINGLE );
  ...
  m_router->StartRouting( point, startItem, layer );   // pick startItem via
                                                         // QueryHoverItems(point)
  m_router->Move( point, endItem );                     // push a coordinate
  m_router->FixRoute( point, endItem, forceFinish, forceCommit );
  m_router->CommitRouting();
  ```
  (`qa/tools/pns/pns_log_player.cpp:44-306`, call sites cross-checked against
  `pcbnew/router/router_tool.cpp:886-2528` where the real interactive GUI
  tool drives the exact same calls from mouse events.)

`PNS::ROUTER`'s public API (`pcbnew/router/pns_router.h:124-268`) is the
paper's described surface almost verbatim: `SetMode(ROUTER_MODE)`,
`StartRouting`, `Move`, `FixRoute`, `CommitRouting`, `SyncWorld`,
`ToggleViaPlacement`, `SwitchLayer`, `QueryHoverItems`.

## Getting real `pcbnew.TRACK`/`PCB_VIA` objects out

This is the one place the reference tooling doesn't do what we need.
`PNS_LOG_PLAYER` deliberately uses the *base* iface
(`PNS_KICAD_IFACE_BASE::Commit()` is a no-op — it's a regression tool that
only diffs PNS-internal items, it never touches the real `BOARD`).

The real GUI iface, `PNS_KICAD_IFACE` (`pcbnew/router/pns_kicad_iface.h:127`),
does the real conversion: `createBoardItem(PNS::ITEM*)` turns a
`PNS::SEGMENT`/`PNS::ARC`/`PNS::VIA` into a real `PCB_TRACK`/`PCB_ARC`/
`PCB_VIA`, and `Commit()` pushes it into the board via `BOARD_COMMIT`
(`pns_kicad_iface.cpp:2212,2250`). The catch: `BOARD_COMMIT` is only
constructed in `SetHostTool(PCB_TOOL_BASE*)`
(`pns_kicad_iface.cpp:2326-2330`), and `BOARD_COMMIT`'s constructors all want
a `TOOL_BASE`/`TOOL_MANAGER`/`EDA_DRAW_FRAME` — GUI-tool machinery we don't
have headlessly.

**What we actually built:** subclass `PNS_KICAD_IFACE_BASE` (not the real
`PNS_KICAD_IFACE` — see below for why), reimplementing `createBoardItem`/
`modifyBoardItem` ourselves (ported from `pns_kicad_iface.cpp`, minus the
drag/footprint-offset bookkeeping) and calling `BOARD::Add()`/`BOARD::Remove()`
directly instead of `BOARD_COMMIT`. `BOARD::Add`/`Remove` are plain public
methods with no GUI dependency — `BOARD_COMMIT` only exists for undo/redo
grouping, which we don't need in a training loop.

**Why not just subclass the real `PNS_KICAD_IFACE`:** tried that first, hit
a real linker problem. `createBoardItem`/`modifyBoardItem` only exist on
`PNS_KICAD_IFACE`, and it and `PNS_KICAD_IFACE_BASE` are both implemented in
the *same* `pns_kicad_iface.cpp` — one compiled object file. Static linking
is all-or-nothing per object file (no per-function granularity, and KiCad's
build doesn't set `-ffunction-sections`, so `--gc-sections` can't help
either). So needing anything from that file — even just the harmless base
class's `SyncWorld()` — pulls in `PNS_KICAD_IFACE::Commit()`'s dead code
too, which calls `BOARD_COMMIT::Push()` (`pcbnew/board_commit.cpp`, part of
the `pcbcommon` library we must link), which looks up `ZONE_FILLER_TOOL`
and `PCB_SELECTION_TOOL` through `TOOL_MANAGER` — real GUI-only tool
classes implemented in `pcbnew/tools/*.cpp` (part of `pcbnew_kiface_objects`,
the full GUI object library, which needs a live wx app and is exactly what
we're avoiding). Confirmed via `nm` on the actual built objects that
`board_commit.cpp.o` is the *only* source of the undefined typeinfo
symbols, and that nothing else in `pcbcommon` needs it — it's pulled in
purely by `pns_kicad_iface.cpp.o` needing to be linked as a whole.

First fix attempt: a narrow, hand-rolled stub (`gui_tool_stubs.cpp`, since
removed) providing minimal standalone `ZONE_FILLER_TOOL`/`PCB_SELECTION_TOOL`
definitions — not inheriting from their real base classes, just matching the
exact method signatures referenced. Two follow-on problems surfaced:

1. **LTO silently stripped it.** KiCad's Release config enables
   `INTERPROCEDURAL_OPTIMIZATION` project-wide. LTO's dead-code analysis is
   scoped to *this target's own source files* — it can't see that a
   separately pre-built static library member (`board_commit.cpp.o`,
   inside `pcbcommon.a`) still needs these symbols, so it "correctly" (from
   its limited view) stripped them. Build and link both reported success;
   the symbols were simply gone from the final `.so`, only surfacing as an
   `ImportError` at import time. Fixed by disabling
   `INTERPROCEDURAL_OPTIMIZATION` for this one target
   (`pcbworld/engine/cpp/CMakeLists.txt`) — negligible cost given the
   target is a handful of small files.

2. **More missing symbols kept surfacing one at a time**
   (`GFootprintTable`, etc.) — each individually traceable, but slow to
   discover via trial and error. Turns out KiCad already solved this exact
   problem: `qa/qa_utils/mocks.cpp` is what makes `qa_pns_regressions`/
   `pns_debug_tool` (`qa/tools/pns/`) link successfully as real executables
   against the same `pnsrouter + pcbcommon + connectivity + gal + common`
   set we use — a *stronger* check than ours, since executables require
   every symbol resolved at link time, not just at import. Adopted an
   adapted copy as `pcbworld/engine/cpp/kicad_headless_mocks.cpp` instead
   of continuing to discover the same missing pieces individually. Dropped
   the dialog (`DIALOG_FIND`/`DIALOG_FILTER_SELECTION`) and board-stackup
   color-list stubs from the original — those are needed by
   `qa_pns_regressions`'s *own* additional source files
   (`pcb_test_frame.cpp`, `stackup_predefined_prms.cpp`), which we don't
   compile, not by `pnsrouter`/`pcbcommon`/`connectivity` themselves.

One symbol turned out not to be dead code at all:
`GetStandardColors(BOARD_STACKUP_ITEM_TYPE)`, called from
`pcbnew/pcb_io/ipc2581/pcb_io_ipc2581.cpp` — part of `PCBNEW_IO_LIBRARIES`,
which we genuinely need for board load/save, not something we can treat as
unreachable. Its real implementation
(`pcbnew/board_stackup_manager/stackup_predefined_prms.cpp`) is a small,
GUI-free pure-data file (standard fabrication color lists), so rather than
mock it we compile the real file directly as an extra source in
`pcbworld/engine/cpp/CMakeLists.txt` — the same thing
`qa/tools/pns/CMakeLists.txt` does for the same reason.

Next: `Kiface()` (`include/kiface_base.h:134`) — KiCad's global "which DSO
am I" accessor, one instance defined per real KIFACE (`pcbnew.cpp`,
`eeschema.cpp`, etc). KiCad has its own headless answer for this too
(`qa/mocks/kicad/common_mocks.cpp`), but it goes through a mock-object
framework (`turtlemocks`, via `qa_utils/wx_utils/unit_test_utils.h`) that
would be a new dependency for just 3 methods. `KIFACE_BASE` only has 3 pure
virtuals (`OnKifaceStart`, `CreateKiWindow`, `IfaceOrAddress`), so
`kicad_headless_mocks.cpp` implements a minimal direct subclass instead of
pulling in the mock framework.

`Pgm()` (`include/pgm_base.h:453`, same "one global accessor per
substrate piece" pattern) is a plausible next candidate — `PGM_BASE` is
referenced as a parameter type in `KIFACE_BASE::OnKifaceStart` and is a
much bigger class (KiCad's own mock stubs ~15 methods). Deliberately not
pre-emptively stubbed: no confirmed evidence yet that anything we actually
link calls it, and guessing wrong on a class that size risks costing more
than waiting for real evidence via the same `nm`-based tracing used
throughout this doc.

All of this is safe for the same reason KiCad's own QA tooling is safe:
none of this GUI-tool-framework code (`PCB_SELECTION_TOOL`,
`ZONE_FILLER_TOOL`, `PCB_TOOL_BASE`, dialogs) is ever actually invoked by
anything the bridge does — we never construct a `TOOL_MANAGER`, so nothing
ever calls into it. It exists purely to satisfy the linker. Linking is
name-based (the linker matches mangled symbol names, not semantic type
compatibility), so a mock with different internals than the real class is
fine as long as the signatures match.

`Pgm()` never actually surfaced as a missing symbol — the bridge imported
cleanly after the `Kiface()` fix, so that speculative concern turned out
to be unnecessary.

**Update, once DRC support needed `pcbnew_kiface_objects`:** everything in
this section describes why `kicad_headless_mocks.cpp` was necessary *at
the time* — when this target only linked `pnsrouter`/`pcbcommon`/
`connectivity`/`gal`/`common`/`scripting`, none of which contain real
`PCB_SELECTION_TOOL`/`ZONE_FILLER_TOOL`/`PCB_TOOL_BASE`/`Kiface()`/
`GFootprintTable` definitions. Adding `pcbnew_kiface_objects` to the link
line later (for `DRC_ENGINE`, since `drc/*.cpp` lives in pcbnew's own
object library, not `pcbcommon`) changed that: `pcbnew_kiface_objects` *is*
pcbnew's full object library, so it already contains real definitions of
every symbol this file mocked, plus its own copy of
`stackup_predefined_prms.cpp` (originally compiled directly into this
target for the same "not a dead code path" reason described below).
Keeping the mocks after that point caused "multiple definition" link
errors against the real ones — `kicad_headless_mocks.cpp` and the extra
`stackup_predefined_prms.cpp` source were both deleted from
`pcbworld/engine/cpp/CMakeLists.txt` once `pcbnew_kiface_objects` made
them redundant. If this bridge is ever split back apart (DRC dropped,
`pcbnew_kiface_objects` unlinked again), these mocks would need to come
back — this section stays as the reference for why, and the git history
around the DRC commit has the working implementation if that ever
happens.

## Runtime crash: stale candidate ids (not a linker issue)

Once the bridge imported successfully, the first real routing attempt
crashed the whole Colab process (no Python traceback — a C++-level fault).
Root cause was in our own code, not KiCad's: `PNS_BRIDGE::QueryHoverItems`
cleared its candidate list on *every* call and returned ids as plain
indices into it. Querying near pad A, then querying again near pad B,
silently invalidated pad A's id (id `0` now pointed at pad B's item) — so
`start_route(pad_a.x, pad_a.y, pad_a.id, ...)` handed the router pad A's
*coordinates* together with pad B's *item pointer*, a mismatch KiCad's
routing geometry code isn't expecting and evidently doesn't handle
gracefully. Traced by reading `PNS_PCBNEW_RULE_RESOLVER::Clearance()`/
`QueryConstraint()` first (both properly null-guard a missing `DRC_ENGINE`,
ruling that out) before finding the actual bug in `pns_bridge.cpp`.

Fixed by making candidate ids stable and persistent: `PNS_BRIDGE` now keeps
an append-only `m_candidateItems` vector plus a
`std::unordered_map<PNS::ITEM*, long long> m_candidateIds` for
deduplication, cleared only in `LoadBoard()` (a genuinely new PNS world),
not on every query. An id from an earlier query stays valid no matter how
many subsequent queries happen in between.

Real bug, worth having fixed — but it turned out **not** to be the actual
cause of the crash: it persisted after this fix too. See below.

## Runtime crash, take 2: `ROUTER::m_settings` is never initialized

Isolated by splitting the routing sequence into one cell per call (load,
net query, both pad queries, mode/width setup all succeeded and printed
fine) — the crash was in `start_route` specifically, KiCad's
`PNS::ROUTER::StartRouting()` itself.

`StartRouting()`'s very first action is `GetRuleResolver()->ClearCaches()`,
then `isStartingPointRoutable()`, whose first line is
`Settings().AllowDRCViolations()`. `ROUTER::Settings()`
(`pcbnew/router/pns_router.h:197`) is a bare `return *m_settings;` —
and `ROUTER::ROUTER()` (`pcbnew/router/pns_router.cpp:75`) sets
`m_settings = nullptr;` in the constructor. It's *only* ever assigned via
`LoadSettings(ROUTING_SETTINGS*)`, an explicit call every real KiCad code
path makes (the GUI's `router_tool.cpp`, and headlessly,
`qa/tools/pns/pns_log_player.cpp: m_router->LoadSettings(m_routingSettings.get())`)
— which our bridge never did. First real call into the router = immediate
null dereference = the whole process dies with no Python-catchable
exception, since it's a C++ fault below pybind11's exception translation
layer.

Fixed in `LoadBoard()`: construct a `PNS::ROUTING_SETTINGS` the same way
`pns_log_player.cpp` does (`new ROUTING_SETTINGS(nullptr, "")` — no parent
JSON_SETTINGS, no path, a standalone in-memory settings object) and call
`m_router->LoadSettings(...)` before `SyncWorld()`.

## Runtime crash, take 3: `pcbworld_pns_bridge` + system `pcbnew` in one process

With routing itself fixed, the notebook's final verification cell —
`import pcbnew` for an independent reload-and-check, in the *same* kernel
that already had `import pcbworld_pns_bridge as bridge` from step 4 —
started crashing. Isolated by splitting that cell line-by-line
(`import pcbnew` alone was the one that died) and by the crash's own
signature: it didn't reproduce in a fresh process (`import pcbnew` alone
works fine, as `scripts/make_toy_board.py` already proved), only when run
after the bridge module was already loaded in the same interpreter.

This is architectural, not a one-line bug. `pcbworld_pns_bridge` statically
links large, overlapping chunks of KiCad's own C++ code (`BOARD`,
`PCB_TRACK`, connectivity, GAL, ...) into its `.so` — including our own
`kicad_headless_mocks.cpp` definitions of `Kiface()` and `GFootprintTable`,
both of which KiCad's own headers document as "KIFACE scope" /
one-instance-per-process globals (`include/kiface_base.h:134`,
`include/fp_lib_table.h:294`). The system `pcbnew` module has its own
*real* versions of the exact same globals, with the same
must-be-unique-per-process contract. KiCad's multi-DSO architecture was
designed around exactly one such set existing per process (one running
`pcbnew`/`eeschema`/etc at a time) — never two independently-compiled
copies coexisting via `dlopen`, which is exactly what two separate Python
`import`s of two different KiCad-derived `.so`s does.

No in-process fix attempted — the correct fix is architectural: never let
both modules land in the same process. `scripts/verify_routed_board.py`
runs the reload-and-check as a genuinely separate script/subprocess
(mirroring `scripts/make_toy_board.py`'s existing pattern), and the
notebook's step 5 now calls it via `%%bash`, not an inline `import`. This
is a hard constraint for the RL environment too, not just this notebook —
see `docs/performance.md`.

## Build plan

Link the same targets KiCad's own headless PNS/DRC unit tests link against
(`qa/tests/pcbnew/CMakeLists.txt:130-142`):
`pcbnew_kiface_objects`, `pnsrouter`, `pcbcommon`, `connectivity`, `gal`,
`common`, `scripting`. Board load/save reuses `PCB_IO_KICAD_SEXPR` (same
plugin `kicad-cli pcb ...` uses) and DRC reuses `DRC_ENGINE` — KiCad already
has a full **headless DRC regression suite**
(`qa/tests/pcbnew/drc/test_drc_copper_conn.cpp` etc., using
`KI_TEST::LoadBoard` + `BOARD_DESIGN_SETTINGS::m_DRCSeverities`), so DRC
access is lower-risk than routing and will be wired up once routing works.

Our bridge (`pcbworld/engine/cpp/`) builds as an extra CMake target inside a
KiCad source checkout (same pattern as the `qa/tools/pns` executables),
producing a pybind11 Python module, not by patching KiCad itself — no fork
needed. This is compiled once per Colab session and can be cached to Drive.

## Open questions to resolve once this compiles

1. Does `PAD`/`PCB_TRACK` hit-testing for `QueryHoverItems` need a working
   `BOARD_DESIGN_SETTINGS`/`NETINFO_LIST` fully populated, or does a bare
   `PCB_IO_KICAD_SEXPR`-loaded board already have that? (Likely yes — DRC
   tests load boards the same way and immediately run full DRC.)
2. Exact linking errors/missing symbols once building the bridge target for
   real — expect iteration here, this is untested C++.
