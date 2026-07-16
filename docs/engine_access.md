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

All of this is safe for the same reason KiCad's own QA tooling is safe:
none of this GUI-tool-framework code (`PCB_SELECTION_TOOL`,
`ZONE_FILLER_TOOL`, `PCB_TOOL_BASE`, dialogs) is ever actually invoked by
anything the bridge does — we never construct a `TOOL_MANAGER`, so nothing
ever calls into it. It exists purely to satisfy the linker. Linking is
name-based (the linker matches mangled symbol names, not semantic type
compatibility), so a mock with different internals than the real class is
fine as long as the signatures match.

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
