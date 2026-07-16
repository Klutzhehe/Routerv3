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

**Plan:** subclass `PNS_KICAD_IFACE` ourselves, keep its (protected)
`createBoardItem`/`modifyBoardItem` conversion logic, but override `Commit()`
to skip `BOARD_COMMIT` entirely and call `BOARD::Add()` / `BOARD::Remove()`
directly on the created items. `BOARD::Add`/`Remove` are plain public methods
with no GUI dependency — `BOARD_COMMIT` only exists for undo/redo grouping,
which we don't need in a training loop. This is new code we're writing (not
copied from KiCad), flagged in `pcbworld/engine/cpp/pns_bridge.cpp` as the
one part that needs verification once it actually compiles and links.

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
