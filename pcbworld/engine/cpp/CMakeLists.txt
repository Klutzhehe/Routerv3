# Builds pcbworld_pns_bridge, a pybind11 module wrapping KiCad's PNS::ROUTER.
#
# NOT standalone: this must be add_subdirectory()'d from *inside* a KiCad
# source checkout's own CMake configure (same pattern KiCad uses for
# qa/tools/pns), so it inherits KiCad's wx/compile flags and can link
# against KiCad's own in-tree targets (pnsrouter, pcbcommon, ...) instead
# of fighting prebuilt-library ABI mismatches. See notebooks/00_setup.ipynb
# for how this gets wired into a KiCad build in Colab.

find_package( pybind11 CONFIG REQUIRED )

pybind11_add_module( pcbworld_pns_bridge
    pns_bridge.cpp
    bindings.cpp
)

# kicad_headless_mocks.cpp and the standalone stackup_predefined_prms.cpp
# compile (both previously listed above) are gone as of the DRC work
# below: kicad_headless_mocks.cpp existed purely to satisfy the linker for
# GUI-tool-framework symbols (PCB_SELECTION_TOOL, ZONE_FILLER_TOOL,
# PCB_TOOL_BASE, Kiface(), GFootprintTable, ROUTER_TOOL::
# NeighboringSegmentFilter) that weren't otherwise linked when this target
# only pulled in pnsrouter/pcbcommon/connectivity/gal/common/scripting.
# Now that pcbnew_kiface_objects (below) is linked for DRC_ENGINE, it
# already provides *real* definitions of every one of those symbols (it's
# pcbnew's own full object library) -- keeping the mocks around caused
# "multiple definition" link errors against the real ones. Same reasoning
# for stackup_predefined_prms.cpp: pcbnew_kiface_objects already compiles
# it for real, so our own extra copy was a duplicate, not a supplement.
# kicad_headless_mocks.cpp is deleted, not just unlisted here -- nothing
# in the current architecture (bridge always links pcbnew_kiface_objects)
# can need it again without reintroducing the exact conflict just fixed.

# KiCad's Release config enables LTO project-wide. Originally disabled here
# because an earlier, narrower version of kicad_headless_mocks.cpp's
# symbols got silently stripped by LTO's dead-code analysis (no visibility
# into separately pre-built static library members needing them). That
# file is gone now, but LTO stays off for this target regardless -- it's a
# handful of small files, the cost is unmeasurable, and it removes a whole
# class of "did LTO just silently strip something" risk for whatever gets
# added here next.
set_target_properties( pcbworld_pns_bridge PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE )

target_include_directories( pcbworld_pns_bridge PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/pcbnew
    ${CMAKE_SOURCE_DIR}/pcbnew/router
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/common/geometry
    ${CMAKE_SOURCE_DIR}/3d-viewer
)

# Same link set as KiCad's own headless PNS/DRC unit tests
# (qa/tests/pcbnew/CMakeLists.txt) minus the Boost.Test / qa_utils pieces
# we don't need, plus PCBNEW_IO_LIBRARIES (pcbnew/CMakeLists.txt:687) --
# pcbcommon's plugin registry references every non-native board-format
# importer/exporter's vtable (PCB_IO_IPC2581 etc.), so they all have to be
# linked in even though we only construct PCB_IO_KICAD_SEXPR ourselves.
#
# pcbnew_kiface_objects added for DRC_ENGINE/DRC_ITEM (pcbnew/drc/*.cpp are
# compiled into pcbnew's own object library, not pcbcommon) -- routing alone
# never needed it (PNS::ROUTER lives in pnsrouter), so it wasn't linked
# until RunDRC() needed symbols from there. Untested: expect this to be the
# first place new missing-symbol errors surface, same as the routing bridge
# did originally (see docs/engine_access.md) -- iterate from real linker
# output, don't guess further ahead of it.
target_link_libraries( pcbworld_pns_bridge PRIVATE
    pcbnew_kiface_objects
    pnsrouter
    pcbcommon
    connectivity
    3d-viewer
    gal
    common
    scripting
    # pcbnew_kiface_objects includes exporters/export_idf.cpp, which needs
    # IDF_SEGMENT/IDF_POINT/etc from utils/idftools/ -- a separate static
    # library (utils/idftools/CMakeLists.txt: add_library(idf3 STATIC ...)),
    # not part of pcbnew_kiface_objects itself. Confirmed by an
    # ImportError at dlopen time (undefined symbol, lazy binding meant the
    # linker didn't catch it), not a link-time failure.
    idf3
    ${PCBNEW_IO_LIBRARIES}
)
