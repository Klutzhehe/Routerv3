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
    kicad_headless_mocks.cpp
)

# KiCad's Release config enables LTO project-wide, which silently stripped
# an earlier, narrower version of kicad_headless_mocks.cpp's symbols (LTO's
# dead-code analysis only sees this target's own files, with no visibility
# into the separately pre-built pcbcommon.a/pnsrouter.a members that
# actually need them -- see the comment at the top of that file).
# Disabling LTO for this tiny target removes that whole risk class; cost is
# unmeasurable given the target is a handful of small files.
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
target_link_libraries( pcbworld_pns_bridge PRIVATE
    pnsrouter
    pcbcommon
    connectivity
    3d-viewer
    gal
    common
    scripting
    ${PCBNEW_IO_LIBRARIES}
)
