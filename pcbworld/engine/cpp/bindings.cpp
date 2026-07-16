#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pns_bridge.h"

namespace py = pybind11;
using pcbworld::PNS_BRIDGE;

PYBIND11_MODULE( pcbworld_pns_bridge, m )
{
    m.doc() = "Headless bridge to KiCad's PNS::ROUTER (see docs/engine_access.md)";

    py::class_<PNS_BRIDGE::Candidate>( m, "Candidate" )
        .def_readonly( "id", &PNS_BRIDGE::Candidate::id )
        .def_readonly( "x", &PNS_BRIDGE::Candidate::x )
        .def_readonly( "y", &PNS_BRIDGE::Candidate::y )
        .def_readonly( "kind", &PNS_BRIDGE::Candidate::kind )
        .def_readonly( "net", &PNS_BRIDGE::Candidate::net );

    py::class_<PNS_BRIDGE>( m, "PNSBridge" )
        .def( py::init<>() )
        .def( "load_board", &PNS_BRIDGE::LoadBoard, py::arg( "path" ) )
        .def( "save_board", &PNS_BRIDGE::SaveBoard, py::arg( "path" ) )
        .def( "net_names", &PNS_BRIDGE::NetNames )
        .def( "query_hover_items", &PNS_BRIDGE::QueryHoverItems, py::arg( "x" ), py::arg( "y" ),
              py::arg( "layer" ) = -1, py::arg( "slop_radius" ) = 100000 )
        .def( "start_route", &PNS_BRIDGE::StartRoute, py::arg( "x" ), py::arg( "y" ),
              py::arg( "item_id" ), py::arg( "layer" ) )
        .def( "push", &PNS_BRIDGE::Push, py::arg( "x" ), py::arg( "y" ),
              py::arg( "item_id" ) = -1 )
        .def( "fix", &PNS_BRIDGE::Fix, py::arg( "x" ), py::arg( "y" ), py::arg( "item_id" ) = -1,
              py::arg( "force_finish" ) = false, py::arg( "force_commit" ) = false )
        .def( "commit_routing", &PNS_BRIDGE::CommitRouting )
        .def( "stop_routing", &PNS_BRIDGE::StopRouting )
        .def( "set_mode", &PNS_BRIDGE::SetMode )
        .def( "set_track_width", &PNS_BRIDGE::SetTrackWidth )
        .def( "set_via_diameter", &PNS_BRIDGE::SetViaDiameter )
        .def( "set_via_drill", &PNS_BRIDGE::SetViaDrill )
        .def( "toggle_via_placement", &PNS_BRIDGE::ToggleViaPlacement )
        .def( "switch_layer", &PNS_BRIDGE::SwitchLayer );

    // PNS::ROUTER_MODE (pcbnew/router/pns_router.h)
    m.attr( "MODE_ROUTE_SINGLE" ) = 1;
    m.attr( "MODE_ROUTE_DIFF_PAIR" ) = 2;
    m.attr( "MODE_TUNE_SINGLE" ) = 3;
    m.attr( "MODE_TUNE_DIFF_PAIR" ) = 4;
    m.attr( "MODE_TUNE_DIFF_PAIR_SKEW" ) = 5;
}
