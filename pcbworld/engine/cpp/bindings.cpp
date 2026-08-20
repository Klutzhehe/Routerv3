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

    py::class_<PNS_BRIDGE::DRCViolation>( m, "DRCViolation" )
        .def_readonly( "error_code", &PNS_BRIDGE::DRCViolation::errorCode )
        .def_readonly( "message", &PNS_BRIDGE::DRCViolation::message )
        .def_readonly( "severity", &PNS_BRIDGE::DRCViolation::severity )
        .def_readonly( "x", &PNS_BRIDGE::DRCViolation::x )
        .def_readonly( "y", &PNS_BRIDGE::DRCViolation::y );

    py::class_<PNS_BRIDGE::NetPad>( m, "NetPad" )
        .def_readonly( "net", &PNS_BRIDGE::NetPad::net )
        .def_readonly( "pad_name", &PNS_BRIDGE::NetPad::padName )
        .def_readonly( "x", &PNS_BRIDGE::NetPad::x )
        .def_readonly( "y", &PNS_BRIDGE::NetPad::y )
        .def_readonly( "layer", &PNS_BRIDGE::NetPad::layer );

    py::class_<PNS_BRIDGE::TrackSegment>( m, "TrackSegment" )
        .def_readonly( "x1", &PNS_BRIDGE::TrackSegment::x1 )
        .def_readonly( "y1", &PNS_BRIDGE::TrackSegment::y1 )
        .def_readonly( "x2", &PNS_BRIDGE::TrackSegment::x2 )
        .def_readonly( "y2", &PNS_BRIDGE::TrackSegment::y2 )
        .def_readonly( "width", &PNS_BRIDGE::TrackSegment::width )
        .def_readonly( "layer", &PNS_BRIDGE::TrackSegment::layer )
        .def_readonly( "net", &PNS_BRIDGE::TrackSegment::net )
        .def_readonly( "is_arc", &PNS_BRIDGE::TrackSegment::isArc );

    py::class_<PNS_BRIDGE::ViaGeom>( m, "ViaGeom" )
        .def_readonly( "x", &PNS_BRIDGE::ViaGeom::x )
        .def_readonly( "y", &PNS_BRIDGE::ViaGeom::y )
        .def_readonly( "diameter", &PNS_BRIDGE::ViaGeom::diameter )
        .def_readonly( "drill", &PNS_BRIDGE::ViaGeom::drill )
        .def_readonly( "layer_top", &PNS_BRIDGE::ViaGeom::layerTop )
        .def_readonly( "layer_bottom", &PNS_BRIDGE::ViaGeom::layerBottom )
        .def_readonly( "net", &PNS_BRIDGE::ViaGeom::net );

    py::class_<PNS_BRIDGE::PadGeom>( m, "PadGeom" )
        .def_readonly( "x", &PNS_BRIDGE::PadGeom::x )
        .def_readonly( "y", &PNS_BRIDGE::PadGeom::y )
        .def_readonly( "size_x", &PNS_BRIDGE::PadGeom::sizeX )
        .def_readonly( "size_y", &PNS_BRIDGE::PadGeom::sizeY )
        .def_readonly( "layer_top", &PNS_BRIDGE::PadGeom::layerTop )
        .def_readonly( "layer_bottom", &PNS_BRIDGE::PadGeom::layerBottom )
        .def_readonly( "net", &PNS_BRIDGE::PadGeom::net )
        .def_readonly( "pad_name", &PNS_BRIDGE::PadGeom::padName );

    py::class_<PNS_BRIDGE::ZoneGeom>( m, "ZoneGeom" )
        .def_readonly( "outline", &PNS_BRIDGE::ZoneGeom::outline )
        .def_readonly( "layer", &PNS_BRIDGE::ZoneGeom::layer )
        .def_readonly( "is_keepout", &PNS_BRIDGE::ZoneGeom::isKeepout )
        .def_readonly( "net", &PNS_BRIDGE::ZoneGeom::net );

    py::class_<PNS_BRIDGE::FootprintBBox>( m, "FootprintBBox" )
        .def_readonly( "x1", &PNS_BRIDGE::FootprintBBox::x1 )
        .def_readonly( "y1", &PNS_BRIDGE::FootprintBBox::y1 )
        .def_readonly( "x2", &PNS_BRIDGE::FootprintBBox::x2 )
        .def_readonly( "y2", &PNS_BRIDGE::FootprintBBox::y2 )
        .def_readonly( "reference", &PNS_BRIDGE::FootprintBBox::reference );

    py::class_<PNS_BRIDGE::EdgeShape>( m, "EdgeShape" )
        .def_readonly( "shape_type", &PNS_BRIDGE::EdgeShape::shapeType )
        .def_readonly( "x1", &PNS_BRIDGE::EdgeShape::x1 )
        .def_readonly( "y1", &PNS_BRIDGE::EdgeShape::y1 )
        .def_readonly( "x2", &PNS_BRIDGE::EdgeShape::x2 )
        .def_readonly( "y2", &PNS_BRIDGE::EdgeShape::y2 )
        .def_readonly( "width", &PNS_BRIDGE::EdgeShape::width );

    py::class_<PNS_BRIDGE::BoardGeometry>( m, "BoardGeometry" )
        .def_readonly( "tracks", &PNS_BRIDGE::BoardGeometry::tracks )
        .def_readonly( "vias", &PNS_BRIDGE::BoardGeometry::vias )
        .def_readonly( "pads", &PNS_BRIDGE::BoardGeometry::pads )
        .def_readonly( "zones", &PNS_BRIDGE::BoardGeometry::zones )
        .def_readonly( "courtyards", &PNS_BRIDGE::BoardGeometry::courtyards )
        .def_readonly( "board_edge", &PNS_BRIDGE::BoardGeometry::boardEdge );

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
        .def( "reset", &PNS_BRIDGE::Reset )
        .def( "net_pads", &PNS_BRIDGE::NetPads )
        .def( "run_drc", &PNS_BRIDGE::RunDRC )
        .def( "get_board_geometry", &PNS_BRIDGE::GetBoardGeometry )
        .def( "set_mode", &PNS_BRIDGE::SetMode )
        .def( "set_collision_mode", &PNS_BRIDGE::SetCollisionMode )
        .def( "set_track_width", &PNS_BRIDGE::SetTrackWidth )
        .def( "set_via_diameter", &PNS_BRIDGE::SetViaDiameter )
        .def( "set_via_drill", &PNS_BRIDGE::SetViaDrill )
        .def( "set_diff_pair_gap", &PNS_BRIDGE::SetDiffPairGap, py::arg( "gap" ) )
        .def( "set_diff_pair_via_gap", &PNS_BRIDGE::SetDiffPairViaGap, py::arg( "gap" ) )
        .def( "set_diff_pair_width", &PNS_BRIDGE::SetDiffPairWidth, py::arg( "width" ) )
        .def( "set_target_length", &PNS_BRIDGE::SetTargetLength, py::arg( "length" ) )
        .def( "set_meander_max_amplitude", &PNS_BRIDGE::SetMeanderMaxAmplitude, py::arg( "max_amp" ) )
        .def( "set_meander_spacing", &PNS_BRIDGE::SetMeanderSpacing, py::arg( "spacing" ) )
        .def( "toggle_via_placement", &PNS_BRIDGE::ToggleViaPlacement )
        .def( "switch_layer", &PNS_BRIDGE::SwitchLayer );


    // PNS::ROUTER_MODE (pcbnew/router/pns_router.h)
    m.attr( "MODE_ROUTE_SINGLE" ) = 1;
    m.attr( "MODE_ROUTE_DIFF_PAIR" ) = 2;
    m.attr( "MODE_TUNE_SINGLE" ) = 3;
    m.attr( "MODE_TUNE_DIFF_PAIR" ) = 4;
    m.attr( "MODE_TUNE_DIFF_PAIR_SKEW" ) = 5;

    // PNS::PNS_MODE (pcbnew/router/pns_routing_settings.h) -- collision
    // response mode, set via set_collision_mode(). LoadBoard() already
    // defaults to RM_MARK_OBSTACLES; these constants are for code that
    // wants to opt back into classical Shove/Walkaround (e.g. a baseline
    // comparison run against the RL-trained agent).
    m.attr( "RM_MARK_OBSTACLES" ) = 0;
    m.attr( "RM_SHOVE" ) = 1;
    m.attr( "RM_WALKAROUND" ) = 2;
}
