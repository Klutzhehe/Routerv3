#include "pns_bridge.h"

#include <board.h>
#include <board_connected_item.h>
#include <board_item_container.h>
#include <board_design_settings.h>
#include <project/net_settings.h>
#include <footprint.h>
#include <netinfo.h>
#include <pad.h>
#include <padstack.h>
#include <pcb_shape.h>
#include <pcb_track.h>
#include <zone.h>

#include <geometry/shape_line_chain.h>
#include <geometry/shape_poly_set.h>

#include <pcb_io/kicad_sexpr/pcb_io_kicad_sexpr.h>

#include <drc/drc_engine.h>
#include <drc/drc_item.h>
#include <rc_item.h>

#include <router/pns_arc.h>
#include <router/pns_item.h>
#include <router/pns_itemset.h>
#include <router/pns_meander.h>
#include <router/pns_meander_placer_base.h>
#include <router/pns_node.h>
#include <router/pns_segment.h>
#include <router/pns_solid.h>
#include <router/pns_via.h>

namespace pcbworld
{

namespace
{
// Casts m_router->Placer() to a MEANDER_PLACER_BASE if a tuning route is
// currently active. Only meaningful after StartRouting() -- PNS::ROUTER
// creates the placer then, there's no meander-capable placer beforehand.
PNS::MEANDER_PLACER_BASE* asMeanderPlacer( PNS::ROUTER* aRouter )
{
    if( !aRouter || !aRouter->Placer() )
        return nullptr;

    return dynamic_cast<PNS::MEANDER_PLACER_BASE*>( aRouter->Placer() );
}
}  // namespace

// ---------------------------------------------------------------------
// PNS_BRIDGE_IFACE
//
// Reimplements the ADD/REMOVE/MODIFY subset of what BOARD_COMMIT::Push
// does (pcbnew/board_commit.cpp), directly against BOARD::Add()/Remove(),
// so we don't need a PCB_TOOL_BASE/TOOL_MANAGER. Undo/redo and VIEW/
// CONNECTIVITY bookkeeping are intentionally skipped -- a training loop
// doesn't need undo, and connectivity is rebuilt explicitly by the caller
// (BOARD::BuildConnectivity()) after a route is committed.
// ---------------------------------------------------------------------

// Ported from PNS_KICAD_IFACE::createBoardItem (pns_kicad_iface.cpp) --
// not inherited, see pns_bridge.h for why. Drops the m_fpOffsets bookkeeping
// (SOLID_T/pad case): that's only for component dragging, which we never do.
BOARD_CONNECTED_ITEM* PNS_BRIDGE_IFACE::createBoardItem( PNS::ITEM* aItem )
{
    BOARD_CONNECTED_ITEM* newBoardItem = nullptr;
    NETINFO_ITEM* net = static_cast<NETINFO_ITEM*>( aItem->Net() );

    if( !net )
        net = NETINFO_LIST::OrphanedItem();

    switch( aItem->Kind() )
    {
    case PNS::ITEM::ARC_T:
    {
        PNS::ARC* arc = static_cast<PNS::ARC*>( aItem );
        PCB_ARC*  new_arc = new PCB_ARC( GetBoard(), static_cast<const SHAPE_ARC*>( arc->Shape( -1 ) ) );
        new_arc->SetWidth( arc->Width() );
        new_arc->SetLayer( GetBoardLayerFromPNSLayer( arc->Layers().Start() ) );
        new_arc->SetNet( net );
        newBoardItem = new_arc;
        break;
    }

    case PNS::ITEM::SEGMENT_T:
    {
        PNS::SEGMENT* seg = static_cast<PNS::SEGMENT*>( aItem );
        PCB_TRACK*    track = new PCB_TRACK( GetBoard() );
        const SEG&    s = seg->Seg();
        track->SetStart( VECTOR2I( s.A.x, s.A.y ) );
        track->SetEnd( VECTOR2I( s.B.x, s.B.y ) );
        track->SetWidth( seg->Width() );
        track->SetLayer( GetBoardLayerFromPNSLayer( seg->Layers().Start() ) );
        track->SetNet( net );
        newBoardItem = track;
        break;
    }

    case PNS::ITEM::VIA_T:
    {
        PCB_VIA*  via_board = new PCB_VIA( GetBoard() );
        PNS::VIA* via = static_cast<PNS::VIA*>( aItem );
        via_board->SetPosition( VECTOR2I( via->Pos().x, via->Pos().y ) );
        via_board->SetWidth( PADSTACK::ALL_LAYERS, via->Diameter( 0 ) );
        via_board->SetDrill( via->Drill() );
        via_board->SetNet( net );
        via_board->SetViaType( via->ViaType() ); // MUST be before SetLayerPair()
        via_board->SetIsFree( via->IsFree() );
        via_board->SetLayerPair( GetBoardLayerFromPNSLayer( via->Layers().Start() ),
                                 GetBoardLayerFromPNSLayer( via->Layers().End() ) );
        newBoardItem = via_board;
        break;
    }

    case PNS::ITEM::SOLID_T:
        // Pads already exist on the board; routing never creates one.
        return nullptr;

    default:
        return nullptr;
    }

    if( net->GetNetCode() <= 0 )
    {
        NETINFO_ITEM* newNetInfo = newBoardItem->GetNet();
        newNetInfo->SetParent( GetBoard() );
        newNetInfo->SetNetClass( GetBoard()->GetDesignSettings().m_NetSettings->GetDefaultNetclass() );
    }

    if( aItem->IsLocked() )
        newBoardItem->SetLocked( true );

    return newBoardItem;
}

// Ported from PNS_KICAD_IFACE::modifyBoardItem -- drops the m_commit->Modify()
// calls (undo bookkeeping we don't need) and the SOLID_T/pad-drag case.
void PNS_BRIDGE_IFACE::modifyBoardItem( PNS::ITEM* aItem )
{
    BOARD_ITEM* board_item = aItem->Parent();

    switch( aItem->Kind() )
    {
    case PNS::ITEM::ARC_T:
    {
        PNS::ARC*        arc = static_cast<PNS::ARC*>( aItem );
        PCB_ARC*         arc_board = static_cast<PCB_ARC*>( board_item );
        const SHAPE_ARC* arc_shape = static_cast<const SHAPE_ARC*>( arc->Shape( -1 ) );

        arc_board->SetStart( VECTOR2I( arc_shape->GetP0() ) );
        arc_board->SetEnd( VECTOR2I( arc_shape->GetP1() ) );
        arc_board->SetMid( VECTOR2I( arc_shape->GetArcMid() ) );
        arc_board->SetWidth( arc->Width() );
        break;
    }

    case PNS::ITEM::SEGMENT_T:
    {
        PNS::SEGMENT* seg = static_cast<PNS::SEGMENT*>( aItem );
        PCB_TRACK*    track = static_cast<PCB_TRACK*>( board_item );
        const SEG&    s = seg->Seg();

        track->SetStart( VECTOR2I( s.A.x, s.A.y ) );
        track->SetEnd( VECTOR2I( s.B.x, s.B.y ) );
        track->SetWidth( seg->Width() );
        break;
    }

    case PNS::ITEM::VIA_T:
    {
        PCB_VIA*  via_board = static_cast<PCB_VIA*>( board_item );
        PNS::VIA* via = static_cast<PNS::VIA*>( aItem );

        via_board->SetPosition( VECTOR2I( via->Pos().x, via->Pos().y ) );
        via_board->SetWidth( PADSTACK::ALL_LAYERS, via->Diameter( 0 ) );
        via_board->SetDrill( via->Drill() );
        via_board->SetNet( static_cast<NETINFO_ITEM*>( via->Net() ) );
        via_board->SetViaType( via->ViaType() ); // MUST be before SetLayerPair()
        via_board->SetIsFree( via->IsFree() );
        via_board->SetLayerPair( GetBoardLayerFromPNSLayer( via->Layers().Start() ),
                                 GetBoardLayerFromPNSLayer( via->Layers().End() ) );
        break;
    }

    case PNS::ITEM::SOLID_T:
        break;  // pad drag -- not driven by routing, nothing to do

    default:
        break;
    }
}

void PNS_BRIDGE_IFACE::AddItem( PNS::ITEM* aItem )
{
    BOARD_CONNECTED_ITEM* boardItem = createBoardItem( aItem );

    if( boardItem )
    {
        aItem->SetParent( boardItem );
        boardItem->ClearFlags();
        m_pendingAdds.push_back( boardItem );
    }
}

void PNS_BRIDGE_IFACE::UpdateItem( PNS::ITEM* aItem )
{
    // Mutates the live BOARD_ITEM's geometry in place via setters that
    // don't require a COMMIT to take effect.
    modifyBoardItem( aItem );
}

void PNS_BRIDGE_IFACE::RemoveItem( PNS::ITEM* aItem )
{
    BOARD_ITEM* parent = aItem->Parent();

    if( aItem->OfKind( PNS::ITEM::SOLID_T ) && parent && parent->Type() == PCB_PAD_T )
    {
        // Pad movement (component dragging) isn't something we drive from
        // the routing API -- nothing to do.
        return;
    }

    if( parent )
        m_pendingRemoves.push_back( parent );
}

void PNS_BRIDGE_IFACE::Commit()
{
    BOARD* board = GetBoard();

    for( BOARD_ITEM* item : m_pendingRemoves )
        board->Remove( item, REMOVE_MODE::BULK );

    for( BOARD_ITEM* item : m_pendingAdds )
        board->Add( item, ADD_MODE::INSERT );

    m_pendingAdds.clear();
    m_pendingRemoves.clear();
}

// ---------------------------------------------------------------------
// PNS_BRIDGE
// ---------------------------------------------------------------------

PNS_BRIDGE::PNS_BRIDGE() = default;
PNS_BRIDGE::~PNS_BRIDGE() = default;

bool PNS_BRIDGE::LoadBoard( const std::string& aPath )
{
    PCB_IO_KICAD_SEXPR io;

    BOARD* raw = nullptr;

    try
    {
        raw = io.LoadBoard( wxString( aPath.c_str(), wxConvUTF8 ), nullptr );
    }
    catch( const std::exception& )
    {
        return false;
    }

    if( !raw )
        return false;

    m_board.reset( raw );
    m_board->BuildListOfNets();
    m_board->BuildConnectivity();

    m_iface = std::make_unique<PNS_BRIDGE_IFACE>();
    m_iface->SetBoard( m_board.get() );

    m_router = std::make_unique<PNS::ROUTER>();
    m_router->SetInterface( m_iface.get() );

    m_routingSettings = std::make_unique<PNS::ROUTING_SETTINGS>( nullptr, "" );
    m_router->LoadSettings( m_routingSettings.get() );

    m_router->ClearWorld();
    m_router->SyncWorld();

    m_candidateItems.clear();
    m_candidateIds.clear();

    return true;
}

bool PNS_BRIDGE::SaveBoard( const std::string& aPath )
{
    if( !m_board )
        return false;

    PCB_IO_KICAD_SEXPR io;

    try
    {
        io.SaveBoard( wxString( aPath.c_str(), wxConvUTF8 ), m_board.get() );
    }
    catch( const std::exception& )
    {
        return false;
    }

    return true;
}

std::vector<std::string> PNS_BRIDGE::NetNames() const
{
    std::vector<std::string> names;

    if( !m_board )
        return names;

    for( NETINFO_ITEM* net : m_board->GetNetInfo() )
        names.push_back( net->GetNetname().ToStdString() );

    return names;
}

std::vector<PNS_BRIDGE::Candidate> PNS_BRIDGE::QueryHoverItems( int aX, int aY, int aLayer,
                                                                 int aSlopRadius )
{
    std::vector<Candidate> out;

    if( !m_router )
        return out;

    PNS::ITEM_SET hits = m_router->QueryHoverItems( VECTOR2I( aX, aY ), aSlopRadius );

    for( PNS::ITEM* item : hits.CItems() )
    {
        if( aLayer >= 0 && !item->Layers().Overlaps( aLayer ) )
            continue;

        long long id;
        auto existing = m_candidateIds.find( item );

        if( existing != m_candidateIds.end() )
        {
            id = existing->second;
        }
        else
        {
            id = static_cast<long long>( m_candidateItems.size() );
            m_candidateItems.push_back( item );
            m_candidateIds.emplace( item, id );
        }

        Candidate c;
        c.id = id;
        VECTOR2I pos = item->Anchor( 0 );
        c.x = pos.x;
        c.y = pos.y;

        switch( item->Kind() )
        {
        case PNS::ITEM::SOLID_T:   c.kind = "pad";     break;
        case PNS::ITEM::VIA_T:     c.kind = "via";     break;
        case PNS::ITEM::SEGMENT_T: c.kind = "segment"; break;
        case PNS::ITEM::ARC_T:     c.kind = "arc";     break;
        default:                   c.kind = "other";   break;
        }

        out.push_back( c );
    }

    return out;
}

PNS::ITEM* PNS_BRIDGE::resolveItem( long long aItemId ) const
{
    if( aItemId < 0 || static_cast<size_t>( aItemId ) >= m_candidateItems.size() )
        return nullptr;

    return m_candidateItems[static_cast<size_t>( aItemId )];
}

bool PNS_BRIDGE::StartRoute( int aX, int aY, long long aItemId, int aLayer )
{
    if( !m_router )
        return false;

    if( !m_router->StartRouting( VECTOR2I( aX, aY ), resolveItem( aItemId ), aLayer ) )
        return false;

    // Re-apply length-tuning config to the placer StartRouting() just
    // created -- it didn't exist yet when SetTargetLength()/etc. were
    // called (see m_meanderSettings' comment in pns_bridge.h).
    if( PNS::MEANDER_PLACER_BASE* placer = asMeanderPlacer( m_router.get() ) )
        placer->UpdateSettings( m_meanderSettings );

    return true;
}

bool PNS_BRIDGE::Push( int aX, int aY, long long aItemId )
{
    if( !m_router )
        return false;

    return m_router->Move( VECTOR2I( aX, aY ), resolveItem( aItemId ) );
}

bool PNS_BRIDGE::Fix( int aX, int aY, long long aItemId, bool aForceFinish, bool aForceCommit )
{
    if( !m_router )
        return false;

    return m_router->FixRoute( VECTOR2I( aX, aY ), resolveItem( aItemId ), aForceFinish,
                                aForceCommit );
}

void PNS_BRIDGE::CommitRouting()
{
    if( m_router )
        m_router->CommitRouting();

    if( m_board )
        m_board->BuildConnectivity();
}

void PNS_BRIDGE::StopRouting()
{
    if( m_router )
        m_router->StopRouting();
}

void PNS_BRIDGE::Reset()
{
    if( !m_board || !m_router )
        return;

    // Drop PNS's own world model of the current routing first -- nothing
    // in it should reference a BOARD_ITEM after this call, so it's safe to
    // delete those items outright below rather than leaving anything that
    // could transiently dangle.
    m_router->ClearWorld();

    // Snapshot pointers before removing -- BOARD::Remove() mutates the
    // container BOARD::Tracks() returns a view over, so removing while
    // iterating it directly would invalidate the iteration.
    std::vector<BOARD_ITEM*> toRemove;
    for( PCB_TRACK* track : m_board->Tracks() )
        toRemove.push_back( track );

    for( BOARD_ITEM* item : toRemove )
    {
        // BOARD::Remove() only unlinks the item from the board's own
        // containers -- it does not free it (ownership transfers back to
        // the caller, same as PNS_BRIDGE_IFACE::Commit()'s handling of
        // m_pendingRemoves elsewhere in this file, which has the same
        // gap). Reset() is meant to be called every RL episode -- without
        // this delete, every track/via/arc ever routed away would leak
        // for the rest of the process's life.
        m_board->Remove( item, REMOVE_MODE::BULK );
        delete item;
    }

    m_board->BuildConnectivity();

    // Same SyncWorld() LoadBoard() does, but against the same BOARD
    // instance (footprint placement, any other board-level state the
    // caller set up survives) instead of reloading from disk.
    m_router->SyncWorld();

    m_candidateItems.clear();
    m_candidateIds.clear();
}

std::vector<PNS_BRIDGE::NetPad> PNS_BRIDGE::NetPads() const
{
    std::vector<NetPad> out;

    if( !m_board )
        return out;

    for( FOOTPRINT* fp : m_board->Footprints() )
    {
        for( PAD* pad : fp->Pads() )
        {
            NetPad np;
            np.net = pad->GetNetname().ToStdString();
            np.padName = fp->GetReference().ToStdString() + ":" + pad->GetNumber().ToStdString();
            VECTOR2I pos = pad->GetPosition();
            np.x = pos.x;
            np.y = pos.y;
            np.layer = -1;
            out.push_back( np );
        }
    }

    return out;
}

// Two-copper-layer simplification shared by pads and zones below: records
// only whether F_Cu/B_Cu are present in a layer set, not every inner layer.
// Correct for every board this codebase currently generates
// (pcbworld/data/generate_board.py is 2-layer only) -- revisit if/when a
// multi-inner-layer generator is added (see ROADMAP.md's generalization
// tests, which explicitly call out "different layer counts").
static void CopperLayerRange( const LSET& aLayers, int& aTop, int& aBottom )
{
    aTop = aLayers.Contains( F_Cu ) ? static_cast<int>( F_Cu ) : static_cast<int>( UNDEFINED_LAYER );
    aBottom = aLayers.Contains( B_Cu ) ? static_cast<int>( B_Cu ) : aTop;
}

PNS_BRIDGE::BoardGeometry PNS_BRIDGE::GetBoardGeometry() const
{
    BoardGeometry geom;

    if( !m_board )
        return geom;

    for( PCB_TRACK* track : m_board->Tracks() )
    {
        if( PCB_VIA* via = dynamic_cast<PCB_VIA*>( track ) )
        {
            ViaGeom v;
            VECTOR2I pos = via->GetPosition();
            v.x = pos.x;
            v.y = pos.y;
            v.diameter = via->GetWidth( PADSTACK::ALL_LAYERS );
            v.drill = via->GetDrillValue();
            v.layerTop = static_cast<int>( via->TopLayer() );
            v.layerBottom = static_cast<int>( via->BottomLayer() );
            v.net = via->GetNetname().ToStdString();
            geom.vias.push_back( v );
            continue;
        }

        TrackSegment seg;
        VECTOR2I start = track->GetStart();
        VECTOR2I end = track->GetEnd();
        seg.x1 = start.x;
        seg.y1 = start.y;
        seg.x2 = end.x;
        seg.y2 = end.y;
        seg.width = track->GetWidth();
        seg.layer = static_cast<int>( track->GetLayer() );
        seg.net = track->GetNetname().ToStdString();
        seg.isArc = dynamic_cast<PCB_ARC*>( track ) != nullptr;
        geom.tracks.push_back( seg );
    }

    for( FOOTPRINT* fp : m_board->Footprints() )
    {
        for( PAD* pad : fp->Pads() )
        {
            PadGeom p;
            VECTOR2I pos = pad->GetPosition();
            VECTOR2I size = pad->GetSize( PADSTACK::ALL_LAYERS );
            p.x = pos.x;
            p.y = pos.y;
            p.sizeX = size.x;
            p.sizeY = size.y;
            CopperLayerRange( pad->GetLayerSet(), p.layerTop, p.layerBottom );
            p.net = pad->GetNetname().ToStdString();
            p.padName = fp->GetReference().ToStdString() + ":" + pad->GetNumber().ToStdString();
            geom.pads.push_back( p );
        }

        BOX2I bbox = fp->GetBoundingBox();
        FootprintBBox b;
        b.x1 = bbox.GetLeft();
        b.y1 = bbox.GetTop();
        b.x2 = bbox.GetRight();
        b.y2 = bbox.GetBottom();
        b.reference = fp->GetReference().ToStdString();
        geom.courtyards.push_back( b );
    }

    for( ZONE* zone : m_board->Zones() )
    {
        if( zone->Outline() == nullptr || zone->Outline()->OutlineCount() == 0 )
            continue;

        ZoneGeom z;
        const SHAPE_LINE_CHAIN& chain = zone->Outline()->Outline( 0 );

        for( int i = 0; i < chain.PointCount(); ++i )
        {
            const VECTOR2I& pt = chain.CPoint( i );
            z.outline.emplace_back( pt.x, pt.y );
        }

        int top, bottom;
        CopperLayerRange( zone->GetLayerSet(), top, bottom );
        z.layer = top >= 0 ? top : static_cast<int>( zone->GetFirstLayer() );
        z.isKeepout = zone->GetIsRuleArea();
        z.net = zone->GetNetname().ToStdString();
        geom.zones.push_back( z );
    }

    for( BOARD_ITEM* item : m_board->Drawings() )
    {
        if( item->GetLayer() != Edge_Cuts )
            continue;

        PCB_SHAPE* shape = dynamic_cast<PCB_SHAPE*>( item );

        if( !shape )
            continue;

        EdgeShape e;
        e.width = shape->GetWidth();
        VECTOR2I start = shape->GetStart();
        VECTOR2I end = shape->GetEnd();
        e.x1 = start.x;
        e.y1 = start.y;
        e.x2 = end.x;
        e.y2 = end.y;

        switch( shape->GetShape() )
        {
        case SHAPE_T::RECTANGLE: e.shapeType = "rect";    break;
        case SHAPE_T::CIRCLE:  e.shapeType = "circle";  break;
        case SHAPE_T::ARC:     e.shapeType = "arc";     break;
        case SHAPE_T::SEGMENT: e.shapeType = "segment"; break;
        default:
        {
            // Polygon/bezier/etc: exact outline points aren't exported
            // (same simplification as ZoneGeom dropping holes) -- fall
            // back to the shape's own bounding box so the edge still
            // constrains a rasterized board-boundary channel, just not
            // exactly along a curved/irregular edge.
            e.shapeType = "polygon";
            BOX2I bbox = shape->GetBoundingBox();
            e.x1 = bbox.GetLeft();
            e.y1 = bbox.GetTop();
            e.x2 = bbox.GetRight();
            e.y2 = bbox.GetBottom();
            break;
        }
        }

        geom.boardEdge.push_back( e );
    }

    return geom;
}

void PNS_BRIDGE::SetMode( int aMode )
{
    if( m_router )
        m_router->SetMode( static_cast<PNS::ROUTER_MODE>( aMode ) );
}

void PNS_BRIDGE::SetTrackWidth( int aWidthNm )
{
    if( m_router )
        m_router->Sizes().SetTrackWidth( aWidthNm );
}

void PNS_BRIDGE::SetViaDiameter( int aDiameterNm )
{
    if( m_router )
        m_router->Sizes().SetViaDiameter( aDiameterNm );
}

void PNS_BRIDGE::SetViaDrill( int aDrillNm )
{
    if( m_router )
        m_router->Sizes().SetViaDrill( aDrillNm );
}

void PNS_BRIDGE::SetDiffPairGap( int aGapNm )
{
    if( m_router )
        m_router->Sizes().SetDiffPairGap( aGapNm );
}

void PNS_BRIDGE::SetDiffPairViaGap( int aGapNm )
{
    if( m_router )
        m_router->Sizes().SetDiffPairViaGap( aGapNm );
}

void PNS_BRIDGE::SetDiffPairWidth( int aWidthNm )
{
    if( m_router )
        m_router->Sizes().SetDiffPairWidth( aWidthNm );
}

void PNS_BRIDGE::SetTargetLength( long long aLengthNm )
{
    m_meanderSettings.SetTargetLength( aLengthNm );

    if( PNS::MEANDER_PLACER_BASE* placer = asMeanderPlacer( m_router.get() ) )
        placer->UpdateSettings( m_meanderSettings );
}

void PNS_BRIDGE::SetMeanderMaxAmplitude( int aAmpNm )
{
    m_meanderSettings.m_maxAmplitude = aAmpNm;

    if( PNS::MEANDER_PLACER_BASE* placer = asMeanderPlacer( m_router.get() ) )
        placer->UpdateSettings( m_meanderSettings );
}

void PNS_BRIDGE::SetMeanderSpacing( int aSpacingNm )
{
    m_meanderSettings.m_spacing = aSpacingNm;

    if( PNS::MEANDER_PLACER_BASE* placer = asMeanderPlacer( m_router.get() ) )
        placer->UpdateSettings( m_meanderSettings );
}

void PNS_BRIDGE::ToggleViaPlacement()

{
    if( m_router )
        m_router->ToggleViaPlacement();
}

bool PNS_BRIDGE::SwitchLayer( int aLayer )
{
    if( !m_router )
        return false;

    return m_router->SwitchLayer( aLayer );
}

std::vector<PNS_BRIDGE::DRCViolation> PNS_BRIDGE::RunDRC()
{
    std::vector<DRCViolation> out;

    if( !m_board )
        return out;

    // Routing since LoadBoard()/CommitRouting() changes net connectivity;
    // several DRC tests (unconnected items, shorts) read it directly.
    m_board->BuildConnectivity();

    BOARD_DESIGN_SETTINGS& designSettings = m_board->GetDesignSettings();

    DRC_ENGINE engine( m_board.get(), &designSettings );

    // DRC_VIOLATION_HANDLER's real signature (drc_engine.h) -- confirmed by
    // the actual Colab compiler error, not guessed a second time: 2nd param
    // is a const VECTOR2I&, and the 4th is an optional marker-creation
    // callback (std::function<void(PCB_MARKER*)>*), not a DRC_CONSTRAINT*
    // as first assumed. We don't need to create markers ourselves -- just
    // collecting violations into `out` -- so it's unused here.
    engine.SetViolationHandler(
        [&]( const std::shared_ptr<DRC_ITEM>& aItem, const VECTOR2I& aPos, int /* aLayer */,
             std::function<void( PCB_MARKER* )>* /* aCreateMarker */ )
        {
            DRCViolation v;
            v.errorCode = aItem->GetErrorCode();
            v.message = aItem->GetErrorMessage().ToStdString();
            v.x = aPos.x;
            v.y = aPos.y;

            switch( designSettings.GetSeverity( v.errorCode ) )
            {
            case RPT_SEVERITY_ERROR:     v.severity = "error";     break;
            case RPT_SEVERITY_WARNING:   v.severity = "warning";   break;
            case RPT_SEVERITY_EXCLUSION: v.severity = "exclusion"; break;
            default:                     v.severity = "ignore";    break;
            }

            out.push_back( v );
        } );

    // Empty wxFileName -- no external rules file, so InitEngine() falls
    // back to KiCad's built-in default rule set (same as a board with no
    // custom DRC rules configured in the GUI).
    engine.InitEngine( wxFileName() );
    engine.RunTests( EDA_UNITS::MM, true, false );

    engine.ClearViolationHandler();

    return out;
}

}  // namespace pcbworld
