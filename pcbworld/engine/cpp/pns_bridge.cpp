#include "pns_bridge.h"

#include <board.h>
#include <board_connected_item.h>
#include <board_item_container.h>
#include <board_design_settings.h>
#include <project/net_settings.h>
#include <netinfo.h>
#include <pad.h>
#include <padstack.h>
#include <pcb_track.h>

#include <pcb_io/kicad_sexpr/pcb_io_kicad_sexpr.h>

#include <router/pns_arc.h>
#include <router/pns_item.h>
#include <router/pns_itemset.h>
#include <router/pns_node.h>
#include <router/pns_segment.h>
#include <router/pns_solid.h>
#include <router/pns_via.h>

namespace pcbworld
{

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

    return m_router->StartRouting( VECTOR2I( aX, aY ), resolveItem( aItemId ), aLayer );
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

}  // namespace pcbworld
