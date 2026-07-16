#include "pns_bridge.h"

#include <board.h>
#include <board_connected_item.h>
#include <board_item_container.h>
#include <pad.h>
#include <pcb_track.h>

#include <pcb_io/kicad_sexpr/pcb_io_kicad_sexpr.h>

#include <router/pns_item.h>
#include <router/pns_itemset.h>
#include <router/pns_node.h>
#include <router/pns_solid.h>

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
    // modifyBoardItem() mutates the live BOARD_ITEM's geometry in place
    // (SetStart/SetEnd/SetWidth/...) via setters that don't require a
    // COMMIT to take effect -- COMMIT::Modify() only exists for undo
    // bookkeeping, which we don't need. See pns_kicad_iface.cpp
    // modifyBoardItem() for what this covers per PNS::ITEM kind.
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

    m_lastCandidates.clear();

    for( PNS::ITEM* item : hits.CItems() )
    {
        if( aLayer >= 0 && !item->Layers().Overlaps( aLayer ) )
            continue;

        m_lastCandidates.push_back( item );

        Candidate c;
        c.id = static_cast<long long>( m_lastCandidates.size() - 1 );
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
    if( aItemId < 0 || static_cast<size_t>( aItemId ) >= m_lastCandidates.size() )
        return nullptr;

    return m_lastCandidates[static_cast<size_t>( aItemId )];
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
