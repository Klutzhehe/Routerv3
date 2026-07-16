// Headless bridge exposing KiCad's PNS::ROUTER to Python.
//
// Architecture and citations: see docs/engine_access.md. Summary:
//   - PNS::ROUTER itself has no GUI dependency (pcbnew/router/pns_router.h).
//   - We drive it the same way qa/tools/pns/pns_log_player.cpp does:
//     SetInterface -> SyncWorld -> StartRouting -> Move* -> FixRoute -> CommitRouting.
//   - The one piece that differs from any existing KiCad code path: turning
//     routed PNS items back into real BOARD_ITEMs without a BOARD_COMMIT
//     (which needs a PCB_TOOL_BASE/TOOL_MANAGER we don't have headlessly).
//
// PNS_BRIDGE_IFACE derives from PNS_KICAD_IFACE_BASE, *not* the real
// PNS_KICAD_IFACE -- deliberately. PNS_KICAD_IFACE_BASE's Commit()/AddItem()
// are harmless no-ops (pns_kicad_iface.h), but the real PNS_KICAD_IFACE
// (used by the GUI) implements them via BOARD_COMMIT, and BOARD_COMMIT::Push
// (pcbnew/board_commit.cpp) looks up ZONE_FILLER_TOOL/PCB_SELECTION_TOOL
// through TOOL_MANAGER -- GUI-only classes with no headless equivalent.
// Since createBoardItem()/modifyBoardItem() (the PNS::ITEM -> BOARD_ITEM
// conversion we actually need) are only defined on the real PNS_KICAD_IFACE,
// and object-file-granularity linking means pulling in *any* symbol from
// that class's translation unit pulls in all of it (including the BOARD_COMMIT
// call), we reimplement those two conversions ourselves below instead of
// inheriting them -- ported directly from pns_kicad_iface.cpp's
// createBoardItem()/modifyBoardItem(), minus the drag/footprint-offset
// bookkeeping (m_fpOffsets/m_itemGroups) which only matters for component
// dragging, a tool interaction we never drive.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <router/pns_kicad_iface.h>
#include <router/pns_router.h>

class BOARD;
class BOARD_CONNECTED_ITEM;

namespace pcbworld
{

class PNS_BRIDGE_IFACE : public PNS_KICAD_IFACE_BASE
{
public:
    PNS_BRIDGE_IFACE() = default;
    ~PNS_BRIDGE_IFACE() override = default;

    void AddItem( PNS::ITEM* aItem ) override;
    void UpdateItem( PNS::ITEM* aItem ) override;
    void RemoveItem( PNS::ITEM* aItem ) override;
    void Commit() override;

private:
    BOARD_CONNECTED_ITEM* createBoardItem( PNS::ITEM* aItem );
    void modifyBoardItem( PNS::ITEM* aItem );

    std::vector<BOARD_ITEM*> m_pendingAdds;
    std::vector<BOARD_ITEM*> m_pendingRemoves;
};

// One board + one router session. Mirrors PNS_LOG_PLAYER::createRouter/
// ReplayLog (qa/tools/pns/pns_log_player.cpp) but driven by discrete Python
// calls instead of a pre-recorded log file.
class PNS_BRIDGE
{
public:
    PNS_BRIDGE();
    ~PNS_BRIDGE();

    bool LoadBoard( const std::string& aPath );
    bool SaveBoard( const std::string& aPath );

    std::vector<std::string> NetNames() const;

    // Candidate items/points near (x, y) on aLayer, for the agent to pick a
    // start/end from -- mirrors PNS::TOOL_BASE::pickSingleItem's use of
    // ROUTER::QueryHoverItems (pcbnew/router/pns_tool_base.cpp:143).
    struct Candidate
    {
        long long id;   // stable handle, valid until the next LoadBoard()
        int x, y;
        std::string kind;   // "pad" | "segment" | "via" | "arc"
        std::string net;
    };
    std::vector<Candidate> QueryHoverItems( int aX, int aY, int aLayer, int aSlopRadius );

    bool StartRoute( int aX, int aY, long long aItemId, int aLayer );
    bool Push( int aX, int aY, long long aItemId );
    bool Fix( int aX, int aY, long long aItemId, bool aForceFinish, bool aForceCommit );
    void CommitRouting();
    void StopRouting();

    void SetMode( int aMode );          // PNS::ROUTER_MODE
    void SetTrackWidth( int aWidthNm );
    void SetViaDiameter( int aDiameterNm );
    void SetViaDrill( int aDrillNm );
    void ToggleViaPlacement();
    bool SwitchLayer( int aLayer );

private:
    PNS::ITEM* resolveItem( long long aItemId ) const;

    std::unique_ptr<BOARD> m_board;
    std::unique_ptr<PNS_BRIDGE_IFACE> m_iface;
    std::unique_ptr<PNS::ROUTER> m_router;

    // PNS::ITEM pointers handed to Python as opaque, stable ids. Previously
    // this was cleared on every QueryHoverItems() call, which silently
    // invalidated ids from an earlier call the moment a second query ran --
    // e.g. querying near pad A then pad B, then trying to use pad A's id,
    // would resolve to pad B's item while still using pad A's coordinates.
    // That mismatch (an item pointer far from the given "start here" point)
    // crashed the router. Now append-only and deduplicated by item pointer,
    // so an id stays valid across any number of queries; only LoadBoard()
    // (a brand new PNS world) invalidates them.
    mutable std::vector<PNS::ITEM*> m_candidateItems;
    mutable std::unordered_map<PNS::ITEM*, long long> m_candidateIds;
};

}  // namespace pcbworld
