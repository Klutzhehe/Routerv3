// Headless bridge exposing KiCad's PNS::ROUTER to Python.
//
// Architecture and citations: see docs/engine_access.md. Summary:
//   - PNS::ROUTER itself has no GUI dependency (pcbnew/router/pns_router.h).
//   - We drive it the same way qa/tools/pns/pns_log_player.cpp does:
//     SetInterface -> SyncWorld -> StartRouting -> Move* -> FixRoute -> CommitRouting.
//   - The one piece that differs from any existing KiCad code path: turning
//     routed PNS items back into real BOARD_ITEMs without a BOARD_COMMIT
//     (which needs a PCB_TOOL_BASE/TOOL_MANAGER we don't have headlessly).
//     PNS_BRIDGE_IFACE below reimplements just the Add/Remove/Update subset
//     of what BOARD_COMMIT::Push does (pcbnew/board_commit.cpp), calling
//     BOARD::Add()/BOARD::Remove() directly. This part is new and unverified
//     until it compiles and links against a real KiCad build.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <router/pns_kicad_iface.h>
#include <router/pns_router.h>

class BOARD;

namespace pcbworld
{

// Reimplements the board-mutation subset of PNS_KICAD_IFACE::Commit()
// without going through BOARD_COMMIT (which requires a live GUI tool).
class PNS_BRIDGE_IFACE : public PNS_KICAD_IFACE
{
public:
    PNS_BRIDGE_IFACE() = default;
    ~PNS_BRIDGE_IFACE() override = default;

    void AddItem( PNS::ITEM* aItem ) override;
    void UpdateItem( PNS::ITEM* aItem ) override;
    void RemoveItem( PNS::ITEM* aItem ) override;
    void Commit() override;

    // PNS_KICAD_IFACE::GetUnits() dereferences a host tool we don't have.
    EDA_UNITS GetUnits() const override { return EDA_UNITS::MM; }

private:
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
        long long id;   // opaque handle, valid until the next router call
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

    // PNS::ITEM pointers handed to Python as opaque ids; valid only within
    // the node they were queried from (router world is rebuilt on SyncWorld).
    mutable std::vector<PNS::ITEM*> m_lastCandidates;
};

}  // namespace pcbworld
