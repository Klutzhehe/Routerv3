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
#include <router/pns_meander.h>
#include <router/pns_router.h>
#include <router/pns_routing_settings.h>

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

    // Removes all routed copper (tracks/vias/arcs -- BOARD::Tracks()
    // includes all three, PCB_VIA/PCB_ARC derive from PCB_TRACK) while
    // leaving footprint placement untouched, then rebuilds the PNS world
    // against the now-bare board. Lets an RL episode retry routing (or
    // move on to a differently-ordered net sequence) without a disk
    // round-trip through LoadBoard(). Roadmap item: "reset() needs to walk
    // the board's tracks/vias and remove them while preserving footprint
    // placement."
    void Reset();

    // Reset() for a single net: removes only the copper belonging to
    // aNet, leaving every other routed net intact. The action an agent
    // needs when a later net cannot reach its pad because an earlier
    // net's copper is sitting on it -- which is not hypothetical here.
    // scripts/measure_layer_hop_rescue.py's data shows every failing net
    // failing IDENTICALLY across all six same-layer approaches (straight
    // line, 5-point polyline, four perpendicular detours), i.e. the
    // contention is at the target pad itself, and no detour can route
    // around that because every attempt still ends at the same (x, y).
    // Tearing out the offending net can.
    //
    // Returns the number of BOARD_ITEMs removed (0 if aNet is unknown or
    // was never routed) so a caller can tell "ripped up nothing" from
    // "ripped up something" without a second GetBoardGeometry() call.
    //
    // Two caveats, both inherited from Reset()'s mechanism:
    //  - Invalidates every QueryHoverItems() candidate id, because it
    //    rebuilds the PNS world (ClearWorld/SyncWorld). ROADMAP.md's
    //    constraint 5 says ids are stable "within a loaded board" and
    //    only LoadBoard() invalidates them -- this widens that. Re-query
    //    after any RipUp().
    //  - Costs a full world resync, not an incremental node edit. Fine
    //    per-rip-up-action (which is a deliberate, occasional macro
    //    move); do not call it in a per-push inner loop.
    //
    // Do not call while a route is in progress -- StopRouting() first.
    // The active placer holds a NODE branched off the world this
    // invalidates.
    int RipUp( const std::string& aNet );

    struct NetPad
    {
        std::string net;
        std::string padName;   // "<footprint reference>:<pad number>"
        int x, y;
        int layer;   // always -1 (pad may span multiple copper layers) --
                      // matches query_hover_items's "any layer" convention
    };

    // Enumerates every pad on the board. The only other way to find a pad
    // is QueryHoverItems() at a specific (x, y), which requires already
    // knowing where every pad is from some external source -- fine for a
    // hand-written toy-board test, not workable for an agent or generator
    // that needs to route an arbitrary board's nets programmatically.
    // Group by `net` in Python to get per-net pad lists for sequencing.
    std::vector<NetPad> NetPads() const;

    struct DRCViolation
    {
        int errorCode;            // DRCE_* (drc_rule.h) -- stable numeric id
        std::string message;      // human-readable, from DRC_ITEM/RC_ITEM
        std::string severity;     // "error" | "warning" | "exclusion" | "ignore"
        int x, y;                 // marker position
    };

    // Runs KiCad's real DRC_ENGINE against the currently loaded board, using
    // whatever BOARD_DESIGN_SETTINGS came in with the loaded .kicad_pcb file
    // and KiCad's built-in default rule set (no external rules file passed
    // to InitEngine). Mirrors the harness pattern KiCad's own headless DRC
    // regression tests use (qa/tests/pcbnew/drc/test_drc_copper_conn.cpp
    // etc.): construct a DRC_ENGINE, install a violation-handler callback,
    // InitEngine(), RunTests(). No dialog/GUI involved -- DRC_ENGINE itself
    // has no GUI dependency, same as PNS::ROUTER.
    std::vector<DRCViolation> RunDRC();

    // Bulk geometry export -- the one call an agent/state-builder needs to
    // rasterize the board into the spatial channels (copper, keepouts,
    // courtyards, vias, edge-distance field) without a per-grid-cell
    // QueryHoverItems() call, which would mean one router call per pixel.
    // Everything here is read directly off the loaded BOARD, so it works
    // whether or not a route is currently in progress.
    struct TrackSegment
    {
        int x1, y1, x2, y2;
        int width;
        int layer;          // PCB_LAYER_ID
        std::string net;
        bool isArc;          // true: (x1,y1)-(x2,y2) is an arc's start/end
                              // chord, not a straight segment -- exact arc
                              // shape (mid point, radius) isn't exported;
                              // callers rasterizing this need to treat arcs
                              // as a coarse straight-line approximation, or
                              // this struct needs a mid-point field added
                              // later if that's not accurate enough.
    };

    struct ViaGeom
    {
        int x, y;
        int diameter, drill;
        int layerTop, layerBottom;
        std::string net;
    };

    struct PadGeom
    {
        int x, y;
        int sizeX, sizeY;   // pad footprint size, not shape-accurate (a
                             // circle/oval/roundrect pad is exported as its
                             // bounding box) -- fine for a rasterized
                             // obstacle channel, not for exact DRC-grade
                             // geometry (use RunDRC() for that).
        int layerTop, layerBottom;
        std::string net;
        std::string padName;   // "<footprint reference>:<pad number>"
    };

    struct ZoneGeom
    {
        std::vector<std::pair<int, int>> outline;  // main outline only
                                                     // (Outline(0)) -- holes
                                                     // and additional
                                                     // outlines in the same
                                                     // zone are dropped.
        int layer;
        bool isKeepout;      // ZONE::GetIsRuleArea()
        std::string net;     // empty for a pure keepout/rule-area zone
    };

    // Coarse per-footprint bounding box, standing in for a real courtyard
    // polygon. FOOTPRINT's actual courtyard cache API
    // (GetCourtyard(PCB_LAYER_ID)) is version-sensitive enough that getting
    // it wrong silently produces an empty/stale polygon rather than a
    // compile error -- a plain bounding box is worse-fidelity but can't be
    // subtly wrong the same way. Revisit if the courtyard channel needs
    // tighter-than-bbox fidelity.
    struct FootprintBBox
    {
        int x1, y1, x2, y2;
        std::string reference;
    };

    struct EdgeShape
    {
        std::string shapeType;   // "segment" | "rect" | "circle" | "arc" | "polygon"
        int x1, y1, x2, y2;      // segment/rect: two endpoints/corners.
                                 // circle: x1,y1 = center, x2,y2 = a point
                                 // on the circumference (radius = distance).
                                 // polygon: bounding box only -- exact
                                 // outline points aren't exported, same
                                 // simplification as ZoneGeom's holes.
        int width;
    };

    struct BoardGeometry
    {
        std::vector<TrackSegment> tracks;
        std::vector<ViaGeom> vias;
        std::vector<PadGeom> pads;
        std::vector<ZoneGeom> zones;
        std::vector<FootprintBBox> courtyards;
        std::vector<EdgeShape> boardEdge;   // Edge_Cuts layer shapes
    };

    BoardGeometry GetBoardGeometry() const;

    // -- In-progress head state ------------------------------------------
    //
    // Everything above reads *committed* board items. Nothing exposed the
    // route currently being placed: the "head" PNS carries between
    // StartRoute() and Fix(). That gap is the reason RL reward shaping has
    // no per-step signal to work with. Push() is ROUTER::Move(), and its
    // bool was measured accepting 72/72 net-attempts across three Colab
    // runs (scripts/measure_waypoint_fidelity.py) while Fix() then
    // rejected ~67% of those same routes -- so a leg is ~10 steps carrying
    // one terminal bit, with no credit assignment across the waypoints
    // that produced it. Nothing learns from that.
    //
    // Reading the head back after each Push() gives the env the two things
    // it actually needs per step: where the router put the head versus
    // where it was asked to (deviation), and whether that head is legal
    // (collision). Both are dense, both are free of a DRC run.
    //
    // NOTE (deliberate asymmetry, matching this file's convention for new
    // C++): GetHeadGeometry() is built from accessors this codebase has
    // effectively already exercised -- Placer() is used by
    // asMeanderPlacer() above, and LINE::CLine()/Width()/Layer() are the
    // same shape-reading pattern GetBoardGeometry() uses on BOARD items.
    // HeadCollides() is the riskier of the two: NODE::CheckColliding()'s
    // exact signature and return type (OPT_OBSTACLE vs bool) are written
    // from the remembered API, not confirmed against a checkout. If the
    // Colab build fails, expect it to fail there -- and note that deleting
    // HeadCollides() plus its binding leaves GetHeadGeometry(), which is
    // the load-bearing half, still standing.
    struct HeadSegment
    {
        int x1, y1, x2, y2;
        int width;
        int layer;   // PCB_LAYER_ID
    };

    struct HeadVia
    {
        int x, y;
        int layerTop, layerBottom;
        // No diameter/drill: PNS::VIA's padstack-aware accessors are
        // exactly the kind of version-sensitive surface that costs a
        // Colab round to get wrong (compare PCB_VIA::GetWidth(
        // PADSTACK::ALL_LAYERS ) in GetBoardGeometry()), and an RL env
        // needs to know a via EXISTS and where -- not how wide it is,
        // which it configured itself via SetViaDiameter().
    };

    struct HeadGeometry
    {
        bool active;                        // false if no route in progress
        std::vector<HeadSegment> segments;  // the in-progress polyline
        std::vector<HeadVia> vias;
        int endX, endY;                     // placer's own CurrentEnd()
        int layer;                          // placer's CurrentLayer()
        double length;                      // summed segment length, nm
    };

    // Snapshot of the route being placed right now. Safe to call with no
    // route in progress -- returns active=false rather than failing, so an
    // env can call it unconditionally after every Push().
    HeadGeometry GetHeadGeometry() const;

    // Does the current head collide with anything in the placer's world?
    //
    // This is the per-step legality signal RM_MarkObstacles was supposed
    // to provide through Push()'s return value and measurably does not.
    // Asks the placer's own NODE directly instead of trusting Move()'s
    // bool.
    //
    // Returns false when no route is in progress. A false from THAT case
    // and a false meaning "head is clean" are deliberately not
    // distinguished: callers should gate on GetHeadGeometry().active,
    // which they need anyway.
    bool HeadCollides() const;

    // -- Design rules ------------------------------------------------------
    //
    // The board's own sizing rules, so a caller stops guessing them.
    //
    // Every env and script in this repo currently hardcodes
    // via_diameter=600000 / via_drill=300000 (pcb_route_env.py,
    // diff_pair_route_env.py, measure_layer_hop_rescue.py) and nothing has
    // ever checked those against the loaded board. They are not connected:
    // SetViaDiameter() writes ROUTER::Sizes(), which is what PNS validates
    // a via against, while RunDRC() judges the committed result against
    // BOARD_DESIGN_SETTINGS, which came from the .kicad_pcb file. And
    // pcbworld/data/generate_board.py sets no design rules at all, so its
    // boards carry whatever a bare pcbnew.BOARD() defaults to.
    //
    // Two failure modes follow, and both have already cost real Colab
    // rounds or are waiting to:
    //  - Sizes() legal, BOARD_DESIGN_SETTINGS stricter -> PNS happily
    //    places the via, RunDRC() then reports a violation the agent gets
    //    penalized for and cannot see the cause of.
    //  - Via size never configured, or below the board's minimum -> no
    //    legal via geometry exists anywhere, so SwitchLayer() fails
    //    uniformly, at every position, for reasons that look positional
    //    but are not. That is precisely the 15/15 rejection
    //    scripts/measure_layer_hop_rescue.py chased across two rounds
    //    (commits 381e1a7, dc9164e).
    //
    // An RL agent placing vias must have them legal by construction, not
    // by a constant that happens to match. Read this at env reset() and
    // feed it straight into SetTrackWidth/SetViaDiameter/SetViaDrill.
    //
    // UNVERIFIED, in the usual place: the default-netclass path
    // (GetDesignSettings().m_NetSettings->GetDefaultNetclass()) is already
    // used and Colab-proven at pns_bridge.cpp's makeNetInfo path, so
    // trackWidth/viaDiameter/viaDrill/clearance are low risk. The
    // BOARD_DESIGN_SETTINGS minimums below are raw public member names
    // (m_TrackMinWidth, m_ViasMinSize, m_MinThroughDrill, m_HoleToHoleMin,
    // m_MinClearance) written from the remembered KiCad 9 API. If the
    // build fails here, they are why -- and dropping the four min* fields
    // leaves the netclass half, which is the half envs actually need to
    // configure from, still working.
    struct DesignRules
    {
        int trackWidth;      // default netclass
        int viaDiameter;
        int viaDrill;
        int clearance;
        int minTrackWidth;   // board-wide hard minimums
        int minViaDiameter;
        int minViaDrill;
        int minHoleToHole;
    };

    DesignRules GetDesignRules() const;

    void SetMode( int aMode );          // PNS::ROUTER_MODE

    // PNS::ROUTING_SETTINGS's own collision-response mode (PNS::PNS_MODE:
    // RM_MarkObstacles / RM_Shove / RM_Walkaround, pns_routing_settings.h) --
    // distinct from SetMode()'s PNS::ROUTER_MODE above, and NOT previously
    // exposed to Python: LoadBoard() built m_routingSettings with whatever
    // ROUTING_SETTINGS's own constructor defaults to, which for KiCad's
    // interactive router is RM_Shove. Shove mode means Push() doesn't just
    // report a collision -- PNS moves the *other* conflicting trace out of
    // the way to make the agent's segment fit, i.e. it silently does a
    // piece of the routing decision for the caller. For RL training this
    // has to be RM_MarkObstacles (report-only, no auto-adjustment) so
    // Push() is a pure geometry/DRC validator. LoadBoard() now defaults to
    // RM_MarkObstacles; call this to opt back into Shove/Walkaround for a
    // non-RL baseline run.
    // UNVERIFIED: written against pns_routing_settings.h's remembered API
    // (ROUTING_SETTINGS::SetMode(PNS_MODE), enum values RM_MarkObstacles=0/
    // RM_Shove/RM_Walkaround) with no KiCad source checkout available
    // locally to confirm against -- same "expect iteration from real Colab
    // compiler output" caveat as everything else new in this file.
    void SetCollisionMode( int aMode );  // PNS::PNS_MODE
    void SetTrackWidth( int aWidthNm );
    void SetViaDiameter( int aDiameterNm );
    void SetViaDrill( int aDrillNm );
    void SetDiffPairGap( int aGapNm );
    void SetDiffPairViaGap( int aGapNm );
    void SetDiffPairWidth( int aWidthNm );
    void SetTargetLength( long long aLengthNm );
    void SetMeanderMaxAmplitude( int aAmpNm );
    void SetMeanderSpacing( int aSpacingNm );
    void ToggleViaPlacement();
    bool SwitchLayer( int aLayer );


private:
    PNS::ITEM* resolveItem( long long aItemId ) const;

    std::unique_ptr<BOARD> m_board;
    std::unique_ptr<PNS_BRIDGE_IFACE> m_iface;
    std::unique_ptr<PNS::ROUTER> m_router;

    // ROUTER::m_settings starts as nullptr (pns_router.cpp ROUTER::ROUTER())
    // and is only ever assigned via LoadSettings() -- there's no built-in
    // default. Without this, ROUTER::Settings() (a bare *m_settings
    // dereference) null-derefs and crashes the whole process the moment
    // StartRouting() calls isStartingPointRoutable(), which reads
    // Settings().AllowDRCViolations() as its very first line. Same
    // nullptr-parent/empty-path construction qa/tools/pns/pns_log_player.cpp
    // uses.
    std::unique_ptr<PNS::ROUTING_SETTINGS> m_routingSettings;

    // Target length / meander shape config, applied to the active
    // PNS::MEANDER_PLACER_BASE when one exists (tuning modes only -- created
    // by StartRouting(), not available beforehand) and re-applied every
    // StartRoute() so settings configured before starting a tuning route
    // (the normal call order: SetMode/SetTargetLength/... then StartRoute)
    // actually take effect. PNS::ROUTER has no meander-settings slot of its
    // own -- unlike Sizes(), which lives on ROUTER itself, meander config is
    // owned per-placer (pns_meander_placer_base.h).
    PNS::MEANDER_SETTINGS m_meanderSettings;

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
