// Adapted from KiCad's own qa/qa_utils/mocks.cpp (same license as the rest
// of KiCad: GPL-2.0-or-later, see AUTHORS.txt in the KiCad source tree).
//
// This is KiCad's own, already-proven answer to the exact problem this
// bridge has: qa_pns_regressions / pns_debug_tool (qa/tools/pns/) link
// pnsrouter + pcbcommon + connectivity + gal + common headlessly, same as
// us, and mocks.cpp is what makes that link succeed as a real executable
// (which requires ALL symbols resolved, a strictly stronger check than our
// .so build, which only fails at import/dlopen time -- see
// docs/engine_access.md for the full trail of individually-discovered
// missing symbols that led here). Rather than keep discovering these one
// at a time, we adopt the proven file instead of a narrower hand-rolled
// version (previously gui_tool_stubs.cpp, now superseded by this).
//
// Every definition below is either a trivial no-op or returns a
// default/empty value. That's correct for us for the same reason it's
// correct for KiCad's own QA tools: none of this GUI-tool-framework code
// (PCB_SELECTION_TOOL, ZONE_FILLER_TOOL, PCB_TOOL_BASE, dialogs) is ever
// actually invoked by anything this bridge does -- we never construct a
// TOOL_MANAGER, so nothing ever calls into these. They exist purely to
// satisfy the linker.
//
// LTO note: pcbworld/engine/cpp/CMakeLists.txt disables
// INTERPROCEDURAL_OPTIMIZATION for this target specifically because an
// earlier, narrower version of this file got silently stripped by LTO
// (its dead-code analysis had no visibility into these symbols being
// needed by an already-built, separate static library). That fix covers
// this file too.

#include <board.h>
#include <fp_lib_table.h>
#include <footprint.h>
#include <router/router_tool.h>
#include <tools/pcb_actions.h>
#include <tools/pcb_selection_tool.h>
#include <tools/zone_filler_tool.h>
#include <zone_filler.h>

FP_LIB_TABLE GFootprintTable;


void ROUTER_TOOL::NeighboringSegmentFilter( const VECTOR2I&, GENERAL_COLLECTOR&, PCB_SELECTION_TOOL* )
{
}


/**
 * Private implementation of firewalled private data
 */
class PCB_SELECTION_TOOL::PRIV
{
public:
};


PCB_SELECTION_TOOL::PCB_SELECTION_TOOL() :
        SELECTION_TOOL( "pcbnew.InteractiveSelection" ),
        m_frame( NULL ),
        m_enteredGroup( NULL ),
        m_nonModifiedCursor( KICURSOR::ARROW ),
        m_priv( nullptr )
{
}


PCB_SELECTION_TOOL::~PCB_SELECTION_TOOL()
{
}


bool PCB_SELECTION_TOOL::Init()
{
    return true;
}


void PCB_SELECTION_TOOL::Reset( RESET_REASON aReason )
{
}


int PCB_SELECTION_TOOL::Main( const TOOL_EVENT& aEvent )
{
    return 0;
}


void PCB_SELECTION_TOOL::EnterGroup()
{
}


void PCB_SELECTION_TOOL::ExitGroup( bool aSelectGroup )
{
}


PCB_SELECTION& PCB_SELECTION_TOOL::GetSelection()
{
    return m_selection;
}


PCB_SELECTION& PCB_SELECTION_TOOL::RequestSelection( CLIENT_SELECTION_FILTER aClientFilter,
                                                 bool aConfirmLockedItems )
{
    return m_selection;
}


const GENERAL_COLLECTORS_GUIDE PCB_SELECTION_TOOL::getCollectorsGuide() const
{
    return GENERAL_COLLECTORS_GUIDE( LSET(), PCB_LAYER_ID::UNDEFINED_LAYER, nullptr );
}


bool PCB_SELECTION_TOOL::selectPoint( const VECTOR2I& aWhere, bool aOnDrag,
                                      bool* aSelectionCancelledFlag,
                                      CLIENT_SELECTION_FILTER aClientFilter )
{
    return false;
}


bool PCB_SELECTION_TOOL::selectCursor( bool aForceSelect, CLIENT_SELECTION_FILTER aClientFilter )
{
    return false;
}


bool PCB_SELECTION_TOOL::selectMultiple()
{
    return false;
}


int PCB_SELECTION_TOOL::CursorSelection( const TOOL_EVENT& aEvent )
{
    return 0;
}


int PCB_SELECTION_TOOL::ClearSelection( const TOOL_EVENT& aEvent )
{
    return 0;
}


int PCB_SELECTION_TOOL::SelectAll( const TOOL_EVENT& aEvent )
{
    return 0;
}


int PCB_SELECTION_TOOL::expandConnection( const TOOL_EVENT& aEvent )
{
    return 0;
}


void PCB_SELECTION_TOOL::selectAllConnectedTracks(
        const std::vector<BOARD_CONNECTED_ITEM*>& aStartItems, STOP_CONDITION aStopCondition )
{
}


void PCB_SELECTION_TOOL::SelectAllItemsOnNet( int aNetCode, bool aSelect )
{
}


int PCB_SELECTION_TOOL::selectNet( const TOOL_EVENT& aEvent )
{
    return 0;
}


void PCB_SELECTION_TOOL::selectAllItemsOnSheet( wxString& aSheetPath )
{
}


void PCB_SELECTION_TOOL::zoomFitSelection()
{
}


int PCB_SELECTION_TOOL::selectSheetContents( const TOOL_EVENT& aEvent )
{
    return 0;
}


int PCB_SELECTION_TOOL::selectSameSheet( const TOOL_EVENT& aEvent )
{
    return 0;
}


bool PCB_SELECTION_TOOL::ctrlClickHighlights()
{
    return false;
}


int PCB_SELECTION_TOOL::filterSelection( const TOOL_EVENT& aEvent )
{
    return 0;
}


void PCB_SELECTION_TOOL::FilterCollectedItems( GENERAL_COLLECTOR& aCollector, bool aMultiSelect )
{
}


bool PCB_SELECTION_TOOL::itemPassesFilter( BOARD_ITEM* aItem, bool aMultiSelect )
{
    return true;
}


void PCB_SELECTION_TOOL::ClearSelection( bool aQuietMode )
{
}


void PCB_SELECTION_TOOL::RebuildSelection()
{
}


bool PCB_SELECTION_TOOL::Selectable( const BOARD_ITEM* aItem, bool checkVisibilityOnly ) const
{
    return false;
}


bool PCB_SELECTION_TOOL::selectionContains( const VECTOR2I& aPoint ) const
{
    return false;
}

void PCB_SELECTION_TOOL::GuessSelectionCandidates( GENERAL_COLLECTOR& aCollector,
                                                   const VECTOR2I& aWhere ) const
{
}


int PCB_SELECTION_TOOL::updateSelection( const TOOL_EVENT& aEvent )
{
    return 0;
}


void PCB_SELECTION_TOOL::setTransitions()
{
}


void PCB_SELECTION_TOOL::select( EDA_ITEM* aItem )
{
}


void PCB_SELECTION_TOOL::unselect( EDA_ITEM* aItem )
{
}


void PCB_SELECTION_TOOL::highlight( EDA_ITEM* aItem, int aHighlightMode,
                                    SELECTION* aGroup )
{
}


void PCB_SELECTION_TOOL::unhighlight( EDA_ITEM* aItem, int aHighlightMode,
                                              SELECTION* aGroup )
{
}

void PCB_TOOL_BASE::doInteractiveItemPlacement( const TOOL_EVENT&        aTool,
                                                INTERACTIVE_PLACER_BASE* aPlacer,
                                                const wxString& aCommitMessage, int aOptions )
{
}


bool PCB_TOOL_BASE::Init()
{
    return true;
}


void PCB_TOOL_BASE::Reset( RESET_REASON aReason )
{
}


void PCB_TOOL_BASE::setTransitions()
{
}


bool PCB_TOOL_BASE::Is45Limited() const
{
    return false;
}


ZONE_FILLER::~ZONE_FILLER()
{
}


ZONE_FILLER_TOOL::ZONE_FILLER_TOOL() :
    PCB_TOOL_BASE( "pcbnew.ZoneFiller" ),
    m_fillInProgress( false )
{
}


ZONE_FILLER_TOOL::~ZONE_FILLER_TOOL()
{
}


void ZONE_FILLER_TOOL::Reset( RESET_REASON aReason )
{
}


void ZONE_FILLER_TOOL::setTransitions()
{
}


PCBNEW_SETTINGS::DISPLAY_OPTIONS& PCB_TOOL_BASE::displayOptions() const
{
    static PCBNEW_SETTINGS::DISPLAY_OPTIONS disp;

    return disp;
}

PCB_DRAW_PANEL_GAL* PCB_TOOL_BASE::canvas() const
{
    return nullptr;
}


const PCB_SELECTION& PCB_TOOL_BASE::selection() const
{
    static PCB_SELECTION sel;

    return sel;
}


PCB_SELECTION& PCB_TOOL_BASE::selection()
{
    static PCB_SELECTION sel;

    return sel;
}

BOX2I PCB_SELECTION::GetBoundingBox() const
{
    return BOX2I();
}


EDA_ITEM* PCB_SELECTION::GetTopLeftItem( bool onlyModules ) const
{
   return nullptr;
}


const std::vector<KIGFX::VIEW_ITEM*> PCB_SELECTION::updateDrawList() const
{
    std::vector<KIGFX::VIEW_ITEM*> items;

   return items;
}
