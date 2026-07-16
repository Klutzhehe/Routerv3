// Dead-code-only stub definitions for GUI tool classes.
//
// pnsrouter (pcbnew/router/router_tool.cpp) and pcbcommon
// (pcbnew/board_commit.cpp) reference ZONE_FILLER_TOOL/PCB_SELECTION_TOOL
// through TOOL_MANAGER::GetTool<T>() inside code paths this headless
// bridge never executes -- we never construct or use a TOOL_MANAGER at
// all (PNS_BRIDGE_IFACE bypasses BOARD_COMMIT entirely; see pns_bridge.h).
// That code is provably unreachable from anything this bridge calls, but
// the linker still needs *some* definition to resolve the symbols it
// references. Typeinfo in particular is an eager/DATA relocation in a
// shared library -- unlike a plain function call, it can't be left
// lazily unresolved the way a never-invoked function pointer can, which
// is why this surfaces as an ImportError rather than only failing if the
// dead code path were ever actually reached.
//
// Deliberately NOT inheriting from the real PCB_TOOL_BASE/SELECTION_TOOL
// (pcbnew/tools/pcb_tool_base.h, pcbnew/tools/selection_tool.h) -- doing
// so would pull in the rest of the GUI tool framework, which is exactly
// what this avoids. Linking is name-based: the mangled symbol for a
// class's typeinfo/methods only encodes the class name and method
// signatures, not its base classes, so the linker is satisfied by this
// stub regardless of the layout/inheritance mismatch with the real
// class. Since the referencing code never actually runs, that mismatch
// never manifests as a behavioral difference.

class TOOL_EVENT;
class ZONE;

class ZONE_FILLER_TOOL
{
public:
    virtual ~ZONE_FILLER_TOOL() = default;
    static bool IsZoneFillAction( const TOOL_EVENT* aEvent ) { return false; }
    void DirtyZone( ZONE* aZone ) {}
};

class PCB_SELECTION_TOOL
{
public:
    virtual ~PCB_SELECTION_TOOL() = default;
    void RebuildSelection() {}
};
