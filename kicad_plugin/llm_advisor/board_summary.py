"""Pulls a small text summary out of the current KiCad BOARD for the LLM.

A couple of field lookups below (DRC marker count in particular) are
wrapped defensively -- pcbnew's SWIG API has shifted method names across
KiCad releases (Markers() vs GetMarkers() vs GetMARKERs()) and this hasn't
been run against a real board yet. Follow the "paste into the Scripting
Console first" step in kicad_plugin/README.md: if a field silently reports
"unavailable", that's the one to fix for your KiCad version.
"""

import pcbnew


def _count_drc_markers(board):
    for method_name in ("Markers", "GetMarkers", "GetMARKERs"):
        method = getattr(board, method_name, None)
        if method is None:
            continue
        try:
            return len(list(method()))
        except TypeError:
            continue
    return None


def summarize_board(board):
    lines = []

    footprints = list(board.GetFootprints())
    lines.append(f"Components: {len(footprints)}")

    nets = [n for n in board.GetNetInfo() if n.GetNetname()]
    lines.append(f"Nets: {len(nets)}")
    if nets:
        sample = ", ".join(n.GetNetname() for n in nets[:15])
        more = "" if len(nets) <= 15 else f", ... (+{len(nets) - 15} more)"
        lines.append(f"Net names (sample): {sample}{more}")

    tracks = list(board.GetTracks())
    lines.append(f"Track/via segments: {len(tracks)}")

    drc_count = _count_drc_markers(board)
    if drc_count is None:
        lines.append("DRC violations: unavailable (run Inspect > Design Rules Checker "
                      "in the GUI first, or fix the marker API lookup for your KiCad version)")
    else:
        lines.append(f"DRC violations (from last-run DRC markers on the board): {drc_count}")

    bbox = board.GetBoardEdgesBoundingBox()
    lines.append(
        f"Board outline bounding box: {pcbnew.ToMM(bbox.GetWidth()):.1f} x "
        f"{pcbnew.ToMM(bbox.GetHeight()):.1f} mm"
    )

    return "\n".join(lines)


def build_prompt(board_summary, user_question=None):
    question = user_question or (
        "Review this PCB for anything a designer should double-check -- "
        "unrouted or suspiciously few nets, DRC violations, an oddly small "
        "or large board outline for the component count, etc. Be concise, "
        "a few bullet points."
    )
    return (
        "You are reviewing a KiCad PCB design. Here is a summary of the "
        f"current board state:\n\n{board_summary}\n\n{question}"
    )
