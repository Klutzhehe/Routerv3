"""LLM Board Advisor -- a KiCad Action Plugin.

Reads a summary of the current board, sends it to an LLM, and reports the
reply back (printed to the Scripting Console + added as a text comment on
the board). See kicad_plugin/README.md for install/usage/testing.

Architecture notes (why it's built this way):
  - The LLM call runs on a background thread with the result marshalled
    back via wx.CallAfter -- ActionPlugin.Run() executes on KiCad's UI
    thread, so a blocking network call in-line would freeze the whole
    application for the duration of the request.
  - Board mutation (adding the comment) happens back on the UI thread via
    that same wx.CallAfter, since BOARD/pcbnew objects aren't meant to be
    touched from a worker thread concurrently with the GUI.
"""

import os
import threading

import pcbnew
import wx

from .board_summary import build_prompt, summarize_board
from .llm_client import LLMError, load_dotenv, query_llm

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


def _add_comment(board, text):
    item = pcbnew.PCB_TEXT(board)
    item.SetText(text)
    item.SetLayer(pcbnew.Cmts_User)
    bbox = board.GetBoardEdgesBoundingBox()
    item.SetPosition(pcbnew.VECTOR2I(bbox.GetLeft(), bbox.GetBottom() + pcbnew.FromMM(5)))
    board.Add(item)

    try:
        pcbnew.Refresh()
    except AttributeError:
        pass  # older/newer API without a bare Refresh() -- comment is still added


class LLMAdvisorPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "LLM Board Advisor"
        self.category = "Analysis"
        self.description = "Send a board summary to an LLM and report its advice"
        self.show_toolbar_button = True
        self.icon_file_name = ""

    def Run(self):
        load_dotenv(os.path.join(_PLUGIN_DIR, ".env"))

        board = pcbnew.GetBoard()
        summary = summarize_board(board)
        prompt = build_prompt(summary)

        print("[LLM Board Advisor] board summary:")
        print(summary)
        print("[LLM Board Advisor] querying LLM...")

        def worker():
            try:
                advice = query_llm(prompt)
                error = None
            except LLMError as e:
                advice = None
                error = str(e)

            wx.CallAfter(self._on_result, board, advice, error)

        threading.Thread(target=worker, daemon=True).start()

    def _on_result(self, board, advice, error):
        if error:
            print(f"[LLM Board Advisor] error: {error}")
            wx.MessageBox(f"LLM request failed:\n\n{error}", "LLM Board Advisor",
                           wx.OK | wx.ICON_ERROR)
            return

        print("[LLM Board Advisor] advice:")
        print(advice)

        _add_comment(board, f"LLM Board Advisor:\n{advice}")
        wx.MessageBox(advice, "LLM Board Advisor", wx.OK | wx.ICON_INFORMATION)


LLMAdvisorPlugin().register()
