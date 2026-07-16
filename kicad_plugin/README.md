# LLM Board Advisor -- KiCad Action Plugin

Reads the current board (component count, nets, DRC marker count, outline
size), sends it to an LLM (OpenAI / Anthropic / DeepSeek), and reports the
reply back as a printed console message + a text comment placed on the
board. No compiling, no headless build -- runs inside your normal installed
KiCad using its standard Python scripting API.

Separate from, and complementary to, `pcbworld/` in this repo: that's the
(unfinished, compile-from-source) path to real `PNS::ROUTER` access for
actual autorouting. This plugin can't drive the interactive router --
`pcbnew`'s standard scripting API doesn't expose it -- so it's an advisor,
not an autorouter.

## 1. Set your API key

```
cd kicad_plugin/llm_advisor
cp .env.example .env
```

Edit `.env` and fill in the key for whichever provider you set
`LLM_PROVIDER` to (`anthropic`, `openai`, or `deepseek`). `.env` is
gitignored at the repo root -- never commit real keys.

## 2. Test in the Scripting Console first

Before installing it as a plugin, confirm the KiCad-to-Python bridge and
the LLM call both work in isolation, with a board already open:

`Tools > Scripting Console`, then:

```python
import sys
sys.path.insert(0, r"<path to this repo>/kicad_plugin/llm_advisor")

from board_summary import summarize_board, build_prompt
from llm_client import load_dotenv, query_llm
import os

load_dotenv(r"<path to this repo>/kicad_plugin/llm_advisor/.env")
board = pcbnew.GetBoard()
print(summarize_board(board))
print(query_llm(build_prompt(summarize_board(board))))
```

If `summarize_board` errors on the DRC marker count, that's a known soft
spot -- see the comment in `board_summary.py`'s `_count_drc_markers`. It's
wrapped to degrade gracefully (reports "unavailable") rather than crash the
whole summary, so fix it there once you know your KiCad version's actual
method name.

## 3. Install as an Action Plugin

Copy (or symlink) the `llm_advisor/` folder into KiCad's plugins directory:

- Windows: `%APPDATA%\kicad\9.0\scripting\plugins\`
- macOS: `~/Documents/KiCad/9.0/scripting/plugins/`
- Linux: `~/.local/share/kicad/9.0/scripting/plugins/`

(Or find the exact path via `Tools > Scripting Console` ->
`pcbnew.GetKicadConfigPath()`.)

Then in the PCB editor: `Tools > External Plugins > Refresh Plugins`, and
it'll appear as "LLM Board Advisor" under `Tools > External Plugins` (and
as a toolbar button).

## Notes

- The LLM call runs on a background thread so it doesn't freeze the KiCad
  UI while waiting on the network; the result comes back via `wx.CallAfter`.
- Swap providers/models by editing `LLM_PROVIDER`/`LLM_MODEL` in `.env` --
  no code changes needed, see `llm_client.py`.
