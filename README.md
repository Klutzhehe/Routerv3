# RouterV3

Recreating an engine-grounded PCB autorouting environment (inspired by the
*PCBWorld* paper, arXiv:2607.05915) built on KiCad's native Push-and-Shove
router (`PNS::ROUTER`), then training a novel routing agent on top of it.

Designed to run on **Google Colab** — see [`notebooks/00_setup.ipynb`](notebooks/00_setup.ipynb)
for a from-scratch environment bring-up (installs KiCad headlessly, verifies
the engine wrapper, and routes a toy board end to end).

## Project layout

```
pcbworld/       # Headless PNS::ROUTER bridge + (future) RL env/agents -- compile-from-source path
  engine/       # Headless wrapper around KiCad's routing/DRC engine
  env/          # Gym-style RL environment built on the engine wrapper
  agents/       # RL (PPO/GRPO) and LLM routing agents
  data/         # Synthetic + real board generation (D1/D2/D3-style datasets)
kicad_plugin/   # LLM Board Advisor -- KiCad Action Plugin, no compiling required
notebooks/      # Colab notebooks (setup, wrapper demo, training, eval)
scripts/        # CLI entry points
tests/          # Unit + integration tests
docs/           # Design notes, findings on KiCad API surface
```

Two parallel tracks: [`kicad_plugin/`](kicad_plugin/) is a working-today LLM
advisor Action Plugin (reads board state, asks an LLM, reports back) using
KiCad's standard scripting API -- no build required, but it can't drive the
interactive router (that API isn't exposed there). [`pcbworld/`](pcbworld/)
is the heavier path to real routing: compiling a bridge to KiCad's actual
`PNS::ROUTER` for an eventual RL routing agent.

## Status

**The hard part is done and verified.** `pcbworld/engine/cpp/` compiles into
a real pybind11 module that drives KiCad's actual `PNS::ROUTER` headlessly —
confirmed end-to-end in Colab: load a board, find pads, start a route, push
a point, fix the route, commit, save, then independently reload with a
*separate* KiCad Python process and confirm a real track landed on disk.
Nothing about this was obvious going in (see below) — start with
[`ROADMAP.md`](ROADMAP.md) before writing more code, especially in a fresh
conversation with no memory of how this was built.

## Next steps

See [`ROADMAP.md`](ROADMAP.md) for the full plan (near-term plumbing,
candidate directions for the "novel SOTA agent," and the operational
gotchas that cost real time to discover — read those before re-deriving
them).
