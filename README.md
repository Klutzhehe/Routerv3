# RouterV3

Recreating an engine-grounded PCB autorouting environment (inspired by the
*PCBWorld* paper, arXiv:2607.05915) built on KiCad's native Push-and-Shove
router (`PNS::ROUTER`), then training a novel routing agent on top of it.

Designed to run on **Google Colab** — see [`notebooks/00_setup.ipynb`](notebooks/00_setup.ipynb)
for a from-scratch environment bring-up (installs KiCad headlessly, verifies
the engine wrapper, and routes a toy board end to end).

## Project layout

```
pcbworld/
  engine/     # Headless wrapper around KiCad's routing/DRC engine
  env/        # Gym-style RL environment built on the engine wrapper
  agents/     # RL (PPO/GRPO) and LLM routing agents
  data/       # Synthetic + real board generation (D1/D2/D3-style datasets)
notebooks/    # Colab notebooks (setup, wrapper demo, training, eval)
scripts/      # CLI entry points
tests/        # Unit + integration tests
docs/         # Design notes, findings on KiCad API surface
```

## Status

Early scaffolding. See [`docs/engine_access.md`](docs/engine_access.md) for the
current findings on how to drive KiCad's router headlessly (IPC API vs.
scripting console vs. custom build) — this is the load-bearing technical
question the rest of the project depends on.

## Milestones

1. **Engine wrapper** — headless Python control of KiCad's router: select a
   net, start a route, push points into the pathfinder, commit resulting
   tracks, run DRC. Round-trips a `.kicad_pcb` file.
2. **RL environment** — Gym-style env over the wrapper, reward = potential
   shaping on DRC violations / wirelength / via count, synthetic board
   generator for training data.
3. **Baseline agents** — PPO/GRPO from scratch on synthetic boards; compare
   against Freerouting and grid-based RL baselines.
4. **Novel agent** — once the above is validated, propose and implement a
   SOTA routing approach beyond the paper's baselines.
