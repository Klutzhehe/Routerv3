"""Curriculum stages -- one mechanism per rung, each with a gate.

`mzr/DESIGN.md` section 7. The stage 0-3 gate is **absolute 100% completion**
(argmax, sustained 3 consecutive evals).

**No solvability pre-filter.** Boards are generated fresh from seeds. If the
policy stalls a few points short of 1.0, the handful of failing eval seeds get
**reviewed by hand** -- `python -m mzr.world.pool --stage S --seeds ...` reports
whether the expert can route each -- rather than being auto-filtered out of the
distribution.

**Pure RL first.** `bc_coef0` is 0 for every stage. If a stage plateaus, raise
it (`--bc-coef`) to blend in expert behaviour cloning, annealed. That decision
is made from a real plateau on this problem, not from a paper about a different
one.

Each stage changes `GeneratorConfig` / `WorldConfig` and nothing in the model.
`gate` is ``("absolute", x)`` for every implemented stage. `kill` is the
pre-committed "stop, the premise was wrong" line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mzr.env.rewards import RewardConfig
from mzr.world.generator import GeneratorConfig
from mzr.world.spec import BoardSpec, LayerStack, PriceRules, RipupRules


@dataclass
class Stage:
    name: str
    height: int
    width: int
    layers: int
    generator: GeneratorConfig
    ripup: RipupRules
    max_macro_steps: int
    gate: tuple[str, float]
    kill: str
    #: BC loss weight at stage start, annealed to 0 as completion rises.
    #: **0 everywhere** -- pure RL first (see module docstring). Raise via
    #: `--bc-coef` only after a measured plateau.
    bc_coef0: float = 0.0
    reward: RewardConfig = field(default_factory=RewardConfig)

    def board_spec(self) -> BoardSpec:
        return BoardSpec(
            height_cells=self.height,
            width_cells=self.width,
            layers=LayerStack(num_layers=self.layers),
        )


_NO_RIPUP = RipupRules(interval=0)
_RIPUP = RipupRules(interval=8, fraction=0.25)


STAGES: dict[str, Stage] = {
    "0": Stage(
        name="0: single net, obstacles",
        height=48, width=48, layers=2,
        generator=GeneratorConfig(
            num_nets=1, num_components=3, pin_pitch_cells=4, num_keepouts=3,
            keepout_max_cells=10,
        ),
        ripup=_NO_RIPUP,
        max_macro_steps=48,
        gate=("absolute", 1.0),
        kill="can't reach 0.95 in 500 updates -> geometry or reward bug, not a hard problem",
    ),
    "1": Stage(
        name="1: 3 simultaneous nets, price on",
        height=48, width=48, layers=2,
        generator=GeneratorConfig(num_nets=3, num_components=3, pin_pitch_cells=4),
        ripup=_RIPUP,
        max_macro_steps=48,
        gate=("absolute", 1.0),
        kill="can't clear 0.90 in 2000 updates -> simultaneous premise is weak; try --bc-coef 0.5",
    ),
    "2": Stage(
        name="2: 5 nets, 4 layers",
        height=64, width=64, layers=4,
        generator=GeneratorConfig(num_nets=5, num_components=4, pin_pitch_cells=4),
        ripup=_RIPUP,
        max_macro_steps=64,
        gate=("absolute", 1.0),
        kill="can't clear 0.90 in 3000 updates -> add h/g/f, or --bc-coef 0.5",
    ),
    "3": Stage(
        name="3: 8 nets, 4 layers, search on",
        height=64, width=64, layers=4,
        generator=GeneratorConfig(num_nets=8, num_components=4, pin_pitch_cells=4),
        ripup=_RIPUP,
        max_macro_steps=64,
        gate=("absolute", 1.0),
        kill="prior can't clear 0.85 -> search is being built on a weak prior; --bc-coef 0.3",
    ),
}


#: Held-out eval seeds. Fixed, and disjoint from any training seed range (the
#: trainer seeds boards from 1000+), so an eval number is never a memorised
#: training board. A failing eval seed is reproducible on its own -- that is the
#: whole point of keeping the set fixed rather than random.
EVAL_SEEDS = list(range(900_000, 900_128))
