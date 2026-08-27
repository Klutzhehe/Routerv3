"""Curriculum stages -- one mechanism per rung, each with a gate.

`mzr/DESIGN.md` section 7. Sized so that **stage 3 can plausibly reach ~100%**:
stages 1-3 run at low net counts where near-perfect completion is achievable,
and the hard scaling work -- where no router hits 100% -- lives in stages 4-8.
"Stage 3 done" is therefore not "the router is finished"; it is "search beats
the prior on a tractable board".

Each stage changes `GeneratorConfig` / `WorldConfig` and nothing in the model.
The `gate` is what a run must clear before advancing:

* ``("absolute", x)``  -- held-out argmax completion >= x
* ``("vs_expert", x)`` -- held-out argmax completion >= expert + x
* ``("vs_prior", x)``  -- (stage 3 only) search completion >= prior + x

`kill` is the pre-committed "the premise was wrong, stop" line.
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
    #: BC loss weight at the start of the stage, annealed to 0 as completion
    #: rises. 0 for stage 0 (nothing to imitate -- one net, greedy is optimal).
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
        gate=("absolute", 0.99),
        kill="can't reach 95% in 500 updates -> geometry or reward bug, not a hard problem",
    ),
    "1": Stage(
        name="1: 3 simultaneous nets, price on",
        height=48, width=48, layers=2,
        generator=GeneratorConfig(num_nets=3, num_components=3, pin_pitch_cells=4),
        ripup=_RIPUP,
        max_macro_steps=48,
        gate=("vs_expert", 0.10),
        kill="no gain over sequential+PathFinder in 2000 updates -> simultaneous premise is wrong",
        bc_coef0=0.5,
    ),
    "2": Stage(
        name="2: 5 nets, 4 layers",
        height=64, width=64, layers=4,
        generator=GeneratorConfig(num_nets=5, num_components=4, pin_pitch_cells=4),
        ripup=_RIPUP,
        max_macro_steps=64,
        gate=("vs_expert", 0.05),
        kill="none -- a model can fail an absolute fidelity test and still serve search; let stage 3 decide",
        bc_coef0=0.5,
    ),
    "3": Stage(
        name="3: 8 nets, 4 layers, search on",
        height=64, width=64, layers=4,
        generator=GeneratorConfig(num_nets=8, num_components=4, pin_pitch_cells=4),
        ripup=_RIPUP,
        max_macro_steps=64,
        gate=("vs_prior", 0.05),
        kill="search shows no gain over prior-only -> ship prior-only (still a complete router)",
        bc_coef0=0.3,
    ),
}


#: Held-out eval seeds. Fixed, and disjoint from any training seed range, so an
#: eval number is never memorised training boards. The tail values are
#: `neuroroute/`'s known-hard seeds -- kept in every eval set by convention so a
#: regression on the cases that were painful last time is visible immediately.
EVAL_SEEDS = list(range(900_000, 900_064)) + [
    9648, 9681, 9764, 9779, 9148, 9251, 9091, 9390, 9535, 9901
]
