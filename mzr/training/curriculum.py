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

Stage `1m` is deliberately **not** on the 0-1-2-3 spine. It holds net count at
stage 1's three and changes exactly one thing -- nets get up to four pins, so a
net is a k-1 leg spanning tree rather than a single connection. That isolates
the one behaviour multi-pin adds: a pin shared by two tree edges hosts two
frontiers and must be routed *through*, not just reached. Comparing it against
stage 1 at equal LEG count separates "branch points are hard" from "more legs
are hard"; run it after stage 1 so there is something to compare against.

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
    #: Quality thresholds the gate enforces ALONGSIDE completion.
    #:
    #: Completion alone passed a policy that double-routed 46.5% of boards at
    #: 2.3x copper, and later one emitting 38% right angles whose direction
    #: head was collapsed on d0 for 317 of 317 actions -- a field follower that
    #: had learned no steering at all. A router that arrives by wandering
    #: should not clear a gate.
    #:
    #: `max_copper` is on the MEDIAN, which sits at 1.000 on a healthy policy;
    #: the mean is dragged by a pathological tail and makes a poor threshold.
    max_copper: float = 1.15
    #: Fab practice replaces every 90-degree corner with two 45s, so this is a
    #: real rule rather than an aesthetic. Measured at 0.38 on the stage-0
    #: policy that "passed".
    max_right_angle: float = 0.15
    #: Ceiling on the fraction of actions that are `d0` -- straight down the
    #: geodesic gradient.
    #:
    #: **Default 1.0 (disabled), and deliberately so.** This started life as
    #: "the steering check", on the hypothesis that a direction head collapsed
    #: on d0 meant the policy had learned no obstacle avoidance. A stage-0 run
    #: tested that hypothesis and refuted it, over 28 evals:
    #:
    #:     group                  n    completion  right-angle  copper
    #:     d0 <= 95% (steering)   7    0.8771      51.3%        1.126x
    #:     d0 >  95% (following)  21   0.9159      47.9%        1.098x
    #:     correlation(d0, completion) = +0.260
    #:
    #: Steering correlated with WORSE results on every axis, and the best eval
    #: of the run -- completion 1.000, copper 1.049x, right-angle 16% -- was a
    #: pure field follower at d0 = 100%. On one net with a correct geodesic
    #: field, following the field IS the optimal policy, and the route quality
    #: comes from choosing step sizes well rather than from deviating. Gating
    #: on d0 at stage 0 would fail a good policy for being right.
    #:
    #: It becomes meaningful from stage 1, where the field cannot see other
    #: nets' live copper and yielding a channel REQUIRES leaving your own
    #: gradient -- so the stages that need it set it explicitly below.
    max_d0_frac: float = 1.0
    #: Secondary: entropy floor, to catch a head that is dead rather than
    #: merely decisive. Kept LOW and not relied on -- the collapsed policy
    #: measured 0.410 against this 0.40 and scraped through, because the
    #: distribution had spread while the argmax never varied. That near-miss
    #: is exactly why `max_d0_frac` above exists and is the primary check.
    min_dir_entropy: float = 0.40

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
        # Steering is required here: the geodesic field does not
        # contain other nets' live copper, so yielding a channel
        # means leaving your own gradient. See Stage.max_d0_frac.
        max_d0_frac=0.95,
    ),
    "2": Stage(
        name="2: 5 nets, 4 layers",
        height=64, width=64, layers=4,
        generator=GeneratorConfig(num_nets=5, num_components=4, pin_pitch_cells=4),
        ripup=_RIPUP,
        max_macro_steps=64,
        gate=("absolute", 1.0),
        kill="can't clear 0.90 in 3000 updates -> add h/g/f, or --bc-coef 0.5",
        # Steering is required here: the geodesic field does not
        # contain other nets' live copper, so yielding a channel
        # means leaving your own gradient. See Stage.max_d0_frac.
        max_d0_frac=0.95,
    ),
    "1m": Stage(
        name="1m: 3 nets, up to 4 pins each -- multi-pin fan-out",
        height=48, width=48, layers=2,
        generator=GeneratorConfig(
            num_nets=3, num_components=4, pin_pitch_cells=4,
            multi_pin_frac=0.6, max_pins_per_net=4,
        ),
        ripup=_RIPUP,
        max_macro_steps=48,
        gate=("absolute", 1.0),
        kill="can't clear 0.85 in 2000 updates -> the branch point is the problem, "
             "not net count; compare against stage 1 at equal leg count",
        # Steering is required here: the geodesic field does not
        # contain other nets' live copper, so yielding a channel
        # means leaving your own gradient. See Stage.max_d0_frac.
        max_d0_frac=0.95,
    ),
    "3": Stage(
        name="3: 8 nets, 4 layers, search on",
        height=64, width=64, layers=4,
        generator=GeneratorConfig(num_nets=8, num_components=4, pin_pitch_cells=4),
        ripup=_RIPUP,
        max_macro_steps=64,
        gate=("absolute", 1.0),
        kill="prior can't clear 0.85 -> search is being built on a weak prior; --bc-coef 0.3",
        # Steering is required here: the geodesic field does not
        # contain other nets' live copper, so yielding a channel
        # means leaving your own gradient. See Stage.max_d0_frac.
        max_d0_frac=0.95,
    ),
}


#: Held-out eval seeds. Fixed, and disjoint from any training seed range (the
#: trainer seeds boards from 1000+), so an eval number is never a memorised
#: training board. A failing eval seed is reproducible on its own -- that is the
#: whole point of keeping the set fixed rather than random.
EVAL_SEEDS = list(range(900_000, 900_128))
