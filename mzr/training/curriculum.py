"""Curriculum stages -- one mechanism per rung, each with a gate.

**Revised 2026-09-02.** Stage 0 sat under its gate for many sessions. The cause
was measured with NON-LEARNED baselines on the first 48 held-out eval seeds, so
no policy was implicated -- `mzr/DESIGN.md` section 7.1 has the full evidence.
Summary of what changed here:

* Stage 0 is **one layer**. It was two, with pads on outer layers, so ~25% of
  boards could not be routed without a via -- while the policy's init bias makes
  its untrained argmax exactly `baselines.greedy`, which never vias and
  therefore failed *exactly and only* those boards (12 of 48, 0 same-layer
  failures). The stage conflated "route around obstacles" with "discover the
  via" and reported one number. The via now has its own rung, `0v`.
* `geodesic_downsample` is 1, not 4. At 4 a 3x3 keepout was invisible to the
  field (0 coarse cells blocked) and a 10x10 could shrink to one.
* `copper_seeded` is on, so each leg grows from one end toward its own live
  copper. Dual-ended growth toward static pads double-routed 24 of 48 boards
  with no policy involved at all.
* `max_macro_steps` doubled: one frontier now covers a whole leg, not half.

Measured on those 48 seeds after the change: `greedy` on stage 0 goes from
completion 0.7500 / copper 2.00 / 24 double-routed / 40.4% right angles to
**1.0000 / 1.000 / 0 / 0.0%**.

**Standing rule:** run `layer_hop` against a stage's gate before training it.
If a parameter-free heuristic clears the gate, the stage is not testing what it
claims; if it cannot, neither can a policy initialised at `greedy`.

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
    #: Resolution the geodesic field is relaxed at. **1 = exact.**
    #:
    #: This was 4 everywhere, inherited from a memory argument that only binds
    #: at 128x128x8. On the 48x48 boards these stages actually use it made the
    #: field a 12x12 grid, and a coarse cell counts as blocked only when every
    #: fine cell in it is -- so obstacles vanished. Measured on a 48x48 board:
    #:
    #:     3x3  keepout ->  0 coarse cells blocked   (invisible)
    #:     6x6  keepout ->  0-1, depending on alignment
    #:     10x10 keepout -> 1-4
    #:
    #: `GeneratorConfig.keepout_max_cells` is 10 and keepouts are sampled from
    #: [3, 10], so the field the whole obstacle-avoidance story rests on could
    #: not see most of the obstacles it was supposed to route around. At these
    #: board sizes the exact field costs a few MB.
    geodesic_downsample: int = 1
    #: Grow each leg from ONE end, toward the net's live copper, instead of
    #: from both pads toward each other.
    #:
    #: Dual-ended growth toward *static pads* is what causes double-routing,
    #: and it does so with no policy involved at all: the non-learned
    #: `layer_hop` baseline double-routes 24 of 48 stage-0 boards at 1.79x
    #: copper, because the two frontiers mirror around each other, swap
    #: positions and each completes the whole run while `completion` reads
    #: 1.000. Four reward patches (`leg_progress`, `tip_progress`,
    #: `leg_budget_frac`, `wirelength` x12) were built to price that out and
    #: none removed it. Seeding the field from live copper removes it by
    #: construction: measured on the same 48 boards, copper_median 1.79 -> 1.00
    #: and doubled 24 -> 0.
    copper_seeded: bool = True
    #: Macro-steps between field refreshes under `copper_seeded`.
    geodesic_refresh: int = 4
    #: PPO discount. **0.999, not the usual 0.99**, and the difference is the
    #: whole route-quality story on stage 0.
    #:
    #: These episodes are short and hard-bounded by `max_macro_steps`, so there
    #: is no reason for a strong time preference -- but 0.99 imposes one. With a
    #: terminal payout of ~12 (`arrival` 2 + `completion` 10), halving
    #: time-to-arrival from 40 steps to 20 is worth
    #:
    #:     0.99**20 * 12 - 0.99**40 * 12  ~=  1.79
    #:
    #: while the right angles that haste causes cost ~0.34 per board at
    #: `corner` 0.25. So finishing fast paid about 5x what quality charged, and
    #: the policy took the trade -- correctly, under the reward it was given.
    #:
    #: Measured on the first two post-fix runs: the step-class histogram came
    #: out {1 cell: 34, 2 cells: 334} and **every one of the 66 right angles
    #: was preceded by a 2-cell segment** -- 66 of 66. Raising `corner` from
    #: 0.08 to 0.25 and `entropy_coef` from 0.004 to 0.02 moved right-angle
    #: 0.22 -> 0.20, i.e. not at all, because neither touches the term that was
    #: actually paying. At 0.999 the same haste is worth ~0.23, which is the
    #: same order as the quality terms.
    gamma: float = 0.999
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
#: Partial retraction, not whole-net rip-up. `interval=8, fraction=0.25` -- the
#: old value -- did nothing at all at stage 1: `ripup_round` takes
#: `floor(3 * 0.25) = 0`, so the negotiation substrate never ran and stage 1's
#: plateau was simply the ceiling of simultaneous GREEDY growth (DESIGN.md 7.3).
#:
#: Measured on stage 1, layer_hop, 48 held-out boards (144 legs), no learning:
#:
#:     no rip-up (the old default, k=0)   0.8542  123/144   RA 0.119
#:     whole-net rip-up, fraction 0.5     0.5625   81/144
#:     retract 1 step every 4             0.9444  136/144   RA 0.084
#:
#: Gentle and frequent wins; retracting 2+ steps every 2 collapses to 0.6181,
#: the same over-destruction whole-net rips showed. That is PathFinder's
#: gradualness argument (DESIGN.md section 3) reproduced on this problem.
_RIPUP = RipupRules(interval=4, retract_steps=1, retract_fraction=0.25)


STAGES: dict[str, Stage] = {
    "0": Stage(
        name="0: single net, one layer, obstacles -- route around things",
        height=48, width=48, layers=1,
        generator=GeneratorConfig(
            num_nets=1, num_components=3, pin_pitch_cells=4, num_keepouts=3,
            keepout_max_cells=10,
        ),
        ripup=_NO_RIPUP,
        max_macro_steps=96,
        gate=("absolute", 1.0),
        kill="can't reach 0.95 in 500 updates -> geometry or reward bug, not a hard problem",
        # The entropy floor is disabled here for exactly the reason
        # `max_d0_frac` is (see Stage.max_d0_frac): on ONE net with an exact
        # geodesic field, following the field IS the optimal policy, so a
        # direction head that collapses onto d0 has learned the right thing.
        # Measured on the first post-fix run: argmax completion 1.000 on every
        # eval with `ent_direction` at 0.003 -- the gate was failing a policy
        # that routed all 64 held-out boards perfectly. It stays armed from
        # stage 1, where leaving your own gradient is the task.
        min_dir_entropy=0.0,
    ),
    "0v": Stage(
        name="0v: single net, two layers -- the via is mandatory",
        height=48, width=48, layers=2,
        generator=GeneratorConfig(
            num_nets=1, num_components=3, pin_pitch_cells=4, num_keepouts=3,
            keepout_max_cells=10,
        ),
        ripup=_NO_RIPUP,
        max_macro_steps=96,
        gate=("absolute", 1.0),
        kill="can't reach 0.95 in 1000 updates -> the via penalty is drowning "
             "discovery; drop RewardConfig.via or seed the layer head from layer_hop",
        # The entropy floor is disabled here for exactly the reason
        # `max_d0_frac` is (see Stage.max_d0_frac): on ONE net with an exact
        # geodesic field, following the field IS the optimal policy, so a
        # direction head that collapses onto d0 has learned the right thing.
        # Measured on the first post-fix run: argmax completion 1.000 on every
        # eval with `ent_direction` at 0.003 -- the gate was failing a policy
        # that routed all 64 held-out boards perfectly. It stays armed from
        # stage 1, where leaving your own gradient is the task.
        min_dir_entropy=0.0,
    ),
    "1": Stage(
        name="1: 3 simultaneous nets, price on",
        height=48, width=48, layers=2,
        generator=GeneratorConfig(num_nets=3, num_components=3, pin_pitch_cells=4),
        ripup=_RIPUP,
        max_macro_steps=96,
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
        max_macro_steps=128,
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
        max_macro_steps=96,
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
        max_macro_steps=128,
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
