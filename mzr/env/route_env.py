"""`RouteEnv` -- the batched RL environment.

Not a `gym.Env`. Gym's single-environment API forces one board per process,
which is the exact constraint that capped every previous thread in this repo at
two workers. This env is vectorised at the tensor level: one `step()` advances
`B` boards and all `B*F` frontiers at once, and everything it returns carries a
leading batch dimension.

Simpler than `neuroroute/env/route_env.py` in one structural way: **there is no
scheduler.** Every net is live from `reset()`. That env's `step()` had to apply
geometry, retire finished heads, then bind pending nets to the freed slots
using a sampled scheduler action -- a four-phase dance whose whole purpose was
to make the scheduler learnable. Here there is nothing to schedule, so `step()`
is: apply the joint action, compute reward, build the next observation.

One ordering subtlety carries over. The observation the caller acted on defines
what ``direction = 0`` meant for that action (it is egocentric, relative to the
frontier's bearing at that moment). Copper moves between the caller receiving
the observation and the action being applied, so the engine must resolve the
move against *that* bearing, not a freshly recomputed one -- `step()` passes
`self._obs.bearing` through explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from mzr.env.observation import Observation, build_observation
from mzr.env.rewards import RewardConfig, failure_penalty, step_reward, terminal_reward
from mzr.world.engine import SimultaneousRouterWorld, WorldConfig
from mzr.world.generator import GeneratorConfig, generate_board
from mzr.world.spec import NUM_ENDS, BoardSpec


@dataclass
class EnvConfig:
    spec: BoardSpec = field(default_factory=BoardSpec)
    world: WorldConfig = field(default_factory=WorldConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    #: Hard cap on macro-steps per episode. An episode also ends early once
    #: every frontier has settled, which is the common case on easy boards.
    max_episode_steps: int = 64
    seed: int = 0
    #: When set, `reset()` samples `batch_size` seeds from this list (a
    #: solvable-board pool, see `world/pool.py`) instead of walking a counter.
    #: This is how stages 0-3 guarantee every training board is 100%-routable.
    board_seeds: list[int] | None = None


@dataclass
class StepOut:
    obs: Observation
    #: (B, F) per-frontier reward -- the signal for the factored action heads.
    reward: torch.Tensor
    #: (B,) board-level reward -- failure penalty per step, plus the terminal
    #: block on the final step. Kept separate because it is a joint outcome and
    #: the value head that consumes it is board-level, not per-frontier.
    board_reward: torch.Tensor
    done: bool
    info: dict


class RouteEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.world = SimultaneousRouterWorld(cfg.spec, cfg.world)
        self.device = self.world.device
        self._seed = cfg.seed
        self._rng = torch.Generator().manual_seed(cfg.seed)
        self._t = 0
        self._obs: Observation | None = None
        self._leg_done_prev: torch.Tensor | None = None
        self._boards: list = []

    # -- lifecycle --------------------------------------------------------

    def reset(self, seeds: Iterable[int] | None = None) -> Observation:
        B = self.cfg.world.batch_size
        if seeds is None:
            if self.cfg.board_seeds:
                pool = self.cfg.board_seeds
                pick = torch.randint(0, len(pool), (B,), generator=self._rng)
                seeds = [pool[int(i)] for i in pick]
            else:
                seeds = [self._seed + i for i in range(B)]
                self._seed += B
        seeds = list(seeds)

        self._boards = [generate_board(self.cfg.spec, self.cfg.generator, s) for s in seeds]
        self.world.load(self._boards)
        self._t = 0
        self._leg_done_prev = self.world.leg_done.clone()
        self._obs = build_observation(self.world)
        return self._obs

    # -- step -----------------------------------------------------------

    def step(self, action: dict[str, torch.Tensor]) -> StepOut:
        """Advance every frontier by one macro-step.

        `action` keys: `direction`, `step`, `layer`, `via`, `width`, `couple`,
        each ``(B, F)`` int64. Missing keys default to zero (the greedy /
        no-op choice), so a partial action is legal -- stage 0 supplies only
        `direction` and `step`.
        """
        assert self._obs is not None, "call reset() before step()"
        w = self.world
        B, F = self.cfg.world.batch_size, w.F
        z = lambda: torch.zeros(B, F, dtype=torch.long, device=self.device)  # noqa: E731
        a = lambda k: action.get(k, z())  # noqa: E731

        res = w.step(
            a("direction"), a("step"), a("layer"), a("via"), a("width"), a("couple"),
            bearing=self._obs.bearing,
        )

        # Per-frontier arrival: a leg that became done this step credits BOTH
        # its frontiers. The engine reports connections only as a board count;
        # credit assignment wants the frontiers that closed the leg.
        newly = w.leg_done & ~self._leg_done_prev            # (B, N, 2)
        self._leg_done_prev = w.leg_done.clone()
        arrived = (
            newly.unsqueeze(-1)
            .expand(B, self.cfg.world.max_nets, self.cfg.world.max_legs, NUM_ENDS)
            .reshape(B, F)
        )

        reward = step_reward(res, arrived, self.cfg.reward)
        board_reward = failure_penalty(res, self.cfg.reward)

        self._t += 1
        done = w.episode_done() or self._t >= self.cfg.max_episode_steps

        info: dict = {
            "t": self._t,
            "nets_done": res.nets_done,
            "nets_failed": res.nets_failed,
            "congestion": res.congestion,
            "rejected": res.rejected,
            "contended": res.contended,
            "moved": res.moved,
            "vias": res.via_placed,
        }

        if done:
            term, metrics = terminal_reward(w, self.cfg.reward)
            board_reward = board_reward + term
            info["terminal"] = metrics
            info["completion"] = metrics["completion"]

        self._obs = build_observation(w)
        return StepOut(
            obs=self._obs,
            reward=reward,
            board_reward=board_reward,
            done=done,
            info=info,
        )

    # -- convenience --------------------------------------------------

    @property
    def t(self) -> int:
        return self._t

    def completion(self) -> torch.Tensor:
        return self.world.completion()
