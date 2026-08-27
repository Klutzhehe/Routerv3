"""`NeuroRouteEnv` -- the batched RL environment.

Not a `gym.Env`. Gym's single-environment API forces one board per process,
which is the exact constraint that has capped every previous thread in this
repo at 2 workers. This env is vectorised at the tensor level instead: one
`step()` advances `B` boards and `B*K` routing heads, and everything it returns
carries a leading batch dimension.

Ordering within a step is deliberate:

1. apply the per-head geometry actions,
2. retire heads whose nets finished or ran out of budget,
3. bind pending nets to the slots that just freed up, using the scheduler
   action that was sampled from *this* step's observation,
4. build the next observation.

Scheduling last is what makes the scheduler head learnable: it acts on an
observation that lists the pending nets and sees the consequence on the very
next step, rather than choosing blind.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Iterable

import numpy as np
import torch

from neuroroute.env.observation import Observation, build_observation
from neuroroute.env.rewards import RewardConfig, step_reward, terminal_reward
from neuroroute.world.engine import BatchedRouterWorld, StepResult, WorldConfig
from neuroroute.world.generator import GeneratorConfig, generate_board, straight_line_demand
from neuroroute.world.spec import BoardSpec


@dataclass
class EnvConfig:
    spec: BoardSpec = dc_field(default_factory=BoardSpec)
    world: WorldConfig = dc_field(default_factory=WorldConfig)
    generator: GeneratorConfig = dc_field(default_factory=GeneratorConfig)
    reward: RewardConfig = dc_field(default_factory=RewardConfig)
    #: Hard cap on env steps per episode. An episode also ends early once every
    #: net is resolved, which is the common case on easy boards.
    max_episode_steps: int = 512
    #: Recompute the straight-line demand channel every N steps. It only
    #: changes when a net is retired, so per-step is pure waste.
    demand_refresh: int = 16
    seed: int = 0


class NeuroRouteEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.world = BatchedRouterWorld(cfg.spec, cfg.world)
        self.device = self.world.device
        self._seed = cfg.seed
        self._t = 0
        self._demand: torch.Tensor | None = None
        self._boards: list = []
        # The observation the caller last acted on. Its `bearing` defines what
        # `direction=0` meant for the action about to be applied, so the engine
        # must resolve the move against *that* bearing, not against a freshly
        # recomputed one -- they can differ, because copper moved in between.
        self._obs = None

    # -- lifecycle ----------------------------------------------------------

    def reset(self, seeds: Iterable[int] | None = None) -> Observation:
        cfg = self.cfg
        B = cfg.world.batch_size
        if seeds is None:
            seeds = [self._seed + i for i in range(B)]
            self._seed += B
        seeds = list(seeds)

        self._boards = [generate_board(cfg.spec, cfg.generator, s) for s in seeds]
        self.world.load(self._boards)
        self._t = 0
        self._refresh_demand()

        # Seed the head slots with the shortest pending nets. The learned
        # scheduler takes over from step 1; this only decides the very first
        # assignment, before any observation exists to condition on.
        self.world.assign(self._greedy_schedule())
        self._obs = build_observation(self.world, self._demand)
        return self._obs

    def _refresh_demand(self) -> None:
        fields = []
        for bi, board in enumerate(self._boards):
            pending = [
                n
                for ni, n in enumerate(board.netlist.nets)
                if ni < self.cfg.world.max_nets
                and int(self.world.net_status[bi, ni]) in (0, 1)
            ]
            from neuroroute.world.spec import Netlist

            fields.append(straight_line_demand(self.cfg.spec, Netlist(pending)))
        d = torch.from_numpy(np.stack(fields, 0)).to(self.device)
        self._demand = (d / d.amax().clamp_min(1.0)).float()

    # -- scheduling ---------------------------------------------------------

    def _greedy_schedule(self) -> torch.Tensor:
        """Shortest-pending-net-first, one per idle slot.

        Used to seed `reset()` and as the `--no-learned-scheduler` fallback.
        Shortest-first is the heuristic `docs/RL_PLAN.md` settled on, kept here
        as the baseline the learned scheduler has to beat rather than as the
        permanent answer.
        """
        w = self.world
        B, K = w.head_net.shape
        src = w.net_src[:, :, 0, 1:].float()
        dst = w.net_dst[:, :, 0, 1:].float()
        span = torch.linalg.vector_norm(dst - src, dim=-1)
        span = torch.where(w.pending_mask(), span, torch.full_like(span, float("inf")))

        order = span.argsort(dim=1)                      # (B, N)
        out = torch.full((B, K), -1, dtype=torch.long, device=self.device)
        idle = w.idle_slots()
        rank = idle.cumsum(dim=1) - 1
        take = idle & (rank < order.shape[1])
        picks = order.gather(1, rank.clamp(0, order.shape[1] - 1))
        bb = torch.arange(B, device=self.device).view(B, 1).expand(B, K)
        finite = torch.isfinite(span[bb, picks])
        return torch.where(take & finite, picks, out)

    # -- step ---------------------------------------------------------------

    def step(self, action: dict[str, torch.Tensor]) -> tuple[Observation, torch.Tensor, torch.Tensor, dict]:
        """Advance one step.

        Returns ``(obs, reward, done, info)`` where `reward` is ``(B, K)`` --
        per head, not per board. Credit lands on the head that earned it, which
        is the only way `K` simultaneous heads can share one value function
        without smearing every head's outcome across all the others.
        """
        w = self.world
        cfg = self.cfg

        res: StepResult = w.step(
            direction=action["direction"],
            step_class=action["step"],
            layer_action=action["layer"],
            via_class=action["via"],
            width_class=action["width"],
            couple=action["couple"],
            bearing=self._obs.bearing if self._obs is not None else None,
        )
        reward = step_reward(res, cfg.reward)

        # (B,) bool -- which boards actually ripped a net up this step. Board
        # level, not per-head: ripup is a board-level decision (like
        # scheduling), and its cost belongs on the board-level reward stream
        # that trains it, not smeared into the per-head geometry reward those
        # heads did not cause.
        did_ripup = torch.zeros(w.head_net.shape[0], dtype=torch.bool, device=self.device)
        if "ripup" in action:
            did_ripup = w.ripup(action["ripup"])

        sched = action.get("schedule")
        if sched is None:
            sched = self._greedy_schedule()
        w.assign(sched)

        self._t += 1
        if self._t % cfg.demand_refresh == 0:
            self._refresh_demand()

        resolved = (~w.pending_mask()).all(dim=1) & (w.head_net < 0).all(dim=1)
        timeout = self._t >= cfg.max_episode_steps
        done = resolved | torch.full_like(resolved, bool(timeout))

        info: dict[str, torch.Tensor] = {
            "rejected": res.rejected,
            "moved": res.moved,
            "via_placed": res.via_placed,
            "arrived": res.arrived,
            "exhausted": res.exhausted,
            "active": res.active,
            "nets_done": res.nets_done,
            "nets_failed": res.nets_failed,
            "completion": w.completion(),
            "did_ripup": did_ripup,
        }
        if bool(done.all()):
            term, metrics = terminal_reward(w, cfg.reward)
            info["terminal_reward"] = term
            info.update({f"final/{k}": v for k, v in metrics.items()})

        self._obs = build_observation(w, self._demand)
        return self._obs, reward, done, info

    # -- convenience --------------------------------------------------------

    def rejected_action_rate(self, info: dict) -> torch.Tensor:
        """Fraction of *acting* heads whose action was rejected.

        The direct analogue of `scripts/`'s Rejected-Action Rate metric on the
        raster thread, where it went 1.51% -> 0.40% when the raycast
        suppression landed [LIVE]. Reported every update so a regression in the
        suppression path is visible immediately rather than as a slow drift in
        completion rate.
        """
        act = info["active"].float().sum().clamp_min(1.0)
        return info["rejected"].float().sum() / act
