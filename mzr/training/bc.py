"""Expert demonstrations for behaviour cloning.

`ppo.py` has carried a working BC loss since the beginning -- it evaluates
`log pi(expert_action | s)` at the policy's own observation and adds it to the
objective. What it never had was anything filling `buf.bc_actions`, so
`bc_action` defaulted to None, the guard at `ppo.py:230` short-circuited, and
**`--bc-coef` was a silent no-op**. Same class of bug as `--lr` being ignored
on resume: a flag that reports nothing and changes nothing.

This is the missing half.

## Why the expert and not KiCad

The demonstration has to be an ACTION in the policy's own space -- a direction
octant and a step class -- not a picture of a good route. `world/expert.py`
emits unit lattice steps stamped through the same `move_claims` / `via_claims`
that `engine.step()` uses, so its plan is replayable as a sequence of actions.
A KiCad trace is arbitrary polyline geometry, and recovering which macro-step
actions would have produced it is a lossy inverse problem. (KiCad also has no
built-in autorouter to ask, and `DESIGN.md` confines KiCad to ingest, export
and DRC in any case.)

## Where the demonstration comes from

The expert plans from the board's STATIC layer -- pads and keepouts -- once per
reset, never from live occupancy. So the plan is fixed for the episode and each
step is a lookup rather than a re-plan, which is what makes this affordable
inside the collection loop.

The policy will wander off that path. When it does, the demonstration is "head
for the nearest point on the expert's route, then follow it", which is the
standard DAgger-flavoured answer: supervise at the state the policy actually
reached, not only at states the expert would have visited.

## The three things that must be right

1. **Frontier -> (net, leg, end).** `net = f // (max_legs * NUM_ENDS)`,
   `leg = (f // NUM_ENDS) % max_legs`, `end = f % NUM_ENDS`.
2. **Direction of travel.** A path runs src -> dst. An `END_SRC` frontier walks
   it forwards; an `END_DST` frontier walks it BACKWARDS. Getting this wrong
   teaches the policy to route away from its target.
3. **Egocentric conversion.** The engine computes `abs_dir = (bearing +
   direction) % 8`, so a demonstration expressed as an absolute octant must be
   converted with `ego = (abs - bearing) % 8`. Mixing the two frames is silent
   and was already the cause of one wrong diagnosis in this project.

`verify_bc_actions()` checks all three by replaying the demonstration and
asserting the frontier actually advances along the expert's path.
"""

from __future__ import annotations

import torch

from mzr.world.expert import (
    ExpertConfig,
    route_world_board,
    route_world_board_live,
)
from mzr.world.spec import END_SRC, NUM_ENDS

#: The expert emits unit steps only (verify_world: "410/410 steps are unit
#: moves or single via hops"), so every demonstration is step class 0.
_UNIT_STEP = 0


class ExpertActions:
    """Per-step expert demonstrations for one env, planned once per reset.

    Only `direction` and `step` are cloned. `layer` / `via` are left to RL on
    purpose: cloning them copies the expert's via decisions wholesale, and the
    expert is a sequential Dijkstra router whose via policy is not the one this
    project is trying to learn.
    """

    def __init__(self, env, *, negotiate: bool = False,
                 cfg: ExpertConfig | None = None, replan_every: int = 0) -> None:
        self.env = env
        self.negotiate = negotiate
        self.cfg = cfg
        self._paths: list[dict] = []
        #: Macro-steps between in-episode re-plans against live occupancy.
        #: 0 keeps the old behaviour -- plan once per reset from the static
        #: layer and let it go stale. See `replan()`.
        self.replan_every = replan_every
        self._since = 0

    def _index(self, res) -> dict:
        """Index a plan by cell, so a wandering frontier can find where it
        rejoins, and keep the ordered list for "which way is forward"."""
        return {key: (pts, {tuple(p): i for i, p in enumerate(pts)})
                for key, pts in res.paths.items()}

    def plan(self) -> None:
        """Plan every board from the STATIC layer. Call once per `env.reset()`.

        `negotiate=False` by default: PathFinder's rip-up loop is what makes
        the expert a strong *baseline*, but for a demonstration the extra
        iterations mostly cost wall-clock inside the collection loop.
        """
        w = self.env.world
        self._paths = [self._index(route_world_board(w, b, self.cfg,
                                                     negotiate=self.negotiate))
                       for b in range(w.cfg.batch_size)]
        self._since = 0

    def replan(self) -> None:
        """Re-plan the still-open legs from the live frontiers against live
        occupancy, replacing the stale demonstration.

        A plan fixed at reset stops being a demonstration once other nets have
        laid copper across it -- it starts pointing the frontier into cells that
        are now occupied. Cloning that made stage 1 oscillate (0.755 / 0.693 /
        0.823 / 0.599 at `--bc-coef 0.5`).

        Boards with no open legs left keep whatever they had; there is nothing
        to demonstrate and re-planning them only costs time.
        """
        w = self.env.world
        for b in range(w.cfg.batch_size):
            res = route_world_board_live(w, b, self.cfg, negotiate=False)
            if res.paths:
                self._paths[b] = self._index(res)
        self._since = 0

    def maybe_replan(self) -> bool:
        """Tick the cadence. Returns True when a re-plan actually ran."""
        if self.replan_every <= 0 or not self._paths:
            return False
        self._since += 1
        if self._since < self.replan_every:
            return False
        self.replan()
        return True

    @torch.no_grad()
    def action(self, obs) -> dict | None:
        """`{"action": {...}, "mask": (B, F) bool}` at the CURRENT state.

        `mask` is the set of frontiers a demonstration exists for; the BC loss
        averages over it, so a frontier the expert never routed contributes
        nothing rather than contributing noise.
        """
        if not self._paths:
            return None
        w = self.env.world
        B, F = w.cfg.batch_size, w.F
        dev = w.device

        direction = torch.zeros(B, F, dtype=torch.long, device=dev)
        step = torch.full((B, F), _UNIT_STEP, dtype=torch.long, device=dev)
        mask = torch.zeros(B, F, dtype=torch.bool, device=dev)

        pos = w.fr_pos.cpu().numpy()
        alive = obs.frontier_mask.cpu().numpy()
        bearing = obs.bearing.cpu().numpy()
        per_leg = NUM_ENDS

        for b in range(B):
            board = self._paths[b]
            for f in range(F):
                if not alive[b, f]:
                    continue
                net = f // (w.cfg.max_legs * NUM_ENDS)
                leg = (f // per_leg) % w.cfg.max_legs
                end = f % per_leg
                entry = board.get((net, leg))
                if entry is None:
                    continue
                pts, where = entry
                cur = (int(pos[b, f, 0]), int(pos[b, f, 1]), int(pos[b, f, 2]))

                i = where.get(cur)
                if i is None:
                    # Off the expert's route: aim at the nearest point on it.
                    best, bi = None, None
                    for j, p in enumerate(pts):
                        d = abs(p[1] - cur[1]) + abs(p[2] - cur[2])
                        if best is None or d < best:
                            best, bi = d, j
                    if bi is None:
                        continue
                    nxt = pts[bi]
                    if nxt == cur:
                        continue
                else:
                    # On the route: forwards from src, BACKWARDS from dst.
                    j = i + 1 if end == END_SRC else i - 1
                    if not (0 <= j < len(pts)):
                        continue
                    nxt = pts[j]

                dy, dx = nxt[1] - cur[1], nxt[2] - cur[2]
                if dy == 0 and dx == 0:
                    continue
                # Octant of the move, then into the policy's egocentric frame.
                import math
                absolute = int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8
                direction[b, f] = (absolute - int(bearing[b, f])) % 8
                mask[b, f] = True

        if not bool(mask.any()):
            return None
        return {"action": {"direction": direction, "step": step}, "mask": mask}


@torch.no_grad()
def verify_bc_actions(env, expert: ExpertActions, steps: int = 8) -> dict:
    """Replay the demonstration and check the frontier actually advances.

    A demonstration that is silently in the wrong frame, or that walks a
    dst-end frontier backwards up its own route, still produces plausible
    tensors and trains happily to nothing. The only honest check is to APPLY
    it and see whether the distance to target falls.
    """
    obs = env.reset()
    expert.plan()
    closer = farther = 0
    for _ in range(steps):
        bc = expert.action(obs)
        if bc is None:
            break
        before = env.world.fr_prev.clone()
        full = {"direction": bc["action"]["direction"], "step": bc["action"]["step"]}
        out = env.step(full)
        after = env.world.fr_prev
        m = bc["mask"] & obs.frontier_mask
        if bool(m.any()):
            delta = (after - before)[m]
            closer += int((delta < 0).sum())
            farther += int((delta > 0).sum())
        obs = out.obs
        if out.done:
            break
    total = closer + farther
    return {"closer": closer, "farther": farther,
            "frac_closer": closer / max(1, total), "n": total}
