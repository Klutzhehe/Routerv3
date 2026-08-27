"""`NeuroRoutePolicy` -- encoder, forecaster and the action heads.

Action structure, and why it is what it is:

* **Factorised, not a product space.** The flat product of every action
  dimension is ~9k discrete actions; factorised it is ~31 logits. A 9k-way
  softmax over a space where most combinations are nonsense is not a learnable
  object.
* **`step` is conditioned on `direction`.** These two are the one pair that
  genuinely interacts: a direction can be clear for 2 cells and blocked at 8,
  so independent marginals would call the pair "safe" while the specific move
  collides. That exact gap is what `docs/WORLD_MODEL_SPATIAL_DESIGN.md`'s
  addendum was written about (per-direction granularity was too coarse and
  showed up as repeated same-spot rejections). Sampling direction first and
  conditioning the step logits on it removes the problem structurally rather
  than compensating for it with a penalty.
* **Log-probs are masked to the dimensions that actually mattered.** When the
  policy places a via, the engine ignores `direction`/`step`/`width`; counting
  their log-probs would inject pure gradient noise on dimensions that had no
  effect on the outcome.

Two mechanisms are carried over from the raster thread because they are the
ones with measured wins, and both are deliberately **not learned**:

* the fixed per-(direction, step) safety suppression -- Rejected-Action Rate
  1.51% -> 0.40% [LIVE];
* the near-zero actor init, so an untrained policy emits action 0 and
  therefore *is* the greedy router, starting training at the baseline instead
  of below it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from neuroroute.env.observation import Observation
from neuroroute.models.encoder import FieldEncoder, HeadCropEncoder, NetEncoder
from neuroroute.models.forecaster import FORECAST_CHANNELS, Forecast, FutureFieldPredictor
from neuroroute.world.spec import NUM_DIRECTIONS, NUM_STEPS

#: Fixed, non-learned logit penalty for an action the geometry proves will
#: collide. A constant, not an `nn.Parameter`: this project has direct prior
#: history (`models/router_policy.py`'s init comment) of a learned bias being
#: outgrown by the weight-driven logits it competed with, becoming negligible
#: exactly when it mattered most. Applied after the head, so training can route
#: around it but never erode it. If every option is unsafe the penalty is
#: uniform, and a uniform shift changes neither softmax nor argmax -- so it can
#: only ever discriminate when it has real information to add.
SAFETY_SUPPRESSION = 8.0


@dataclass
class PolicyOutput:
    actions: dict[str, torch.Tensor]
    log_prob: torch.Tensor      # (B, K)
    entropy: torch.Tensor       # (B, K)
    value: torch.Tensor         # (B, K)
    schedule_log_prob: torch.Tensor  # (B,)
    ripup_log_prob: torch.Tensor     # (B,)
    board_value: torch.Tensor        # (B,) -- schedule's and ripup's shared
                                      # critic; see _ripup's docstring for why
                                      # neither ever trained before this
    forecast: Forecast
    latent: torch.Tensor


class NeuroRoutePolicy(nn.Module):
    def __init__(
        self,
        field_channels: int,
        head_features: int,
        net_features: int,
        num_layers: int,
        num_via_classes: int,
        num_width_classes: int,
        width: int = 64,
        head_width: int = 256,
        crop: int = 16,
        use_forecast: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_forecast = use_forecast

        self.field = FieldEncoder(field_channels, width=width)
        self.crop = HeadCropEncoder(field_channels, num_layers, crop=crop)
        self.nets = NetEncoder(net_features, self.field.global_dim)
        self.forecaster = FutureFieldPredictor(width)

        per_head = (
            width                                  # latent gathered at the head cell
            + self.crop.out_dim                    # native-resolution local crop
            + head_features                        # exact geometry: raycast/safety/geodesic
            + self.field.global_dim                # whole-board context
            + (FORECAST_CHANNELS if use_forecast else 0)
        )
        self.trunk = nn.Sequential(
            nn.Linear(per_head, head_width),
            nn.SiLU(),
            nn.Linear(head_width, head_width),
            nn.SiLU(),
            # LayerNorm is load-bearing, not decoration. The action heads are
            # initialised at gain 0.01 so that the BIAS decides the untrained
            # action -- that is what makes an untrained policy behave like the
            # greedy router. Without normalisation the trunk's activations are
            # large enough that 0.01 * sqrt(head_width) * |activation| still
            # rivals the 2.0 direction bias, content noise wins the argmax, and
            # the deterministic untrained policy collapses: measured at 84%
            # rejected actions and 6.7% completion against greedy's 0.16% and
            # 27.5%. With the norm the head start is real.
            nn.LayerNorm(head_width),
        )

        self.h_dir = nn.Linear(head_width, NUM_DIRECTIONS)
        self.dir_embed = nn.Embedding(NUM_DIRECTIONS, 32)
        self.h_step = nn.Linear(head_width + 32, NUM_STEPS)
        self.h_layer = nn.Linear(head_width, 1 + num_layers)
        self.h_via = nn.Linear(head_width, num_via_classes)
        self.h_width = nn.Linear(head_width, num_width_classes)
        self.h_couple = nn.Linear(head_width, 2)
        self.h_value = nn.Linear(head_width, 1)

        self.h_schedule = nn.Sequential(
            nn.Linear(self.nets.out_dim + self.field.global_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        # Same shape as h_schedule -- a per-net score over the OPPOSITE mask
        # (routed, not pending). See _ripup for why it needs its own
        # NetEncoder pass rather than reusing the scheduler's tokens.
        self.h_ripup = nn.Sequential(
            nn.Linear(self.nets.out_dim + self.field.global_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        # The learned logit for "rip up nothing this step" -- see _ripup.
        self.h_ripup_none = nn.Parameter(torch.tensor(4.0))
        # Schedule and ripup are board-level decisions; nothing about them is
        # per-head, so they need their own critic rather than the per-head
        # h_value (which is masked to active heads and is not meaningful on,
        # e.g., a board that is fully idle between nets -- exactly when a
        # scheduling decision matters most).
        self.h_board_value = nn.Linear(self.field.global_dim, 1)
        self._init_actor()

    def _init_actor(self) -> None:
        """Near-zero weights, with the bias tilted toward the greedy action.

        Every action head starts almost content-independent, and the bias
        makes the resulting near-uniform-input choice be: go in the reference
        bearing direction, stay on this layer, keep a pair coupled. That is
        exactly the greedy baseline, so training starts *at* it. Losing this
        init throws away a free head start -- which is the whole reason the
        egocentric frame was chosen in the first place (docs/HANDOVER.md).
        """
        for head in (self.h_dir, self.h_step, self.h_layer, self.h_via, self.h_width, self.h_couple):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.h_dir.bias[0] = 2.0        # index 0 == down the geodesic gradient
            # Index 0 == stay on this layer (no via). The bias is derived from
            # the layer count so that P(stay) at init is ~0.75 REGARDLESS of L:
            # softmax with `num_layers` via options and bias b gives
            # P(stay) = e^b / (e^b + L), so b = log(3L) fixes P(stay) at 3/4.
            #
            # A fixed constant cannot do that. 4.0 was chosen when the problem
            # was an untrained policy attempting mostly-impossible through-vias
            # (92.6% rejected actions), and it over-corrected: it put P(stay)
            # at ~0.97 on a 2-layer board, and under argmax -- which is what
            # eval uses -- it meant "never place a via" until the learned
            # logits could overcome 4.0.
            #
            # Measured consequence on stage 0: 4/16 boards have their two pads
            # on different layers and are unroutable without a via, so a
            # via-less policy is capped at exactly 75.0% -- which is precisely
            # where held-out eval sat for three consecutive evals. The reward
            # was never the problem: a correct via already scores +0.190
            # against +0.021 for a lateral move. The policy simply never
            # proposed one.
            #
            # The real defence against impossible vias is `via_safe`
            # suppression in `_suppressed_layer_logits`, which is exact and
            # non-learned. This bias only has to stop vias being the *default*.
            self.h_layer.bias[0] = math.log(3.0 * self.num_layers)
            self.h_couple.bias[1] = 1.0     # keep differential pairs coupled by default
            # Index 0 means "the width this net actually requires" -- the
            # engine takes max(action, net_width), so class 0 is never a
            # violation. Without this bias the near-zero weights decide the
            # argmax arbitrarily, and an untrained policy picked class 1 on
            # 612 of 627 actions: 0.3 mm traces, three lattice cells wide,
            # on a congested board. Almost everything collided (88% rejected
            # actions) while `Observation.safety` -- computed at the net's
            # REQUIRED width -- was still reporting those moves as clear.
            # Widening is a real capability, but it has to be a decision the
            # policy makes on purpose, not its default.
            self.h_width.bias[0] = 2.0
            self.h_via.bias[0] = 2.0        # smallest via unless asked otherwise
            self.h_step.bias[0] = 0.5       # short steps are the safe default
        nn.init.orthogonal_(self.h_value.weight, gain=1.0)
        nn.init.zeros_(self.h_value.bias)
        nn.init.orthogonal_(self.h_board_value.weight, gain=1.0)
        nn.init.zeros_(self.h_board_value.bias)

    # -- feature assembly ---------------------------------------------------

    def _gather_at_heads(self, grid: torch.Tensor, head_pos: torch.Tensor, fine_hw: tuple[int, int]) -> torch.Tensor:
        """Sample a (B, C, L, h, w) grid at each head's cell. -> (B, K, C)"""
        B, C, L, h, w = grid.shape
        K = head_pos.shape[1]
        sy = max(1, fine_hw[0] // h)
        sx = max(1, fine_hw[1] // w)
        b = torch.arange(B, device=grid.device).view(B, 1).expand(B, K)
        lay = head_pos[..., 0].clamp(0, L - 1)
        gy = (head_pos[..., 1] // sy).clamp(0, h - 1)
        gx = (head_pos[..., 2] // sx).clamp(0, w - 1)
        return grid[b, :, lay, gy, gx]

    def encode(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Forecast]:
        z, g = self.field(obs.field)
        forecast = self.forecaster(z)

        B, K = obs.head_mask.shape
        fine_hw = (obs.field.shape[-2], obs.field.shape[-1])

        parts = [
            self._gather_at_heads(z, obs.head_pos, fine_hw),
            self.crop(obs.field, obs.head_pos, obs.head_mask),
            obs.heads,
            g.unsqueeze(1).expand(B, K, g.shape[-1]),
        ]
        if self.use_forecast:
            # Detached: the forecaster is trained by its own supervised losses
            # on completed episodes. Letting the RL gradient reshape it would
            # turn a model of the future into whatever makes the critic's job
            # easiest this update.
            parts.append(self._gather_at_heads(forecast.as_channels().detach(), obs.head_pos, fine_hw))

        feat = self.trunk(torch.cat(parts, dim=-1))
        return feat, g, z, forecast

    # -- action -------------------------------------------------------------

    def _suppressed_dir_logits(self, feat: torch.Tensor, safety: torch.Tensor) -> torch.Tensor:
        logits = self.h_dir(feat)
        any_safe = safety.any(dim=-1)                       # (B, K, D)
        return logits - SAFETY_SUPPRESSION * (~any_safe).float()

    def _suppressed_layer_logits(self, feat: torch.Tensor, via_safe: torch.Tensor) -> torch.Tensor:
        """Layer logits with impossible vias suppressed.

        Index 0 is "stay", which is always available. Index `j+1` places a via
        to layer `j`, and a through via has to be free on every layer at once
        -- on a populated board most of those are impossible. Suppressing them
        with the same fixed constant used for direction and step took the
        untrained rejected-action rate from 92.6% to a fraction of that, and it
        is not something the policy should have to learn from reward when the
        geometry is available for free.
        """
        logits = self.h_layer(feat)
        stay = torch.zeros_like(logits[..., :1])
        via = -SAFETY_SUPPRESSION * (~via_safe).float()
        return logits + torch.cat([stay, via], dim=-1)

    def _suppressed_step_logits(
        self, feat: torch.Tensor, safety: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        emb = self.dir_embed(direction)
        logits = self.h_step(torch.cat([feat, emb], dim=-1))
        chosen = safety.gather(
            2, direction.unsqueeze(-1).unsqueeze(-1).expand(*direction.shape, 1, safety.shape[-1])
        ).squeeze(2)                                        # (B, K, NUM_STEPS)
        return logits - SAFETY_SUPPRESSION * (~chosen).float()

    def act(self, obs: Observation, deterministic: bool = False) -> PolicyOutput:
        feat, g, z, forecast = self.encode(obs)

        dir_logits = self._suppressed_dir_logits(feat, obs.safety)
        d_dist = Categorical(logits=dir_logits)
        direction = dir_logits.argmax(-1) if deterministic else d_dist.sample()

        step_logits = self._suppressed_step_logits(feat, obs.safety, direction)
        s_dist = Categorical(logits=step_logits)
        step = step_logits.argmax(-1) if deterministic else s_dist.sample()

        dists = {
            "layer": Categorical(logits=self._suppressed_layer_logits(feat, obs.via_safe)),
            "via": Categorical(logits=self.h_via(feat)),
            "width": Categorical(logits=self.h_width(feat)),
            "couple": Categorical(logits=self.h_couple(feat)),
        }
        actions = {"direction": direction, "step": step}
        for k, dist in dists.items():
            actions[k] = dist.logits.argmax(-1) if deterministic else dist.sample()

        logp, entropy = self._score(
            d_dist, s_dist, dists, actions, obs.head_is_pair, obs.head_mask
        )
        value = self.h_value(feat).squeeze(-1) * obs.head_mask.float()
        board_value = self.h_board_value(g).squeeze(-1)

        schedule, sched_logp = self._schedule(obs, g, deterministic)
        actions["schedule"] = schedule

        ripup, ripup_logp = self._ripup(obs, g, deterministic)
        actions["ripup"] = ripup

        return PolicyOutput(
            actions=actions,
            log_prob=logp,
            entropy=entropy,
            value=value,
            schedule_log_prob=sched_logp,
            ripup_log_prob=ripup_logp,
            board_value=board_value,
            forecast=forecast,
            latent=z,
        )

    def evaluate(
        self,
        obs: Observation,
        actions: dict[str, torch.Tensor],
        return_forecast: bool = False,
    ):
        """Re-score stored actions under the current parameters, for PPO.

        `return_forecast` hands back the forecaster output from the *same*
        encode. The forecaster's supervised loss needs it, and encoding twice
        would double the cost of the single most expensive part of the model
        for information already computed.

        Also re-scores `schedule` and `ripup` (via each one's `given=`) so
        both board-level heads actually receive a PPO gradient here -- see
        `_schedule`'s and `_ripup`'s docstrings for why that was missing.
        """
        feat, g, _z, forecast = self.encode(obs)

        d_dist = Categorical(logits=self._suppressed_dir_logits(feat, obs.safety))
        s_dist = Categorical(
            logits=self._suppressed_step_logits(feat, obs.safety, actions["direction"])
        )
        dists = {
            "layer": Categorical(logits=self._suppressed_layer_logits(feat, obs.via_safe)),
            "via": Categorical(logits=self.h_via(feat)),
            "width": Categorical(logits=self.h_width(feat)),
            "couple": Categorical(logits=self.h_couple(feat)),
        }
        logp, entropy = self._score(
            d_dist, s_dist, dists, actions, obs.head_is_pair, obs.head_mask
        )
        value = self.h_value(feat).squeeze(-1) * obs.head_mask.float()
        board_value = self.h_board_value(g).squeeze(-1)

        _, sched_logp = self._schedule(obs, g, deterministic=False, given=actions["schedule"])
        _, ripup_logp = self._ripup(obs, g, deterministic=False, given=actions["ripup"])

        if return_forecast:
            return logp, entropy, value, sched_logp, ripup_logp, board_value, forecast
        return logp, entropy, value, sched_logp, ripup_logp, board_value

    def _score(self, d_dist, s_dist, dists, actions, is_pair, head_mask):
        """Sum log-probs and entropies over the dimensions that had an effect.

        `layer > 0` means a via was placed, and the engine then ignores
        direction, step and width entirely. Including their log-probs would
        credit or blame the policy for choices that provably did not influence
        the outcome -- pure variance, and it grows with the number of action
        dimensions, which is exactly the direction this action space went.
        """
        placed_via = actions["layer"] > 0
        moved = ~placed_via
        relevant = {
            "direction": moved,
            "step": moved,
            "width": moved,
            "via": placed_via,
            "layer": torch.ones_like(placed_via),
            "couple": is_pair,
        }

        logp = d_dist.log_prob(actions["direction"]) * relevant["direction"].float()
        ent = d_dist.entropy() * relevant["direction"].float()
        logp = logp + s_dist.log_prob(actions["step"]) * relevant["step"].float()
        ent = ent + s_dist.entropy() * relevant["step"].float()
        for k, dist in dists.items():
            m = relevant[k].float()
            logp = logp + dist.log_prob(actions[k]) * m
            ent = ent + dist.entropy() * m

        m = head_mask.float()
        return logp * m, ent * m

    # -- scheduler ----------------------------------------------------------

    def _schedule(
        self, obs: Observation, g: torch.Tensor, deterministic: bool,
        given: torch.Tensor | None = None,
    ):
        """Choose a pending net for each idle head slot.

        Learned rather than shortest-first. At tens of nets ordering barely
        matters and `docs/RL_PLAN.md` was right to fix it as a heuristic and
        remove the combinatorial dimension; at thousands of nets it is most of
        the problem, because the cost of routing a net is dominated by what was
        routed before it.

        Slots choose independently, which can pick the same net twice.
        `BatchedRouterWorld.assign` drops the duplicate, so the only cost is a
        wasted slot -- which the reward already penalises, so the policy has a
        gradient telling it not to. That is much simpler than a
        sample-without-replacement scheme and costs almost nothing.

        `given`, if provided, re-scores a STORED `(B, K)` action instead of
        sampling a new one -- this is what lets `evaluate()` compute this
        head's log-prob under the CURRENT parameters for PPO. Before this
        existed, nothing ever called `_schedule` with a stored action:
        `RolloutBuffer` explicitly dropped "schedule" from what it kept
        (`if k != "schedule"`), so `h_schedule`'s output never reached any
        loss and the scheduler never actually trained, on any run in this
        project's history.
        """
        B, K = obs.head_mask.shape
        tokens = self.nets(obs.nets, g, obs.net_mask)
        ctx = g.unsqueeze(1).expand(B, tokens.shape[1], g.shape[-1])
        scores = self.h_schedule(torch.cat([tokens, ctx], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~obs.net_mask, float("-inf"))

        none_pending = ~obs.net_mask.any(dim=1)
        safe = scores.masked_fill(none_pending.unsqueeze(1), 0.0)
        dist = Categorical(logits=safe)

        idle = ~obs.head_mask
        picks = []
        logp = torch.zeros(B, device=scores.device)
        for k in range(K):
            if given is None:
                choice = safe.argmax(-1) if deterministic else dist.sample()
            else:
                # `given[:, k] == -1` means "no pick happened here" for a
                # reason that has nothing to do with net index 0 specifically
                # (the slot wasn't idle, or nothing was pending at all) --
                # `clamp_min(0)` just needs SOME in-range index to re-score,
                # and can land on net 0 even when `safe[..., 0]` is `-inf`
                # (net 0 exists but is not pending for this board). Unlike
                # `_ripup`, there is no dedicated always-finite "none" index
                # here to route to instead, so the mask below has to be
                # applied via `where`, not multiplication -- see the comment
                # on `logp` just below for why that distinction is load-bearing.
                choice = given[:, k].clamp_min(0)
            take = idle[:, k] & ~none_pending
            picks.append(torch.where(take, choice, torch.full_like(choice, -1)))
            # NOT `dist.log_prob(choice) * take.float()`. When `choice` lands
            # on a masked (-inf-scored) index -- which only the `given`
            # branch above can cause -- `log_prob` is itself `-inf`, and
            # `-inf * 0.0 = NaN` in IEEE 754 regardless of `take` being
            # False. `torch.where` selects between two already-computed
            # finite-or-not values without multiplying either by the other,
            # so it cannot produce that NaN. Found live on the very first
            # GPU update once `given` re-scoring started actually running --
            # not exercised by any local CPU smoke test at toy scale, which
            # is the honest reason it wasn't caught before real training hit
            # a wide enough batch to make the coincidence likely.
            step_logp = dist.log_prob(choice)
            logp = logp + torch.where(take, step_logp, torch.zeros_like(step_logp))
        return torch.stack(picks, dim=1), logp

    # -- rip-up ---------------------------------------------------------------

    def _ripup(
        self, obs: Observation, g: torch.Tensor, deterministic: bool,
        given: torch.Tensor | None = None,
    ):
        """Choose at most one already-routed net per board to rip up.

        Complementary to the scheduler, over the opposite mask: the scheduler
        decides which PENDING net fills an idle slot; this decides whether an
        earlier decision -- a net that finished routing but now sits
        somewhere blocking others -- should be undone and returned to
        pending. `BatchedRouterWorld.ripup()` and the environment wiring for
        it (`NeuroRouteEnv.step`) already existed; the policy never emitted
        the action that would use them.

        A SEPARATE `NetEncoder` pass is used, not the scheduler's `tokens`:
        `NetEncoder.forward` zeroes its OUTPUT outside `mask` (see its
        docstring), so reusing the scheduler's pending-masked tokens would
        hand this head an all-zero embedding for every net it is actually
        choosing between.

        `-1` ("rip up nothing") is index `N` -- one past the last real net --
        scored by a single LEARNED bias (`h_ripup_none`) rather than folded
        into an ordinary masked softmax. Two reasons: an ordinary masked
        categorical would force a pick among whatever routed nets exist even
        when none of them should be touched, and a board with zero routed
        nets would leave every real option at `-inf` with nothing finite to
        fall back on. Routing `none` through its own always-finite logit
        avoids both. Initialised strongly positive (matching every other
        head's tuned-default-bias pattern) so an untrained policy never rips
        anything up, keeping "untrained policy == greedy baseline" true here
        too.

        `given` re-scores a stored `(B,)` action the same way `_schedule`'s
        `given` does, for the same reason -- see `_schedule`'s docstring.
        """
        B, N = obs.routed_mask.shape
        tokens = self.nets(obs.nets, g, obs.routed_mask)
        ctx = g.unsqueeze(1).expand(B, N, g.shape[-1])
        raw = self.h_ripup(torch.cat([tokens, ctx], dim=-1)).squeeze(-1)
        raw = raw.masked_fill(~obs.routed_mask, float("-inf"))
        none_score = self.h_ripup_none.expand(B, 1)
        scores = torch.cat([raw, none_score], dim=1)
        dist = Categorical(logits=scores)

        if given is None:
            choice = scores.argmax(-1) if deterministic else dist.sample()
        else:
            choice = torch.where(given < 0, torch.full_like(given, N), given)
        logp = dist.log_prob(choice)
        net_idx = torch.where(choice == N, torch.full_like(choice, -1), choice)
        return net_idx, logp
