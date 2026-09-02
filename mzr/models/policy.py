"""`PriorPolicy` -- the search-free policy/value network.

This is what stages 0-1 train and what stage 3's MuZero search uses as its
prior. It is deliberately *not* the MuZero net: `h`/`g`/`f` and the latent
dynamics come at stage 2. If simultaneous growth cannot beat sequential
routing with just this, adding search is adding it to a broken foundation --
which is why the build order puts a working prior first.

Shape discipline, from `mzr/DESIGN.md` section 5 and the four failed lookahead
attempts it summarises:

* The **field** is encoded once per macro-step and shared by every frontier --
  the expensive convolution does not scale with frontier count.
* Every frontier is a **token**. The token network is cardinality-agnostic:
  attention with a padding mask over `F`, a shared MLP head. A netlist of 5 or
  5000 is the same code and the same weights. (Bucketed local attention for the
  O(F^2) term switches on at stage 4; full attention is fine at F <= 20.)
* Exact local geometry -- raycasts, safety, geodesic and price lookahead --
  reaches the head as **input features**, never as something decoded from a
  pooled embedding. That decode is what failed four times.

Untrained, this policy is the greedy baseline: near-zero head weights mean the
**bias** is the whole signal at init (a lesson that cost real time in
`neuroroute/` -- a zero-bias width head picked 3-cell traces on 88% of
actions). Biases are set so argmax is "step one cell down the geodesic
gradient, stay on this layer, minimum width" -- exactly `world.baselines.greedy`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mzr.env.observation import FIELD_CHANNELS, Observation, frontier_feature_dim
from mzr.models.encoder import FieldEncoder, FrontierCropEncoder, _norm
from mzr.world.spec import NUM_DIRECTIONS, NUM_KINDS, NUM_STEPS

_BIG = 30.0  # logit suppression magnitude for a fixed, non-learned mask


@dataclass
class PolicyOutput:
    #: (B, F, A) concatenated factored logits, already suppression-masked.
    logits: dict[str, torch.Tensor]
    #: (B,) board value V(s).
    value: torch.Tensor
    #: (B, F) per-frontier value, zero where the frontier is dead.
    value_f: torch.Tensor


def _head_sizes(num_layers: int) -> dict[str, int]:
    return {
        "direction": NUM_DIRECTIONS,
        "step": NUM_STEPS,
        "layer": 1 + num_layers,   # 0 = stay; j>0 = via to layer j-1
        "via": 4,
        "width": 4,
        "couple": 2,
    }


class FrontierEncoder(nn.Module):
    """Per-frontier features + board context -> a token per frontier.

    Full self-attention over frontiers with a padding mask. `src_key_padding_mask`
    marks positions to IGNORE, so it is the negation of the alive mask --
    reversing that silently makes every real frontier invisible and attends
    only to padding.
    """

    def __init__(
        self,
        feat_dim: int,
        crop_dim: int,
        z_dim: int,
        global_dim: int,
        width: int = 192,
        depth: int = 2,
        heads: int | None = None,
    ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(feat_dim + crop_dim + z_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        heads = heads or max(1, width // 32)
        self.ctx = nn.Linear(global_dim, width)
        block = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=3 * width, batch_first=True, dropout=0.0,
            activation="gelu", norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, depth) if depth > 0 else None
        self.width = width

    def forward(
        self,
        feat: torch.Tensor,       # (B, F, feat_dim)
        crop: torch.Tensor,       # (B, F, crop_dim)
        z_gather: torch.Tensor,   # (B, F, z_dim)
        g: torch.Tensor,          # (B, global_dim)
        mask: torch.Tensor,       # (B, F) bool -- alive
    ) -> torch.Tensor:
        t = self.embed(torch.cat([feat, crop, z_gather], dim=-1))
        t = t + self.ctx(g).unsqueeze(1)
        # depth=0: per-frontier features only, no cross-frontier attention.
        # The right size for stages 0-1 (1-3 nets) -- there is almost nothing
        # for frontiers to coordinate about, and the transformer is a real
        # fwd+bwd cost.
        if self.blocks is None:
            return torch.nan_to_num(t) * mask.unsqueeze(-1).float()
        # A batch row with zero live frontiers makes the padding mask all-True,
        # which NaNs softmax. Guard by leaving at least the first slot
        # attention-visible; its output is masked to zero below anyway.
        safe_mask = mask.clone()
        safe_mask[~mask.any(dim=1), 0] = True
        out = self.blocks(t, src_key_padding_mask=~safe_mask)
        return torch.nan_to_num(out) * mask.unsqueeze(-1).float()


class PriorPolicy(nn.Module):
    def __init__(
        self,
        num_layers: int,
        field_width: int = 64,
        token_width: int = 192,
        crop: int = 16,
        encoder_levels: int = 2,
        token_depth: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.sizes = _head_sizes(num_layers)
        self.field = FieldEncoder(FIELD_CHANNELS, width=field_width, levels=encoder_levels)
        self.cropper = FrontierCropEncoder(FIELD_CHANNELS, num_layers, crop=crop)

        feat_dim = frontier_feature_dim(num_layers)
        self.tokens = FrontierEncoder(
            feat_dim=feat_dim,
            crop_dim=self.cropper.out_dim,
            z_dim=field_width,
            global_dim=self.field.global_dim,
            width=token_width,
            depth=token_depth,
        )

        total = sum(self.sizes.values())
        self.head = nn.Linear(token_width, total)
        # The critic reads the frontier tokens too, not just `g`.
        #
        # `g` is `global_proj(z.mean(dim=(2,3,4)))` -- the field embedding
        # *globally mean-pooled*. It carries no frontier position and no
        # distance-to-target, and this repo has four recorded failures at
        # decoding distance out of a pooled embedding (jepa x3,
        # models/fast_lookahead.py). A critic built on it alone measured
        # explained_variance +0.0001 with value_std 0.0008 against return_std
        # 1.74 -- V(s) was constant, so every advantage PPO saw was noise, which
        # is what the kl 0.29 / clip 0.64 thrash in the stage-0 log was.
        #
        # The masked mean over tokens carries the per-frontier geometry
        # (including the `dist` scalar) straight to the value head, by the same
        # principle the action features already follow: hand it the geometry,
        # do not ask it to reconstruct it.
        self.value = nn.Sequential(
            nn.Linear(self.field.global_dim + token_width, token_width),
            nn.SiLU(),
            nn.Linear(token_width, 1),
        )
        # Per-FRONTIER value, for per-agent advantage estimation.
        #
        # MAPPO assumes A_i(s,a) = A_global(s,a) for every agent, which here
        # means one board advantage broadcast over every frontier: it cannot
        # express "frontier B specifically should have stopped". GPAE (arXiv
        # 2603.02654) replaces that with a per-agent value; on 5m_vs_6m it took
        # a 3.1% MAPPO win rate to 93.7%, for +6% wall-clock and a cost that
        # does not grow with agent count -- which is the property that matters
        # if this is ever to run thousands of nets, where MAPPO's per-frontier
        # signal-to-noise falls as 1/N.
        self.value_frontier = nn.Linear(token_width, 1)
        self._init_heads()

    def _init_heads(self) -> None:
        """Near-zero weights, bias tuned so untrained argmax == greedy."""
        nn.init.normal_(self.head.weight, std=0.01)
        with torch.no_grad():
            self.head.bias.zero_()
            off = 0
            bias = {}
            for name, n in self.sizes.items():
                bias[name] = self.head.bias[off : off + n]
                off += n
            bias["direction"][0] = 2.0                          # toward the target
            bias["step"][0] = 1.0                               # one cell at a time
            # P(stay) = 3L / (3L + L) = 0.75 for any layer count.
            bias["layer"][0] = math.log(3.0 * self.num_layers)
            bias["width"][0] = 2.0                              # minimum width
            # via, couple: leave at 0 -- via is only read when layer != stay,
            # couple only when the net is a pair.

    # -- internals --------------------------------------------------------

    def _split(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        out, off = {}, 0
        for name, n in self.sizes.items():
            out[name] = flat[..., off : off + n]
            off += n
        return out

    def _suppress(self, logits: dict[str, torch.Tensor], obs: Observation) -> dict[str, torch.Tensor]:
        """Fixed, non-learned action masking.

        `safety` and `via_safe` are recomputed from the occupancy grid every
        step (see `env/observation.py`), so this cannot be trained away -- which
        is the whole point. A learned bias with the same intent decayed to
        nothing in `neuroroute/` once weight-driven logits grew during training.
        """
        d = logits["direction"]                                  # (B, F, 8)
        safe = obs.safety                                        # (B, F, 8, 3) bool
        # A direction is available if it is safe at *some* step length. That
        # marginal is correct for the direction head.
        #
        # The STEP head is deliberately NOT masked here. Its marginal ("legal
        # in some direction") is unsound, because direction and step are
        # sampled independently from factored heads: marginalising the joint
        # mask lets the policy pick a direction that is legal at 1 cell and a
        # step of 4 cells that is legal in some *other* direction, and land on
        # a combination neither marginal forbids.
        #
        # That is not hypothetical -- it is the stage-0 livelock. Measured on
        # the failing seeds: 23 of 24 actions legal, and argmax picked the one
        # illegal (direction, step) pair on all 16 steps until `max_stuck_steps`
        # retired the net. Because a rejected move writes nothing, the next
        # observation is identical and a deterministic policy re-picks it
        # forever. `_step_logits_given_direction` applies the joint constraint
        # exactly, conditioned on the direction actually chosen.
        dir_ok = safe.any(dim=-1)                                # (B, F, 8)
        # A frontier with no legal direction at all is genuinely entombed.
        # Masking everything would make the logits all -inf and the Categorical
        # NaN, so leave it unmasked and let the engine's stuck counter retire
        # it -- which is that counter's correct remaining job.
        dir_ok = torch.where(dir_ok.any(dim=-1, keepdim=True), dir_ok,
                             torch.ones_like(dir_ok))
        logits["direction"] = d - _BIG * (~dir_ok).float()

        via_ok = obs.via_safe.float()                            # (B, F, L)
        stay = torch.ones_like(via_ok[..., :1])
        layer_ok = torch.cat([stay, via_ok], dim=-1)             # (B, F, 1+L)
        logits["layer"] = logits["layer"] - _BIG * (layer_ok < 0.5).float()
        return logits

    def forward(self, obs: Observation) -> PolicyOutput:
        z, g = self.field(obs.field)                             # z:(B,d,L,h,w) g:(B,Gd)
        B, d, L, h, w = z.shape
        crop = self.cropper(obs.field, obs.frontier_pos, obs.frontier_mask)

        # Gather the shared latent at each frontier's (downsampled) cell.
        ds_h = obs.field.shape[-2] // h
        ds_w = obs.field.shape[-1] // w
        fy = (obs.frontier_pos[..., 1] // ds_h).clamp(0, h - 1)
        fx = (obs.frontier_pos[..., 2] // ds_w).clamp(0, w - 1)
        fl = obs.frontier_pos[..., 0].clamp(0, L - 1)
        bidx = torch.arange(B, device=z.device).view(B, 1)
        z_gather = z[bidx, :, fl, fy, fx]                        # (B, F, d)

        tok = self.tokens(obs.frontiers, crop, z_gather, g, obs.frontier_mask)
        logits = self._suppress(self._split(self.head(tok)), obs)
        m = obs.frontier_mask.unsqueeze(-1).float()
        pooled = (tok * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        value = self.value(torch.cat([g, pooled], dim=-1)).squeeze(-1)
        # Dead frontiers must value exactly zero: their future reward is zero,
        # so the TD error has to vanish rather than carry a learned constant.
        value_f = self.value_frontier(tok).squeeze(-1) * obs.frontier_mask.float()
        return PolicyOutput(logits=logits, value=value, value_f=value_f)

    # -- rollout / update API -------------------------------------------

    @staticmethod
    def _step_logits_given_direction(
        step_logits: torch.Tensor, safety: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        """Mask the step head by the legality of (direction, step) JOINTLY.

        `safety` is egocentric-indexed, and so is the `direction` head, so this
        is a direct gather with no rotation.

        Applied identically in `act()` and `evaluate()`. That is not a detail:
        the PPO ratio is exp(evaluate_logp - act_logp), so if only one of them
        conditioned the step head the two would be different distributions and
        the ratio would be noise with nothing raising an error --  the failure
        `verify_world`'s "act() and evaluate() give identical log-probs" check
        exists to catch.
        """
        B, F, _, n_steps = safety.shape
        idx = direction.view(B, F, 1, 1).expand(B, F, 1, n_steps)
        ok = torch.gather(safety, 2, idx).squeeze(2)             # (B, F, n_steps)
        # No legal step in the chosen direction: leave unmasked rather than
        # emit all -inf. Only reachable when the direction head was itself
        # unmasked because every direction was blocked (true entombment).
        ok = torch.where(ok.any(dim=-1, keepdim=True), ok, torch.ones_like(ok))
        return step_logits - _BIG * (~ok).float()

    def _dists(self, logits: dict[str, torch.Tensor]) -> dict[str, torch.distributions.Categorical]:
        return {k: torch.distributions.Categorical(logits=v) for k, v in logits.items()}

    @torch.no_grad()
    def act(self, obs: Observation, deterministic: bool = False) -> dict:
        out = self.forward(obs)
        logits = dict(out.logits)
        action, logp = {}, {}

        # Direction first: the step head's mask depends on which direction was
        # actually chosen, so the action is drawn autoregressively over exactly
        # these two heads. Everything else stays conditionally independent.
        pick = (lambda dist: dist.probs.argmax(dim=-1)) if deterministic else (lambda dist: dist.sample())
        d_dist = torch.distributions.Categorical(logits=logits["direction"])
        action["direction"] = pick(d_dist)
        logp["direction"] = d_dist.log_prob(action["direction"])

        logits["step"] = self._step_logits_given_direction(
            logits["step"], obs.safety, action["direction"]
        )
        for k, dist in self._dists(
            {k: v for k, v in logits.items() if k != "direction"}
        ).items():
            a = pick(dist)
            action[k] = a
            logp[k] = dist.log_prob(a)
        m = obs.frontier_mask.float()
        # `couple` only means anything for a pair; `via` only when a via is
        # actually being placed. Zero their log-prob elsewhere so a dead
        # dimension does not add noise to the ratio.
        logp["couple"] = logp["couple"] * obs.is_pair.float()
        logp["via"] = logp["via"] * (action["layer"] > 0).float()
        total_logp = sum(logp.values()) * m
        return {
            "action": action,
            "logp": total_logp,          # (B, F)
            "value": out.value,          # (B,)
            "value_f": out.value_f,      # (B, F)
            "mask": obs.frontier_mask,
        }

    def evaluate(self, obs: Observation, action: dict[str, torch.Tensor],
                 out: "PolicyOutput | None" = None) -> dict:
        """Score `action` under the policy at `obs`.

        `out` lets a caller supply an already-computed forward pass. Behaviour
        cloning scores TWO action sets at the same observation -- the policy's
        own and the expert's -- and running `forward()` twice re-runs the 3-D
        field encoder on identical input. Measured on stage 0, that doubled
        the PPO phase from 2.6s to 5.3s per update while `collect` was
        unchanged at 1.3s. The encoder is already known to dominate here (see
        `stack_observations`: "~30 s of the ~40 s update").
        """
        if out is None:
            out = self.forward(obs)
        logits = dict(out.logits)
        # Condition on the direction that was actually taken, so this is the
        # same distribution act() drew from. See _step_logits_given_direction.
        logits["step"] = self._step_logits_given_direction(
            logits["step"], obs.safety, action["direction"]
        )
        dists = self._dists(logits)
        m = obs.frontier_mask.float()
        logp = torch.zeros_like(m)
        ent = torch.zeros_like(m)
        # Score only the heads the caller supplied. PPO passes a full action
        # and is unaffected; behaviour cloning passes `direction` and `step`
        # alone, because the expert is a sequential Dijkstra router whose via
        # policy is not the one being learned -- so its layer/via choices must
        # not be cloned. Iterating every head regardless raised KeyError on
        # the first BC update.
        for k, dist in dists.items():
            if k not in action:
                continue
            lp = dist.log_prob(action[k])
            e = dist.entropy()
            if k == "couple":
                lp = lp * obs.is_pair.float()
                e = e * obs.is_pair.float()
            elif k == "via":
                # `via` is only meaningful when a layer change was requested;
                # with no `layer` supplied there is nothing to gate on, so the
                # head is simply not scored.
                if "layer" not in action:
                    continue
                gate = (action["layer"] > 0).float()
                lp = lp * gate
                e = e * gate
            logp = logp + lp
            ent = ent + e
        return {
            "logp": logp * m,            # (B, F)
            "entropy": (ent * m).sum() / m.sum().clamp_min(1.0),
            "value": out.value,          # (B,)
            "value_f": out.value_f,      # (B, F)
        }
