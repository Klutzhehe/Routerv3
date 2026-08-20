"""CFPPolicy -- sampling and PPO-side scoring on top of CFPNet.

The action is a pair, sampled autoregressively within a single env step:

    a_net   ~ Categorical(pointer_logits)          # which net, and whether
                                                   #   to route it or rip
                                                   #   it out
    a_field ~ Normal(mean(.| a_net), std)          # the cost field to route
                                                   #   that net under

so the joint log-probability is just the sum, and PPO needs no special
handling for the hybrid action space beyond adding the two terms. The field
is conditioned on the sampled net (via FiLM in CFPNet.field_params), which
is why sampling has to happen in this order and why the two log-probs
cannot be computed in one pass.

The field the env actually uses is clamped to +/-FIELD_CLAMP before it
reaches the A* planner, but the log-probability is always taken on the
*unclamped* sample. Scoring the clamped value would be wrong -- the density
of a clamped Gaussian is not Gaussian. This is the same convention
stable-baselines uses for clipped continuous actions.

Nothing here touches the router; the env contract is: give this class a
CFPObservation, get back an action to execute.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn

from pcbworld.agents.cfp.model import CFPConfig, CFPNet, Encoded
from pcbworld.agents.cfp.spec import ACTION_RIPUP, ACTION_ROUTE, CFPObservation

# The planner treats the field as a roughly unit-scale cost bias; anything
# beyond this is saturation, not signal, and letting it run away just makes
# A* degenerate into "follow the one cheap cell".
FIELD_CLAMP = 4.0

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclasses.dataclass
class CFPScore:
    """The two entropies are kept apart on purpose.

    The field is num_field_planes * field_size^2 Gaussian dimensions -- 768
    at the default config -- so its entropy is ~700 nats while the pointer
    categorical's is ~2. Summing them and applying one PPO entropy
    coefficient means any coefficient large enough to affect *which net to
    route* is enormous for the field, and any coefficient sane for the field
    does nothing for net choice. Worse, the field entropy's gradient only
    reaches field_log_std, so the joint term is mostly a std regularizer
    wearing an exploration-bonus costume. The trainer gets two coefficients.

    entropy is kept as the plain sum for logging and for callers that
    genuinely want the joint quantity; it is not what the loss should use.
    """

    log_prob: torch.Tensor        # (B,) joint
    cat_entropy: torch.Tensor     # (B,) pointer categorical, ~O(1) nats
    field_entropy: torch.Tensor   # (B,) field Gaussian, ~O(100) nats
    value: torch.Tensor           # (B,)

    @property
    def entropy(self) -> torch.Tensor:
        return self.cat_entropy + self.field_entropy


@dataclasses.dataclass
class CFPAction:
    """One sampled action, plus everything PPO needs to score it later.

    action_index: (B,) long, flat index into the (NUM_ACTION_KINDS * N)
      pointer space. Use split() to recover (kind, net_slot).
    field:        (B, P, F, F) float, the raw (unclamped) Gaussian sample.
    score:        log-prob / entropies / value at sampling time.
    """

    action_index: torch.Tensor
    field: torch.Tensor
    score: CFPScore

    @property
    def log_prob(self) -> torch.Tensor:
        return self.score.log_prob

    @property
    def value(self) -> torch.Tensor:
        return self.score.value

    def split(self, num_nets: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(kind, net_slot); kind is ACTION_ROUTE or ACTION_RIPUP."""
        return self.action_index // num_nets, self.action_index % num_nets

    def planner_field(self) -> torch.Tensor:
        """The clamped field to hand the A* planner. See module docstring on
        why this is never the tensor that gets scored."""
        return self.field.clamp(-FIELD_CLAMP, FIELD_CLAMP)


def _masked_categorical_stats(
    logits: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(log_probs, entropy) for a masked categorical, without NaNs.

    torch.distributions.Categorical computes entropy as sum(p * log p) and
    produces NaN wherever p == 0 and log p == -inf, which for a masked
    action space is most of the row. Computing it with an explicit where()
    over the valid mask sidesteps that entirely.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    p_log_p = torch.where(valid, probs * log_probs, torch.zeros_like(probs))
    return log_probs, -p_log_p.sum(dim=-1)


def _gaussian_log_prob(
    x: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
) -> torch.Tensor:
    """Diagonal-Gaussian log density, summed over every non-batch dim."""
    var = torch.exp(2.0 * log_std)
    per_dim = -0.5 * (x - mean) ** 2 / var - log_std - _LOG_SQRT_2PI
    return per_dim.flatten(1).sum(dim=-1)


def _gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    per_dim = log_std + _LOG_SQRT_2PI + 0.5
    return per_dim.flatten(1).sum(dim=-1)


class CFPPolicy(nn.Module):
    """Thin actor-critic wrapper. Owns no optimizer and no rollout buffer --
    those belong to the trainer, which isn't written yet; this is only the
    parts the trainer would otherwise have to reimplement."""

    def __init__(self, config: CFPConfig | None = None) -> None:
        super().__init__()
        self.net = CFPNet(config)
        self.config = self.net.config

    # -- acting -------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: CFPObservation, deterministic: bool = False) -> CFPAction:
        """Sample (or take the mode of) an action for rollout collection."""
        return self._act(obs, deterministic=deterministic)

    def _act(self, obs: CFPObservation, deterministic: bool) -> CFPAction:
        encoded = self.net.encode(obs)
        valid = obs.action_mask.flatten(1)
        log_probs, cat_entropy = _masked_categorical_stats(encoded.pointer_logits, valid)

        if deterministic:
            action_index = log_probs.argmax(dim=-1)
        else:
            action_index = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)

        net_slot = action_index % obs.num_nets
        mean, log_std = self.net.field_params(encoded, net_slot)
        field = mean if deterministic else mean + torch.randn_like(mean) * log_std.exp()

        cat_lp = log_probs.gather(-1, action_index[:, None]).squeeze(-1)
        field_lp = _gaussian_log_prob(field, mean, log_std)

        return CFPAction(
            action_index=action_index,
            field=field,
            score=CFPScore(
                log_prob=cat_lp + field_lp,
                cat_entropy=cat_entropy,
                field_entropy=_gaussian_entropy(log_std),
                value=encoded.value,
            ),
        )

    # -- scoring (PPO update) ----------------------------------------------

    def evaluate_actions(
        self,
        obs: CFPObservation,
        action_index: torch.Tensor,
        field: torch.Tensor,
    ) -> CFPScore:
        """Re-score stored actions under the current parameters.

        This is the call PPO's ratio, entropy bonus, and value loss are all
        built from. See CFPScore on why the entropy comes back in two pieces
        rather than one.
        """
        encoded = self.net.encode(obs)
        valid = obs.action_mask.flatten(1)
        log_probs, cat_entropy = _masked_categorical_stats(encoded.pointer_logits, valid)

        net_slot = action_index % obs.num_nets
        mean, log_std = self.net.field_params(encoded, net_slot)

        cat_lp = log_probs.gather(-1, action_index[:, None]).squeeze(-1)
        field_lp = _gaussian_log_prob(field, mean, log_std)

        return CFPScore(
            log_prob=cat_lp + field_lp,
            cat_entropy=cat_entropy,
            field_entropy=_gaussian_entropy(log_std),
            value=encoded.value,
        )

    @torch.no_grad()
    def value(self, obs: CFPObservation) -> torch.Tensor:
        """Critic only -- for GAE bootstrapping at a rollout boundary."""
        return self.net.encode(obs).value

    # -- introspection ------------------------------------------------------

    def describe(self) -> str:
        c = self.config
        total = self.net.num_parameters()
        return (
            f"CFPPolicy: {total / 1e6:.1f}M params | dim={c.dim} heads={c.num_heads} "
            f"net_layers={c.net_layers} fusion_rounds={c.fusion_rounds} "
            f"field={c.num_field_planes}x{c.field_size}x{c.field_size} "
            f"(actions: {ACTION_ROUTE}=route, {ACTION_RIPUP}=ripup)"
        )
