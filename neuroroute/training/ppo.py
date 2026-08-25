"""PPO over the batched world, with the forecaster's supervised losses attached.

Two things here are not standard PPO and both are forced by this environment.

**Observation storage.** A rollout observation is dominated by the field
tensor: at ``B=32, C=6, L=8, H=W=128`` that is 25 M floats, 100 MB per step, so
a 64-step rollout in fp32 is 6.4 GB of nothing but board snapshots. Every
channel except `demand` is binary and `demand` is already in [0, 1], so the
buffer stores the field **quantised to uint8** -- 1/4 the memory of fp16, 1/16
of fp32, with a quantisation error of 1/255 on the one channel that is not
already exactly representable.

**Per-head credit.** Rewards and values are ``(B, K)``: one value per routing
head, not one per board. A head slot is its own MDP -- it is handed a net,
routes it, is handed another -- and the observation carries which net that is,
so the slot-level value function is well defined. Pooling `K` heads into one
board-level value would smear every head's outcome across all the others, and
with `K=8` that is an 8x variance penalty for no benefit.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroroute.env.observation import Observation
from neuroroute.models.forecaster import forecast_losses


@dataclass
class PPOConfig:
    rollout_steps: int = 64
    epochs: int = 4
    minibatches: int = 4
    clip: float = 0.2
    value_clip: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    entropy_final: float = 0.001
    value_coef: float = 0.5
    forecast_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    #: Where rollout observations live. "cpu" trades PCIe bandwidth for VRAM
    #: and is the right default on a 16 GB card.
    store_device: str = "cpu"
    normalise_advantage: bool = True


def _quantise(field: torch.Tensor) -> torch.Tensor:
    return (field.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def _dequantise(field: torch.Tensor) -> torch.Tensor:
    return field.float() / 255.0


class RolloutBuffer:
    """Stores a fixed-length rollout, observations included."""

    def __init__(self, cfg: PPOConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.store = torch.device(cfg.store_device)
        self.clear()

    def clear(self) -> None:
        self.obs: list[dict] = []
        self.actions: list[dict] = []
        self.log_prob: list[torch.Tensor] = []
        self.value: list[torch.Tensor] = []
        self.reward: list[torch.Tensor] = []
        self.done: list[torch.Tensor] = []
        self.mask: list[torch.Tensor] = []

    def add(self, obs: Observation, actions: dict, log_prob, value, reward, done, mask) -> None:
        s = self.store
        self.obs.append(
            {
                "field": _quantise(obs.field).to(s),
                "heads": obs.heads.half().to(s),
                "head_pos": obs.head_pos.to(s),
                "head_mask": obs.head_mask.to(s),
                "head_is_pair": obs.head_is_pair.to(s),
                "via_safe": obs.via_safe.to(s),
                "nets": obs.nets.half().to(s),
                "net_mask": obs.net_mask.to(s),
                "safety": obs.safety.to(s),
                "bearing": obs.bearing.to(s),
                "geo_layer": obs.geo_layer.half().to(s),
            }
        )
        self.actions.append({k: v.to(s) for k, v in actions.items() if k != "schedule"})
        self.log_prob.append(log_prob.detach().to(s))
        self.value.append(value.detach().to(s))
        self.reward.append(reward.detach().to(s))
        self.done.append(done.to(s))
        self.mask.append(mask.to(s))

    def observation_at(self, t: int) -> Observation:
        o = self.obs[t]
        d = self.device
        return Observation(
            field=_dequantise(o["field"].to(d)),
            heads=o["heads"].to(d).float(),
            head_pos=o["head_pos"].to(d),
            head_mask=o["head_mask"].to(d),
            head_is_pair=o["head_is_pair"].to(d),
            via_safe=o["via_safe"].to(d),
            nets=o["nets"].to(d).float(),
            net_mask=o["net_mask"].to(d),
            safety=o["safety"].to(d),
            bearing=o["bearing"].to(d),
            geo_layer=o["geo_layer"].to(d).float(),
        )

    def __len__(self) -> int:
        return len(self.obs)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over (T, B, K).

    `dones` is per board, broadcast over heads: a board finishing ends every
    one of its head slots' trajectories at once, which is correct -- the next
    observation comes from a freshly generated board with no relationship to
    this one.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t].float()
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


def ppo_update(
    policy: nn.Module,
    optimiser: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOConfig,
    entropy_coef: float,
    forecast_targets: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """One PPO phase over a collected rollout. Returns scalar diagnostics."""
    d = buffer.device
    T = len(buffer)
    old_logp = torch.stack(buffer.log_prob).to(d)
    old_value = torch.stack(buffer.value).to(d)
    masks = torch.stack(buffer.mask).to(d)

    if cfg.normalise_advantage:
        live = masks > 0
        if bool(live.any()):
            mu = advantages[live].mean()
            sd = advantages[live].std().clamp_min(1e-6)
            advantages = torch.where(live, (advantages - mu) / sd, torch.zeros_like(advantages))

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "forecast": 0.0}
    n_updates = 0
    steps_per_mb = max(1, T // cfg.minibatches)

    for _ in range(cfg.epochs):
        order = torch.randperm(T).tolist()
        for start in range(0, T, steps_per_mb):
            idx = order[start : start + steps_per_mb]
            if not idx:
                continue

            p_loss = v_loss = ent_term = clip_frac = 0.0
            f_loss = torch.zeros((), device=d)
            total = torch.zeros((), device=d)

            for t in idx:
                obs = buffer.observation_at(t)
                act = {k: v.to(d) for k, v in buffer.actions[t].items()}
                logp, entropy, value = policy.evaluate(obs, act)

                m = masks[t]
                denom = m.sum().clamp_min(1.0)
                ratio = torch.exp((logp - old_logp[t]) * m)
                a = advantages[t]

                unclipped = -a * ratio
                clipped = -a * ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip)
                pl = (torch.maximum(unclipped, clipped) * m).sum() / denom

                v_unc = (value - returns[t]) ** 2
                v_cl = (old_value[t] + (value - old_value[t]).clamp(-cfg.value_clip, cfg.value_clip) - returns[t]) ** 2
                vl = 0.5 * (torch.maximum(v_unc, v_cl) * m).sum() / denom

                ent = (entropy * m).sum() / denom
                total = total + pl + cfg.value_coef * vl - entropy_coef * ent

                p_loss += float(pl)
                v_loss += float(vl)
                ent_term += float(ent)
                clip_frac += float((((ratio - 1.0).abs() > cfg.clip).float() * m).sum() / denom)

            # The forecaster is supervised, not reinforced. It is trained on
            # the *terminal* state of the episode this rollout came from, so
            # it gets one gradient per minibatch rather than one per step --
            # its labels do not vary within an episode.
            if forecast_targets is not None:
                obs0 = buffer.observation_at(idx[0])
                _feat, _g, latent, forecast = policy.encode(obs0)
                losses = forecast_losses(forecast, forecast_targets)
                f_loss = sum(losses.values())
                total = total + cfg.forecast_coef * f_loss

            total = total / len(idx)
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimiser.step()

            n = len(idx)
            stats["policy_loss"] += p_loss / n
            stats["value_loss"] += v_loss / n
            stats["entropy"] += ent_term / n
            stats["clip_frac"] += clip_frac / n
            stats["forecast"] += float(f_loss)
            n_updates += 1

    return {k: v / max(1, n_updates) for k, v in stats.items()}
