"""PPO for the factored per-frontier policy with a shared board critic.

The action is the *joint* move of every live frontier, factored into
independent per-frontier categoricals. Credit assignment follows MAPPO: one
board value `V(s)`, one board advantage `A_t` per macro-step, and every
frontier's action on that step is updated against the same `A_t`. The
per-frontier reward decomposition (`env/rewards.py`) still matters -- it puts a
dense progress signal into the return -- but the policy gradient sees the shared
advantage, not a per-frontier one. A per-frontier critic was tried in
`neuroroute/` and the honest read there was that the board-level decision needs
a board-level value; the same applies when *every* action is board-level.

Not chunked yet. At stage 0-1 (small boards, `F <= 20`) the stored field
tensors fit; the chunked-restack path that `neuroroute/training/ppo.py` needed
at 8-layer 128x128 is a knob to add when the curriculum reaches that scale, not
before.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    lr: float = 3e-4
    #: Behaviour-cloning loss weight, annealed by the trainer. 0 disables it.
    bc_coef: float = 0.0


def compute_gae(
    rewards: torch.Tensor,      # (T, B)
    values: torch.Tensor,       # (T, B)
    last_value: torch.Tensor,   # (B,)
    dones: torch.Tensor,        # (T, B) 1.0 on the final step of an episode
    cfg: PPOConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Board-level GAE. Returns (advantages, returns), each (T, B)."""
    T, B = rewards.shape
    adv = torch.zeros(T, B, device=rewards.device)
    gae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + cfg.gamma * next_v * nonterminal - values[t]
        gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


class RolloutBuffer:
    """Holds one batch of rollout data, transitions flattened to (T*B, ...)."""

    def __init__(self):
        self.obs: list = []
        self.actions: list[dict] = []
        self.logp: list[torch.Tensor] = []      # (B, F)
        self.masks: list[torch.Tensor] = []     # (B, F)
        self.rewards: list[torch.Tensor] = []   # (B,) board reward = frontier-sum + board term
        self.values: list[torch.Tensor] = []    # (B,)
        self.dones: list[torch.Tensor] = []     # (B,)
        self.bc_actions: list[dict | None] = []

    def add(self, obs, action, logp, mask, board_reward, value, done, bc_action=None):
        self.obs.append(obs)
        self.actions.append(action)
        self.logp.append(logp.detach())
        self.masks.append(mask)
        self.rewards.append(board_reward.detach())
        self.values.append(value.detach())
        self.dones.append(done)
        self.bc_actions.append(bc_action)

    def __len__(self) -> int:
        return len(self.obs)


def ppo_update(
    policy,
    optimizer,
    buf: RolloutBuffer,
    last_value: torch.Tensor,
    cfg: PPOConfig,
) -> dict:
    """One PPO update over a filled buffer. Returns a metrics dict."""
    T = len(buf)
    B = buf.rewards[0].shape[0]
    dev = buf.rewards[0].device

    rewards = torch.stack(buf.rewards)                 # (T, B)
    values = torch.stack(buf.values)                   # (T, B)
    dones = torch.stack(buf.dones).float()             # (T, B)
    adv, returns = compute_gae(rewards, values, last_value, dones, cfg)

    # Normalise advantages across the whole batch (standard PPO).
    adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-6))

    idx = list(range(T))
    metrics = {k: 0.0 for k in ("policy_loss", "value_loss", "entropy", "clip_frac", "bc_loss", "approx_kl")}
    n_updates = 0

    for _ in range(cfg.epochs):
        perm = torch.randperm(T).tolist()
        chunk = max(1, T // cfg.minibatches)
        for c in range(0, T, chunk):
            steps = perm[c : c + chunk]
            if not steps:
                continue

            pol_loss = val_loss = ent_term = bc_term = kl_term = clip_term = 0.0
            for t in steps:
                obs = buf.obs[t]
                act = buf.actions[t]
                mask = buf.masks[t].float()            # (B, F)
                n_live = mask.sum().clamp_min(1.0)

                ev = policy.evaluate(obs, act)
                new_logp = ev["logp"]                  # (B, F)
                old_logp = buf.logp[t]

                ratio = torch.exp((new_logp - old_logp).clamp(-20, 20))
                a_t = adv[t].view(B, 1)                # board advantage, broadcast
                unclipped = ratio * a_t
                clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a_t
                pol_loss = pol_loss + -(torch.minimum(unclipped, clipped) * mask).sum() / n_live

                v = ev["value"]                        # (B,)
                val_loss = val_loss + (v - returns[t]).pow(2).mean()
                ent_term = ent_term + ev["entropy"]

                with torch.no_grad():
                    kl_term = kl_term + ((old_logp - new_logp) * mask).sum() / n_live
                    clip_term = clip_term + (
                        ((ratio - 1.0).abs() > cfg.clip).float() * mask
                    ).sum() / n_live

                if cfg.bc_coef > 0.0 and buf.bc_actions[t] is not None:
                    bc = buf.bc_actions[t]
                    evb = policy.evaluate(obs, bc["action"])
                    # Cross-entropy to the expert action, on frontiers the
                    # expert actually moved.
                    w = bc["mask"].float()
                    bc_term = bc_term + -(evb["logp"] * w).sum() / w.sum().clamp_min(1.0)

            k = len(steps)
            loss = (
                pol_loss / k
                + cfg.value_coef * val_loss / k
                - cfg.entropy_coef * ent_term / k
                + cfg.bc_coef * bc_term / k
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            metrics["policy_loss"] += float((pol_loss / k).detach())
            metrics["value_loss"] += float((val_loss / k).detach())
            metrics["entropy"] += float((ent_term / k).detach())
            metrics["clip_frac"] += float(clip_term / k)
            metrics["bc_loss"] += float((bc_term / k).detach()) if cfg.bc_coef > 0 and not isinstance(bc_term, float) else 0.0
            metrics["approx_kl"] += float(kl_term / k)
            metrics["grad_norm"] = float(gn)
            n_updates += 1

    for key in ("policy_loss", "value_loss", "entropy", "clip_frac", "bc_loss", "approx_kl"):
        metrics[key] /= max(1, n_updates)
    metrics["return_mean"] = float(returns.mean())
    metrics["adv_std"] = float(adv.std())
    return metrics
