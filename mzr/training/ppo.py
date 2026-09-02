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

from mzr.env.observation import stack_observations


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    value_coef: float = 0.5
    #: Clip the value update to +/- this around the old prediction, like the
    #: policy ratio is clipped. Standard PPO; stops one bad-scale rollout batch
    #: from yanking the critic (and, through the shared encoder, the policy).
    value_clip: float = 10.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    lr: float = 3e-4
    #: Behaviour-cloning loss weight, annealed by the trainer. 0 disables it.
    bc_coef: float = 0.0
    #: Per-FRONTIER advantage instead of one board advantage broadcast to every
    #: frontier. See `compute_gae_frontier`.
    per_frontier_adv: bool = False


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


def compute_gae_frontier(
    rewards: torch.Tensor,      # (T, B, F) per-frontier reward
    values: torch.Tensor,       # (T, B, F) per-frontier value, 0 where dead
    last_value: torch.Tensor,   # (B, F)
    dones: torch.Tensor,        # (T, B) 1.0 on the final step of an episode
    masks: torch.Tensor,        # (T, B, F) 1.0 where the frontier is alive
    cfg: PPOConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frontier GAE. Returns (advantages, returns), each (T, B, F).

    MAPPO gives every agent the same advantage -- stated in the GPAE paper
    (arXiv 2603.02654) as ``A_i(s,a) = A_global(s,a)`` for all i -- which here
    is one board scalar broadcast over every frontier. That signal cannot
    express "frontier B should have stopped while frontier A was right to
    move", and its per-frontier signal-to-noise falls as 1/N: 2 frontiers share
    it at one net, ~2000 at a thousand nets. It is the one blocker in this
    design that gets *worse* as the problem gets bigger.

    `env/rewards.py::step_reward` already returns a per-frontier reward; the
    trainer was summing it away one line before PPO saw it. This keeps it.

    A dead frontier values exactly zero (masked in the policy), so its TD error
    vanishes and it contributes no gradient rather than a learned constant.
    """
    T, B, F = rewards.shape
    adv = torch.zeros(T, B, F, device=rewards.device)
    gae = torch.zeros(B, F, device=rewards.device)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = (1.0 - dones[t]).unsqueeze(-1)
        delta = rewards[t] + cfg.gamma * next_v * nonterminal - values[t]
        gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
        adv[t] = gae * masks[t]
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
        self.f_rewards: list = []   # (B, F) per-frontier reward
        self.f_values: list = []    # (B, F) per-frontier value

    def add(self, obs, action, logp, mask, board_reward, value, done, bc_action=None,
            frontier_reward=None, frontier_value=None):
        self.obs.append(obs)
        self.actions.append(action)
        self.logp.append(logp.detach())
        self.masks.append(mask)
        self.rewards.append(board_reward.detach())
        self.values.append(value.detach())
        self.f_rewards.append(None if frontier_reward is None else frontier_reward.detach())
        self.f_values.append(None if frontier_value is None else frontier_value.detach())
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
    last_value_f: torch.Tensor | None = None,
) -> dict:
    """One PPO update over a filled buffer. Returns a metrics dict."""
    T = len(buf)
    B = buf.rewards[0].shape[0]
    dev = buf.rewards[0].device

    rewards = torch.stack(buf.rewards)                 # (T, B)
    values = torch.stack(buf.values)                   # (T, B)
    dones = torch.stack(buf.dones).float()             # (T, B)
    adv, returns = compute_gae(rewards, values, last_value, dones, cfg)
    adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-6))

    per_frontier = cfg.per_frontier_adv and buf.f_rewards and buf.f_rewards[0] is not None
    if per_frontier:
        f_rew = torch.stack(buf.f_rewards)                          # (T, B, F)
        f_val = torch.stack(buf.f_values)                           # (T, B, F)
        f_mask = torch.stack(buf.masks).float()                     # (T, B, F)
        f_adv, f_ret = compute_gae_frontier(
            f_rew, f_val, last_value_f, dones, f_mask, cfg
        )
        # Normalise over LIVE frontiers only -- dead ones are exact zeros and
        # would drag the mean and shrink the std toward nothing.
        live = f_mask > 0.5
        if bool(live.any()):
            mu = f_adv[live].mean()
            sd = f_adv[live].std().clamp_min(1e-6)
            f_adv = ((f_adv - mu) / sd) * f_mask

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

            k = len(steps)
            # One batched evaluate for the whole minibatch: stack k timesteps
            # into a (k*B, ...) observation and action, run the model once.
            mb_obs = stack_observations([buf.obs[t] for t in steps])
            mb_act = {
                key: torch.cat([buf.actions[t][key] for t in steps], dim=0)
                for key in buf.actions[steps[0]]
            }
            mb_mask = torch.cat([buf.masks[t] for t in steps], dim=0).float()   # (k*B, F)
            mb_oldlp = torch.cat([buf.logp[t] for t in steps], dim=0)           # (k*B, F)
            mb_adv = adv[steps].reshape(k * B, 1)
            mb_ret = returns[steps].reshape(k * B)
            if per_frontier:
                # The whole point: each frontier is scored on its OWN advantage
                # rather than the board's, so "this one should have stopped" is
                # expressible.
                mb_adv = torch.cat([f_adv[t] for t in steps], dim=0)     # (k*B, F)
                mb_fret = torch.cat([f_ret[t] for t in steps], dim=0)
                mb_foldval = torch.cat([buf.f_values[t] for t in steps], dim=0)

            mb_oldval = torch.cat([buf.values[tt] for tt in steps], dim=0)     # (k*B,)

            # One forward pass, scored twice: the PPO ratio uses the policy's
            # own action, behaviour cloning the expert's. Both at the same
            # observation, so the (heavy) field encoder runs once.
            _fwd = policy.forward(mb_obs)
            ev = policy.evaluate(mb_obs, mb_act, out=_fwd)
            new_logp = ev["logp"]                                              # (k*B, F)
            n_live = mb_mask.sum().clamp_min(1.0)

            ratio = torch.exp((new_logp - mb_oldlp).clamp(-20, 20))
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * mb_adv
            pol_loss = -(torch.minimum(unclipped, clipped) * mb_mask).sum() / n_live
            v = ev["value"]
            v_clipped = mb_oldval + (v - mb_oldval).clamp(-cfg.value_clip, cfg.value_clip)
            val_loss = torch.maximum((v - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()
            if per_frontier:
                vf = ev["value_f"]
                vf_c = mb_foldval + (vf - mb_foldval).clamp(-cfg.value_clip, cfg.value_clip)
                vf_loss = torch.maximum((vf - mb_fret).pow(2), (vf_c - mb_fret).pow(2))
                val_loss = val_loss + (vf_loss * mb_mask).sum() / n_live
            ent_term = ev["entropy"]

            with torch.no_grad():
                kl_term = ((mb_oldlp - new_logp) * mb_mask).sum() / n_live
                clip_term = (((ratio - 1.0).abs() > cfg.clip).float() * mb_mask).sum() / n_live

            bc_term = torch.zeros((), device=dev)
            # A demonstration can be missing for SOME steps of a minibatch --
            # `ExpertActions.action()` returns None whenever no live frontier
            # sits on the expert's route. Guarding on `steps[0]` alone and then
            # indexing every `t` raised TypeError on the first BC run. Weight
            # the absent steps to zero instead, so the BC loss averages over the
            # frontiers that actually have a demonstration.
            have = [t for t in steps if buf.bc_actions[t] is not None]
            if cfg.bc_coef > 0.0 and have:
                zero = {k: torch.zeros_like(v)
                        for k, v in buf.bc_actions[have[0]]["action"].items()}
                blank_m = torch.zeros_like(buf.bc_actions[have[0]]["mask"])
                mb_bc = {
                    key: torch.cat(
                        [(buf.bc_actions[t]["action"][key] if buf.bc_actions[t] is not None
                          else zero[key]) for t in steps], dim=0)
                    for key in buf.bc_actions[have[0]]["action"]
                }
                w = torch.cat(
                    [(buf.bc_actions[t]["mask"] if buf.bc_actions[t] is not None
                      else blank_m) for t in steps], dim=0).float()
                # Reuse the forward pass the PPO term already computed --
                # same observation, so the field encoder must not run twice.
                evb = policy.evaluate(mb_obs, mb_bc, out=_fwd)
                bc_term = -(evb["logp"] * w).sum() / w.sum().clamp_min(1.0)

            loss = (
                pol_loss
                + cfg.value_coef * val_loss
                - cfg.entropy_coef * ent_term
                + cfg.bc_coef * bc_term
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            metrics["policy_loss"] += float(pol_loss.detach())
            metrics["value_loss"] += float(val_loss.detach())
            metrics["entropy"] += float(ent_term.detach())
            metrics["clip_frac"] += float(clip_term)
            metrics["bc_loss"] += float(bc_term.detach()) if cfg.bc_coef > 0 else 0.0
            metrics["approx_kl"] += float(kl_term)
            metrics["grad_norm"] = float(gn)
            n_updates += 1

    for key in ("policy_loss", "value_loss", "entropy", "clip_frac", "bc_loss", "approx_kl"):
        metrics[key] /= max(1, n_updates)
    metrics["return_mean"] = float(returns.mean())
    metrics["adv_std"] = float(adv.std())
    return metrics
