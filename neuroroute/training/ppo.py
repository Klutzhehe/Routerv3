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

import time
from contextlib import contextmanager
from dataclasses import dataclass, field as dc_field, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroroute.env.observation import Observation
from neuroroute.models.forecaster import forecast_losses


@contextmanager
def _timed(bucket: dict[str, float] | None, name: str, device: torch.device):
    """Accumulate wall time for one sub-phase into `bucket[name]`, CUDA-accurate.

    Mirrors `Telemetry.section`'s synchronize-before-and-after pattern -- CUDA
    ops are launched async, so an unsynchronised `perf_counter()` around them
    times how fast Python could *submit* work, not how long the GPU took to do
    it. `bucket=None` skips both the sync and the dict write, so callers that
    do not want per-phase detail pay nothing extra.
    """
    if bucket is None:
        yield
        return
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize()
        bucket[name] = bucket.get(name, 0.0) + (time.perf_counter() - t)


@dataclass
class PPOConfig:
    rollout_steps: int = 64
    epochs: int = 4
    minibatches: int = 4
    #: Timesteps folded into ONE forward/backward pass. This is the GPU
    #: utilisation knob. Each rollout timestep is only `B` boards, which is far
    #: too small to saturate a modern GPU; stacking `chunk` of them along the
    #: batch dimension turns `epochs * minibatches * steps_per_mb` tiny passes
    #: into `epochs * ceil(T/chunk)` big ones. Measured on a T4 at stage 0, the
    #: update phase was 68% of wall time with 128 passes per update.
    #: Lower it first if you hit OOM -- it trades memory for utilisation
    #: linearly and changes nothing about the maths.
    chunk: int = 8
    #: fp16 autocast for the forward pass, loss and optimiser step in fp32.
    #: OFF by default and UNVERIFIED -- see the note in `ppo_update`.
    amp: bool = False
    clip: float = 0.2
    value_clip: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    entropy_final: float = 0.001
    value_coef: float = 0.5
    forecast_coef: float = 0.5
    #: Separate from `value_coef` on purpose. `board_value` regresses
    #: against board-level returns -- a SUM over up to K=8 heads' rewards,
    #: not one head's -- so its natural squared-error scale is much larger
    #: than the per-head value loss's, even after normalising its input
    #: (see h_board_value's LayerNorm fix). Sharing `value_coef` measured
    #: at 97.7% of the ENTIRE total loss from one real logged update
    #: (bv=718 vs vl=16, pl effectively 0%) -- the routing policy was
    #: getting essentially no gradient at all while this dominated. This
    #: default (0.01) is a first-pass calibration from that one data point,
    #: not a derived constant -- watch `bv` on the next run and retune if
    #: it is still lopsided.
    board_value_coef: float = 0.01
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
        # Board-level (schedule + ripup) bookkeeping. Separate from the
        # per-head lists above because both actions, and the critic that
        # scores them, operate on the whole board rather than one head --
        # see PolicyOutput.board_value and models/policy.py::_ripup.
        self.sched_log_prob: list[torch.Tensor] = []
        self.ripup_log_prob: list[torch.Tensor] = []
        self.board_value: list[torch.Tensor] = []
        self.board_reward: list[torch.Tensor] = []

    def add(
        self, obs: Observation, actions: dict, log_prob, value, reward, done, mask,
        sched_log_prob, ripup_log_prob, board_value, board_reward,
    ) -> None:
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
                "routed_mask": obs.routed_mask.to(s),
                "safety": obs.safety.to(s),
                "bearing": obs.bearing.to(s),
                "geo_layer": obs.geo_layer.half().to(s),
            }
        )
        # Every action kept now, including "schedule" and "ripup" -- both
        # used to be dropped here (`if k != "schedule"`), which is exactly
        # why neither ever received a PPO gradient before this change.
        self.actions.append({k: v.to(s) for k, v in actions.items()})
        self.log_prob.append(log_prob.detach().to(s))
        self.value.append(value.detach().to(s))
        self.reward.append(reward.detach().to(s))
        self.done.append(done.to(s))
        self.mask.append(mask.to(s))
        self.sched_log_prob.append(sched_log_prob.detach().to(s))
        self.ripup_log_prob.append(ripup_log_prob.detach().to(s))
        self.board_value.append(board_value.detach().to(s))
        self.board_reward.append(board_reward.detach().to(s))

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
            routed_mask=o["routed_mask"].to(d),
            safety=o["safety"].to(d),
            bearing=o["bearing"].to(d),
            geo_layer=o["geo_layer"].to(d).float(),
        )

    def __len__(self) -> int:
        return len(self.obs)


def stack_observations(obs_list: list[Observation], device: torch.device) -> Observation:
    """Fold several timesteps' observations into one batch.

    Every field concatenates along dim 0, and every module in the policy is
    batch-agnostic (convolutions, per-row MLPs, attention over K/N), so a
    stacked observation is just a bigger batch -- the maths is identical.
    `NeuroRoutePolicy._schedule`/`_ripup` only ever use `B` as a size (to
    bound their per-slot loop and shape a zeros tensor), never as an
    identity or index, so `evaluate()` calling them on a stacked, chunk*B
    wide observation is exactly as safe as everything else here.

    This is what turns a rollout timestep (B boards, e.g. 16) into a forward
    pass wide enough to actually occupy a GPU.
    """
    cat = lambda key: torch.cat([getattr(o, key) for o in obs_list], dim=0)  # noqa: E731
    return Observation(
        field=cat("field"),
        heads=cat("heads"),
        head_pos=cat("head_pos"),
        head_mask=cat("head_mask"),
        head_is_pair=cat("head_is_pair"),
        via_safe=cat("via_safe"),
        nets=cat("nets"),
        net_mask=cat("net_mask"),
        routed_mask=cat("routed_mask"),
        safety=cat("safety"),
        bearing=cat("bearing"),
        geo_layer=cat("geo_layer"),
    )


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
    board_advantages: torch.Tensor,
    board_returns: torch.Tensor,
    forecast_targets: dict[str, torch.Tensor] | None = None,
    scaler: "torch.amp.GradScaler | None" = None,
    phase_timer: dict[str, float] | None = None,
) -> dict[str, float]:
    """One PPO phase over a collected rollout. Returns scalar diagnostics.

    Timesteps are processed in **chunks stacked along the batch dimension**,
    not one at a time. A single rollout timestep is only `B` boards wide, which
    leaves a GPU almost idle -- measured at 1,247 decisions/sec and 0.9 GB of
    15.6 GB on a T4, with this phase taking 68% of wall time. Stacking `chunk`
    timesteps makes each pass `chunk * B` wide for the same total work.

    The maths is unchanged: losses are still masked per head and averaged over
    live heads, so a chunk produces the same gradient as the mean of its
    timesteps did.

    `phase_timer`, if given, accumulates CUDA-accurate wall time per sub-phase
    (`prep`/`forward`/`loss`/`backward`/`step`) across every call the caller
    makes it -- pass the same dict in every update to build a cumulative,
    run-long breakdown of what "update" (already known to dominate wall time)
    is actually spending its time on, instead of guessing.

    `board_advantages`/`board_returns` are the GAE outputs for the
    board-level (schedule + ripup) reward stream -- see `PolicyOutput.
    board_value`'s docstring for why that stream, and its own value
    function, has to be separate from the per-head one.
    """
    d = buffer.device
    T = len(buffer)
    old_logp = torch.stack(buffer.log_prob).to(d)
    old_value = torch.stack(buffer.value).to(d)
    masks = torch.stack(buffer.mask).to(d)
    old_sched_logp = torch.stack(buffer.sched_log_prob).to(d)
    old_ripup_logp = torch.stack(buffer.ripup_log_prob).to(d)
    old_board_value = torch.stack(buffer.board_value).to(d)
    amp = bool(cfg.amp and d.type == "cuda")

    if cfg.normalise_advantage:
        live = masks > 0
        if bool(live.any()):
            mu = advantages[live].mean()
            sd = advantages[live].std().clamp_min(1e-6)
            advantages = torch.where(live, (advantages - mu) / sd, torch.zeros_like(advantages))
        # No mask needed here: unlike a head, which can be idle, a board
        # always has SOME state every timestep, so every entry is live.
        board_advantages = (board_advantages - board_advantages.mean()) / \
            board_advantages.std().clamp_min(1e-6)

    stats = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0,
        "forecast": 0.0, "sched_loss": 0.0, "ripup_loss": 0.0, "board_value_loss": 0.0,
    }
    n_updates = 0
    chunk = max(1, min(cfg.chunk, T))
    # Did GradScaler itself decline to apply an update this call? Distinct
    # from a genuinely corrupted model: GradScaler already refuses to step
    # the optimiser when its unscaled gradients contain inf (routine,
    # expected while its scale factor is still calibrating, especially in
    # the first updates after --amp is turned on), but it does not clear
    # `.grad` afterward -- so the stale inf sits there for whoever inspects
    # gradients next. The caller uses this to tell "GradScaler already
    # caught and discarded this" apart from "a real non-finite gradient
    # that WOULD have corrupted the model."
    amp_step_skipped = False

    for _ in range(cfg.epochs):
        order = torch.randperm(T).tolist()
        for start in range(0, T, chunk):
            idx = order[start : start + chunk]
            if not idx:
                continue
            n = len(idx)

            with _timed(phase_timer, "prep", d):
                obs = stack_observations([buffer.observation_at(t) for t in idx], d)
                act = {
                    k: torch.cat([buffer.actions[t][k].to(d) for t in idx], dim=0)
                    for k in buffer.actions[idx[0]]
                }
                rows = lambda src: torch.cat([src[t] for t in idx], dim=0)  # noqa: E731
                m = rows(masks)
                denom = m.sum().clamp_min(1.0)

            want_f = forecast_targets is not None
            with _timed(phase_timer, "forward", d):
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    out = policy.evaluate(obs, act, return_forecast=want_f)
                logp, entropy, value = out[0], out[1], out[2]
                sched_logp, ripup_logp, board_value = out[3], out[4], out[5]
                forecast = out[6] if want_f else None
                if want_f:
                    # Same reasoning as the logp/entropy/value cast below, and
                    # this one was missed the first time AMP was actually run:
                    # `forecast_losses` calls BCE-with-logits and an exp() on
                    # raw conv output that autocast leaves in fp16, and it is
                    # called *outside* the autocast context, so nothing else
                    # casts it back. Confirmed as the cause of the first real
                    # AMP crash -- `h_value.weight`'s gradient going
                    # non-finite despite the value head having nothing to do
                    # with the forecaster; a NaN forecast loss poisons the
                    # whole shared backward graph, not just its own branch.
                    forecast = replace(
                        forecast,
                        final_occupancy=forecast.final_occupancy.float(),
                        contention=forecast.contention.float(),
                        jam_risk=forecast.jam_risk.float(),
                    )

            with _timed(phase_timer, "loss", d):
                # Loss in fp32 regardless of autocast: a PPO ratio is an
                # exponential of a difference of log-probs, and fp16 there is
                # a good way to get a silent inf.
                logp, entropy, value = logp.float(), entropy.float(), value.float()
                sched_logp = sched_logp.float()
                ripup_logp = ripup_logp.float()
                board_value = board_value.float()

                ratio = torch.exp((logp - rows(old_logp)) * m)
                a = rows(advantages)
                unclipped = -a * ratio
                clipped = -a * ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip)
                pl = (torch.maximum(unclipped, clipped) * m).sum() / denom

                ret = rows(returns)
                ov = rows(old_value)
                v_unc = (value - ret) ** 2
                v_cl = (ov + (value - ov).clamp(-cfg.value_clip, cfg.value_clip) - ret) ** 2
                vl = 0.5 * (torch.maximum(v_unc, v_cl) * m).sum() / denom

                ent = (entropy * m).sum() / denom
                total = pl + cfg.value_coef * vl - entropy_coef * ent

                # Board-level PPO terms for schedule and ripup, sharing one
                # advantage (`board_a`): both are decisions about the SAME
                # board state at the SAME timestep, so "did the board do
                # well after this" is one signal, not two. No per-element
                # mask needed -- unlike the per-head terms above, every row
                # here is a real, always-defined board-level decision (see
                # `_ripup`'s docstring for why its log-prob is well-defined
                # even when there is nothing to rip up).
                board_a = rows(board_advantages)

                sched_ratio = torch.exp(sched_logp - rows(old_sched_logp))
                sched_unclipped = -board_a * sched_ratio
                sched_clipped = -board_a * sched_ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip)
                sched_loss = torch.maximum(sched_unclipped, sched_clipped).mean()

                ripup_ratio = torch.exp(ripup_logp - rows(old_ripup_logp))
                ripup_unclipped = -board_a * ripup_ratio
                ripup_clipped = -board_a * ripup_ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip)
                ripup_loss = torch.maximum(ripup_unclipped, ripup_clipped).mean()

                board_ret = rows(board_returns)
                old_bv = rows(old_board_value)
                bv_unc = (board_value - board_ret) ** 2
                bv_cl = (old_bv + (board_value - old_bv).clamp(-cfg.value_clip, cfg.value_clip)
                         - board_ret) ** 2
                board_vl = 0.5 * torch.maximum(bv_unc, bv_cl).mean()

                total = total + sched_loss + ripup_loss + cfg.board_value_coef * board_vl

                # The forecaster is supervised, not reinforced, and its labels
                # are the terminal state of the episode this rollout came from
                # -- one label set per board, so they repeat across the
                # chunk's timesteps.
                f_loss = torch.zeros((), device=d)
                if want_f:
                    tgt = {k: v.repeat(n, *([1] * (v.dim() - 1))) for k, v in forecast_targets.items()}
                    f_loss = sum(forecast_losses(forecast, tgt).values())
                    total = total + cfg.forecast_coef * f_loss

            with _timed(phase_timer, "backward", d):
                optimiser.zero_grad(set_to_none=True)
                if scaler is not None and amp:
                    scaler.scale(total).backward()
                else:
                    total.backward()

            with _timed(phase_timer, "step", d):
                if scaler is not None and amp:
                    scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    scale_before = scaler.get_scale()
                    scaler.step(optimiser)
                    scaler.update()
                    # A shrinking scale after `update()` is GradScaler's own
                    # signal that it saw inf and skipped the step just now --
                    # there is no other public way to ask "did the last step
                    # actually apply". `<` not `!=`: growth_interval can also
                    # *raise* the scale on a healthy step, which must not be
                    # read as a skip.
                    if scaler.get_scale() < scale_before:
                        amp_step_skipped = True
                else:
                    nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    optimiser.step()

            stats["policy_loss"] += float(pl.detach())
            stats["value_loss"] += float(vl.detach())
            stats["entropy"] += float(ent.detach())
            stats["clip_frac"] += float(
                (((ratio - 1.0).abs() > cfg.clip).float() * m).sum().detach() / denom
            )
            stats["forecast"] += float(f_loss.detach())
            stats["sched_loss"] += float(sched_loss.detach())
            stats["ripup_loss"] += float(ripup_loss.detach())
            stats["board_value_loss"] += float(board_vl.detach())
            n_updates += 1

    out = {k: v / max(1, n_updates) for k, v in stats.items()}
    # Not averaged like the metrics above -- a flag, not a per-iteration
    # quantity. `check_model_health`'s caller uses it to tell a routine
    # GradScaler skip apart from a genuinely corrupted model.
    out["amp_step_skipped"] = 1.0 if amp_step_skipped else 0.0
    return out
