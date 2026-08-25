"""PCBRouterNet: Full Reinforcement Learning Actor-Critic Architecture.

Combines:
1. Spatial PCBEncoder (CNN + Transformer Backbone, plus local-attention
   pool, non-learned raycast sensor, and local-crop CNN -- see
   models/pcb_encoder.py and docs/WORLD_MODEL_SPATIAL_DESIGN.md -> combined
   latent)
2. NetSelectorHead (Optional Net Attention)
3. 96-Action Router Policy Head (Categorical Action Distribution), plus two
   direct non-learned-geometry -> logit-bias paths: a per-direction one and
   a finer per-(direction, distance) one, so a blocked direction/distance
   is discouraged by construction, not only by hoping the rest of the
   network learns it
4. Value Critic Head (Scalar Board Return Estimate)
"""

from __future__ import annotations

import copy
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.distributions import Categorical

from models.pcb_encoder import (
    PCBEncoder,
    RAYCAST_NUM_DIRS,
    DIST_SAFETY_DIM,
    DIST_SAFETY_SUPPRESSION,
    combined_latent_dim,
)
from models.net_selector import NetSelectorHead


class PCBRouterNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        action_dim: int = 96,
        d_model: int = 512,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model

        # 1. State Encoder (CNN + Transformer)
        self.encoder = PCBEncoder(
            in_channels=in_channels,
            d_model=d_model,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
        )

        # 2. Net Selector Head
        self.net_selector = NetSelectorHead(d_model=d_model)

        # PCBEncoder now returns a wider latent (whole-board mean-pool +
        # local-attention pool + raycast + per-(dir,dist) safety mask +
        # local-crop CNN, see models/pcb_encoder.py) -- size the heads from
        # that combined width instead of d_model directly, since it's
        # derived, not d_model itself.
        head_input_dim = combined_latent_dim(d_model)

        # 3. Router Action Policy Head (96 Discrete Growth Actions)
        self.policy_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )

        # 4. Value Critic Head (Predicts Future Expected PCB Return)
        self.value_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # 5. Direct raycast -> logit-bias path (see
        # docs/WORLD_MODEL_SPATIAL_DESIGN.md's confidence assessment for why
        # this is separate from policy_head rather than only concatenated
        # into its input). action_dim must divide evenly by the 8 dir_idx
        # values -- true for both the 24-action space (dir*3) and the
        # 96-action space (dir*12).
        assert action_dim % RAYCAST_NUM_DIRS == 0, (
            f"action_dim={action_dim} must be a multiple of {RAYCAST_NUM_DIRS} "
            "(one block of actions per dir_idx)"
        )
        self.actions_per_dir = action_dim // RAYCAST_NUM_DIRS
        self.raycast_to_logit_bias = nn.Linear(RAYCAST_NUM_DIRS, RAYCAST_NUM_DIRS)
        # Init so a direction's raycast reading directly discourages/favors
        # its own dir_idx from step 0 -- raycast in [0,1] (0=blocked now,
        # 1=free through the full cap), centered at 0.5 so a blocked
        # direction gets a negative bias and a free one gets a positive
        # bias, not just "less positive". scale=2.0 spans a ~7.4x
        # (e^2) relative-weight swing between fully-blocked and fully-free
        # -- a real nudge, same spirit as (and roughly the same order of
        # magnitude as) the dir_idx==0 +0.5 tilt below, but per-board and
        # per-step instead of a fixed constant. This does NOT depend on the
        # rest of the network learning the "blocked direction -> avoid it"
        # association -- it's structurally present before any training
        # happens, and training only has to refine it.
        with torch.no_grad():
            nn.init.eye_(self.raycast_to_logit_bias.weight)
            self.raycast_to_logit_bias.weight.mul_(2.0)
            nn.init.constant_(self.raycast_to_logit_bias.bias, -1.0)

        # 6. Direct per-(direction, distance) collision-suppression path --
        # a finer-grained sibling of raycast_to_logit_bias above. That path
        # only discriminates by DIRECTION, so it cannot express "dist=2 is
        # fine here but dist=8 collides" -- exactly the gap that showed up
        # as repeated same-spot REJECTED-collision retries in practice
        # (checkpoints_stage2_v8_spatial, seed 9764 trace). dist_safe is an
        # exact (non-learned) boolean per (dir_idx, dist_idx) pair, so this
        # is a FIXED additive bias, not a learned Linear like the one above
        # -- see DIST_SAFETY_SUPPRESSION's docstring in pcb_encoder.py for
        # why this one specifically should not be trainable away.
        assert action_dim % DIST_SAFETY_DIM == 0, (
            f"action_dim={action_dim} must be a multiple of {DIST_SAFETY_DIM} "
            "(one block of actions per (dir_idx, dist_idx) pair)"
        )
        self.actions_per_distpair = action_dim // DIST_SAFETY_DIM

        # Initialize policy head near zero to prevent violent initial divergence.
        # gain=0.01 (not smaller) matters here in a way it wouldn't without
        # the bias below: measured directly, gain=0.01 made the weight-driven
        # logit contribution std=0.0017 -- ~1200x smaller than the +2.0 bias
        # that used to sit on top of it. Two genuinely different boards then
        # produced final-logit differences of ~0.0004, so the deterministic
        # policy was, in effect, ALWAYS the fixed bias regardless of what the
        # encoder learned: gradient descent would have had to grow this
        # layer's weights by three orders of magnitude to compete, which
        # ~1000-1500 minibatch updates measurably did not do (an untrained
        # random model and four independently-trained checkpoints all scored
        # 37/50 on the same fixed eval boards). gain=0.1 puts the
        # weight-driven contribution within reach of training instead.
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.1)
        nn.init.constant_(self.policy_head[-1].bias, 0.0)

        # Bias the dir_idx == 0 actions ("toward the target", or around
        # whatever the geodesic field says is in the way -- see
        # PCBRouterEnv._bearing_vector) higher at init. This is
        # the discrete analogue of line_route_env.py's continuous
        # mean-zero-action trick: instead of an untrained policy sampling
        # uniformly over 8 board-pose-dependent directions, it starts
        # already preferring the one direction that generalizes across
        # every board. +0.5 logit ~= e^0.5 ~= 1.65x relative weight -- a
        # real nudge, not the +2.0 (~7.4x, dominating ~51% of initial mass
        # on 3/24 actions) that made this un-overcomable in the first place.
        with torch.no_grad():
            bias = self.policy_head[-1].bias
            if action_dim == 24:      # dir_idx*3 + dist_idx
                bias[0:3] += 0.5
            elif action_dim == 96:    # dir_idx*12 + dist_idx*4 + layer*2 + via
                bias[0:12] += 0.5

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[Categorical, torch.Tensor]:
        """
        Forward pass for action distribution and state value.
        Args:
            obs: (B, 10, 256, 256) spatial observation tensor
        Returns:
            dist: Categorical distribution over 96 discrete actions
            value: (B, 1) state value estimate
        """
        pcb_latent, _patch_tokens, raycast_vector, dist_safe = self.encoder(obs)

        # Compute Action Logits & Distribution
        action_logits = self.policy_head(pcb_latent)
        dir_bias = self.raycast_to_logit_bias(raycast_vector)  # (B, 8)
        action_logits = action_logits + dir_bias.repeat_interleave(self.actions_per_dir, dim=-1)

        B = dist_safe.shape[0]
        distpair_bias = (dist_safe.reshape(B, -1) - 1.0) * DIST_SAFETY_SUPPRESSION  # (B, 24): 0 if safe, -SUPPRESSION if not
        action_logits = action_logits + distpair_bias.repeat_interleave(self.actions_per_distpair, dim=-1)

        dist = Categorical(logits=action_logits)

        # Compute State Value
        value = self.value_head(pcb_latent)

        return dist, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate or sample action for PPO / Actor-Critic step.
        Returns:
            action: (B,) sampled or passed action
            log_prob: (B,) log probability of action
            entropy: (B,) distribution entropy
            value: (B, 1) state value baseline
        """
        dist, value = self.forward(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


def select_deterministic_action(dist: Categorical, forbidden: set[int]) -> int:
    """argmax, but skipping actions already tried and rejected at the
    CURRENT position since the last successful move.

    Plain argmax retries the identical action forever once a move is
    rejected: a rejected move barely changes the observation (only the
    Channel-9 rejection marker's intensity ticks up), so nothing shifts
    which action wins the argmax. Measured directly on 5/5 boards that got
    stuck: the deterministic policy picked the exact same action 6 times in
    a row before max_consecutive_collisions gave up on the net -- the
    collision-retry budget PCBRouterEnv provides is real for a STOCHASTIC
    policy (training's sampling naturally varies), but structurally inert
    for a deterministic one. `forbidden` is the caller's job to maintain:
    add the action just rejected, clear it the instant a move succeeds.

    Training's rollout collection deliberately does NOT use this -- it
    would decouple the sampled action from dist.log_prob(action), corrupting
    the PPO ratio between collection and update. This is for deterministic
    evaluation/deployment only, where there is no log-prob to keep
    consistent.
    """
    logits = dist.logits.squeeze(0) if dist.logits.dim() > 1 else dist.logits
    for idx in torch.argsort(logits, descending=True).tolist():
        if idx not in forbidden:
            return idx
    return int(torch.argmax(logits).item())  # every action forbidden; shouldn't happen


def lookahead_select_action(
    model: "PCBRouterNet",
    env,
    obs_np,
    device_str: str,
    forbidden: set,
    top_k: int = 4,
    horizon: int = 4,
) -> int:
    """Pick an action via a shallow forward search instead of committing to
    the single best immediate one.

    select_deterministic_action is purely reactive: one observation in, one
    action out, no lookahead beyond what that single observation encodes.
    Measured directly (render_episode.py --verbose traces on seeds 9148/9251
    under an earlier checkpoint): at a tight multi-obstacle intersection, a
    fully-trained deterministic policy can get caught in a stable CYCLE
    between the same few cells, because from wherever it currently stands
    the locally-best action leads to a position whose own locally-best
    action leads right back. A non-learned oracle graph search never hits
    this, because it can look several steps ahead and discover a direction
    that looks worse RIGHT NOW pays off in 2-3 steps -- exactly what a
    single-step-reactive choice structurally cannot represent.

    This borrows that same idea without touching the trained weights: for
    each of the policy's top-K candidate first actions, simulate `horizon`
    steps forward on a throwaway deep copy of the env (continuing greedily
    with the SAME policy for the simulated steps too), and commit to
    whichever real first action's simulated rollout got closest to the
    target (or actually connected, or avoided leading to failure). Same
    network, same weights -- more compute spent at decision time instead of
    learned in advance. Materially slower per step (~top_k*horizon extra
    env steps and forward passes) -- meant for targeted investigation of
    specific hard boards, not routine bulk benchmarking.

    Only reasons about ONE net's own trajectory: if round-robin (num_nets >
    1) rotates control to a different net mid-simulation, the simulated
    rollout for that candidate stops there rather than feed an action meant
    for this net to whichever net actually became active.
    """
    idx = env.current_net_idx
    if idx is None:
        return 0
    active_net = env.board.nets[idx]

    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device_str).unsqueeze(0)
    with torch.no_grad():
        dist, _ = model(obs_t)
    logits = dist.logits.squeeze(0) if dist.logits.dim() > 1 else dist.logits
    ranked = [a for a in torch.argsort(logits, descending=True).tolist() if a not in forbidden]
    if not ranked:
        ranked = torch.argsort(logits, descending=True).tolist()
    candidates = ranked[:top_k]

    best_action = candidates[0]
    best_score = float("inf")

    for cand in candidates:
        sim_env = copy.deepcopy(env)
        sim_forbidden: set = set()
        state = sim_env.net_states[idx]
        sim_prev_head = (state.head_x, state.head_y)
        action = cand
        score = float("inf")

        for _ in range(horizon):
            if sim_env.current_net_idx != idx:
                break
            sim_obs, _reward, term, trunc, info = sim_env.step(action)
            if active_net.net_id in sim_env.completed_nets:
                score = 0.0
                break
            if active_net.net_id in sim_env.failed_nets:
                score = float("inf")
                break
            state = sim_env.net_states[idx]
            score = min(score, sim_env._geo_dist_at(state.geodesic_cache, state.head_x, state.head_y))
            if term or trunc:
                break
            new_head = info["acted_head_pos"][:2]
            if new_head == sim_prev_head:
                sim_forbidden.add(action)
            else:
                sim_forbidden = set()
            sim_prev_head = new_head
            sim_obs_t = torch.as_tensor(sim_obs, dtype=torch.float32, device=device_str).unsqueeze(0)
            with torch.no_grad():
                sim_dist, _ = model(sim_obs_t)
            action = select_deterministic_action(sim_dist, sim_forbidden)

        if score < best_score:
            best_score = score
            best_action = cand

    return best_action
