"""Analytic (non-learned) fast lookahead action selection.

An additive, opt-in alternative to `lookahead_select_action`'s real-simulator
forward search (`models/router_policy.py`) -- same idea (rank the policy's
top-K candidate actions by where they actually lead, instead of committing
to the single best immediate one) but without simulating anything or
learning anything.

Why this exists, and why it doesn't touch a neural network for scoring:
`jepa/`'s four independent negative results (see `jepa/README.md` and
`models/fast_lookahead.py`'s docstring) all tried to DECODE distance-to-
target from some embedding -- pooled, per-token, even the checkpoint's own
co-trained value_head -- and all four failed, on real data, including a
clean 100%-completion run that ruled out the earlier short-episode
confound. The distance value was never actually missing, though: it's
already a materialized numpy array. `PCBRouterEnv` computes
`state.geodesic_cache` ONCE per net attempt (`environment.py`'s
`_init_net_state`, unchanged for the rest of that attempt) and a candidate
action's resulting position is pure deterministic geometry
(`environment.py`'s `step`, ~lines 545-572) -- neither is a network's job
to reconstruct from pixels. So instead of predicting, this module REPLAYS:
for each of the policy's top-K candidate actions, it computes exactly
where that action would land using the environment's own read-only helper
methods (not reimplemented -- called directly, to avoid drift from the
proven `step()` logic), then reads the already-computed geodesic field at
that landing spot. No `copy.deepcopy(env)`, no real `env.step()`, no extra
network forward pass beyond the ONE encoder+policy pass already needed to
produce the candidate list (shared with plain argmax).

Horizon=1 only, deliberately: this scores each candidate's single
immediate landing spot, not a chained multi-step rollout like
`lookahead_select_action`'s horizon=4. This directly targets the failure
mode this project measured and named: a 2-cycle where the locally-best
action at A leads to B, and B's own locally-best action leads right back
to A. Plain `select_deterministic_action` never compares alternatives at
all (it just emits the top logit); a 1-step comparison across the top-K
candidates already breaks a 2-cycle, because whichever candidate doesn't
regress the geodesic distance wins. Chaining this further (horizon>1)
would need a small "shadow state" carried across hypothetical steps
(position, layer, heading-smoothing history, hypothetically-drawn copper)
without deep-copying the whole env -- a real, separately-scoped extension,
deliberately not built here. If horizon=1 doesn't clear the known hard
seeds, that is the natural next increment, not a sign to abandon this
approach.

Mutation safety -- the one real subtlety here, worth stating plainly:
`PCBRouterEnv._smoothed_descent_dir` is the ONLY helper method this module
calls that has a side effect -- it writes `state.smoothed_descent_dir` (an
exponential moving average over this net's recent steps). Calling it
naively once per candidate would over-smooth that history on every call
(each call treats the PREVIOUS call's output as "history"), and leaving
its mutation in place after a peek would leak into the real env's next
real `step()` call -- silently changing the actual trajectory versus a run
without this selector, which this project's "the proven path must stay
provably unchanged" discipline would not tolerate. Fixed two ways: (1) the
smoothed direction is computed ONCE per real decision, not once per
candidate -- all top-K candidates share the same current position, so they
would get the identical value anyway, exactly what real `step()` itself
does; (2) that one call snapshots `state.smoothed_descent_dir` first and
restores it immediately after reading the result, so the peek is fully
read-only from the real environment's point of view. `_check_line_collision`,
`_geo_dist_at`, `_bilinear_sample`, and `decode_action` are all read-only by
inspection (this module never calls `_rasterize_line`, the only
copper-grid-mutating method, or anything else that writes board/net state).

Does NOT touch `pcbworld/environment.py`, `models/router_policy.py`'s
existing functions, or `jepa/` -- purely additive, same isolation
discipline `models/fast_lookahead.py` already established. If this doesn't
pan out either, `--lookahead` remains the proven fallback (100%/1000 on the
full stage-2 benchmark, just slow).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from pcbworld.environment import DIST_STEPS


def _smoothed_direction_readonly(env, state, x: float, y: float) -> Tuple[float, float]:
    """Reads the current smoothed descent direction via the real (mutating)
    `_smoothed_descent_dir`, then restores `state.smoothed_descent_dir` to
    its pre-call value -- see module docstring's "Mutation safety" section.
    """
    saved = state.smoothed_descent_dir
    gdx, gdy = env._smoothed_descent_dir(state, x, y)
    state.smoothed_descent_dir = saved
    return gdx, gdy


def _peek_candidate(
    env, state, active_net, action: int, smoothed_gdx: float, smoothed_gdy: float,
) -> Tuple[int, int, int, bool, bool, float]:
    """Read-only: computes where `action` would move the active net's head,
    without mutating env/state or drawing anything into the copper grid.
    Mirrors `PCBRouterEnv.step` (environment.py, ~lines 515-611) exactly,
    calling its own read-only helper methods rather than reimplementing
    their math -- see module docstring.

    Returns (new_x, new_y, new_layer, is_collided, is_connected, dist_geo).
    `dist_geo` is only meaningful when not `is_collided` -- a collided
    candidate's landing spot was never actually reached.
    """
    target = active_net.target_pad
    dir_idx, dist_idx, layer_change, via_flag = env.decode_action(action)
    step_dist = DIST_STEPS[dist_idx]
    prev_x, prev_y, prev_layer = state.head_x, state.head_y, state.head_layer

    dir_x, dir_y = env._bearing_vector(
        smoothed_gdx, smoothed_gdy, dir_idx, prev_x, prev_y, target.x, target.y
    )
    new_x = int(round(prev_x + dir_x * step_dist))
    new_y = int(round(prev_y + dir_y * step_dist))
    # Via/layer-change applies unconditionally in the real step(), even if
    # the move itself then collides (see environment.py lines 538-543) --
    # mirrored here the same way, not gated on is_collided below.
    new_layer = (1 - prev_layer) if (via_flag == 1 or layer_change == 1) else prev_layer

    out_of_bounds = new_x < 0 or new_x >= env.grid_size or new_y < 0 or new_y >= env.grid_size
    is_collided = out_of_bounds or env._check_line_collision(
        prev_x, prev_y, new_x, new_y, new_layer, active_net.net_id
    )

    dist_geo = env._geo_dist_at(state.geodesic_cache, new_x, new_y)
    euclid_to_target = math.hypot(new_x - target.x, new_y - target.y)
    is_connected = (
        not is_collided and euclid_to_target <= env.snap_radius and new_layer == target.layer
    )

    return new_x, new_y, new_layer, is_collided, is_connected, dist_geo


def analytic_lookahead_select_action(
    model,
    env,
    obs_np,
    device_str: str,
    forbidden: set,
    top_k: int = 4,
) -> int:
    """Same top-K candidate-ranking shape as `lookahead_select_action`, but
    scores each candidate by replaying the environment's own deterministic
    movement/collision math against the already-computed geodesic field --
    no simulation, no learned predictor, no extra encoder forward pass
    beyond the one shared with plain argmax. See module docstring.

    Only reasons about the CURRENT decision (horizon=1) -- unlike
    `lookahead_select_action`, there is no multi-step rollout to interrupt
    if round-robin (num_nets > 1) rotates control elsewhere, since nothing
    beyond this one step is simulated.
    """
    idx = env.current_net_idx
    if idx is None:
        return 0
    state = env.net_states[idx]
    active_net = env.board.nets[idx]

    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device_str).unsqueeze(0)
    with torch.no_grad():
        dist, _ = model(obs_t)
    logits = dist.logits.squeeze(0) if dist.logits.dim() > 1 else dist.logits
    ranked = [a for a in torch.argsort(logits, descending=True).tolist() if a not in forbidden]
    if not ranked:
        ranked = torch.argsort(logits, descending=True).tolist()
    candidates = ranked[:top_k]

    # Shared across all candidates -- they all start from the same current
    # position, so real step() would compute this exact same value
    # regardless of which candidate is eventually chosen. See
    # _smoothed_direction_readonly's docstring for why this must be read
    # exactly once per decision, not once per candidate.
    smoothed_gdx, smoothed_gdy = _smoothed_direction_readonly(env, state, state.head_x, state.head_y)

    best_action = candidates[0]
    best_score = float("inf")
    for cand in candidates:
        _nx, _ny, _nl, is_collided, is_connected, dist_geo = _peek_candidate(
            env, state, active_net, cand, smoothed_gdx, smoothed_gdy
        )
        if is_connected:
            score = -1.0  # strictly best -- an immediate connect beats any distance comparison
        elif is_collided:
            score = float("inf")  # illegal move -- never preferred over a legal one
        else:
            score = dist_geo
        if score < best_score:
            best_score = score
            best_action = cand

    return best_action
