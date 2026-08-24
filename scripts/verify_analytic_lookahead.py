"""Validates models/analytic_lookahead.py's `_peek_candidate` against real
`PCBRouterEnv.step()` output, BEFORE trusting it inside any action
selector -- same "verify cheaply before building on top of it" discipline
this project has used throughout (see jepa/README.md).

Needs no trained checkpoint and no GPU: the peek function only depends on
board geometry and action decoding, not on any policy, so this drives the
environment with RANDOM actions across many fresh boards and compares, at
every decision, the peek's predicted (new_x, new_y, new_layer, is_collided,
is_connected) against what a real `copy.deepcopy(env).step(action)` (an
untouched, throwaway copy so the ACTUAL episode being driven is unaffected)
actually produces.

is_collided ground truth: `step()` only updates `state.head_x/head_y` when
the move was NOT collided (environment.py ~line 589, inside `if not
is_collided:`) -- so "the real head's (x, y) is unchanged after the step"
is used as the collision proxy. head_layer updates unconditionally
(via/layer-change is applied before the collision check), so it is
compared directly rather than gated the same way. This proxy can misfire
only in the vanishing edge case where a genuinely legal move's rounded
landing cell happens to equal the starting cell exactly, which the minimum
DIST_STEPS value (2) makes effectively impossible here.

Restart handling: a collided action can ALSO push `state.collision_run`
past `max_consecutive_collisions` within that same `step()` call, which
fires `_restart_net` and resets `head_x/head_y` to the net's source pad
(environment.py ~line 326) -- a real, deliberate multi-step-accumulated
side effect a single-step peek has no way to predict (and does not need
to, for the actual selector's purposes: a colliding candidate is already
scored as worst-possible regardless of whether it also happened to be the
one that crossed the restart threshold). Detected here via `restart_count`
(incremented in `step()` right before `_restart_net` fires, and NOT reset
by `_restart_net` itself) and excluded from the position/collision
comparison -- comparing a single-step geometric replay against a
multi-step accumulated jam-and-reset would be testing the wrong thing, not
finding a bug in the peek.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys

from pcbworld.environment import PCBRouterEnv
from models.analytic_lookahead import _peek_candidate, _smoothed_direction_readonly
from scripts.train_ai_router import STAGE_CONFIG, action_dim_for_stage


def run(stage: int, num_episodes: int, max_steps: int, seed_offset: int, seed_py: int):
    random.seed(seed_py)
    stage_cfg = STAGE_CONFIG[stage]
    action_dim = action_dim_for_stage(stage_cfg)

    total_checked = 0
    mismatches = []

    for ep in range(num_episodes):
        env = PCBRouterEnv(
            num_nets=stage_cfg["num_nets"],
            num_obstacles=stage_cfg["num_obstacles"],
            enable_layer_via=stage_cfg["enable_layer_via"],
            max_net_restarts=2,
        )
        obs, _info = env.reset(seed=seed_offset + ep)

        for _step in range(max_steps):
            if env.current_net_idx is None:
                break
            idx = env.current_net_idx
            state = env.net_states[idx]
            active_net = env.board.nets[idx]
            action = random.randrange(action_dim)

            prev_x, prev_y, prev_layer = state.head_x, state.head_y, state.head_layer
            prev_restart_count = state.restart_count
            smoothed_gdx, smoothed_gdy = _smoothed_direction_readonly(env, state, prev_x, prev_y)
            pred_x, pred_y, pred_layer, pred_collided, pred_connected, _pred_dist = _peek_candidate(
                env, state, active_net, action, smoothed_gdx, smoothed_gdy
            )
            pre_call_smoothed = state.smoothed_descent_dir

            sim_env = copy.deepcopy(env)
            sim_env.step(action)
            sim_state = sim_env.net_states[idx]
            restarted = sim_state.restart_count > prev_restart_count
            real_x, real_y, real_layer = sim_state.head_x, sim_state.head_y, sim_state.head_layer
            real_collided = restarted or (real_x == prev_x and real_y == prev_y)
            real_connected = active_net.net_id in sim_env.completed_nets

            total_checked += 1
            if restarted:
                # This action's collision also crossed max_consecutive_collisions
                # within the same step() call -- a multi-step-accumulated jam
                # response the peek is not meant to predict (see module
                # docstring). Skip the position/collision comparison for
                # this decision; still confirm the peek left no residue.
                ok = state.smoothed_descent_dir == pre_call_smoothed
            else:
                ok = (
                    pred_layer == real_layer
                    and pred_collided == real_collided
                    and pred_connected == real_connected
                    and (real_collided or (pred_x == real_x and pred_y == real_y))
                    and state.smoothed_descent_dir == pre_call_smoothed
                )
            if not ok:
                mismatches.append({
                    "episode": ep, "step": _step, "action": action, "restarted": restarted,
                    "prev": (prev_x, prev_y, prev_layer),
                    "pred": (pred_x, pred_y, pred_layer, pred_collided, pred_connected),
                    "real": (real_x, real_y, real_layer, real_collided, real_connected),
                })

            # Drive the REAL episode forward with the same action so
            # trajectories stay realistic (not just first-decision-ever).
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break

    print(f"Checked {total_checked} decisions across {num_episodes} episodes (stage {stage}).")
    if mismatches:
        print(f"*** {len(mismatches)} MISMATCHES -- analytic peek does NOT match real step() ***")
        for m in mismatches[:10]:
            print(f"  ep={m['episode']} step={m['step']} action={m['action']} prev={m['prev']}")
            print(f"    pred={m['pred']}")
            print(f"    real={m['real']}")
        sys.exit(1)
    else:
        print("All decisions matched exactly. analytic_lookahead's peek is verified faithful to real step().")


def main():
    parser = argparse.ArgumentParser(description="Verify models/analytic_lookahead.py's peek against real PCBRouterEnv.step()")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed-offset", type=int, default=500000, help="Disjoint from every other seed block used in this project.")
    parser.add_argument("--seed-py", type=int, default=0)
    args = parser.parse_args()
    run(args.stage, args.num_episodes, args.max_steps, args.seed_offset, args.seed_py)


if __name__ == "__main__":
    main()
