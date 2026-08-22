"""Does LineRouteEnv behave against the REAL router, not just the fake?

Everything in pcbworld/env/line_route_env.py is verified against
tests/fake_bridge.py, which never rejects a push() or a fix(). That proves
sequencing and reward bookkeeping, and proves nothing about real router
behaviour. This is the smallest run that closes that gap, before any GPU time
is spent on PPO.

## The check that actually means something

The env's action convention is that **a = 0 walks straight at the target**,
so a policy that always emits 0 IS the greedy straight-line router. And that
router's completion rate on a 24-net board has already been measured
independently: `scripts/measure_waypoint_fidelity.py` got **9/24** direct
straight-push successes (0/24 rescued by its detour ladder).

So the greedy run below has a PREDICTED answer, roughly 9/24. That makes this
a real test rather than a smoke test:

  - Landing near 9/24 means the whole stack -- observation frame, heading
    maths, snap/fix logic, per-net sequencing -- is wired correctly against
    real geometry.
  - Landing near 0/24 means something structural is broken (most likely the
    heading convention, the snap radius vs step size relationship, or pad
    candidate resolution), and PPO would be training against a broken env.
  - Landing much ABOVE 9/24 is not automatically good news: the env takes
    many 1mm steps where the fidelity script took one big push, so some
    improvement is expected from incremental pushing alone -- but a large
    jump is worth understanding before trusting it.

The random run exists as a contrast: it should do clearly worse than greedy.
If random matches greedy, the action is not affecting outcomes and nothing
can learn from it.

Per-step wall clock is also reported. docs/RL_PLAN.md budgets ~0.035ms per
step on the assumption that get_board_geometry() is fetched once per net
rather than per step; this measures whether that held.

Bridge-only: never import pcbnew in this process.

Usage (after notebooks/00_setup.ipynb has built the bridge):
    python3 pcbworld/data/generate_board.py board24.kicad_pcb --num-nets 24 --seed 0
    python3 scripts/smoke_line_env.py board24.kicad_pcb
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

from pcbworld.env.line_obs import LineObsConfig
from pcbworld.env.line_route_env import LineRouteEnv
from scripts.measure_waypoint_fidelity import _load_bridge

MM = 1_000_000
GREEDY_BASELINE = 9  # measured, 24 nets, scripts/measure_waypoint_fidelity.py


def _run_episode(env: LineRouteEnv, policy: str, rng: np.random.Generator):
    obs, info = env.reset()
    assert env.observation_space.contains(obs), "reset() observation is outside the declared space"

    rewards: list[float] = []
    step_times: list[float] = []
    collisions = 0
    guard = 0
    terminated = False
    max_steps = env.max_steps_per_net * len(env._nets) + 10

    while not terminated and guard < max_steps:
        action = (
            np.zeros(1, dtype=np.float32)
            if policy == "greedy"
            else rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
        )
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_times.append(time.perf_counter() - t0)

        assert np.isfinite(obs).all(), f"non-finite observation at step {guard}"
        assert env.observation_space.contains(obs), f"observation left the space at step {guard}"
        assert np.isfinite(reward), f"non-finite reward at step {guard}"
        rewards.append(float(reward))
        collisions += int(bool(info["collides"]))
        guard += 1

    return {
        "policy": policy,
        "completed": list(info["completed"]),
        "failed": list(info["failed"]),
        "num_nets": len(env._nets),
        "steps": guard,
        "terminated": terminated,
        "total_reward": sum(rewards),
        "collision_steps": collisions,
        "step_times": step_times,
    }


def _report(result: dict) -> None:
    done, total = len(result["completed"]), result["num_nets"]
    ms = [t * 1e3 for t in result["step_times"]]
    print(f"\n--- {result['policy']} ---")
    print(f"  routed          : {done}/{total}")
    print(f"  steps taken     : {result['steps']} (terminated={result['terminated']})")
    print(f"  total reward    : {result['total_reward']:.2f}")
    print(f"  steps colliding : {result['collision_steps']}/{result['steps']}")
    if ms:
        print(
            f"  wall clock/step : mean={statistics.mean(ms):.3f}ms "
            f"median={statistics.median(ms):.3f}ms max={max(ms):.3f}ms"
        )
    if result["failed"]:
        print(f"  failed nets     : {result['failed'][:12]}"
              f"{' ...' if len(result['failed']) > 12 else ''}")


def run(board_path: str, num_nets: int, bridge_dir: str | None, seed: int) -> dict:
    _load_bridge(bridge_dir)  # puts pcbworld_pns_bridge on sys.path before the env imports it

    rng = np.random.default_rng(seed)
    results = {}

    for policy in ("greedy", "random"):
        env = LineRouteEnv(
            board_path,
            max_nets=num_nets,
            obs_config=LineObsConfig(k_nearest=32, max_steps=80),
            step_size_nm=1 * MM,
            snap_radius_nm=500_000,   # >= step/2, see LineRouteEnv's docstring
            max_steps_per_net=80,
        )
        try:
            results[policy] = _run_episode(env, policy, rng)
            _report(results[policy])
        finally:
            env.close()

    greedy, random_ = results["greedy"], results["random"]
    g_done, r_done = len(greedy["completed"]), len(random_["completed"])
    total = greedy["num_nets"]

    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    print(
        f"  greedy (a=0) routed {g_done}/{total}; the independently measured straight-line\n"
        f"  baseline on a 24-net board is {GREEDY_BASELINE}/24 "
        f"(scripts/measure_waypoint_fidelity.py).\n"
    )

    if g_done == 0:
        print(
            "  BROKEN: the greedy policy routed nothing. a=0 is meant to BE the straight-line\n"
            "  router, so this is structural -- check the heading convention, snap_radius_nm vs\n"
            "  step_size_nm, and pad candidate resolution before any PPO run. Training against\n"
            "  this env would learn nothing.\n"
        )
    elif g_done < max(1, GREEDY_BASELINE // 3):
        print(
            f"  SUSPECT: {g_done}/{total} is well below the {GREEDY_BASELINE}/24 baseline. The env\n"
            "  is doing something, but not what the straight-line router does. Worth\n"
            "  understanding before training.\n"
        )
    else:
        print(
            "  HEALTHY: greedy is in the range the independent measurement predicts, so the\n"
            "  observation frame, heading maths, snap/fix logic and net sequencing all line up\n"
            "  against real geometry.\n"
        )

    if r_done >= g_done:
        print(
            f"  WARNING: random routed {r_done}/{total}, no worse than greedy. The action may not\n"
            "  be affecting outcomes -- if so there is no gradient for PPO to follow.\n"
        )
    else:
        print(f"  Random routed {r_done}/{total}, below greedy -- the action affects outcomes.\n")

    all_ms = [t * 1e3 for t in greedy["step_times"]]
    if all_ms:
        median = statistics.median(all_ms)
        print(
            f"  Median {median:.3f}ms/step against docs/RL_PLAN.md's ~0.035ms budget "
            f"({'within' if median <= 0.2 else 'ABOVE'} expectation).\n"
            f"  If it is far above, get_board_geometry() is probably being called more than once\n"
            f"  per net -- that is the assumption the budget rests on."
        )
    print("=" * 78)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board_path", help=".kicad_pcb from generate_board.py")
    parser.add_argument("--num-nets", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bridge-dir", default=None)
    args = parser.parse_args()

    run(args.board_path, args.num_nets, args.bridge_dir, args.seed)
    sys.exit(0)
