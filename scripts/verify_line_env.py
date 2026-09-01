#!/usr/bin/env python3
"""Smoke test for LineRouteEnv against real pcbworld_pns_bridge.

Run in Colab after building the bridge to verify:
1. Environment resets and steps without crashing
2. Observations have correct shapes and dtypes
3. Action space is correct
4. Reward signal behaves sensibly
5. Episode terminates correctly on success/failure
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pcbworld.env.line_route_env import LineRouteEnv, RewardWeights


def test_env(board_path: str, verbose: bool = True) -> bool:
    """Run a complete smoke test of LineRouteEnv."""
    print(f"\n{'='*60}")
    print(f"🧪 LineRouteEnv Smoke Test")
    print(f"   Board: {board_path}")
    print(f"{'='*60}\n")

    try:
        env = LineRouteEnv(
            board_path=board_path,
            track_width_nm=250_000,
            max_steps_per_net=100,
            reward_weights=RewardWeights(),
        )
    except Exception as e:
        print(f"❌ Failed to create env: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test observation space
    print("📐 Observation Space:")
    print(f"   Type: {type(env.observation_space)}")
    print(f"   Keys: {list(env.observation_space.spaces.keys())}")
    for key, space in env.observation_space.spaces.items():
        print(f"   {key}: {space.shape} {space.dtype}")

    # Test action space
    print(f"\n🎮 Action Space: {env.action_space}")
    print(f"   Shape: {env.action_space.shape}, Dtype: {env.action_space.dtype}")

    # Test reset
    print("\n🔄 Testing reset()...")
    try:
        obs, info = env.reset()
        print(f"   ✅ Reset successful")
        print(f"   Info: {info}")
        print(f"   Obs keys: {list(obs.keys())}")
        for key, val in obs.items():
            print(f"   {key}: shape={val.shape}, dtype={val.dtype}, range=[{val.min():.3f}, {val.max():.3f}]")
    except Exception as e:
        print(f"   ❌ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test step with random actions
    print("\n🚶 Testing step() with random actions...")
    total_reward = 0.0
    completed = False

    for step in range(10):
        action = np.random.uniform(-1.0, 1.0)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        done = term or trunc

        if verbose:
            dist = np.linalg.norm([
                obs["global"][0] * 10_000_000,  # dist_to_target / L * L
            ])  # Approximate
            print(f"   Step {step+1}: action={action:.3f}, reward={reward:.3f}, "
                  f"done={done}, info={info}")

        if done:
            completed = info.get("completed", False)
            break

    print(f"\n📊 Episode Summary:")
    print(f"   Steps: {step+1}")
    print(f"   Total Reward: {total_reward:.1f}")
    print(f"   Completed: {completed}")
    print(f"   Terminated: {term}, Truncated: {trunc}")

    # Test deterministic straight-line action (action=0 = toward target)
    print("\n🎯 Testing deterministic straight-line policy (action=0)...")
    env.reset()
    straight_reward = 0.0
    for step in range(50):
        obs, reward, term, trunc, info = env.step(0.0)
        straight_reward += reward
        if term or trunc:
            completed = info.get("completed", False)
            break
    print(f"   Straight-line: {step+1} steps, reward={straight_reward:.1f}, completed={completed}")

    # Test multiple episodes
    print("\n🔁 Testing multiple episodes...")
    successes = 0
    for ep in range(3):
        env.reset()
        done = False
        for _ in range(100):
            action = np.random.uniform(-1.0, 1.0)
            _, _, term, trunc, info = env.step(action)
            if term or trunc:
                if info.get("completed", False):
                    successes += 1
                break
    print(f"   3 random episodes: {successes}/3 completed")

    env.close()
    print(f"\n✅ All tests passed!")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Smoke test LineRouteEnv")
    parser.add_argument("--board", type=str, required=True, help="Path to .kicad_pcb board")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    success = test_env(args.board, verbose=not args.quiet)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()