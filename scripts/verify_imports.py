#!/usr/bin/env python3
"""Verify all imports work correctly for the line-geometry PCB router.

Run this in Colab after installing requirements to verify the codebase loads.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    print("🧪 Verifying imports...\n")

    # Core packages
    try:
        import numpy as np
        print(f"  ✅ numpy {np.__version__}")
    except ImportError as e:
        print(f"  ❌ numpy: {e}")
        return False

    try:
        import gymnasium as gym
        print(f"  ✅ gymnasium {gym.__version__}")
    except ImportError as e:
        print(f"  ❌ gymnasium: {e}")
        return False

    try:
        import torch
        print(f"  ✅ torch {torch.__version__} (cuda={torch.cuda.is_available()})")
    except ImportError as e:
        print(f"  ❌ torch: {e}")
        return False

    try:
        import matplotlib
        print(f"  ✅ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ❌ matplotlib: {e}")
        return False

    # Project modules
    modules_to_test = [
        ("pcbworld", "pcbworld"),
        ("pcbworld.env", "pcbworld.env"),
        ("models", "models"),
        ("training", "training"),
    ]

    for mod_name, import_path in modules_to_test:
        try:
            __import__(import_path)
            print(f"  ✅ {import_path}")
        except ImportError as e:
            print(f"  ❌ {import_path}: {e}")
            return False

    # Specific classes
    classes_to_test = [
        ("LineRouteEnv", "pcbworld.env.line_route_env"),
        ("LineDiffPairTuneEnv", "pcbworld.env.line_diff_pair_tune_env"),
        ("LineGeometryPolicy", "models.line_geometry_policy"),
        ("RolloutBuffer", "training.replay_buffer"),
        ("RewardScaler", "training.reward_scaling"),
        ("train_line_policy", "training.train_line_policy"),
    ]

    for class_name, module_path in classes_to_test:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            getattr(mod, class_name)
            print(f"  ✅ {module_path}.{class_name}")
        except (ImportError, AttributeError) as e:
            print(f"  ❌ {module_path}.{class_name}: {e}")
            return False

    # Test observation/action space creation
    print("\n🔧 Testing environment spaces...")
    try:
        from pcbworld.env.line_route_env import LineRouteEnv
        # Just test space creation without bridge
        obs_space = LineRouteEnv.__new__(LineRouteEnv).observation_space
        act_space = LineRouteEnv.__new__(LineRouteEnv).action_space
        print(f"  ✅ LineRouteEnv observation_space: {obs_space}")
        print(f"  ✅ LineRouteEnv action_space: {act_space}")
    except Exception as e:
        print(f"  ❌ Space test: {e}")
        return False

    try:
        from pcbworld.env.line_diff_pair_tune_env import LineDiffPairTuneEnv
        obs_space = LineDiffPairTuneEnv.__new__(LineDiffPairTuneEnv).observation_space
        act_space = LineDiffPairTuneEnv.__new__(LineDiffPairTuneEnv).action_space
        print(f"  ✅ LineDiffPairTuneEnv observation_space: {obs_space}")
        print(f"  ✅ LineDiffPairTuneEnv action_space: {act_space}")
    except Exception as e:
        print(f"  ❌ LineDiffPairTuneEnv space test: {e}")
        return False

    # Test model instantiation
    print("\n🧠 Testing model instantiation...")
    try:
        from models.line_geometry_policy import LineGeometryPolicy
        model = LineGeometryPolicy()
        print(f"  ✅ LineGeometryPolicy: {sum(p.numel() for p in model.parameters()):,} params")
    except Exception as e:
        print(f"  ❌ Model test: {e}")
        return False

    print("\n✅ All imports verified successfully!")
    return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)