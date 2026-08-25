"""Validates models/pcb_encoder.py's new spatial-awareness pieces (head
position recovery, raycast sensor, per-(direction,distance) safety mask,
local-crop extraction) against independent brute-force reference
implementations, BEFORE trusting them inside any training run -- same
"verify cheaply before building on top of it" discipline as
scripts/verify_analytic_lookahead.py.

Needs no trained checkpoint and no GPU: everything here depends only on
board/observation geometry (built via real PCBRouterEnv episodes, same as
verify_analytic_lookahead.py) and the encoder's forward pass on random
weights, not on any trained policy.

Four independent checks:
1. Head position recovery: argmax over Channel 3 must exactly reproduce the
   env's own ground-truth (state.head_x, state.head_y).
2. Raycast sensor: the vectorized torch implementation
   (PCBEncoder._raycast_sensor) must exactly match a brute-force Python
   pixel-walk along the same 8 bearings, written independently (loops, not
   gather) so the two implementations can't share a bug.
3. Per-(direction, distance) safety mask: same brute-force pixel-walk,
   additionally checked against each of the environment's actual
   DIST_STEPS values -- this is the finer-grained collision-reduction
   signal (see DIST_SAFETY_SUPPRESSION in pcb_encoder.py), independent from
   #2's coarser per-direction-only reading.
4. Local-crop extraction: PCBEncoder._local_crop must exactly match a
   brute-force numpy pad+slice, including at board edges/corners (head near
   x=0/255 or y=0/255) where off-by-one errors are most likely.

Also runs a full PCBRouterNet forward pass (random weights, both the
d_model=256 config every training/eval script in this repo actually uses,
and the d_model=512 class default) end-to-end to confirm shapes and check
for NaN/Inf, since a shape or dtype mistake in the new code paths would
otherwise only surface deep inside a Colab training run.
"""

from __future__ import annotations

import argparse
import math
import random
import sys

import numpy as np
import torch

from pcbworld.environment import PCBRouterEnv, DIST_STEPS
from models.pcb_encoder import PCBEncoder, RAYCAST_MAX_STEPS, RAYCAST_NUM_DIRS, LOCAL_CROP_PAD, LOCAL_CROP_SIZE
from models.router_policy import PCBRouterNet


def brute_force_raycast(obs: np.ndarray, head_x: int, head_y: int):
    """Returns (raycast, dist_safe): raycast is the coarse per-direction
    [0,1] reading; dist_safe[d][k] is whether a hop of DIST_STEPS[k] cells
    along direction d is collision-free. Both derived from the same
    independently-computed `first_blocked` per direction, matching
    PCBEncoder._raycast_sensor's semantics exactly but via plain Python
    loops instead of vectorized gather."""
    H, W = obs.shape[1], obs.shape[2]

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    x0 = clamp(head_x - 1, 0, W - 1)
    x1 = clamp(head_x + 1, 0, W - 1)
    y0 = clamp(head_y - 1, 0, H - 1)
    y1 = clamp(head_y + 1, 0, H - 1)
    field7 = obs[7]
    ddx = float(field7[head_y, x0]) - float(field7[head_y, x1])
    ddy = float(field7[y0, head_x]) - float(field7[y1, head_x])
    norm = math.hypot(ddx, ddy)
    if norm < 1e-6:
        gdx, gdy = 1.0, 0.0
    else:
        gdx, gdy = ddx / norm, ddy / norm
    ref_bearing = math.atan2(gdy, gdx)

    raycast = []
    dist_safe = []
    for d in range(RAYCAST_NUM_DIRS):
        angle = ref_bearing + d * (math.pi / 4.0)
        dx, dy = math.cos(angle), math.sin(angle)
        first_blocked = None
        for s in range(1, RAYCAST_MAX_STEPS + 1):
            sx = clamp(int(round(head_x + dx * s)), 0, W - 1)
            sy = clamp(int(round(head_y + dy * s)), 0, H - 1)
            if obs[0, sy, sx] > 0.5 or obs[1, sy, sx] > 0.5:
                first_blocked = s
                break
        if first_blocked is None:
            first_blocked = RAYCAST_MAX_STEPS + 1
        raycast.append((first_blocked - 1) / float(RAYCAST_MAX_STEPS))
        dist_safe.append([1.0 if first_blocked > step else 0.0 for step in DIST_STEPS])
    return raycast, dist_safe


def brute_force_crop(obs: np.ndarray, head_x: int, head_y: int) -> np.ndarray:
    C, H, W = obs.shape
    pad = LOCAL_CROP_PAD
    padded = np.zeros((C, H + 2 * pad, W + 2 * pad), dtype=obs.dtype)
    padded[:, pad:pad + H, pad:pad + W] = obs
    padded[1, :, :pad] = 1.0
    padded[1, :, -pad:] = 1.0
    padded[1, :pad, :] = 1.0
    padded[1, -pad:, :] = 1.0
    return padded[:, head_y:head_y + LOCAL_CROP_SIZE, head_x:head_x + LOCAL_CROP_SIZE]


def check_geometry(num_episodes: int, max_steps: int, seed_offset: int, seed_py: int) -> int:
    random.seed(seed_py)
    encoder = PCBEncoder(in_channels=10, d_model=32, num_transformer_layers=1, num_heads=2)
    encoder.eval()

    total_checked = 0
    head_mismatches = []
    raycast_mismatches = []
    dist_safe_mismatches = []
    crop_mismatches = []

    for ep in range(num_episodes):
        env = PCBRouterEnv(num_nets=3, num_obstacles=6, enable_layer_via=True, max_net_restarts=2)
        obs, _info = env.reset(seed=seed_offset + ep)

        for step in range(max_steps):
            if env.current_net_idx is None:
                break
            idx = env.current_net_idx
            state = env.net_states[idx]
            head_x, head_y = state.head_x, state.head_y

            x_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                rec_x, rec_y = PCBEncoder._head_position(x_t)
                raycast_t, dist_safe_t = encoder._raycast_sensor(x_t, rec_x, rec_y)
                crop_t = encoder._local_crop(x_t, rec_x, rec_y)
            rec_x, rec_y = int(rec_x.item()), int(rec_y.item())

            total_checked += 1
            if (rec_x, rec_y) != (head_x, head_y):
                head_mismatches.append((ep, step, (head_x, head_y), (rec_x, rec_y)))

            expected_ray, expected_dist_safe = brute_force_raycast(obs, head_x, head_y)
            got_ray = raycast_t.squeeze(0).tolist()
            if any(abs(a - b) > 1e-5 for a, b in zip(expected_ray, got_ray)):
                raycast_mismatches.append((ep, step, expected_ray, got_ray))

            got_dist_safe = dist_safe_t.squeeze(0).tolist()
            if expected_dist_safe != got_dist_safe:
                dist_safe_mismatches.append((ep, step, expected_dist_safe, got_dist_safe))

            expected_crop = brute_force_crop(obs, head_x, head_y)
            got_crop = crop_t.squeeze(0).numpy()
            if not np.allclose(expected_crop, got_crop, atol=1e-5):
                bad = np.argwhere(~np.isclose(expected_crop, got_crop, atol=1e-5))
                crop_mismatches.append((ep, step, len(bad), bad[:3].tolist()))

            action = random.randrange(24 if not env.enable_layer_via else 96)
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break

    print(f"Checked {total_checked} decisions across {num_episodes} episodes.")
    ok = True
    if head_mismatches:
        ok = False
        print(f"*** {len(head_mismatches)} HEAD POSITION MISMATCHES ***")
        for m in head_mismatches[:5]:
            print(f"  {m}")
    else:
        print("Head position recovery: all matched exactly.")

    if raycast_mismatches:
        ok = False
        print(f"*** {len(raycast_mismatches)} RAYCAST MISMATCHES ***")
        for m in raycast_mismatches[:5]:
            print(f"  ep={m[0]} step={m[1]}\n    expected={m[2]}\n    got={m[3]}")
    else:
        print("Raycast sensor: all matched brute-force reference exactly.")

    if dist_safe_mismatches:
        ok = False
        print(f"*** {len(dist_safe_mismatches)} DIST-SAFETY MISMATCHES ***")
        for m in dist_safe_mismatches[:5]:
            print(f"  ep={m[0]} step={m[1]}\n    expected={m[2]}\n    got={m[3]}")
    else:
        print("Per-(direction,distance) safety mask: all matched brute-force reference exactly.")

    if crop_mismatches:
        ok = False
        print(f"*** {len(crop_mismatches)} LOCAL-CROP MISMATCHES ***")
        for m in crop_mismatches[:5]:
            print(f"  ep={m[0]} step={m[1]} num_bad_elems={m[2]} sample_bad_idx={m[3]}")
    else:
        print("Local-crop extraction: all matched brute-force reference exactly.")

    return 0 if ok else 1


def check_edge_positions() -> int:
    """Force head positions to the four corners and edge midpoints -- the
    off-by-one-prone cases random episodes may not reliably hit."""
    encoder = PCBEncoder(in_channels=10, d_model=32, num_transformer_layers=1, num_heads=2)
    encoder.eval()
    grid = 256
    positions = [
        (0, 0), (grid - 1, 0), (0, grid - 1), (grid - 1, grid - 1),
        (0, 128), (grid - 1, 128), (128, 0), (128, grid - 1),
    ]
    rng = np.random.default_rng(12345)
    ok = True
    for hx, hy in positions:
        obs = np.zeros((10, grid, grid), dtype=np.float32)
        obs[1] = (rng.random((grid, grid)) < 0.1).astype(np.float32)  # random obstacles
        obs[0] = (rng.random((grid, grid)) < 0.05).astype(np.float32)  # random copper
        y_coords, x_coords = np.ogrid[:grid, :grid]
        dist_sq = (x_coords - hx) ** 2 + (y_coords - hy) ** 2
        obs[3] = np.exp(-0.5 * dist_sq / 16.0).astype(np.float32)
        obs[7] = (rng.random((grid, grid))).astype(np.float32)  # arbitrary smooth-ish field

        x_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            rec_x, rec_y = PCBEncoder._head_position(x_t)
            raycast_t, dist_safe_t = encoder._raycast_sensor(x_t, rec_x, rec_y)
            crop_t = encoder._local_crop(x_t, rec_x, rec_y)
        rec_x, rec_y = int(rec_x.item()), int(rec_y.item())

        if (rec_x, rec_y) != (hx, hy):
            print(f"*** EDGE HEAD MISMATCH at ({hx},{hy}): recovered ({rec_x},{rec_y}) ***")
            ok = False
            continue

        expected_ray, expected_dist_safe = brute_force_raycast(obs, hx, hy)
        got_ray = raycast_t.squeeze(0).tolist()
        if any(abs(a - b) > 1e-5 for a, b in zip(expected_ray, got_ray)):
            print(f"*** EDGE RAYCAST MISMATCH at ({hx},{hy}): expected={expected_ray} got={got_ray} ***")
            ok = False

        got_dist_safe = dist_safe_t.squeeze(0).tolist()
        if expected_dist_safe != got_dist_safe:
            print(f"*** EDGE DIST-SAFETY MISMATCH at ({hx},{hy}): expected={expected_dist_safe} got={got_dist_safe} ***")
            ok = False

        expected_crop = brute_force_crop(obs, hx, hy)
        got_crop = crop_t.squeeze(0).numpy()
        if not np.allclose(expected_crop, got_crop, atol=1e-5):
            print(f"*** EDGE CROP MISMATCH at ({hx},{hy}) ***")
            ok = False

    if ok:
        print(f"Edge/corner positions ({len(positions)} tested): all matched exactly.")
    return 0 if ok else 1


def check_forward_pass() -> int:
    ok = True
    for d_model, num_layers, num_heads, action_dim in [
        (256, 2, 4, 96),
        (256, 2, 4, 24),
        (512, 4, 8, 96),
    ]:
        model = PCBRouterNet(
            in_channels=10, action_dim=action_dim, d_model=d_model,
            num_transformer_layers=num_layers, num_heads=num_heads,
        )
        model.eval()
        obs = torch.randn(3, 10, 256, 256)
        # Channel 3 needs a real peak for head-position recovery to be
        # meaningful (random noise would give an arbitrary argmax, which is
        # still a valid *shape* test but not a valid *geometry* test).
        for b in range(3):
            hx, hy = np.random.randint(0, 256, size=2)
            y_coords, x_coords = np.ogrid[:256, :256]
            dist_sq = (x_coords - hx) ** 2 + (y_coords - hy) ** 2
            obs[b, 3] = torch.as_tensor(np.exp(-0.5 * dist_sq / 16.0), dtype=torch.float32)
        with torch.no_grad():
            dist, value = model(obs)
        if dist.logits.shape != (3, action_dim):
            print(f"*** BAD LOGITS SHAPE for d_model={d_model}, action_dim={action_dim}: {dist.logits.shape} ***")
            ok = False
        if value.shape != (3, 1):
            print(f"*** BAD VALUE SHAPE for d_model={d_model}, action_dim={action_dim}: {value.shape} ***")
            ok = False
        if not torch.isfinite(dist.logits).all() or not torch.isfinite(value).all():
            print(f"*** NaN/Inf in output for d_model={d_model}, action_dim={action_dim} ***")
            ok = False
    if ok:
        print("Full PCBRouterNet forward pass: shapes and finiteness OK for all tested configs.")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description="Verify the spatial-awareness additions in models/pcb_encoder.py")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed-offset", type=int, default=600000, help="Disjoint from every other seed block used in this project.")
    parser.add_argument("--seed-py", type=int, default=0)
    args = parser.parse_args()

    results = [
        check_geometry(args.num_episodes, args.max_steps, args.seed_offset, args.seed_py),
        check_edge_positions(),
        check_forward_pass(),
    ]
    sys.exit(1 if any(results) else 0)


if __name__ == "__main__":
    main()
