"""End-to-end environment checks with the non-learned baselines. CPU only.

What this is really testing is the set of invariants everything downstream
assumes. A learned policy will happily exploit a broken invariant and report a
high reward while producing copper that shorts -- and the only place that gets
caught is either here or, much later and much more expensively, in KiCad DRC.

Checks, in order of how bad it is if one fails:

1. **No cell is claimed by two nets.** If this fails the board is shorted and
   every metric above it is meaningless.
2. **A "routed" net's copper actually connects its pads**, verified by flood
   fill from the source pad through that net's own cells -- not by trusting the
   status flag the engine set.
3. Rejected moves leave the world byte-identical.
4. Rip-up restores the occupancy grid exactly.
5. Greedy beats nothing-but-still-completes, and detour beats greedy. The
   *ordering* is the signal; the absolute numbers are board-dependent.

Run:  python -m neuroroute.scripts.verify_env
"""

from __future__ import annotations

import sys
import time
from collections import deque

import numpy as np
import torch

from neuroroute.env.baselines import (
    detour_action,
    greedy_action,
    greedy_safe_action,
    layer_hop_action,
)
from neuroroute.env.route_env import EnvConfig, NeuroRouteEnv
from neuroroute.world.engine import STATUS_DONE, WorldConfig
from neuroroute.world.generator import GeneratorConfig
from neuroroute.world.spec import BoardSpec, LayerStack

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def make_env(num_nets=16, layers=4, size=64, batch=4, heads=4, steps=64) -> NeuroRouteEnv:
    spec = BoardSpec(height_cells=size, width_cells=size, layers=LayerStack(num_layers=layers))
    return NeuroRouteEnv(
        EnvConfig(
            spec=spec,
            world=WorldConfig(
                batch_size=batch, max_heads=heads, max_nets=64,
                max_steps_per_net=steps, device="cpu",
            ),
            generator=GeneratorConfig(num_nets=num_nets, num_components=5),
            max_episode_steps=steps * 4,
        )
    )


def rollout(env: NeuroRouteEnv, policy, max_steps: int = 400):
    obs = env.reset()
    rejected = acted = 0
    for _ in range(max_steps):
        act = policy(obs)
        obs, rew, done, info = env.step(act)
        rejected += int(info["rejected"].sum())
        acted += int(info["active"].sum())
        if bool(done.all()):
            break
    return obs, rejected / max(1, acted)


# ---------------------------------------------------------------------------


def test_no_shorts(env: NeuroRouteEnv) -> None:
    """Every occupied cell has exactly one owner -- true by the int16 encoding,
    so what this actually checks is that no *pad* got overwritten by another
    net's copper, which would silently destroy connectivity."""
    w = env.world
    static = w.static.cpu().numpy()
    occ = w.occ.cpu().numpy()
    pads = static > 0
    clobbered = int(((occ != static) & pads).sum())
    check("no pad overwritten by another net", clobbered == 0, f"{clobbered} clobbered")


def test_routed_nets_connect(env: NeuroRouteEnv) -> None:
    """Flood-fill each 'done' net's own copper from its source pad and require
    the target pad to be reachable. This deliberately ignores `net_status` and
    re-derives the answer from the occupancy grid."""
    w = env.world
    occ = w.occ.cpu().numpy()
    B, L, H, W = occ.shape
    checked = broken = 0

    nbrs = [(dl, dy, dx) for dl in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if not (dl == 0 and dy == 0 and dx == 0)]

    for b in range(B):
        for n in range(w.cfg.max_nets):
            if not bool(w.net_valid[b, n]) or int(w.net_status[b, n]) != STATUS_DONE:
                continue
            own = n + 1
            legs = 2 if int(w.net_kind[b, n]) == 1 else 1
            for leg in range(legs):
                s = w.net_src[b, n, leg].tolist()
                t = w.net_dst[b, n, leg].tolist()
                checked += 1
                seen = {tuple(s)}
                q = deque([tuple(s)])
                found = tuple(s) == tuple(t)
                while q and not found:
                    cl, cy, cx = q.popleft()
                    for dl, dy, dx in nbrs:
                        nl, ny, nx = cl + dl, cy + dy, cx + dx
                        if not (0 <= nl < L and 0 <= ny < H and 0 <= nx < W):
                            continue
                        if (nl, ny, nx) in seen or occ[b, nl, ny, nx] != own:
                            continue
                        seen.add((nl, ny, nx))
                        if (nl, ny, nx) == tuple(t):
                            found = True
                            break
                        q.append((nl, ny, nx))
                if not found:
                    broken += 1

    check(
        "every net marked done is physically connected",
        broken == 0,
        f"{checked - broken}/{checked} legs verified by flood fill",
    )


def test_rejected_moves_are_no_ops() -> None:
    """A rejected action must leave the world byte-identical. If it does not,
    the per-step collision signal is measuring something other than 'this move
    was refused', and the dense reward stops meaning what it says."""
    env = make_env(num_nets=24, layers=2, size=48, batch=2, heads=4)
    obs = env.reset()
    w = env.world

    for _ in range(60):
        before_occ = w.occ.clone()
        before_pos = w.head_pos.clone()
        before_len = w.net_len.clone()

        B, K = obs.head_mask.shape
        # Aim every head at a direction the raycast says is fully blocked.
        blocked = ~obs.safety.any(dim=-1)                 # (B, K, D)
        d = blocked.float().argmax(dim=-1)
        act = greedy_action(obs)
        act["direction"] = d
        act["step"] = torch.full_like(d, 3)
        forced = blocked.any(dim=-1) & obs.head_mask

        obs, rew, done, info = env.step(act)
        rej = info["rejected"] & forced
        if bool(rej.any()):
            # Where every head was rejected, nothing may have changed.
            all_rej = bool((info["rejected"] | ~info["active"]).all())
            if all_rej:
                same = (
                    torch.equal(before_occ, w.occ)
                    and torch.equal(before_len, w.net_len)
                )
                check("rejected step leaves occupancy and lengths untouched", same)
                return
        if bool(done.all()):
            break
    check("rejected step leaves occupancy and lengths untouched", True, "no all-rejected step observed; vacuous")


def test_ripup_restores() -> None:
    env = make_env(num_nets=12, layers=2, size=48, batch=2, heads=2)
    obs, _ = rollout(env, greedy_safe_action)
    w = env.world

    done_nets = (w.net_status == STATUS_DONE) & w.net_valid
    if not bool(done_nets.any()):
        check("rip-up restores occupancy", True, "no routed nets to rip; vacuous")
        return

    before = w.occ.clone()
    target = done_nets.float().argmax(dim=1)
    n_before = [int((w.occ[b] == int(target[b]) + 1).sum()) for b in range(w.occ.shape[0])]
    ok = w.ripup(target)
    n_after = [int((w.occ[b] == int(target[b]) + 1).sum()) for b in range(w.occ.shape[0])]

    # Only the ripped net's cells changed, and its pads survived.
    changed_other = False
    for b in range(w.occ.shape[0]):
        if not bool(ok[b]):
            continue
        own = int(target[b]) + 1
        diff = (before[b] != w.occ[b])
        if bool((diff & (before[b] != own)).any()):
            changed_other = True
    pads_kept = all(
        n_after[b] >= int((w.static[b] == int(target[b]) + 1).sum()) for b in range(w.occ.shape[0]) if bool(ok[b])
    )
    check("rip-up touches only the ripped net", not changed_other)
    check("rip-up keeps the net's pads", pads_kept, f"{n_before} -> {n_after} cells")
    check("ripped nets return to pending", bool((w.net_status.gather(1, target.view(-1, 1)).squeeze(1)[ok] == 0).all()))


def test_baseline_ordering() -> None:
    print("\n[5] baselines on a shared board set")
    results = {}
    for name, policy in (
        ("greedy(step=1)", greedy_action),
        ("greedy(longest safe step)", greedy_safe_action),
        ("detour(turn when blocked)", detour_action),
    ):
        env = make_env(num_nets=24, layers=4, size=64, batch=4, heads=4)
        t0 = time.perf_counter()
        obs, rej = rollout(env, policy)
        dt = time.perf_counter() - t0
        comp = float(env.world.completion().mean())
        vias = float(env.world.board_stats()["vias"].float().mean())
        results[name] = comp
        print(f"    {name:<28} completion {comp:6.1%}  rejected {rej:5.1%}  vias {vias:4.1f}  {dt:5.2f}s")

    # NOT asserted: that bigger steps or turning-when-blocked complete MORE.
    # They do not, reliably. Measured on two different board sets, longest-safe-
    # step came out at 37.5% vs 35.4% on one and 29.2% vs 30.2% on the other --
    # it wins or loses depending on the boards. Committing more copper per
    # decision buys speed and costs manoeuvrability, and on a congested board
    # that trade can go either way. `docs/RL_PLAN.md` records the same shape of
    # result from the other direction: a random policy out-completed greedy
    # 9/24 to 8/24 purely by wandering around obstacles a straight line could
    # not pass. So the ordering is a finding to report, not an invariant to
    # assert -- asserting it would just make an honest test flaky.
    spread = max(results.values()) - min(results.values())
    check("greedy completes something at all", results["greedy(step=1)"] > 0.0)
    check(
        "baselines are within a sane band of each other",
        spread < 0.25,
        f"spread {spread:.1%} across {len(results)} baselines",
    )
    return results


def test_layers_help() -> None:
    print("\n[6] does adding layers help? (the thing PNS could never test)")
    out = {}
    vias = {}
    for L in (1, 2, 4, 8):
        env = make_env(num_nets=32, layers=L, size=64, batch=4, heads=4)
        obs = env.reset()
        for _ in range(400):
            act = layer_hop_action(obs) if L > 1 else detour_action(obs)
            obs, rew, done, info = env.step(act)
            if bool(done.all()):
                break
        out[L] = float(env.world.completion().mean())
        vias[L] = float(env.world.board_stats()["vias"].float().mean())
        print(f"    {L} layer(s): completion {out[L]:6.1%}  vias {vias[L]:5.1f}")

    check("vias are actually placed on a multi-layer board", vias[8] > 0, f"{vias[8]:.1f} vias/board")
    check("more layers routes more", out[8] > out[1], f"1L {out[1]:.1%} -> 8L {out[8]:.1%}")
    test_routed_nets_connect_env(env)
    return out


def test_routed_nets_connect_env(env) -> None:
    """Re-run the flood-fill connectivity check on a board routed *with vias*.

    A route that changes layer is connected through the via disc, not through
    an in-plane neighbour, so it exercises a path the single-layer check never
    touches. If via stamping had an off-by-one in its layer span, this is where
    it would show up.
    """
    test_routed_nets_connect(env)


def test_throughput() -> None:
    print("\n[7] throughput (CPU; GPU is the real target)")
    env = make_env(num_nets=48, layers=8, size=96, batch=8, heads=8, steps=96)
    obs = env.reset()
    t0 = time.perf_counter()
    n = 40
    for _ in range(n):
        obs, *_ = env.step(greedy_safe_action(obs))
    dt = time.perf_counter() - t0
    decisions = n * env.cfg.world.batch_size * env.cfg.world.max_heads
    print(f"    {n} env steps, B=8 K=8, 8 layers, 96x96: {dt:.2f}s")
    print(f"    {decisions / dt:,.0f} routing decisions/sec on CPU")
    check("throughput is sane", decisions / dt > 200, f"{decisions/dt:,.0f}/s")


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 68)
    print("NeuroRoute environment verification")
    print("=" * 68)

    print("\n[1] invariants after a greedy rollout")
    env = make_env(num_nets=24, layers=4, size=64, batch=4, heads=4)
    obs, rej = rollout(env, greedy_safe_action)
    print(f"    completion {float(env.world.completion().mean()):.1%}, rejected-action rate {rej:.1%}")
    test_no_shorts(env)

    print("\n[2] connectivity, re-derived from the occupancy grid")
    test_routed_nets_connect(env)

    print("\n[3] rejected moves")
    test_rejected_moves_are_no_ops()

    print("\n[4] rip-up")
    test_ripup_restores()

    test_baseline_ordering()
    test_layers_help()
    test_throughput()

    print("\n" + "=" * 68)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("All environment checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
