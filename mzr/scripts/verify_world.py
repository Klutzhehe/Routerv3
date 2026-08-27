"""Local verification of the simultaneous-frontier world. No GPU, no KiCad.

`mzr/DESIGN.md` build order step 1. The discipline this follows is the one that
found every real bug in NeuroRoute: **re-derive the answer from the occupancy
grid rather than trusting the engine's own status flags.** Every bug in that
project's list shared one shape -- a metric that looked plausible while the
thing under it was broken -- and the checks that caught them all worked by
independently recomputing what the engine claimed.

Specifically, `test_done_nets_are_connected` is the check that found the worst
bug in NeuroRoute: with several frontiers acting in one batched step, two both
passed a legality test against the pre-step occupancy and both wrote the same
cell. One silently lost, its frontier advanced anyway, and its route was left
with a hole in it. No reward curve would ever have shown that. A flood fill
does, immediately.

Run:
    python -m mzr.scripts.verify_world
"""

from __future__ import annotations

import argparse
import sys
from collections import deque

import numpy as np
import torch

from mzr.world.engine import (
    STATUS_DONE,
    STATUS_ROUTING,
    SimultaneousRouterWorld,
    WorldConfig,
)
from mzr.world.generator import GeneratorConfig, generate_board
from mzr.world.spec import (
    END_DST,
    END_SRC,
    KIND_DIFF_PAIR,
    NUM_ENDS,
    BoardSpec,
    LayerStack,
    PriceRules,
    RipupRules,
)

_FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _FAILED.append(name)


def build(
    *,
    nets: int = 6,
    layers: int = 2,
    size: int = 48,
    batch: int = 4,
    seed0: int = 0,
    ripup_interval: int = 0,
    steps: int = 48,
) -> SimultaneousRouterWorld:
    spec = BoardSpec(
        height_cells=size, width_cells=size, layers=LayerStack(num_layers=layers)
    )
    gcfg = GeneratorConfig(num_nets=nets, num_components=4, pin_pitch_cells=4)
    boards = [generate_board(spec, gcfg, seed=seed0 + i) for i in range(batch)]
    wcfg = WorldConfig(
        batch_size=batch,
        max_nets=max(8, nets + 4),
        max_macro_steps=steps,
        max_steps_per_frontier=steps,
        ripup=RipupRules(interval=ripup_interval),
        price=PriceRules(),
    )
    world = SimultaneousRouterWorld(spec, wcfg)
    world.load(boards)
    return world


def greedy_actions(world: SimultaneousRouterWorld, *, via: bool = False):
    """All-zero actions = direction 0 (down the geodesic gradient), step 1.

    This is the greedy baseline the egocentric frame is designed to make the
    zero action, so it doubles as a check that the frame is wired correctly:
    if action 0 were not "toward the target", completion here would be near 0.
    """
    B, F = world.cfg.batch_size, world.F
    z = torch.zeros(B, F, dtype=torch.long, device=world.device)
    if not via:
        return z, z, z, z, z, z
    # layer_hop: drop to the target's layer when the frontier is not on it.
    tgt_layer = world._target_pad()[..., 0]
    cur_layer = world.fr_pos[..., 0]
    la = torch.where(tgt_layer != cur_layer, tgt_layer + 1, torch.zeros_like(tgt_layer))
    return z, z, la, z, z, z


def rollout(world: SimultaneousRouterWorld, *, via: bool = False, max_steps: int = 200):
    n = 0
    while not world.episode_done() and n < max_steps:
        world.step(*greedy_actions(world, via=via))
        n += 1
    return n


# ---------------------------------------------------------------------------
# Structural checks -- these are about the *simultaneous* design specifically
# ---------------------------------------------------------------------------


def test_all_nets_live_at_load() -> None:
    """Every valid net is routing from step 0, with both ends seeded.

    This is the architecture, stated as a test. NeuroRoute had `K` head slots
    and a scheduler deciding who occupied them; that scheduler never trained,
    so it degenerated to greedy sequential routing. Here there is no queue and
    no assignment -- if this check ever fails, the design has quietly regressed
    to the thing it was built to replace.
    """
    w = build(nets=6, batch=3)
    valid = w.net_valid
    routing = (w.net_status == STATUS_ROUTING) & valid
    check(
        "every valid net is ROUTING at load",
        bool((routing == valid).all()),
        f"{int(routing.sum())}/{int(valid.sum())} nets",
    )

    legs = w.leg_valid.unsqueeze(-1).expand(-1, -1, -1, NUM_ENDS).reshape(w.cfg.batch_size, w.F)
    expect = legs & valid.view(*valid.shape, 1, 1).expand(-1, -1, 2, NUM_ENDS).reshape(
        w.cfg.batch_size, w.F
    )
    check(
        "every valid leg has both frontiers alive",
        bool((w.fr_alive == expect).all()),
        f"{int(w.fr_alive.sum())} frontiers, expected {int(expect.sum())}",
    )

    pos = w.fr_pos.view(w.cfg.batch_size, w.cfg.max_nets, 2, NUM_ENDS, 3)
    on_pad = (pos == w.net_pad).all(dim=-1)
    live = w.leg_valid.unsqueeze(-1) & valid.unsqueeze(-1).unsqueeze(-1)
    check(
        "every frontier starts on its own pad",
        bool((on_pad | ~live).all()),
        "",
    )


def test_frontiers_target_opposite_pads() -> None:
    """END_SRC aims at the dst pad and END_DST at the src pad.

    Getting this backwards would make both frontiers walk away from each other
    and nothing would ever meet -- but completion would degrade smoothly rather
    than crash, so it is exactly the kind of error that hides.
    """
    w = build(nets=4, batch=2)
    B = w.cfg.batch_size
    tgt = w._target_pad().view(B, w.cfg.max_nets, 2, NUM_ENDS, 3)
    ok_src = (tgt[:, :, :, END_SRC] == w.net_pad[:, :, :, END_DST]).all()
    ok_dst = (tgt[:, :, :, END_DST] == w.net_pad[:, :, :, END_SRC]).all()
    check("frontier targets are the opposite end's pad", bool(ok_src and ok_dst))


# ---------------------------------------------------------------------------
# Occupancy-derived checks -- ignore the engine's flags, recompute the truth
# ---------------------------------------------------------------------------


def test_no_pad_overwritten() -> None:
    """No net's copper sits on another net's pad.

    Every occupied cell has exactly one owner by the int16 encoding, so what
    this really checks is that a pad -- which is an obstacle from step 0 --
    never got clobbered, which would silently destroy connectivity for a net
    that has not started yet.
    """
    w = build(nets=8, layers=2, batch=4)
    rollout(w, via=True)
    static = w.static.cpu().numpy()
    occ = w.occ.cpu().numpy()
    pads = static > 0
    clobbered = int(((occ != static) & pads).sum())
    check("no pad overwritten by another net", clobbered == 0, f"{clobbered} clobbered")


def _flood_connected(occ: np.ndarray, b: int, own: int, src, dst) -> bool:
    """Is `dst` reachable from `src` through cells owned by `own` alone?

    26-connected, so a via -- which stamps the same net id on a disc across
    every layer it spans -- is traversed the same way an in-plane neighbour is.
    Connectivity through vias is therefore *derived*, never taken from a flag.
    """
    L, H, W = occ.shape[1:]
    nbrs = [
        (dl, dy, dx)
        for dl in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dl == 0 and dy == 0 and dx == 0)
    ]
    s, t = tuple(src), tuple(dst)
    if s == t:
        return True
    seen = {s}
    q = deque([s])
    while q:
        cl, cy, cx = q.popleft()
        for dl, dy, dx in nbrs:
            nl, ny, nx = cl + dl, cy + dy, cx + dx
            if not (0 <= nl < L and 0 <= ny < H and 0 <= nx < W):
                continue
            p = (nl, ny, nx)
            if p in seen or occ[b, nl, ny, nx] != own:
                continue
            if p == t:
                return True
            seen.add(p)
            q.append(p)
    return False


def test_done_nets_are_connected(*, layers: int = 2, via: bool = True) -> None:
    """Every net marked DONE is physically connected, end to end.

    THE check. It deliberately ignores `net_status`/`leg_done` and re-derives
    the answer from `occ`. This is what caught NeuroRoute's simultaneous-write
    bug (30/31 legs connected -- one route with an invisible hole), and no
    aggregate metric would have shown it.
    """
    w = build(nets=8, layers=layers, batch=4, ripup_interval=0)
    rollout(w, via=via)
    occ = w.occ.cpu().numpy()
    checked = broken = 0
    for b in range(w.cfg.batch_size):
        for n in range(w.cfg.max_nets):
            if not bool(w.net_valid[b, n]) or int(w.net_status[b, n]) != STATUS_DONE:
                continue
            legs = 2 if int(w.net_kind[b, n]) == KIND_DIFF_PAIR else 1
            for leg in range(legs):
                src = w.net_pad[b, n, leg, END_SRC].tolist()
                dst = w.net_pad[b, n, leg, END_DST].tolist()
                checked += 1
                if not _flood_connected(occ, b, n + 1, src, dst):
                    broken += 1
    check(
        f"every DONE net is connected by flood fill ({layers}L, via={via})",
        broken == 0,
        f"{checked - broken}/{checked} legs verified",
    )


def test_polyline_matches_copper() -> None:
    """Every recorded polyline vertex sits on copper owned by its own net.

    The polyline is what the KiCad exporter emits, so if it drifts from `occ`
    the exported board is not the board that was routed -- and the DRC gate
    would be validating a fiction.
    """
    w = build(nets=8, layers=2, batch=3)
    rollout(w, via=True)
    occ = w.occ.cpu().numpy()
    v = w.route_v.cpu().numpy()
    cnt = w.route_n.cpu().numpy()
    bad = total = 0
    for b in range(w.cfg.batch_size):
        for f in range(w.F):
            n = f // (2 * NUM_ENDS)
            if not bool(w.net_valid[b, n]):
                continue
            for i in range(int(cnt[b, f])):
                lay, y, x = (int(t) for t in v[b, f, i])
                total += 1
                if occ[b, lay, y, x] != n + 1:
                    bad += 1
    check(
        "every polyline vertex sits on its own net's copper",
        bad == 0,
        f"{total - bad}/{total} vertices",
    )


# ---------------------------------------------------------------------------
# Step-mechanics checks
# ---------------------------------------------------------------------------


def test_rejected_steps_are_no_ops() -> None:
    """An all-illegal macro-step must leave the board byte-identical.

    Tested as a per-frontier invariant on ordinary boards rather than by
    contriving a board where everything fails, because "everything fails" is
    surprisingly hard to arrange and easy to get wrong: sealing every free cell
    to keepout does *not* stop a frontier, since a net may legally cross its
    own copper and so can always retrace its own trace. That attempt reported
    "44 moved, 0 rejected" on a fully sealed board and looked like an engine
    bug; it was the test misunderstanding the rule.

    What would fail here is a rejected action that advances its frontier
    anyway. That is the dangerous direction: the frontier walks on while its
    copper does not, leaving a route with a hole in it that only a flood fill
    would ever find -- the exact bug simultaneous stepping caused in NeuroRoute.

    Frontiers that connected this step are excluded: `_try_snap` / `_try_meet`
    legitimately move a frontier onto its target after the move phase, so a
    frontier can be both "rejected" and correctly relocated in the same step.
    """
    w = build(nets=12, layers=2, batch=4)
    B, F = w.cfg.batch_size, w.F
    z = torch.zeros(B, F, dtype=torch.long)

    checked = violations = 0
    for _ in range(24):
        if w.episode_done():
            break
        pos_before = w.fr_pos.clone()
        occ_before = w.occ.clone()
        done_before = w.leg_done.clone()
        res = w.step(z, z, z, z, z, z)

        newly = (w.leg_done & ~done_before).unsqueeze(-1)
        newly = newly.expand(B, w.cfg.max_nets, 2, NUM_ENDS).reshape(B, F)
        stuck = res.rejected & ~newly
        checked += int(stuck.sum())
        violations += int((stuck & (w.fr_pos != pos_before).any(dim=-1)).sum())

        # Copper may only be added, never reassigned: a cell that already
        # belonged to a net must still belong to the same net.
        had = occ_before != 0
        violations += int((had & (w.occ != occ_before)).sum())

    check(
        "a rejected frontier does not advance, and copper is never reassigned",
        violations == 0,
        f"{checked} rejected frontier-steps examined, {violations} violations",
    )


def test_determinism() -> None:
    """Same boards + same actions -> byte-identical occupancy.

    Arbitration resolves contested cells by lowest net id specifically so the
    outcome does not depend on scatter order, which is not deterministic. If
    this fails, replaying a trajectory for a PPO/MuZero update would not
    reproduce the trajectory that generated it.
    """
    occs, cmps = [], []
    for _ in range(2):
        w = build(nets=10, layers=2, batch=3, seed0=100)
        rollout(w, via=True)
        occs.append(w.occ.clone())
        cmps.append(w.completion().clone())
    check(
        "two identical rollouts produce identical boards",
        bool(torch.equal(occs[0], occs[1])) and bool(torch.equal(cmps[0], cmps[1])),
        f"completion {cmps[0].mean():.4f} vs {cmps[1].mean():.4f}",
    )


def test_price_tracks_contention() -> None:
    """The congestion price rises where and only where nets actually contend.

    The price is the negotiation substrate -- if it never fires, the whole
    simultaneous design degenerates into an unmanaged traffic jam, and that
    failure would be invisible in a completion number alone.
    """
    w = build(nets=20, layers=2, size=48, batch=2)
    saw_contention = False
    peak = 0.0
    for _ in range(24):
        if w.episode_done():
            break
        res = w.step(*greedy_actions(w, via=True))
        if int(res.contended.sum()) > 0:
            saw_contention = True
        peak = max(peak, float(res.congestion.max()))
    check(
        "dense boards produce measured contention",
        saw_contention,
        f"peak congestion {peak:.2f}",
    )

    w2 = build(nets=1, layers=1, size=48, batch=2)
    rollout(w2, via=False)
    check(
        "a one-net board produces no contention",
        float(w2.price.present.max()) == 0.0 and float(w2.price.history.max()) == 0.0,
        f"present max {float(w2.price.present.max()):.3f}",
    )


def test_price_decays() -> None:
    """Present congestion decays fast, history slowly.

    Without decay on `present`, a single early collision would read as
    permanent congestion; without *slow* decay on `history`, the price could
    not break the two-net oscillation it exists to break.
    """
    w = build(nets=20, layers=2, size=48, batch=2)
    for _ in range(10):
        if w.episode_done():
            break
        w.step(*greedy_actions(w, via=True))
    p0 = float(w.price.present.sum())
    h0 = float(w.price.history.sum())
    if p0 <= 0.0:
        check("price decay observable", False, "no contention occurred to decay")
        return
    for _ in range(4):
        w.price.decay()
    p1 = float(w.price.present.sum())
    h1 = float(w.price.history.sum())
    check(
        "present congestion decays faster than history",
        p1 < p0 and h1 < h0 and (p1 / p0) < (h1 / h0),
        f"present {p0:.2f}->{p1:.2f}, history {h0:.2f}->{h1:.2f}",
    )


def test_ripup_frees_copper_and_keeps_history() -> None:
    """Rip-up returns copper and nets to routing, but does NOT clear history.

    That is the entire mechanism: a corridor that has been fought over must
    stay expensive after the retreat, or the same nets regrow into it and
    collide in exactly the same place. Clearing history here would turn
    negotiation into an infinite reset loop.
    """
    w = build(nets=16, layers=2, size=48, batch=2, ripup_interval=0)
    for _ in range(16):
        if w.episode_done():
            break
        w.step(*greedy_actions(w, via=True))
    hist_before = float(w.price.history.sum())
    copper_before = int((w.occ > 0).sum())
    ripped = w.ripup_round()
    copper_after = int((w.occ > 0).sum())
    hist_after = float(w.price.history.sum())

    check(
        "rip-up removed copper",
        int(ripped.sum()) == 0 or copper_after < copper_before,
        f"{int(ripped.sum())} nets ripped, copper {copper_before}->{copper_after}",
    )
    check(
        "rip-up preserved historical congestion",
        hist_after >= hist_before - 1e-6,
        f"history {hist_before:.3f}->{hist_after:.3f}",
    )
    static_pads = int((w.static > 0).sum())
    kept = int(((w.occ > 0) & (w.static > 0)).sum())
    check(
        "rip-up restored the ripped nets' pads",
        kept == static_pads,
        f"{kept}/{static_pads} pad cells present",
    )


def test_horizon_is_independent_of_net_count() -> None:
    """The load-bearing claim of DESIGN.md section 1.

    A sequential router's episode grows with net count; this one's should not,
    because every frontier moves every macro-step. If the steps-to-settle here
    scaled with `n`, the horizon argument that makes a learned latent model
    viable would be false -- and it is far cheaper to find that out now than
    after building the model.

    Measured as macro-steps to reach **half of the completion that config
    eventually achieves** -- a settling time, normalised per config.

    Two weaker measures were tried first and both are traps worth naming.
    *Steps until the episode ends* saturates at the step cap, because a few
    stuck frontiers oscillate until their budget runs out however easy the
    board was; it reports a flat line whatever the truth is. *Steps to 50%
    absolute completion* measures board difficulty instead of settling speed --
    greedy scores ~28% on a 20-net board (NeuroRoute, [LIVE]), so it never
    reaches the threshold at all and the test fails for a reason that has
    nothing to do with the horizon claim.
    """
    CAP = 120
    steps: dict[int, int] = {}
    finals: dict[int, float] = {}
    for n in (2, 8, 24):
        w = build(nets=n, layers=1, size=64, batch=4, steps=CAP)
        curve = []
        for _ in range(CAP):
            if w.episode_done():
                break
            w.step(*greedy_actions(w, via=False))
            curve.append(float(w.completion().mean()))
        final = curve[-1] if curve else 0.0
        finals[n] = final
        half = 0.5 * final
        steps[n] = next((i + 1 for i, c in enumerate(curve) if c >= half), CAP)

    # The claim is directional -- the horizon must not GROW with net count --
    # so the test is one-sided. A sequential router's episode grows linearly:
    # 12x the nets would be ~12x the steps. Anything markedly sublinear
    # supports the claim; the horizon coming out *shorter* at 24 nets does not
    # violate it.
    ratio_steps = steps[24] / max(1, steps[2])
    ratio_nets = 24 / 2
    check(
        "macro-steps to settle grow sublinearly in net count",
        ratio_steps < 0.5 * ratio_nets,
        f"{steps} macro-steps for {ratio_nets:.0f}x the nets "
        f"= {ratio_steps:.2f}x the steps (sequential would be ~{ratio_nets:.0f}x); "
        f"final completion { {k: round(v, 2) for k, v in finals.items()} }",
    )


def test_greedy_baseline_is_sane() -> None:
    """The zero action must *be* the greedy router.

    The egocentric frame exists so an untrained, near-zero-init policy starts
    at the greedy baseline rather than below it. If direction 0 were not "down
    the geodesic gradient", completion here would collapse toward zero -- so
    this doubles as a wiring check on the whole observation/action frame.
    """
    w = build(nets=4, layers=1, size=48, batch=8, steps=48)
    rollout(w, via=False)
    c = float(w.completion().mean())
    check(
        "greedy (all-zero action) completes most single-layer nets",
        c > 0.5,
        f"completion {c:.3f}",
    )

    w2 = build(nets=6, layers=4, size=48, batch=8, steps=64)
    rollout(w2, via=True)
    c2 = float(w2.completion().mean())
    w3 = build(nets=6, layers=4, size=48, batch=8, steps=64)
    rollout(w3, via=False)
    c3 = float(w3.completion().mean())
    check(
        "layer_hop beats no-via greedy on a multi-layer board",
        c2 > c3,
        f"layer_hop {c2:.3f} vs greedy {c3:.3f} -- the gap is cross-layer nets",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="skip the slower scale checks")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    print("\n== structure: the simultaneous design itself ==")
    test_all_nets_live_at_load()
    test_frontiers_target_opposite_pads()

    print("\n== occupancy-derived truth ==")
    test_no_pad_overwritten()
    test_done_nets_are_connected(layers=1, via=False)
    test_done_nets_are_connected(layers=4, via=True)
    test_polyline_matches_copper()

    print("\n== step mechanics ==")
    test_rejected_steps_are_no_ops()
    test_determinism()

    print("\n== congestion price ==")
    test_price_tracks_contention()
    test_price_decays()
    test_ripup_frees_copper_and_keeps_history()

    print("\n== design claims ==")
    test_greedy_baseline_is_sane()
    if not args.quick:
        test_horizon_is_independent_of_net_count()

    print()
    if _FAILED:
        print(f"FAILED ({len(_FAILED)}): " + ", ".join(_FAILED))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
