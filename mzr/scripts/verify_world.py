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

import mzr.world.geometry as geo

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
    STEP_LENGTHS,
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


def test_exported_segments_are_backed_by_copper() -> None:
    """Every segment the exporter draws is backed by copper owned by that net.

    A regression test for a real bug this repo's DRC gate caught on its first
    run. Each leg's copper is two polylines -- one per frontier -- and joining
    them unconditionally drew a phantom segment straight across the board
    whenever a leg finished by reaching the far *pad* rather than by the two
    frontiers meeting: the far pad connected to the far end of the other half's
    stub, backed by nothing. KiCad 9.0.2 reported 30 `clearance`, 11
    `shorting_items` and 5 `tracks_crossing`.

    The lattice was innocent -- the exporter was describing a board that had
    never been routed. That is the more dangerous kind of failure, because it
    would have condemned a correct geometry model.

    This check needs no KiCad, so it runs everywhere and runs fast; the DRC gate
    is the authority, but this is what catches the same class of bug in a second
    rather than a minute.

    Checked against the segment's **centre line**, not against
    `geometry.segment_claims`. That distinction cost a debugging round and is
    worth stating: `segment_claims` is a deliberately *conservative* footprint
    used to decide whether a move is legal -- it samples the line, adds the
    diagonal corner guards, and dilates, which for a one-cell diagonal at
    minimum width yields 27 cells including cells *behind* the start that no
    move ever stamps. Using it as a record of "what was written" reported 67
    false violations on copper KiCad had already passed as clean.
    """
    from mzr.eval.kicad_export import _leg_runs

    def centreline(y0, x0, y1, x1):
        n = max(abs(y1 - y0), abs(x1 - x0))
        if n == 0:
            return [(y0, x0)]
        return [
            (round(y0 + (y1 - y0) * k / n), round(x0 + (x1 - x0) * k / n))
            for k in range(n + 1)
        ]

    w = build(nets=24, layers=4, size=64, batch=4, steps=64)
    rollout(w, via=True)
    occ = w.occ
    bad = total = 0
    gaps = vias = stubs = 0
    # A via is placed in place, so a layer change must not also move; an
    # in-plane hop cannot exceed one action's reach.
    reach = max(max(STEP_LENGTHS), w.cfg.snap_radius)
    for b in range(w.cfg.batch_size):
        routes = w.route_v[b].cpu()
        counts = w.route_n[b].cpu()
        for n in range(w.cfg.max_nets):
            if not bool(w.net_valid[b, n]) or int(w.net_status[b, n]) != STATUS_DONE:
                continue
            legs = 2 if int(w.net_kind[b, n]) == KIND_DIFF_PAIR else 1
            wclass = int(w.net_width[b, n])
            for leg in range(legs):
                if not bool(w.leg_done[b, n, leg]):
                    continue
                f_src = (n * 2 + leg) * NUM_ENDS + END_SRC
                f_dst = (n * 2 + leg) * NUM_ENDS + END_DST
                runs = _leg_runs(routes, counts, f_src, f_dst)
                if len(runs) > 1:
                    stubs += 1
                for pts in runs:
                    for i in range(len(pts) - 1):
                        l0, y0, x0 = pts[i]
                        l1, y1, x1 = pts[i + 1]
                        # Contiguity. This is what actually catches the phantom
                        # join, and it must come before the layer-change skip:
                        # a stub tip and the far pad usually sit on DIFFERENT
                        # layers -- that is precisely why the leg ended by
                        # pad-snap instead of meeting -- so the bogus jump is
                        # emitted as a *via*, and a check that skipped layer
                        # changes waved it straight through.
                        if l0 != l1:
                            vias += 1
                            if (y0, x0) != (y1, x1):
                                gaps += 1
                            continue
                        if max(abs(y1 - y0), abs(x1 - x0)) > reach:
                            gaps += 1
                            continue
                        if (y0, x0) == (y1, x1):
                            continue
                        total += 1
                        cells = [
                            int(occ[b, l0, cy, cx]) for cy, cx in centreline(y0, x0, y1, x1)
                        ]
                        if any(c != n + 1 for c in cells):
                            bad += 1

    # Guard against the check silently going vacuous: if the boards stop
    # producing stub legs, the phantom-join bug would no longer be reachable
    # and a green tick here would mean nothing.
    check(
        "export check actually exercises stub legs",
        stubs > 0,
        f"{stubs} legs exported as two runs, {vias} vias, {total} segments",
    )
    check(
        "no exported run jumps -- every vertex pair is contiguous",
        gaps == 0,
        f"{gaps} discontinuities",
    )
    check(
        "every exported segment is backed by its own net's copper",
        bad == 0,
        f"{total - bad}/{total} segments verified",
    )


def test_expert_paths_are_replayable() -> None:
    """Expert routes are made of single engine actions, and are legal copper.

    This is the property that makes the expert a *demonstration source* rather
    than just a baseline number. Behaviour cloning needs (observation, action)
    pairs, so every step of an expert route has to be exactly one thing the
    policy could have done: one of the 8 in-plane directions at a legal step
    length, or one layer change. A planner emitting arbitrary jumps would score
    well and teach nothing.

    Also checks the expert's copper is actually on the board -- it stamps
    through `geometry.move_claims` / `via_claims`, the same functions
    `engine.step()` uses, so a mismatch here means the two have drifted apart.
    """
    from mzr.world.expert import ExpertConfig, route_world_board
    from mzr.world.spec import DIRECTION_VECTORS

    dirs = {(int(dy), int(dx)) for dy, dx in DIRECTION_VECTORS}
    w = build(nets=10, layers=4, size=48, batch=2)
    bad_step = bad_copper = steps = legs = 0
    completion = 0.0
    for b in range(w.cfg.batch_size):
        res = route_world_board(w, b, ExpertConfig(iterations=3))
        completion += res.completion / w.cfg.batch_size
        for (net, leg), path in res.paths.items():
            legs += 1
            for (l0, y0, x0), (l1, y1, x1) in zip(path, path[1:]):
                steps += 1
                if l0 != l1:
                    # A layer change is a via: it must not also move.
                    if (y0, x0) != (y1, x1) or abs(l1 - l0) != 1:
                        bad_step += 1
                elif (y1 - y0, x1 - x0) not in dirs:
                    bad_step += 1

    check(
        "expert routes are single engine actions end to end",
        bad_step == 0,
        f"{steps - bad_step}/{steps} steps over {legs} legs are unit moves or single via hops",
    )
    check(
        "the expert is actually expert -- it beats the layer_hop baseline",
        completion > 0.75,
        f"expert completion {completion:.3f}",
    )


def test_observation_is_well_formed() -> None:
    """The policy's input: right shape, finite, dead rows silent, not aliased.

    `frontier_pos` being a **clone** is the one that matters most and the one
    that would never be noticed. `world.fr_pos` is replaced in place by
    `step()`; an aliasing observation made NeuroRoute's `policy.evaluate()`
    silently disagree with `policy.act()`, which breaks the PPO importance
    ratio with no error raised anywhere and no obviously wrong number to chase.
    """
    from mzr.env.observation import build_observation, frontier_feature_dim
    from mzr.world.baselines import layer_hop

    w = build(nets=12, layers=4, size=48, batch=3)
    for _ in range(8):
        if w.episode_done():
            break
        w.step(*layer_hop(w))
    obs = build_observation(w)

    D = frontier_feature_dim(w.num_layers)
    check(
        "observation shapes are as declared",
        obs.frontiers.shape == (w.cfg.batch_size, w.F, D)
        and obs.field.shape[0] == w.cfg.batch_size,
        f"frontiers {tuple(obs.frontiers.shape)} (D={D}), field {tuple(obs.field.shape)}",
    )
    check(
        "observation is finite everywhere",
        bool(torch.isfinite(obs.frontiers).all()) and bool(torch.isfinite(obs.field).all()),
        f"range [{float(obs.frontiers.min()):.2f}, {float(obs.frontiers.max()):.2f}]",
    )
    check(
        "dead frontiers contribute nothing",
        float(obs.frontiers[~obs.frontier_mask].abs().sum()) == 0.0,
        "",
    )

    before = obs.frontier_pos.clone()
    w.step(*layer_hop(w))
    check(
        "frontier_pos is a clone, not a view of live engine state",
        bool(torch.equal(obs.frontier_pos, before)),
        "an aliased observation breaks the PPO ratio silently",
    )


def test_price_reaches_the_policy() -> None:
    """Congestion actually shows up in what the policy sees, and scales.

    The price is the negotiation substrate -- if it never reaches the
    observation, simultaneous growth is an unmanaged traffic jam and the whole
    design premise is dead, but completion alone would never say so.

    Measured by **volume**, not by the maximum. The max is uninformative here:
    one contested cell pins it to `present_rate / max_present` regardless of
    how much contention there is, so 8-net and 30-net boards report identical
    maxima while differing 25x in actual contested cell-events.
    """
    from mzr.env.observation import CH_PRICE_HISTORY, CH_PRICE_PRESENT, build_observation
    from mzr.world.baselines import layer_hop

    vols = {}
    for nets in (8, 30):
        w = build(nets=nets, layers=4, size=48, batch=4, steps=48)
        total = 0.0
        for _ in range(24):
            if w.episode_done():
                break
            w.step(*layer_hop(w))
            obs = build_observation(w)
            total += float(obs.field[:, CH_PRICE_PRESENT].sum())
            total += float(obs.field[:, CH_PRICE_HISTORY].sum())
        vols[nets] = total

    check(
        "congestion price reaches the observation",
        vols[30] > 0.0,
        f"price mass: {vols[8]:.1f} at 8 nets, {vols[30]:.1f} at 30 nets",
    )
    check(
        "congestion price grows with board density",
        vols[30] > 2.0 * max(vols[8], 1e-6),
        f"{vols[30] / max(vols[8], 1e-6):.1f}x more price mass at 30 nets than at 8",
    )


def test_route_env_reward_is_well_formed() -> None:
    """The RL env: rewards finite, dead frontiers earn nothing, arrivals paid.

    The arrival check is the one that matters. The engine reports a connection
    only as a board-level count; the env recovers *which* frontiers closed a
    leg by diffing `leg_done` across the step, and credits both. If that diff
    is wrong, the dense progress signal still works and completion still rises,
    so nothing looks broken -- but the policy never learns that *finishing* is
    the point, only that getting closer is, which is exactly the failure this
    project is trying to leave behind.
    """
    from mzr.env.route_env import EnvConfig, RouteEnv
    from mzr.world.baselines import layer_hop_action
    from mzr.world.engine import WorldConfig
    from mzr.world.generator import GeneratorConfig

    cfg = EnvConfig(
        spec=BoardSpec(height_cells=48, width_cells=48, layers=LayerStack(num_layers=4)),
        world=WorldConfig(
            batch_size=4, max_nets=14, max_macro_steps=48,
            max_steps_per_frontier=48, ripup=RipupRules(interval=8),
        ),
        generator=GeneratorConfig(num_nets=8, num_components=4, pin_pitch_cells=4),
        max_episode_steps=48,
    )
    env = RouteEnv(cfg)
    obs = env.reset(seeds=[900000 + i for i in range(4)])

    # A frontier that was already masked out at the START of a step must earn
    # nothing from it. Checking against the POST-step mask would be wrong: a
    # frontier that connects this step is live during it, correctly earns the
    # arrival bonus, and only then drops out of the mask.
    prev_mask = obs.frontier_mask.clone()
    settled_earned = 0.0
    arrivals_paid = 0
    finite = True
    steps = 0
    while True:
        out = env.step({"layer": layer_hop_action(env.world)})
        finite = finite and bool(torch.isfinite(out.reward).all())
        finite = finite and bool(torch.isfinite(out.board_reward).all())
        settled_earned += float(out.reward[~prev_mask].abs().sum())
        arrivals_paid += int((out.reward > 3.0).sum())
        prev_mask = out.obs.frontier_mask.clone()
        steps += 1
        if out.done:
            break

    final_done = int(
        (env.world.net_valid & (env.world.net_status == STATUS_DONE)).sum()
    )
    check("route-env reward is finite over a whole episode", finite, f"{steps} steps")
    check(
        "route-env pays nothing to a frontier already settled at step start",
        settled_earned < 1.0,
        f"{settled_earned:.3f} stray reward on already-settled frontiers",
    )
    check(
        "route-env credits arrivals, ~2 frontiers per connected leg",
        arrivals_paid >= final_done,
        f"{arrivals_paid} arrival payouts for {final_done} connected legs",
    )


def test_route_env_is_deterministic() -> None:
    """Same seeds + same actions -> identical rewards and completion.

    A replay for a PPO or MuZero update has to reproduce the trajectory that
    generated it. The engine's determinism is already checked; this confirms
    the env layer -- observation build, reward, arrival diff -- adds no
    nondeterminism on top.
    """
    from mzr.env.route_env import EnvConfig, RouteEnv
    from mzr.world.baselines import layer_hop_action
    from mzr.world.engine import WorldConfig
    from mzr.world.generator import GeneratorConfig

    def run():
        cfg = EnvConfig(
            spec=BoardSpec(height_cells=48, width_cells=48, layers=LayerStack(num_layers=2)),
            world=WorldConfig(
                batch_size=3, max_nets=16, max_macro_steps=40,
                max_steps_per_frontier=40, ripup=RipupRules(interval=0),
            ),
            generator=GeneratorConfig(num_nets=10, num_components=4, pin_pitch_cells=4),
            max_episode_steps=40,
        )
        env = RouteEnv(cfg)
        env.reset(seeds=[7, 8, 9])
        r = torch.zeros(3, env.world.F)
        while True:
            out = env.step({"layer": layer_hop_action(env.world)})
            r += out.reward
            if out.done:
                return r, out.info["completion"].clone()

    r0, c0 = run()
    r1, c1 = run()
    check(
        "route-env replays identically",
        bool(torch.equal(r0, r1)) and bool(torch.equal(c0, c1)),
        f"reward delta {float((r0 - r1).abs().max()):.2e}, completion {c0.mean():.3f} vs {c1.mean():.3f}",
    )


def test_prior_policy_is_greedy_at_init_and_ppo_consistent() -> None:
    """Untrained argmax == the greedy baseline, and act() == evaluate().

    Two properties the whole training setup depends on:

    * **Greedy at init.** Near-zero head weights make the *bias* the entire
      signal, and the biases are set so argmax is "one cell down the geodesic
      gradient, stay on this layer, minimum width". If this drifts, training
      starts *below* the baseline instead of at it -- a zero-bias width head in
      `neuroroute/` picked 3-cell traces on 88% of actions and never recovered.
    * **act() and evaluate() return identical log-probs for the same action.**
      The PPO importance ratio is `exp(evaluate_logp - act_logp)`; if the two
      forward passes disagree, the ratio is noise and the update is meaningless
      -- with no error raised. Dropout in the transformer caused exactly this
      and is why it is set to 0.
    """
    from mzr.env.route_env import EnvConfig, RouteEnv
    from mzr.models.policy import PriorPolicy
    from mzr.world.engine import WorldConfig
    from mzr.world.generator import GeneratorConfig

    torch.manual_seed(0)
    cfg = EnvConfig(
        spec=BoardSpec(height_cells=48, width_cells=48, layers=LayerStack(num_layers=4)),
        world=WorldConfig(
            batch_size=4, max_nets=10, max_macro_steps=40,
            max_steps_per_frontier=40, ripup=RipupRules(interval=0),
        ),
        generator=GeneratorConfig(num_nets=5, num_components=3, pin_pitch_cells=4),
        max_episode_steps=40,
    )
    env = RouteEnv(cfg)
    obs = env.reset(seeds=[1, 2, 3, 4])
    pol = PriorPolicy(num_layers=4, field_width=48, token_width=128)

    det = pol.act(obs, deterministic=True)["action"]
    live = obs.frontier_mask
    greedy = (
        float((det["direction"][live] == 0).float().mean()) == 1.0
        and float((det["step"][live] == 0).float().mean()) == 1.0
        and float((det["layer"][live] == 0).float().mean()) == 1.0
    )
    check("untrained argmax is exactly the greedy action", greedy)

    sampled = pol.act(obs, deterministic=False)
    ev = pol.evaluate(obs, sampled["action"])
    check(
        "act() and evaluate() give identical log-probs (PPO ratio is meaningful)",
        bool(torch.allclose(ev["logp"], sampled["logp"], atol=1e-5)),
        f"max delta {float((ev['logp'] - sampled['logp']).abs().max()):.1e}",
    )

    # Progress shaping must telescope. `_commit` computes fr_prev - new_dist,
    # so over an episode it must sum to (initial distance - final distance).
    # That identity is what makes the shaping potential-based and therefore
    # policy-invariant, and it silently stops holding the moment the potential
    # itself changes -- which copper-seeding does every `geodesic_refresh`
    # steps. Measured before `_rebaseline_fr_prev` existed: 64 boards at 100%
    # completion summed progress to -51.56 cells, with +991.87 / -1043.44 of
    # churn, so most of the navigation signal was refresh artefact.
    import mzr.world.engine as _E

    caught = []
    _orig_step = _E.SimultaneousRouterWorld.step

    def _spy(self, *a, **k):
        r = _orig_step(self, *a, **k)
        caught.append(float((r.progress * r.live.float()).sum()))
        return r

    _E.SimultaneousRouterWorld.step = _spy
    try:
        env3 = RouteEnv(cfg)
        obs3 = env3.reset(seeds=[21, 22, 23, 24])
        d0_sum = float(env3.world.fr_prev.sum())
        for _ in range(cfg.max_episode_steps):
            s3 = env3.step(pol.act(obs3, deterministic=True)["action"])
            obs3 = s3.obs
            if s3.done:
                break
        dend_sum = float(env3.world.fr_prev.sum())
    finally:
        _E.SimultaneousRouterWorld.step = _orig_step

    prog_sum = sum(caught)
    # Retired frontiers stop contributing, so this is a bound rather than an
    # equality: what must not happen is a large NEGATIVE sum, which means the
    # potential moved under the policy.
    check(
        "progress shaping telescopes (potential is stationary across refreshes)",
        prog_sum > -1.0,
        f"sum(progress) {prog_sum:+.2f} cells, fr_prev {d0_sum:.1f} -> {dend_sum:.1f}",
    )

    # -- the quality instrumentation itself ------------------------------
    #
    # These check the METRICS, not the policy. An untrained policy is
    # deliberately collapsed on d0 (see the greedy-at-init check above), so
    # "is it steering" is meaningless here -- but the machinery that will
    # answer that question later has already been wrong twice, and both times
    # it was wrong in the direction of reporting something plausible:
    #
    #   * `route_quality` was absent entirely, so completion certified a
    #     policy that double-routed 46.5% of boards at 2.3x copper.
    #   * the action profile was sampled AFTER the rollout, averaging over
    #     zero live frontiers, and reported d0_frac 0% / entropy 0.000 for
    #     every policy -- failing a healthy one and a collapsed one alike.
    #
    # An instrument that reads plausibly and wrongly is worse than none, so
    # the instrument gets a gate too.
    from mzr.eval.quality import ProfileAccumulator, quality_verdict, route_quality

    acc = ProfileAccumulator()
    obs2 = env.reset(seeds=[11, 12, 13, 14])
    for _ in range(12):
        a = pol.act(obs2, deterministic=True)["action"]
        acc.update(pol, obs2, a)
        stp = env.step(a)
        obs2 = stp.obs
        if stp.done:
            break
    prof = acc.result()
    check(
        "action profile samples live frontiers (not an empty post-rollout set)",
        prof["actions_seen"] > 0,
        f"{prof['actions_seen']} actions seen, d0 {prof['dir_d0_frac']:.0%}, "
        f"{prof['dir_distinct']} distinct directions",
    )

    q = route_quality(env.world)
    check(
        "route_quality returns finite copper ratios",
        q["copper_median"] == q["copper_median"] and q["copper_median"] > 0,
        f"median {q['copper_median']:.3f}x, mean {q['copper_mean']:.3f}x, "
        f"right-angle {q['right_angle_frac']:.0%}, doubled {q['doubled']}",
    )

    # The verdict must actually reject the failure modes it exists for.
    good_q = {"copper_median": 1.02, "right_angle_frac": 0.05, "doubled": 0}
    good_p = {"actions_seen": 100, "dir_d0_frac": 0.70, "dir_distinct": 5,
              "ent_direction": 1.2}
    ok_good, _ = quality_verdict(good_q, good_p, max_copper=1.15,
                                 max_right_angle=0.15, min_dir_entropy=0.4,
                                 max_d0_frac=0.95)
    collapsed_p = dict(good_p, dir_d0_frac=1.0, dir_distinct=1, ent_direction=0.41)
    ok_bad, why_bad = quality_verdict(good_q, collapsed_p, max_copper=1.15,
                                      max_right_angle=0.15, min_dir_entropy=0.4,
                                      max_d0_frac=0.95)
    empty_p = dict(good_p, actions_seen=0)
    ok_empty, why_empty = quality_verdict(good_q, empty_p, max_copper=1.15,
                                          max_right_angle=0.15, min_dir_entropy=0.4,
                                          max_d0_frac=0.95)
    check(
        "quality gate passes a healthy profile and rejects a collapsed one",
        ok_good and not ok_bad and not ok_empty,
        f"healthy={ok_good} collapsed={ok_bad} ({why_bad[:48]}) empty={ok_empty}",
    )
    # The near-miss that motivated max_d0_frac: entropy alone would have
    # passed d0=100% at 0.41 against a 0.40 floor.
    _, why_ent_only = quality_verdict(good_q, collapsed_p, max_copper=1.15,
                                      max_right_angle=0.15, min_dir_entropy=0.4,
                                      max_d0_frac=1.0)
    check(
        "entropy alone would NOT have caught the collapse (why d0_frac is primary)",
        "collapsed" not in why_ent_only,
        f"entropy-only verdict: {why_ent_only or '(passed)'}",
    )

    # The joint (direction, step) mask, checked as a JOINT constraint.
    #
    # `obs.safety` is (B, F, 8, 3) and agrees with the engine's own `_plan`
    # exactly (measured: 3384/3384 entries, 0 disagreements either way). The
    # bug was never the mask, it was applying its two MARGINALS to two heads
    # that sample independently: direction 3 legal at 1 cell, step 4 legal in
    # some other direction, and the pair (3, 4-cell) forbidden by neither.
    #
    # That is the stage-0 livelock. A rejected move writes nothing, so the next
    # observation is identical, and a deterministic policy re-picks the same
    # illegal action until `max_stuck_steps` retires the net -- turning 23-of-24
    # legal options into a guaranteed failure. Measured cost before the fix:
    # 14 of 512 held-out boards, every one at 0.000 completion with 0 copper.
    live = obs.frontier_mask
    for arm, det in (("argmax", True), ("sampled", False)):
        a = pol.act(obs, deterministic=det)["action"]
        ok = torch.gather(
            obs.safety, 2,
            a["direction"].view(*a["direction"].shape, 1, 1)
            .expand(*a["direction"].shape, 1, obs.safety.shape[-1]),
        ).squeeze(2)                                   # (B, F, n_steps)
        chosen = torch.gather(ok, 2, a["step"].unsqueeze(-1)).squeeze(-1)
        # Only frontiers that HAVE a legal move can be held to this.
        has_legal = obs.safety.any(dim=-1).any(dim=-1)
        must = live & has_legal & (a["layer"] == 0)    # a via has its own mask
        bad = int((must & ~chosen).sum())
        check(
            f"{arm} never selects a jointly-illegal (direction, step)",
            bad == 0,
            f"{bad} illegal of {int(must.sum())} live non-via actions",
        )

    # The per-frontier value head is only reached through `value_f`, so a loss
    # built from `value` alone leaves it grad-less and this check calls a
    # healthy head dead. It fired that way from the moment --per-frontier-adv
    # landed: ppo.py trains value_frontier at the `if per_frontier:` branch,
    # and the synthetic loss here was never widened to match.
    loss = (
        -ev["logp"].sum()
        + ev["value"].pow(2).sum()
        + ev["value_f"].pow(2).sum()
        - 0.01 * ev["entropy"]
    )
    loss.backward()
    no_grad = [n for n, q in pol.named_parameters() if q.grad is None]
    check("every policy parameter receives gradient", not no_grad, f"{len(no_grad)} without grad")


def test_training_loop_runs_without_nan() -> None:
    """rollout -> GAE -> PPO update -> eval, two updates, all finite.

    An integration check, not a learning check. It catches the class of bug
    that only appears once every piece is wired together: a NaN in the reward
    that poisons the whole shared backward graph, an advantage that is all
    zeros because the value bootstrap is misaligned, an act/evaluate shape
    mismatch that silently drops half the frontiers from the ratio.
    """
    import tempfile

    from mzr.env.route_env import EnvConfig, RouteEnv
    from mzr.models.policy import PriorPolicy
    from mzr.training.curriculum import STAGES
    from mzr.training.ppo import PPOConfig, RolloutBuffer, ppo_update
    from mzr.training.run import collect
    from mzr.world.engine import WorldConfig

    stage = STAGES["0"]
    env = RouteEnv(EnvConfig(
        spec=stage.board_spec(),
        world=WorldConfig(
            batch_size=4, max_nets=stage.generator.num_nets + 6,
            max_macro_steps=24, max_steps_per_frontier=24, ripup=stage.ripup,
        ),
        generator=stage.generator, max_episode_steps=24,
    ))
    pol = PriorPolicy(num_layers=stage.layers, field_width=32, token_width=64)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    cfg = PPOConfig(epochs=2, minibatches=2)

    obs = env.reset()
    finite = True
    for _ in range(2):
        buf = RolloutBuffer()
        obs = collect(env, pol, buf, 8, obs)
        with torch.no_grad():
            last_v = pol.act(obs, deterministic=False)["value"]
        m = ppo_update(pol, opt, buf, last_v, cfg)
        finite = finite and all(
            v == v and abs(v) < 1e6 for v in m.values()  # NaN != NaN
        )
    for q in pol.parameters():
        finite = finite and bool(torch.isfinite(q).all())

    check(
        "full training loop runs two updates with finite params and metrics",
        finite,
        f"policy_loss {m['policy_loss']:.3f} value_loss {m['value_loss']:.3f} "
        f"kl {m['approx_kl']:.3f} grad_norm {m.get('grad_norm', 0):.2f}",
    )


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
    test_exported_segments_are_backed_by_copper()
    test_expert_paths_are_replayable()
    test_observation_is_well_formed()
    test_price_reaches_the_policy()
    test_route_env_reward_is_well_formed()
    test_route_env_is_deterministic()
    test_prior_policy_is_greedy_at_init_and_ppo_consistent()
    test_training_loop_runs_without_nan()

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
