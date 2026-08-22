# Session handover

Written at the end of a long Claude Code session. Everything below is
committed and pushed to `main` (latest: `4b0791e`), 180/180 local tests
passing.

**Read order for a fresh session:** this file → `docs/RL_PLAN.md` (the live
plan) → `docs/ROUTER_CAPABILITIES.md` (what the engine can actually do, every
claim tagged LIVE/LOCAL/BROKEN) → `ROADMAP.md` (history, including two
superseded directions kept deliberately).

---

## Who does what

`AGENTS.md` is unchanged and still governs: **Antigravity runs Colab and
reports real output; it does not edit tracked source.** Everything in
"What's next" below marked *(logic)* belongs to a Claude Code session.
Everything marked *(Colab)* is a handoff.

This matters right now because the next substantial piece — the policy
network — is logic work, not a Colab errand.

---

## Where things stand: Stage 0 is COMPLETE

| Gate | Result |
|---|---|
| Dense per-step signal | **PASSED** — +1.00 separation, n=99 |
| Waypoint fidelity | **PASSED** — 0.0000mm deviation, mean and max |
| Layer changing | **NEGATIVE, closed** — no primitive works |
| Env on real geometry | **HEALTHY** — greedy routed 8/24 against a predicted 9/24 |

The environment is validated and the plan's riskiest assumptions are
measured rather than hoped for. Nothing blocks a training run except the
policy network.

### The three numbers worth memorising

- **`head_collides()` fired on 100% of attempts whose `fix()` was later
  rejected (n=90) and 0% of those accepted (n=9).** This is why a per-step
  collision penalty is real signal, and why the 1-D action space works.
- **Greedy (`a = 0`) routes 8/24; the independently measured straight-line
  baseline is 9/24.** Because `a = 0` means "walk straight at the target" by
  construction, that single number validates the observation frame, heading
  maths, snap/fix logic and net sequencing together. It is the regression
  check for any change to the env.
- **`T_pns` median 0.86ms/net; `run_drc()` 267ms; `get_board_geometry()`
  0.13ms.** DRC once per episode, board geometry once per net. Never per step.

---

## What was built this session

| Path | What |
|---|---|
| `docs/RL_PLAN.md` | The live plan. Supersedes `docs/AI_ARCHITECTURE.md` (CFP) |
| `docs/ROUTER_CAPABILITIES.md` | Paste-into-a-fresh-session capability briefing |
| `pcbworld/env/line_obs.py` | Line-geometry observation. Pure numpy, no bridge — fully testable locally |
| `pcbworld/env/line_route_env.py` | The env: 1-D heading action, Gate-B-validated reward |
| `scripts/diagnose_layer_switch.py` | Gate A. Ran, answered: `switch_layer()` is dead |
| `scripts/diagnose_via_hop.py` | Gate A follow-up. Ran, answered: no via hop either |
| `scripts/smoke_line_env.py` | Env validation against the real bridge |
| `pcbworld/engine/cpp/pns_bridge.cpp` | **Use-after-free fix in `LoadBoard()`** (see below) |
| `pcbworld/data/generate_board.py` | `--pad-type tht` |

---

## What's next, in order

1. **(Colab, pending — the only thing currently runnable)** Re-run
   `scripts/smoke_line_env.py`. The obstacle list is now cached per net
   instead of rebuilt per step; that is an unverified ~20× perf claim *and* a
   refactor that must be behaviour-neutral. **Greedy must still be exactly
   8/24** — if it moves, the cache is going stale somewhere the env needs it
   fresh, which matters more than the speed. Prompt is in
   `docs/HANDOFF_GATE_RUN.md`.

2. **(logic) The policy network.** The last thing between here and training.
   `pcbworld/agents/ppo_baseline.py`'s plain MLP cannot consume this
   observation — it must reshape the flat tail into `(k_nearest, 12)` and pool
   over rows where the `valid` column is 1. Spec from `docs/RL_PLAN.md`:

   ```
   per-segment MLP   12 → 64 → 64          (shared weights)
   masked max-pool + masked mean-pool  →  128
   global MLP         8 → 64
   concat 192 → 128 → 128
     actor  → 1 mean + 1 learned log_std
     critic → 1
   ```

   ~35k parameters, small enough to train on CPU beside the env workers.
   **Initialise the actor's final layer at near-zero gain** so an untrained
   policy emits `a ≈ 0` — that is what puts it *at* the 8/24 greedy baseline
   rather than below it, and it is the whole reason the action space was
   designed this way. Losing that init throws the free head start away.

3. **(logic) Wire PPO.** `ppo_baseline.py` already has GAE and the clipped
   surrogate working against the fake bridge. Needs the new policy, plus
   observation normalisation on the global 8 only (segment coords are already
   scaled), and Drive checkpointing every N updates — Colab sessions die.

4. **(Colab) Stage 1 training run.** One net, empty board. The bar from the
   plan: **~100% within about ten minutes of training.** Anything less means
   the plumbing is wrong, not the idea. Do not skip to stage 2 on a partial
   result.

5. Then stages 2–3 per `docs/RL_PLAN.md`'s curriculum. **Track completion
   rate, not just reward** — see the open question below.

---

## Things that will cost real time if re-derived

Beyond `ROADMAP.md`'s existing hard-constraints section, which still applies
in full:

- **Layer changing is closed. Do not reopen it casually.** `switch_layer()`
  accepts only a no-op, at every candidate `PCB_LAYER_ID`, with via sizes
  unset / 0.6-0.3 / 0.4-0.2, idle and mid-route, from an SMD pad *and* a THT
  pad. `toggle_via_placement()` places a via but cannot continue a route on
  the far side — a mid-route `fix(force_finish=False)` is rejected outright.
  Three sessions have gone here. One protocol remains untested and is
  deliberately parked: chaining `start_route()` from a committed via on the
  far layer. **Stages 1–3 are single-layer; rip-up-and-reroute, not vias, is
  the lever for stage 3.**

- **`LoadBoard()` twice on one `PNS_BRIDGE` used to be a use-after-free.** It
  freed the `BOARD` while the old interface and router still pointed at it,
  corrupting the heap and segfaulting inside the *next* `start_route()`.
  Fixed (destroy router → interface → settings → board first) and since
  verified across 12 trials on 2 boards. The sampled board pool for
  multi-worker training depends on this holding.

- **`snap_radius_nm >= step_size_nm / 2`.** The head advances a fixed
  distance, so a larger step jumps clean over the snap zone and orbits the
  target forever. The symptom — "the agent never finishes nets" — reads as a
  learning failure and is a config one.

- **Pad obstacles come from `get_board_geometry().pads` (`PadGeom`), not
  `net_pads()` (`NetPad`).** Only the former carries `size_x/size_y`, and an
  obstacle's size is what matters for avoiding it. Both are fetched
  deliberately in the env, each for its own job.

- **`query_hover_items()` is unsorted.** Filter by `kind == "pad"`. Taking
  `candidates[0]` cost a Colab round once already. `LineRouteEnv` does this
  correctly; `simple_route_env.py`, `pcb_route_env.py` and
  `diff_pair_route_env.py` still carry the old pattern.

- **Test isolation:** a test file that replaces `pcbworld_pns_bridge` in
  `sys.modules` must restore the *original module object*, not just a working
  module — `tests/test_diff_pair_route_env.py` binds it at import time and
  mutates that object. `tests/test_diagnose_layer_switch.py` has the correct
  fixture and a comment explaining why. Two older test files still work only
  by alphabetical luck; there is a background task chip open for that.

---

## Open questions to carry into training

1. **Reward and completion count are not perfectly aligned.** In the smoke
   run, random scored −330 reward against greedy's −177 and collided on 34%
   of steps against 18% — yet routed *9*/24 to greedy's 8/24. Straight-line
   is not the best completion strategy on a dense board: wandering sometimes
   stumbles around an obstacle a straight line cannot pass. That is headroom a
   policy can learn to exploit, but **track completion rate during training,
   and reconsider `RewardWeights.collision` (0.5/step) if the policy turns
   timid** — 20 colliding steps currently cancel an entire completion bonus.

2. **15/24 nets are unreachable on one layer** with straight-line routing,
   and the detour ladder rescued 0 of them. With layer changing closed, the
   plan's answer for stage 3 is a rip-up-and-reroute action. That is
   unexercised and is the largest remaining unknown in the curriculum.

3. **`nproc` = 2 on Colab.** Worker count is capped at 1–2. At ~0.03ms/step
   post-caching this is not a throughput problem, but it does mean the
   original "8 workers" figure in early planning is unreachable.
