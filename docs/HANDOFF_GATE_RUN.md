# Handoff: via-hop protocol + env smoke run

Two independent things in one Colab trip. No C++ changed since the last
rebuild, so a restored Drive cache is fine.

Paste the block below to Antigravity.

---

> **Run two scripts in Colab and report their real output.**
> Do not edit tracked source — if something fails, report the failure with
> its full output instead of fixing it (`AGENTS.md`).
>
> **Setup:** `notebooks/00_setup.ipynb` steps 1–4. Don't skip step 2
> (`git pull`) — both scripts are new. No C++ changed since the last run, so
> a cached build is fine.
>
> **1 — Via-hop protocol.** Gate A proved `switch_layer()` is dead and that
> `toggle_via_placement()` commits a via, but not that a route can change
> layer and keep going. This tests that.
>
> ```bash
> python3 pcbworld/data/generate_board.py smd1.kicad_pcb --num-nets 1 --seed 0 && python3 scripts/diagnose_via_hop.py smd1.kicad_pcb 2>&1 | tee via_hop.log
> ```
>
> **2 — Env smoke run.** First time the new RL environment touches the real
> router. It has a predicted answer, so it is a test rather than a smoke
> check: the env's `a = 0` action means "walk straight at the target", so the
> greedy policy IS the straight-line router, and that router already measured
> **9/24** on this board.
>
> ```bash
> python3 pcbworld/data/generate_board.py board24.kicad_pcb --num-nets 24 --seed 0 && python3 scripts/smoke_line_env.py board24.kicad_pcb 2>&1 | tee smoke_env.log
> ```
>
> Keep each `generate_board.py` call as its own process exactly as written —
> it uses system `pcbnew`, which crashes if it shares a process with the
> bridge.
>
> **Report back — real output, not a summary:**
> - `via_hop.log` in full, especially the `VERDICT` block verbatim and the
>   `committed:` note naming how many vias and the per-layer track counts —
>   that line is the only part that proves anything on disk.
> - `smoke_env.log` in full: both the greedy and random `---` blocks and the
>   whole `VERDICT`. The numbers I need are **routed n/24 for greedy**,
>   **routed n/24 for random**, and the **median wall clock per step**.
> - Any `AssertionError` with its full traceback — the smoke script asserts
>   on every observation, so an assertion means the env disagrees with the
>   real router about something specific.
> - If anything segfaults: the last line printed before the crash plus the
>   `faulthandler`/`gdb` traceback, same as before.

---

## What the outcomes mean

**Via hop:**

| Result | Consequence |
|---|---|
| `LAYER HOPPING CONFIRMED ON DISK` | Two-layer routing works; a place-via action goes into the env for stage 3 |
| `PARTIAL` | Head state looked right, copper disagreed — the exact overclaim the script exists to catch |
| `NO MID-ROUTE HOP` | Stay single-layer; revisit only if stage 3 plateaus |

**Env smoke:**

| Greedy result | Consequence |
|---|---|
| Near 9/24 | The whole stack lines up against real geometry — start PPO on stage 1 |
| 0/24 | Structural break (heading convention, snap radius vs step size, pad candidates). PPO would train against a broken env |
| Well below 9/24 | The env does something, but not what the straight-line router does. Understand before training |
| Well above 9/24 | Not automatically good — incremental 1mm pushes should beat one big push somewhat, but a large jump wants explaining |

Random must come in **below** greedy. If it ties, the action isn't affecting
outcomes and there's no gradient for PPO to follow — that's a stop-and-fix,
not a curiosity.
