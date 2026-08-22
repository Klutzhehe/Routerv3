# Handoff: via-hop protocol run (round 3)

Gate A and Gate B are both **done**. Round 2 proved `switch_layer()` is dead
(accepts only a no-op; every hypothesis eliminated) and that
`toggle_via_placement()` commits a real via — but **not** that a route can
change layer and keep going, which is the half stage 3+ needs.

This run tests one specific protocol. No C++ changed since round 2, so a
cached build is fine.

Paste the block below to Antigravity.

---

> **Run one diagnostic in Colab and report its real output.**
> Do not edit tracked source — if something fails, report the failure with
> its full output instead of fixing it (`AGENTS.md`).
>
> **Setup:** `notebooks/00_setup.ipynb` steps 1–4. Don't skip step 2
> (`git pull`) — the script is new. No C++ changed since the last run, so a
> restored Drive cache is fine this time.
>
> ```bash
> python3 pcbworld/data/generate_board.py smd1.kicad_pcb --num-nets 1 --seed 0 && python3 scripts/diagnose_via_hop.py smd1.kicad_pcb 2>&1 | tee via_hop.log
> ```
>
> **Report back — real output, not a summary:**
> - `via_hop.log` in full. Every row prints as it completes, so send a
>   partial log if it crashes.
> - The `VERDICT` block verbatim.
> - Any row showing `ERROR`, with its exception text.
> - Pay particular attention to the `full_two_hop` rows: the `committed:`
>   note naming how many vias and the per-layer track counts is the only
>   part of this that proves anything on disk. Send that line exactly.
> - If it segfaults: the last trial name printed, plus the
>   `faulthandler`/`gdb` traceback, same as before.

---

## What the outcomes mean

| Result | Consequence |
|---|---|
| `LAYER HOPPING CONFIRMED ON DISK` — vias committed **and** tracks on two layers | Two-layer routing works. A place-via action goes into the env; stage 3 can use it |
| `PARTIAL` — route committed but single-layer copper | Head state looked right and the copper disagreed. That is precisely the overclaim round 2 made; the script is built to catch it |
| `NO MID-ROUTE HOP` | Stay single-layer. Stages 1–3 don't need vias; revisit only if stage 3 plateaus against the 9/24 baseline |

None of these block the trainer — stages 1 and 2 are single-layer by design.
