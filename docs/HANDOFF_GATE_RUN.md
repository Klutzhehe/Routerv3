# Handoff: Gate A / Gate B Colab run

Paste the block below to Antigravity.

---

> **Run two diagnostic scripts in Colab and report their real output.**
> Do not edit tracked source — if something fails, report the failure with
> its full output instead of fixing it (`AGENTS.md`).
>
> **Setup:** `notebooks/00_setup.ipynb` steps 1-4. **Do not skip step 2
> (`git pull`)** — these scripts were added in the latest commit. Prefer a
> real rebuild over an old Drive cache: one binding used here
> (`get_head_obstacle()`) is newer than the last documented build.
>
> **Step 0 — what the build actually has:**
> ```bash
> nproc
> python3 -c "import pcbworld_pns_bridge as b; p=b.PNSBridge(); print([m for m in ('get_head_obstacle','head_collides','get_head_geometry','switch_layer','toggle_via_placement') if hasattr(p,m)])"
> ```
>
> **Gate A — why `switch_layer()` has never succeeded:**
> ```bash
> python3 pcbworld/data/generate_board.py smd1.kicad_pcb --num-nets 1 --seed 0
> python3 pcbworld/data/generate_board.py tht1.kicad_pcb --num-nets 1 --seed 0 --pad-type tht
> python3 scripts/diagnose_layer_switch.py smd1.kicad_pcb --tht-board tht1.kicad_pcb 2>&1 | tee gate_a.log
> ```
>
> **Gate B — timing, and whether collisions predict failures:**
> ```bash
> python3 pcbworld/data/generate_board.py board24.kicad_pcb --num-nets 24 --seed 0
> python3 scripts/measure_waypoint_fidelity.py board24.kicad_pcb 2>&1 | tee gate_b_traced.log
> python3 scripts/measure_waypoint_fidelity.py board24.kicad_pcb --no-collision-trace 2>&1 | tee gate_b_control.log
> ```
>
> Keep each `generate_board.py` call as its own process exactly as written —
> it uses system `pcbnew`, which crashes if it shares a process with the
> bridge. Keep `--num-nets 24` and the default board size. The second Gate B
> run is a control and is not optional.
>
> **Report back — real output, not a summary:**
> - Step 0: both outputs. Say so explicitly if `get_head_obstacle` is absent.
> - `gate_a.log`: the whole thing, especially the `VERDICT` block verbatim and
>   any row showing `ERROR`.
> - `gate_b_traced.log`: the `FIDELITY` section, the `GATE B` section
>   (both probabilities **and** their sample sizes), the full `PER-CALL WALL
>   CLOCK` table, and the `T_pns` line.
> - `gate_b_control.log`: just the `FIDELITY` section — I am comparing
>   `direct straight-push success` against the traced run. If they differ,
>   report that plainly rather than picking the better run.
> - On failure: the `=== STAGE: ... ===` marker it stopped at, or the
>   assertion message plus the values printed before it. Full traceback.
