# Handoff: Gate A re-run (round 2)

Gate B is **done and passed** — no need to run it again. Round 1's Gate A
segfaulted on its second trial; that was a real use-after-free in
`PNS_BRIDGE::LoadBoard()`, now fixed. **The C++ changed, so this round needs
a real rebuild, not a cache restore.**

Paste the block below to Antigravity.

---

> **Rebuild the bridge and re-run one diagnostic in Colab.**
> Do not edit tracked source — if something fails, report the failure with
> its full output instead of fixing it (`AGENTS.md`).
>
> **Setup:** `notebooks/00_setup.ipynb` steps 1–4. **Do not skip step 2
> (`git pull`)** — `pcbworld/engine/cpp/pns_bridge.cpp` changed since the
> last run, and this run is pointless without it. The build step will
> recompile the changed file; if it reports nothing to do, the pull did not
> land and the run should stop there.
>
> **Step 0 — confirm the rebuild actually happened:**
> ```bash
> ls -l --time-style=+%Y-%m-%dT%H:%M /content/kicad-src/build/pcbworld_bridge/pcbworld_pns_bridge*.so
> date
> ```
> Report both. If the `.so` timestamp is older than this session, the new
> C++ is not in the running module and everything below is meaningless.
>
> **Gate A:**
> ```bash
> python3 pcbworld/data/generate_board.py smd1.kicad_pcb --num-nets 1 --seed 0
> python3 pcbworld/data/generate_board.py tht1.kicad_pcb --num-nets 1 --seed 0 --pad-type tht
> python3 scripts/diagnose_layer_switch.py smd1.kicad_pcb --tht-board tht1.kicad_pcb 2>&1 | tee gate_a.log
> ```
>
> Keep each `generate_board.py` call as its own process exactly as written —
> it uses system `pcbnew`, which crashes if it shares a process with the
> bridge.
>
> **Report back — real output, not a summary:**
> - Step 0: both outputs.
> - `gate_a.log`: the whole thing. Every trial row now prints as it
>   completes, so even a crash leaves a usable log — send whatever the log
>   contains, including a partial one.
> - The `VERDICT` block verbatim. It prints conclusions, not just booleans —
>   that block is the actual deliverable.
> - Any row showing `ERROR`, with its exception text.
> - **If it segfaults again**, say which trial name was the last one printed
>   before the crash, and include the `faulthandler`/`gdb` traceback the same
>   way you did last time — that was exactly what made the previous bug
>   findable.
