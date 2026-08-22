# Handoff: smoke re-run after the per-step caching fix

Small run. The previous smoke run measured 0.745ms/step against a 0.035ms
budget, because the obstacle list was being rebuilt every step when nothing
in it changes within a net. That is now cached per net. Two things need
confirming: the speedup is real, and the refactor changed no behaviour.

Cached build is fine — no C++ changed.

---

> **Re-run one script in Colab and report its output.**
> Do not edit tracked source — report failures with their full output
> instead of fixing them (`AGENTS.md`).
>
> **Setup:** `notebooks/00_setup.ipynb` steps 1–4. Don't skip step 2
> (`git pull`) — the fix being tested is in the latest commit. A restored
> Drive cache is fine, no C++ changed.
>
> ```bash
> python3 pcbworld/data/generate_board.py board24.kicad_pcb --num-nets 24 --seed 0 && python3 scripts/smoke_line_env.py board24.kicad_pcb 2>&1 | tee smoke_env2.log
> ```
>
> Same board, same seed as last time, so the numbers are directly comparable.
>
> **Report back:**
> - `smoke_env2.log` in full — both `---` blocks and the whole `VERDICT`.
> - The three numbers I am comparing against the previous run:
>   **greedy routed n/24** (was 8/24), **random routed n/24** (was 9/24), and
>   **median wall clock per step** (was 0.745ms).
> - Any `AssertionError` with its full traceback.

---

## What the numbers mean

| | Previous | Expected now |
|---|---|---|
| greedy routed | 8/24 | **exactly 8/24** |
| random routed | 9/24 | 9/24 |
| median ms/step | 0.745 | well below — the point of the change |

**Greedy moving off 8/24 is the finding, not the speed.** The change was pure
caching: the obstacle set is identical, just built once per net instead of
once per step. If completions move, the cache is going stale somewhere the
env still needs it fresh, and that matters far more than the millisecond.

Random can drift by a net or two — it is seeded, but it consumes a different
number of RNG draws if any episode length changes.
