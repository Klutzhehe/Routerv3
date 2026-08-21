# Agent boundary

This repo is worked on by more than one agent. This file exists so that fact
doesn't have to be re-explained in chat every session. Read it before touching
anything.

## The split

- **Claude Code** owns the logic: everything under `pcbworld/`, `tests/`,
  `scripts/`, `docs/`, `ROADMAP.md`. It writes the C++ bindings, the tool
  layer, the agent loop, the tests, the commit messages.
- **Antigravity's job here is Colab execution only**: pull this repo inside
  the Colab notebook (`notebooks/00_setup.ipynb`), run the build/verification
  cells, and report the actual output back — stdout, stack traces, assertion
  failures, whichever `[STAGE]` marker it stopped at. Not a summary written
  from memory of what the cells were supposed to do; the real output.

## Do not edit tracked source files

If a build or verification cell fails, **report the failure**, don't patch
around it. A fix belongs in a Claude Code session that can see the reasoning
behind the design it would be touching — most of this codebase carries
deliberate, non-obvious decisions in its docstrings and commit messages
specifically so a fix doesn't have to be re-derived blind (see
`ROADMAP.md`'s "hard constraints" section for why this matters: several past
bugs here came from a plausible-looking local fix that missed a project-wide
constraint, e.g. two engine modules that can never share a process, or
settings that must be applied in a specific order before a router call).

This applies even when the fix looks small and obviously correct. Report it
instead. If Colab-side context (an error only reproducible against the real
compiled bridge) would help diagnose it, include that in the report — that's
exactly the information this split is designed to get back to where the fix
belongs.

## What "report back" means concretely

- Full output of whichever cell failed, not a paraphrase.
- If a `%%bash` build cell stops partway, the `=== STAGE: ... ===` line it
  stopped at (see `ROADMAP.md`'s "How to run it") — that's the fast path to
  the cause.
- If a verification script (`scripts/verify_head_bindings.py`,
  `scripts/measure_waypoint_fidelity.py`, etc.) asserts, the assertion
  message and the values it printed before failing.
- If everything passes, the output showing that — numbers, not just "it
  worked."

## Why this file exists

Written after a session where this boundary was crossed: local source files
changed with new logic (a real, useful fix in that instance) instead of the
failure being reported for a fix on the Claude Code side. The fix itself was
fine — reviewed, tested, kept — but the process isn't something to rely on
working out by luck a second time. This file is the durable version of that
correction, so it doesn't depend on being re-stated in chat and re-relayed
correctly every session.
