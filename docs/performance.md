# Performance / avoiding a CPU bottleneck

Two separate workloads here, and they scale on different hardware:

## 1. The router itself: CPU-only, no way around it

`PNS::ROUTER` is plain C++ geometry/graph search -- there's no GPU path for
it, in KiCad or in our bridge. So "don't get CPU-bottlenecked" doesn't mean
"move routing to GPU," it means **don't serialize routing steps that could
run in parallel**. Concretely, for RL training throughput:

- **One OS process per environment instance, not threads.** We construct
  `PNS::ROUTER`/`BOARD` objects directly rather than through KiCad's
  `ROUTER::GetInstance()` singleton, so multiple independent instances
  *should* be safe in principle -- but KiCad's codebase wasn't written with
  concurrent multi-instance use in mind (undocumented global/static state is
  plausible anywhere in a codebase this size, e.g. KIID generation,
  locale/number formatting). Multiprocessing (`multiprocessing` /
  `concurrent.futures.ProcessPoolExecutor`, or a vectorized-env wrapper like
  `gymnasium.vector.AsyncVectorEnv`) sidesteps that risk entirely and also
  sidesteps Python's GIL, which would otherwise serialize the C++ calls
  anyway even if the C++ side were thread-safe.
- **Confirmed, not just theoretical:** `pcbworld_pns_bridge` and the system
  `pcbnew` module crash if loaded into the *same* process (see
  `docs/engine_access.md`'s crash-hunting log) -- both statically link
  overlapping chunks of KiCad's own C++ code and both define KiCad's
  "exactly one instance per process" globals (`Kiface()`,
  `GFootprintTable`). This is a hard constraint for the env design, not
  just a nice-to-have: **never `import pcbnew` (the system module) in a
  worker process that has `pcbworld_pns_bridge` loaded**, for anything --
  board generation, independent verification, whatever. If a worker needs
  system-`pcbnew`-only functionality, do it in a genuinely separate
  process/subprocess call (`scripts/verify_routed_board.py` is the
  reference pattern), never inline in the same interpreter.
- **Release build flags** are already set in `notebooks/00_setup.ipynb`
  (`-DCMAKE_BUILD_TYPE=Release`) -- a debug KiCad build is dramatically
  slower and would be the first thing to check if routing feels slow.
- **Keep the hot loop in C++.** Each `push`/`fix` call crosses the
  Python/C++ boundary once per coordinate, not per intermediate geometry
  operation -- the actual push-and-shove search stays inside
  `PNS::ROUTER::Move`/`FixRoute`, called once per env step, not looped over
  from Python.
- Target: number of parallel env worker processes ~= CPU core count minus
  headroom for the training process itself.

## 2. The policy (RL/LLM agent): GPU

The PPO/GRPO policy network training itself is standard GPU work (PyTorch,
`.to("cuda")`), decoupled from the env workers above -- workers produce
transitions on CPU, the learner batches and trains on GPU. Not a concern
until we're past the current milestone (getting the engine wrapper working
at all); noting it here so the env interface (`pcbworld/env/`, not built
yet) gets designed as multi-process-friendly from the start rather than
retrofitted later.
