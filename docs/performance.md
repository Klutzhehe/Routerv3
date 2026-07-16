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
