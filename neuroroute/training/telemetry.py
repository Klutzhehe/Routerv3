"""Live telemetry, health checks and crash reporting for training runs.

This exists because of how this project is actually operated: training runs on
Colab, driven by an agent whose job is to **report real output back**, not to
diagnose (`AGENTS.md`). That only works if the run emits enough to diagnose
from -- after the fact, from a session that has probably already died.

So three things are non-negotiable here:

1. **Everything is flushed immediately.** Colab buffers stdout, and a run that
   dies with 200 unflushed lines has told you nothing. Every write flushes.
2. **Structured history is appended, not held in memory.** `train_log.jsonl` is
   one JSON object per update, `fsync`-ed on write, so a hard crash or a killed
   VM still leaves a complete record up to the last completed update.
   `history.json` (rewritten wholesale) is convenient but useless if the
   process dies mid-run.
3. **Failures dump state, not just a traceback.** A stack trace says where it
   died; `crash_report.txt` also carries the config, the last updates' metrics,
   tensor shapes, and NaN/inf locations -- which is what actually identifies
   the cause when nobody can re-run it interactively.

The health checks are deliberately loud. Silent NaNs are the most expensive
failure mode in RL: the loss goes to `nan`, every gradient follows, and the run
continues burning GPU hours producing a checkpoint of garbage.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unavailable"


def environment_report() -> dict[str, Any]:
    """Everything needed to reproduce or explain a run.

    Reported at startup and embedded in every crash report. The commit hash and
    the dirty flag matter most: this repo is worked on by more than one agent,
    and "which code was actually running" is otherwise unanswerable after the
    fact.
    """
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_gb"] = round(props.total_memory / 1e9, 2)
        info["cuda_version"] = torch.version.cuda
    return info


@dataclass
class HealthReport:
    """Result of one health sweep. `fatal` means stop, do not checkpoint."""

    ok: bool = True
    fatal: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.ok = False

    def fail(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False
        self.fatal = True


def check_model_health(model: torch.nn.Module, amp_skip_ok: bool = False) -> HealthReport:
    """Look for NaN/inf parameters and dead or exploding gradients.

    Run *after* the optimiser step, so it catches a corrupted update before the
    next rollout is collected against a broken policy.

    `amp_skip_ok`: GradScaler itself already declined to apply this update
    (it saw inf in the unscaled gradients and skipped the optimiser step --
    routine while its scale factor is still calibrating, especially right
    after --amp is turned on), but it never clears `.grad` afterward. Without
    this flag every one of those routine skips looks identical to a real
    corruption and kills the run. It only softens the **gradient** check to a
    warning -- a non-finite **parameter** stays fully fatal regardless, since
    that is exactly the failure GradScaler's skip exists to prevent; if it
    happens anyway, something got past the safety GradScaler is supposed to
    provide, and that is a strictly worse signal than the one this flag
    exists to tolerate.

    It also reattributes "gradient norm is exactly zero" on the same update,
    rather than leaving it to print as an unrelated-looking second alarm.
    `clip_grad_norm_` computes ONE norm across every parameter combined, so a
    single inf anywhere makes that combined norm inf too, and the resulting
    clip coefficient (`max_norm / inf`) is exactly 0 -- which then multiplies
    *every other parameter's gradient* by zero in the same step,
    deterministically, not by chance. So whenever `amp_skip_ok` is true, a
    zero gradient norm is a guaranteed downstream mechanical consequence of
    the one inf gradient GradScaler already caught, not a second, independent
    "nothing is learning" event.
    """
    rep = HealthReport()
    total_sq = 0.0
    n_with_grad = 0
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            n_bad = int((~torch.isfinite(p)).sum())
            rep.fail(f"parameter '{name}' has {n_bad} non-finite values")
        if p.grad is not None:
            n_with_grad += 1
            if not torch.isfinite(p.grad).all():
                if amp_skip_ok:
                    rep.warn(f"gradient of '{name}' is non-finite "
                             f"(GradScaler already skipped this step)")
                else:
                    rep.fail(f"gradient of '{name}' is non-finite")
            else:
                total_sq += float(p.grad.detach().pow(2).sum())
    if n_with_grad == 0:
        rep.warn("no parameter received a gradient this update")
    else:
        gnorm = total_sq ** 0.5
        if gnorm == 0.0:
            if amp_skip_ok:
                rep.warn("gradient norm is exactly zero -- the same skipped "
                         "GradScaler step above, not a separate stall (its "
                         "inf zeroed every other parameter's gradient via "
                         "clip_grad_norm_'s combined norm)")
            else:
                rep.warn("gradient norm is exactly zero -- nothing is learning")
        elif gnorm > 1e4:
            rep.warn(f"gradient norm {gnorm:.1f} is very large")
    return rep


def check_metrics_health(row: dict[str, Any]) -> HealthReport:
    """Sanity-check one update's metrics.

    The thresholds are not tuned; they are there to make a silent pathology
    loud. An entropy that has collapsed to zero by update 50 is not
    convergence, it is a policy that stopped exploring, and it looks identical
    to healthy training in a reward curve.
    """
    rep = HealthReport()
    for k, v in row.items():
        if isinstance(v, float) and v != v:  # NaN
            rep.fail(f"metric '{k}' is NaN")
        elif isinstance(v, float) and abs(v) == float("inf"):
            rep.fail(f"metric '{k}' is infinite")

    ent = row.get("entropy")
    if isinstance(ent, float) and ent < 1e-3 and row.get("update", 0) > 20:
        rep.warn(f"entropy {ent:.2e} has collapsed -- the policy stopped exploring")
    vloss = row.get("value_loss")
    if isinstance(vloss, float) and vloss > 1e4:
        rep.warn(f"value loss {vloss:.1f} is exploding")
    # `board_value_loss` was added alongside the scheduler/ripup fix and this
    # check was not -- it explored 30 -> 490,703 in 6 updates on the first
    # real run and nothing here would have said so; the generic NaN/inf loop
    # above only catches it once it goes non-finite, well after "exploding"
    # would have been the more useful, earlier word for it.
    bvloss = row.get("board_value_loss")
    if isinstance(bvloss, float) and bvloss > 1e4:
        rep.warn(f"board value loss {bvloss:.1f} is exploding")
    clip = row.get("clip_frac")
    if isinstance(clip, float) and clip > 0.5:
        rep.warn(f"clip fraction {clip:.2f} -- updates are too large for the PPO trust region")
    rar = row.get("rejected_action_rate")
    if isinstance(rar, float) and rar > 0.5 and row.get("update", 0) > 20:
        rep.warn(f"rejected-action rate {rar:.1%} -- the policy is mostly proposing illegal moves")
    return rep


class Telemetry:
    """Console + JSONL logging, timing, and crash capture for one run."""

    def __init__(self, out_dir: str | Path, run_name: str = "neuroroute"):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.jsonl = self.dir / "train_log.jsonl"
        self.console_path = self.dir / "console.log"
        self._console = self.console_path.open("a", encoding="utf-8")
        self.history: list[dict] = []
        self.timings: dict[str, float] = {}
        self.t0 = time.perf_counter()

    # -- output -------------------------------------------------------------

    def print(self, *parts: Any) -> None:
        """Print to stdout AND the on-disk mirror, flushing both.

        The mirror is what survives a dead Colab session; the flush is what
        makes the live view actually live.
        """
        msg = " ".join(str(p) for p in parts)
        print(msg, flush=True)
        # Never let the on-disk mirror be the thing that kills a training run.
        # stdout has already received the message; a closed or full file is not
        # a reason to lose it or to raise.
        try:
            self._console.write(msg + "\n")
            self._console.flush()
        except (ValueError, OSError):
            pass

    def banner(self, title: str, config: dict[str, Any]) -> None:
        env = environment_report()
        self.print("=" * 78)
        self.print(f"  {title}")
        self.print("=" * 78)
        for k, v in env.items():
            self.print(f"  {k:<18} {v}")
        if env.get("git_dirty"):
            self.print("  !! working tree is DIRTY -- the running code is not the commit above")
        self.print("  " + "-" * 74)
        for k, v in config.items():
            self.print(f"  {k:<18} {v}")
        self.print("=" * 78)
        (self.dir / "run_config.json").write_text(
            json.dumps({"env": env, "config": config}, indent=2, default=str), encoding="utf-8"
        )

    def log(self, row: dict[str, Any]) -> HealthReport:
        """Append one update's metrics, run health checks, return the report.

        Written with `fsync` so the record survives a hard kill -- Colab VMs are
        reclaimed without warning, and a training log that only exists in a
        Python list is lost with them.
        """
        row = {"wall_clock_s": round(time.perf_counter() - self.t0, 2), **row}
        self.history.append(row)
        with self.jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        rep = check_metrics_health(row)
        for w in rep.warnings:
            self.print(f"    [WARN] {w}")
        for e in rep.errors:
            self.print(f"    [FATAL] {e}")
        return rep

    # -- timing -------------------------------------------------------------

    @contextmanager
    def section(self, name: str):
        """Time a phase. Cumulative totals are reported in the progress line so
        a run that is unexpectedly slow says *which part* is slow."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - t)

    def timing_summary(self) -> str:
        total = sum(self.timings.values()) or 1.0
        return "  ".join(f"{k} {v / total:.0%}" for k, v in sorted(self.timings.items()))

    @staticmethod
    def gpu_memory() -> str:
        if not torch.cuda.is_available():
            return "cpu"
        used = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{used:.1f}/{total:.1f}GB"

    # -- failure ------------------------------------------------------------

    def crash(self, exc: BaseException, extra: dict[str, Any] | None = None) -> Path:
        """Write a self-contained crash report and echo it to the console.

        Self-contained matters: whoever reads this cannot re-run the session.
        The report carries the environment, the config, the last ten updates'
        metrics and any caller-supplied tensor shapes alongside the traceback.
        """
        path = self.dir / "crash_report.txt"
        lines = [
            "=" * 78,
            "NEUROROUTE CRASH REPORT",
            "=" * 78,
            f"run: {self.run_name}",
            f"elapsed: {time.perf_counter() - self.t0:.1f}s",
            f"updates completed: {len(self.history)}",
            "",
            "--- environment ---",
            json.dumps(environment_report(), indent=2, default=str),
            "",
            "--- traceback ---",
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        ]
        if extra:
            lines += ["--- state at failure ---", json.dumps(extra, indent=2, default=str), ""]
        if self.history:
            lines += ["--- last 10 updates ---"]
            lines += [json.dumps(r, default=float) for r in self.history[-10:]]
        lines += ["", "--- timing ---", self.timing_summary(), "=" * 78]

        text = "\n".join(lines)
        path.write_text(text, encoding="utf-8")
        self.print("\n" + text)
        self.print(f"\ncrash report written to {path}")
        return path

    def close(self) -> None:
        try:
            self._console.close()
        except Exception:
            pass


def tensor_debug(**tensors: torch.Tensor) -> dict[str, Any]:
    """Summarise tensors for a crash report: shape, dtype, range, NaN count.

    Pass whatever was in scope when things went wrong. A shape mismatch and a
    NaN look identical in a traceback and completely different here.
    """
    out: dict[str, Any] = {}
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            out[name] = repr(t)[:200]
            continue
        finite = torch.isfinite(t.float())
        entry: dict[str, Any] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "n_nan": int(torch.isnan(t.float()).sum()),
            "n_inf": int(torch.isinf(t.float()).sum()),
        }
        if bool(finite.any()):
            vals = t.float()[finite]
            entry["min"] = round(float(vals.min()), 6)
            entry["max"] = round(float(vals.max()), 6)
            entry["mean"] = round(float(vals.mean()), 6)
        out[name] = entry
    return out
