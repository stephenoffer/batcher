"""How one engine's query is timed, and what is kept of the timings.

The headline cell is best-of-N, and a minimum is an optimistic tail rather than a typical
latency: it rewards the one quiet run. So every repetition is kept, and the median, the
95th percentile and the in-process CPU time are carried beside the minimum rather than
thrown away once it is known. A small win that does not clear the run-to-run spread is then
visible as one.

Two further limits are stated rather than hidden. The *first* call is timed separately,
because every timed run follows it and has therefore met warm caches, a warm plan cache and
whatever the engine learned from the correctness run; in an isolated child (`--isolate`)
that first call is the only cold one. And the CPU figure is `time.process_time`, which sees
every thread of *this* process and nothing outside it: an engine whose work runs in another
process (a Spark JVM, Ray workers) under-reports, and the figure is labelled in-process for
that reason.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["Timing", "bench", "bench_samples", "timed_call"]


@dataclass
class Timing:
    """Every repetition of one engine's query, wall and in-process CPU, in milliseconds."""

    wall_ms: list[float] = field(default_factory=list)
    cpu_ms: list[float] = field(default_factory=list)

    @property
    def best(self) -> float:
        return min(self.wall_ms) if self.wall_ms else math.inf

    def quantile(self, q: float) -> float | None:
        """The `q` quantile of the wall times (nearest-rank), or ``None`` with no samples."""
        if not self.wall_ms:
            return None
        ordered = sorted(self.wall_ms)
        rank = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        return ordered[rank]


def timed_call(fn: Callable[[], object]) -> tuple[object, float, float]:
    """Call `fn` once. Returns ``(result, wall_ms, in_process_cpu_ms)``."""
    c0, t0 = time.process_time(), time.perf_counter()
    out = fn()
    wall = (time.perf_counter() - t0) * 1000.0
    return out, wall, (time.process_time() - c0) * 1000.0


def bench_samples(fn: Callable[[], object], runs: int = 5, warmup: bool = True) -> Timing:
    """Time `fn` `runs` times, after one untimed warm-up unless `warmup` is false."""
    if warmup:
        fn()
    timing = Timing()
    for _ in range(runs):
        _, wall, cpu = timed_call(fn)
        timing.wall_ms.append(wall)
        timing.cpu_ms.append(cpu)
    return timing


def bench(fn: Callable[[], object], runs: int = 5) -> float:
    """Time ``fn`` best-of-``runs`` in milliseconds (one warm-up first)."""
    return bench_samples(fn, runs).best
