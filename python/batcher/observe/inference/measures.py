"""Stateless computations behind the live-progress snapshot and its diagnostics.

Split out of `progress` on the seam its author had already marked: everything here is a
pure function of a job's accumulated numbers, with no lock, no bus, and no mutation. That
separation is what lets the folding logic next door stay about *folding*, and it keeps
either half readable on its own as the event vocabulary grows.

The two number formatters this module used to define were byte-identical copies of the
console's, which is how the engine came to have three byte formatters that disagreed. They
are gone; `_internal.humanize` is the one implementation, and `progress` imports it
directly rather than through here.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from batcher.observe.inference.progress import _Job

#: Smoothing for the displayed rate and latency: steady enough to read, quick enough
#: to track.
_ALPHA = 0.3


def _smooth(current: float, sample: float) -> float:
    """Exponentially smooth `sample` into `current`, seeding on the first reading."""
    return sample if current == 0.0 else current + _ALPHA * (sample - current)


def _partition_totals(job: _Job) -> tuple[int, int | None]:
    """Aggregate partition ``done`` and ``total`` across a job's stages.

    ``total`` is known only when every stage that has reported a total has one; a single
    unbudgeted stage makes the aggregate total unknown rather than an undercount.
    """
    done = sum(s.done for s in job.stages.values())
    totals = [s.total for s in job.stages.values()]
    total = sum(t for t in totals if t is not None) if totals and None not in totals else None
    return done, total


def _mean_util(job: _Job) -> float | None:
    """Mean current utilization across the job's devices, or `None` with no sample."""
    if not job.gpus:
        return None
    return sum(g.util_pct for g in job.gpus.values()) / len(job.gpus)


def _mean_vram(job: _Job) -> float | None:
    """Mean used-VRAM fraction across devices that report a total, or `None`."""
    fracs = [g.mem_fraction for g in job.gpus.values() if g.mem_fraction is not None]
    return sum(fracs) / len(fracs) if fracs else None


def _blocked_rising(trend: deque[float]) -> bool:
    """Whether blocked time is trending up: the second half averages well above the first."""
    if len(trend) < 6:
        return False
    values = list(trend)
    half = len(values) // 2
    early = sum(values[:half]) / half
    late = sum(values[half:]) / (len(values) - half)
    return early > 0 and late > early * 1.5


def _finding(severity: str, code: str, message: str) -> dict[str, Any]:
    """One diagnostic finding as a plain dict."""
    return {"severity": severity, "code": code, "message": message}
