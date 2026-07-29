"""Stateless computations behind the live-progress snapshot and its diagnostics.

Split out of `progress` on the seam its author had already marked: everything here is a
pure function of a job's accumulated numbers, with no lock, no bus, and no mutation. That
separation is what lets the folding logic next door stay about *folding*, and it keeps
either half readable on its own as the event vocabulary grows.
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


def _pct(fraction: float) -> str:
    """A clamped integer percentage, e.g. ``62%``; ``<1%`` for a small-but-present share."""
    value = max(fraction, 0.0) * 100
    if 0 < value < 1:
        return "<1%"
    return f"{value:.0f}%"


def _count(n: float) -> str:
    """A compact SI-style count: ``1.2K``, ``3.4M``, ``5.6B``."""
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= limit:
            return f"{n / limit:.1f}{suffix}"
    return f"{n:.0f}"
