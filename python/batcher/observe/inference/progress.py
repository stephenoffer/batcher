"""Live progress for a long-running distributed or batch-inference job.

The `ActivityStore` answers "which operator, how far, how fast" for a *query*. A multi-hour
inference job over a Ray cluster needs three things that vocabulary cannot hold: how many
*partitions* of a stage are done while it is still running, how each GPU is loaded *right
now*, and how many rows were *silently* dropped. This store folds the distributed
observability events (`PARTITION`, `GPU`, `INFER`, `SKIPPED`, `POOL`) into a snapshot that
renders "N of M partitions, X rows/s, GPU Y%", and into the field-guided diagnostics that
turn those numbers into a verdict a person can act on.

Bounded by construction, because a 12-hour job emits events forever. Every structure is
keyed by something whose cardinality is fixed by the *plan* or the *hardware* — a stage
label, a GPU device, a skip reason — never by time or by event count, so the footprint after
a billion events equals the footprint after ten. GPU utilization lives in a short ring
buffer per device, just deep enough to see the 0->100->0 oscillation that means the GPU is
starved rather than slow; every other reading is a running counter or a smoothed rate.

Thread-safe because the bus is: samples arrive from many worker threads while a reader
renders a line. Every mutation and every read takes one lock.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from batcher._internal import events
from batcher._internal.mathx import safe_div
from batcher.observe.inference.measures import (
    _blocked_rising,
    _count,
    _finding,
    _mean_util,
    _mean_vram,
    _partition_totals,
    _pct,
    _smooth,
)

__all__ = ["InferenceProgress"]

#: How many concurrent jobs (distinct query ids) to retain before the oldest is evicted. A
#: batch-inference run is usually one job; the cap only guards a long-lived process that
#: launches many, and keeps the store bounded across jobs as well as within one.
DEFAULT_MAX_JOBS = 32
#: GPU utilization samples kept per device — enough to see a starvation oscillation, not a
#: history. This is the one per-device buffer, and it is why the store stays bounded.
_GPU_WINDOW = 16
#: Blocked-time snapshots kept per job, to judge whether the pipeline bottleneck is worsening.
_BLOCKED_WINDOW = 16

# GPU utilization bands (percent), from Ray's field guidance. Below `_UTIL_SEVERE` the
# accelerator is being wasted; the target band is the expensive hardware actually earning out.
_UTIL_SEVERE = 30.0
_UTIL_LOW = 70.0
_UTIL_TARGET = 85.0
# GPU memory bands (fraction of VRAM). The good band leaves headroom for a batch-size spike;
# past `_MEM_HIGH` an autobatcher is one large batch away from an out-of-memory kill.
_MEM_GOOD_LO = 0.70
_MEM_HIGH = 0.90
# A device whose recent utilization swings below this floor and above this ceiling within the
# window is cycling — fed in bursts, idle between them — which is data starvation, not a slow
# model. This is the signal a single averaged number hides.
_STARVE_LO = 10.0
_STARVE_HI = 90.0


@dataclass(slots=True)
class _Stage:
    """One stage's partition tally — the ``N of M`` for a single operator."""

    done: int = 0
    total: int | None = None
    rows: int = 0


@dataclass(slots=True)
class _Gpu:
    """One device's latest load plus a short utilization history for starvation detection."""

    util_pct: float = 0.0
    mem_used_bytes: int = 0
    mem_total_bytes: int = 0
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=_GPU_WINDOW))

    @property
    def mem_fraction(self) -> float | None:
        """Used VRAM as a fraction of total, or `None` when total is unknown."""
        return safe_div(self.mem_used_bytes, self.mem_total_bytes, None)

    @property
    def starved(self) -> bool:
        """Whether utilization is oscillating between idle and saturated — data starvation."""
        if len(self.recent) < 4:
            return False
        return min(self.recent) < _STARVE_LO and max(self.recent) > _STARVE_HI


@dataclass(slots=True)
class _Job:
    """Everything known about one running job, all of it bounded in size."""

    query_id: str
    label: str = ""
    started_ts: float = 0.0
    last_ts: float = 0.0
    stages: dict[str, _Stage] = field(default_factory=dict)
    gpus: dict[str, _Gpu] = field(default_factory=dict)
    rows_per_sec: float = 0.0
    total_rows: int = 0
    infer_batches: int = 0
    latency_ms: float = 0.0
    blocked_ms: float = 0.0
    blocked_trend: deque[float] = field(default_factory=lambda: deque(maxlen=_BLOCKED_WINDOW))
    pool_size: int = 0
    pool_pending: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    skipped_total: int = 0
    #: Fault-tolerance actions by `RECOVERY` discriminator (bounded by `RECOVERY_EVENTS`);
    #: `workers_lost` counts rather than lists, since ids are unbounded over a long run.
    recovery: dict[str, int] = field(default_factory=dict)
    workers_lost: int = 0


class InferenceProgress:
    """A bus sink that turns distributed observability events into live inference progress.

    Attach it with `attach`, which returns the detach callable, or feed it events directly
    with `handle` (what a test does). Read it three ways: `snapshot` for the JSON a dashboard
    serves, `render` for the one-line terminal status, and `diagnostics` for the field-guided
    verdicts (GPU under-use, data starvation, a worsening pipeline bottleneck).

    The store is bounded regardless of how long the job runs. It keeps at most `max_jobs`
    jobs, and within each job every structure is keyed by stage, device, or skip reason, so
    no amount of elapsed time or event volume grows it.
    """

    def __init__(self, *, max_jobs: int = DEFAULT_MAX_JOBS) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._max_jobs = max_jobs

    # --- lifecycle ----------------------------------------------------------

    def attach(self) -> Callable[[], None]:
        """Subscribe this store to the event bus; returns the detach callable."""
        return events.subscribe(self.handle)

    # --- ingest -------------------------------------------------------------

    def handle(self, event: events.Event) -> None:
        """Fold one bus event into the store. This is the sink handed to `subscribe`."""
        kind = event.kind
        if kind not in _INGEST:
            return
        with self._lock:
            job = self._job_for(event)
            _INGEST[kind](self, job, event)

    def _job_for(self, event: events.Event) -> _Job:
        """The job for this event's query id, created (and the oldest evicted) on first sight."""
        job = self._jobs.get(event.query_id)
        if job is None:
            if len(self._jobs) >= self._max_jobs:
                # Evict the least-recently-created job so the store stays bounded across many
                # runs, not only within one.
                oldest = next(iter(self._jobs))
                del self._jobs[oldest]
            job = _Job(
                query_id=event.query_id,
                label=str(event.fields.get("label") or event.name or event.query_id),
                started_ts=event.ts,
                last_ts=event.ts,
            )
            self._jobs[event.query_id] = job
        return job

    def _on_partition(self, job: _Job, event: events.Event) -> None:
        stage = job.stages.setdefault(event.name or "stage", _Stage())
        stage.done += 1
        total = event.fields.get("total")
        if total is not None:
            stage.total = int(total)
        rows = int(event.fields.get("rows", 0))
        stage.rows += rows
        self._advance_rate(job, event.ts, rows)

    def _on_infer(self, job: _Job, event: events.Event) -> None:
        fields = event.fields
        job.infer_batches += 1
        job.latency_ms = _smooth(job.latency_ms, float(fields.get("latency_ms", 0.0)))
        blocked = float(fields.get("blocked_ms", 0.0))
        job.blocked_ms = _smooth(job.blocked_ms, blocked)
        job.blocked_trend.append(job.blocked_ms)
        self._advance_rate(job, event.ts, int(fields.get("rows", 0)))

    def _on_gpu(self, job: _Job, event: events.Event) -> None:
        fields = event.fields
        device = str(fields.get("device", fields.get("actor", "gpu0")))
        gpu = job.gpus.get(device)
        if gpu is None:
            gpu = _Gpu()
            job.gpus[device] = gpu
        gpu.util_pct = float(fields.get("util_pct", 0.0))
        gpu.mem_used_bytes = int(fields.get("mem_used_bytes", 0))
        gpu.mem_total_bytes = int(fields.get("mem_total_bytes", gpu.mem_total_bytes))
        gpu.recent.append(gpu.util_pct)

    def _on_skipped(self, job: _Job, event: events.Event) -> None:
        count = int(event.fields.get("count", 0))
        reason = str(event.fields.get("reason", "read_error"))
        job.skipped_total += count
        # Cap the reason cardinality so a per-row unique reason cannot grow the dict; extra
        # reasons fold into one bucket rather than leaking memory over a long run.
        if reason not in job.skipped and len(job.skipped) >= 64:
            reason = "other"
        job.skipped[reason] = job.skipped.get(reason, 0) + count

    def _on_recovery(self, job: _Job, event: events.Event) -> None:
        """Tally one fault-tolerance action, so recovery is not mistaken for slowness.

        Without it a query that survived losing two workers and one that was simply four
        times too slow look identical — the distinction a spot-capacity decision needs.
        """
        name = str(event.fields.get("event", "unknown"))
        job.recovery[name] = job.recovery.get(name, 0) + 1
        if name == "worker_lost":
            job.workers_lost += 1

    def _on_pool(self, job: _Job, event: events.Event) -> None:
        job.pool_size = int(event.fields.get("size", job.pool_size))
        job.pool_pending = int(event.fields.get("pending", job.pool_pending))

    def _advance_rate(self, job: _Job, ts: float, rows: int) -> None:
        """Fold `rows` produced at monotonic `ts` into the smoothed pool-wide rows/sec.

        Events interleave across workers, so the gap between successive events on the bus is
        the pool's real cadence — smoothing instantaneous ``rows / dt`` over it gives a
        pool-level rate, not a single worker's.
        """
        job.total_rows += rows
        dt = ts - job.last_ts
        job.last_ts = ts
        if dt > 0 and rows > 0:
            job.rows_per_sec = _smooth(job.rows_per_sec, rows / dt)

    # --- read ---------------------------------------------------------------

    def _pick(self, query_id: str | None) -> _Job | None:
        if query_id is not None:
            return self._jobs.get(query_id)
        # Default to the most recently active job — the one a person watching is watching.
        return max(self._jobs.values(), key=lambda j: j.last_ts, default=None)

    def snapshot(self, query_id: str | None = None) -> dict[str, Any] | None:
        """The job's live state as a JSON-encodable dict, or `None` if there is no such job.

        With no `query_id` the most recently active job is used. The shape carries partition
        progress per stage, the aggregate ``done``/``total``/``fraction``, the smoothed
        ``rows_per_sec``, per-device GPU load, the actor-pool size, the skipped-row tally, and
        the `diagnostics` list, so a dashboard needs one call per poll.

        Args:
            query_id: The job to read, or None for the most recently active one.

        Returns:
            The job snapshot, or None when no matching job is retained.
        """
        with self._lock:
            job = self._pick(query_id)
            if job is None:
                return None
            done, total = _partition_totals(job)
            return {
                "query_id": job.query_id,
                "label": job.label,
                "partitions": {
                    "done": done,
                    "total": total,
                    "fraction": safe_div(done, total, None),
                    "stages": {
                        name: {"done": s.done, "total": s.total, "rows": s.rows}
                        for name, s in job.stages.items()
                    },
                },
                "rows_per_sec": job.rows_per_sec,
                "total_rows": job.total_rows,
                "inference": {
                    "batches": job.infer_batches,
                    "latency_ms": job.latency_ms,
                    "blocked_ms": job.blocked_ms,
                },
                "gpu": {
                    device: {
                        "util_pct": g.util_pct,
                        "mem_used_bytes": g.mem_used_bytes,
                        "mem_total_bytes": g.mem_total_bytes,
                        "mem_fraction": g.mem_fraction,
                        "starved": g.starved,
                    }
                    for device, g in job.gpus.items()
                },
                "pool": {"size": job.pool_size, "pending": job.pool_pending},
                "skipped": {"total": job.skipped_total, "by_reason": dict(job.skipped)},
                "recovery": {"events": dict(job.recovery), "workers_lost": job.workers_lost},
                "diagnostics": self._diagnostics(job),
            }

    def render(self, query_id: str | None = None) -> str:
        """The one-line terminal status: ``N of M partitions, X rows/s, GPU Y%``.

        Each clause appears only when it is known — no partition total yields a bare count, no
        GPU sample drops the GPU clause — so the line never fabricates a denominator or a
        reading it does not have. Returns ``""`` when there is no matching job.

        Args:
            query_id: The job to render, or None for the most recently active one.

        Returns:
            The status line, or an empty string when no matching job is retained.
        """
        with self._lock:
            job = self._pick(query_id)
            if job is None:
                return ""
            parts: list[str] = [job.label] if job.label else []
            done, total = _partition_totals(job)
            if total:
                parts.append(f"{done} of {total} partitions ({_pct(done / total)})")
            elif done:
                parts.append(f"{done} partitions")
            if job.rows_per_sec > 0:
                parts.append(f"{_count(job.rows_per_sec)} rows/s")
            util = _mean_util(job)
            if util is not None:
                parts.append(f"GPU {util:.0f}%")
            vram = _mean_vram(job)
            if vram is not None:
                parts.append(f"VRAM {vram * 100:.0f}%")
            if job.pool_size:
                parts.append(f"{job.pool_size} actors")
            if job.skipped_total:
                parts.append(f"{_count(job.skipped_total)} skipped")
            # In the status line, not only the metrics: this is the moment a reader asks
            # why the job is slow, and worker loss is the answer.
            if job.workers_lost:
                parts.append(f"recovering ({job.workers_lost} lost)")
        return "  ".join(parts)

    def diagnostics(self, query_id: str | None = None) -> list[dict[str, Any]]:
        """Field-guided findings for the job: what the numbers mean and how severe it is.

        Each finding is a dict with ``severity`` (``"info"``/``"warning"``/``"critical"``),
        ``code``, and a human ``message``. The bands follow Ray's guidance: a GPU under 30%
        is severely under-used, a GPU oscillating between idle and saturated is starved of
        input rather than slow, VRAM past 90% is one batch from an out-of-memory kill, a
        rising blocked time means the pipeline is the bottleneck, and any skipped rows are
        silent data loss worth surfacing.

        Args:
            query_id: The job to judge, or None for the most recently active one.

        Returns:
            The findings, most severe first; empty when the job is healthy or absent.
        """
        with self._lock:
            job = self._pick(query_id)
            return self._diagnostics(job) if job is not None else []

    def _diagnostics(self, job: _Job) -> list[dict[str, Any]]:
        """Build the findings for `job`. Assumes the lock is held."""
        out: list[dict[str, Any]] = []
        for device, gpu in job.gpus.items():
            if gpu.starved:
                out.append(
                    _finding(
                        "warning",
                        "gpu_starved",
                        f"GPU {device} is cycling between idle and saturated: the pipeline "
                        f"cannot feed it fast enough (data starvation, not a slow model).",
                    )
                )
            elif gpu.util_pct < _UTIL_SEVERE:
                out.append(
                    _finding(
                        "critical",
                        "gpu_underused",
                        f"GPU {device} at {gpu.util_pct:.0f}% — severe under-use; the "
                        f"accelerator is mostly idle.",
                    )
                )
            elif gpu.util_pct < _UTIL_LOW:
                out.append(
                    _finding(
                        "info",
                        "gpu_below_target",
                        f"GPU {device} at {gpu.util_pct:.0f}% — below the 70-85% target band.",
                    )
                )
            frac = gpu.mem_fraction
            if frac is not None and frac > _MEM_HIGH:
                out.append(
                    _finding(
                        "warning",
                        "gpu_memory_high",
                        f"GPU {device} VRAM at {frac * 100:.0f}% — near the limit; a batch-size "
                        f"spike risks an out-of-memory kill.",
                    )
                )
        if _blocked_rising(job.blocked_trend):
            out.append(
                _finding(
                    "warning",
                    "pipeline_bottleneck",
                    "Iteration blocked time is rising: workers are waiting on input, so the "
                    "pipeline feeding them is the bottleneck.",
                )
            )
        if job.skipped_total:
            out.append(
                _finding(
                    "warning",
                    "skipped_rows",
                    f"{job.skipped_total} rows were skipped under on_read_error='skip' — "
                    f"silent data loss.",
                )
            )
        order = {"critical": 0, "warning": 1, "info": 2}
        out.sort(key=lambda f: order.get(f["severity"], 3))
        return out


# The kind -> handler table, built once. `handle` dispatches through it, so an event kind the
# store does not care about is a single dict miss with no work.
_INGEST: dict[str, Callable[[InferenceProgress, _Job, events.Event], None]] = {
    events.PARTITION: InferenceProgress._on_partition,
    events.INFER: InferenceProgress._on_infer,
    events.GPU: InferenceProgress._on_gpu,
    events.SKIPPED: InferenceProgress._on_skipped,
    events.POOL: InferenceProgress._on_pool,
    events.RECOVERY: InferenceProgress._on_recovery,
}
