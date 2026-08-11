"""Process-wide counters and timings, as a plain dict.

The event bus carries everything the engine does, but subscribing to it means learning its
kinds, its field names, and its threading rules. Most people exporting metrics want none of
that: they want a flat dict of numbers to hand to Prometheus, OpenTelemetry, StatsD, or a
log line, once a minute, forever.

That is this module. A single bus sink folds every query into a fixed set of counters and
histograms, `metrics_snapshot` returns them as nested plain data, and `prometheus_text`
renders the same numbers in the Prometheus text exposition format. The sink costs a few
integer adds per event and holds a bounded amount of state regardless of how many queries
run.

Collection starts on the first snapshot, or on an explicit `start_metrics()`. It is not
always-on because attaching *any* bus sink signals to the engine that per-query profiles
are being consumed, which makes it assemble one on every query — a cost a process that
exports no metrics should not pay.

Counters are cumulative from that point, like every counter-based metrics system, so a
monitoring backend can compute rates by differencing. `reset_metrics` exists for tests and
for a long-lived service that would rather report per-interval numbers itself.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from batcher._internal import events
from batcher.observe.accelerators.gauges import accelerator_gauges
from batcher.observe.collector import _Collector
from batcher.observe.node_metrics import (
    NODE_CONDITION_HELP,
    device_gauges,
)

__all__ = [
    "metrics_snapshot",
    "prometheus_text",
    "reset_metrics",
    "start_metrics",
    "stop_metrics",
]

_collector = _Collector()
_detach: Callable[[], None] | None = None
_attach_lock = threading.Lock()


def start_metrics() -> None:
    """Begin collecting engine counters, so later snapshots include everything since.

    Counting is opt-in rather than always-on, and deliberately so. Attaching any sink to
    the event bus tells the engine that something is consuming per-query profiles, which
    makes it assemble one on every query — a real cost on the small-query path that a
    process exporting no metrics should not pay.

    `metrics_snapshot` and `prometheus_text` call this for you, so the usual path needs
    nothing. Call it explicitly at startup when you want the first scrape to include the
    queries that ran before it. Idempotent and cheap on the repeat path.

    Examples:
        .. doctest::

            >>> from batcher.observe import metrics_snapshot, start_metrics
            >>> start_metrics()
            >>> metrics_snapshot()["queries"]["total"] >= 0
            True

    Returns:
        None.
    """
    global _detach
    if _detach is not None:
        return
    with _attach_lock:
        if _detach is None:
            _detach = events.subscribe(_collector.handle)


def stop_metrics() -> None:
    """Stop collecting, and leave the module able to start again.

    The counterpart `start_metrics` needs and did not have. Detaching by calling the stored
    unsubscribe function is not enough on its own: the handle stays set, and `start_metrics`
    treats a non-`None` handle as "already attached", so collection can never be resumed in
    that process. A test that detached to avoid leaking a subscriber therefore silenced every
    later one instead, which is a worse leak in the other direction.

    Idempotent: stopping a collector that is not running does nothing.

    Returns:
        None.
    """
    global _detach
    with _attach_lock:
        detach, _detach = _detach, None
    if detach is not None:
        detach()


def metrics_snapshot() -> dict[str, Any]:
    """Every engine counter and timing as a nested dict of plain numbers.

    The dependency-free metrics export: no Batcher types, no event-bus knowledge, nothing
    that needs closing. Counters are cumulative since collection started, so a scrape loop
    differences successive snapshots to get rates. The first call starts collection, so
    call `start_metrics` at startup if the first scrape should cover earlier queries.

    The top-level keys are ``uptime_seconds``, ``queries`` (counts, a live ``active`` gauge,
    a duration histogram, and failures broken out by exception type), ``rows``, ``bytes``,
    ``spills`` (count and volume), ``operators`` (per operator kind), ``logs``
    (records per level), ``data_quality`` (constraint results, in total and per constraint), and
    ``recovery``. A distributed or batch-inference job adds
    ``partitions``, ``skipped`` (inputs dropped under ``on_error="skip"`` — unreadable files
    or splits, not rows: an unreadable file's row count is exactly what is unknown), ``inference``
    (batches, rows, and latency), and ``gpu`` (peak plus per-device utilization and VRAM).

    ``cpu``, ``memory``, and ``io`` are what the query cost the *machine*, summed from the
    per-operator measurements the engine already takes: CPU milliseconds across every worker
    thread, resident-set high-water, page faults, involuntary context switches (contention
    for cores this process was told it had), and real block-device bytes with page-cache
    hits excluded. ``backends`` splits operators by which tier ran the per-row work
    (``interp``, ``jit``, ``interp+jit``), which is the only way to see the JIT silently
    falling back.

    ``writes`` is the counterpart of ``rows``/``bytes``: what the job *produced* — files,
    rows, and bytes on storage — overall and per sink format. Its bytes are the size after
    encoding and compression, so they are deliberately not comparable with the Arrow
    in-memory ``bytes.scanned_total``.

    ``streaming`` carries one entry per continuous query, keyed by its name: micro-batches
    and rows as counters, and throughput, retained state, and ``behind_by_ms`` — how much
    longer the last micro-batch took than its trigger cadence — as levels. Empty until a
    stream has completed a batch, which is the one section that stays absent rather than
    zeroed: a zero here would be indistinguishable from a stopped query.

    ``resources`` is the level rather than the total: Carbonite's buffer-pool envelope and
    its high-water mark, the spill store's per-tier bytes and free disk, the result cache's
    hit rate, the admission limiter's queue depth. These are **gauges** — each reading
    replaces the last — so differencing them gives noise, not a rate. The section is empty
    until a query has completed under a resource manager.

    ``node`` carries the hardware conditions worth alerting on rather than counting: devices
    on a degraded host link, devices whose memory is failing, an NVLink fabric that is down,
    and the RDMA ports' own error counters. Facts rather than verdicts — whether a device is
    schedulable is a subsystem's decision, and this layer does not ask. Every one of these
    conditions leaves a job correct and slow, so none shows up in any of the counters above —
    which is exactly why they belong in the thing an operator has already wired an alert to.
    All zero where nothing could be read, so a CPU-only host exports the section as zeros
    rather than omitting it and making a scrape config conditional.

    Examples:
        .. doctest::

            >>> from batcher.observe import metrics_snapshot
            >>> snap = metrics_snapshot()
            >>> sorted(snap)  # doctest: +NORMALIZE_WHITESPACE
            ['backends', 'bytes', 'cpu', 'data_quality', 'gpu', 'inference', 'io', 'logs',
             'memory', 'node', 'operators', 'partitions', 'queries', 'recovery',
             'resources', 'rows', 'skipped', 'spills', 'streaming', 'uptime_seconds',
             'writes']
            >>> snap["queries"]["total"] >= 0
            True

    Returns:
        A nested dict of counters, safe to `json.dumps` and to mutate.
    """
    start_metrics()
    return _collector.snapshot()


def _escape_label(value: str) -> str:
    """Escape a label value for the Prometheus text format.

    Constraint names carry the characters the format reserves — a regex constraint's name
    embeds the pattern, quotes and backslashes included — and an unescaped one produces a
    line no scraper can parse, silently dropping the whole exposition.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def prometheus_text() -> str:
    """The same counters rendered in the Prometheus text exposition format.

    Serve this from your application's existing ``/metrics`` endpoint and Batcher's numbers
    join whatever you already scrape. Batcher owns no HTTP server for this and pulls in no
    client library; the string is built directly, so there is nothing to configure.

    Every series is prefixed ``batcher_``, counters carry the conventional ``_total``
    suffix, and query duration is exported as a real histogram with ``_bucket``, ``_sum``,
    and ``_count`` series.

    Examples:
        .. doctest::

            >>> from batcher.observe import prometheus_text
            >>> "batcher_queries_total" in prometheus_text()
            True

    Returns:
        The metrics as Prometheus exposition text, ending in a newline.
    """
    snap = metrics_snapshot()
    out: list[str] = []

    def counter(name: str, value: float, help_text: str, unit: str = "counter") -> None:
        out.append(f"# HELP batcher_{name} {help_text}")
        out.append(f"# TYPE batcher_{name} {unit}")
        out.append(f"batcher_{name} {value}")

    counter("uptime_seconds", snap["uptime_seconds"], "Seconds since metrics started", "gauge")
    counter("queries_total", snap["queries"]["total"], "Queries executed")
    counter("queries_failed_total", snap["queries"]["failed"], "Queries that raised")
    counter("queries_active", snap["queries"]["active"], "Queries executing right now", "gauge")
    counter("rows_scanned_total", snap["rows"]["scanned_total"], "Rows read from sources")
    counter("rows_out_total", snap["rows"]["out_total"], "Rows returned to callers")
    counter("bytes_scanned_total", snap["bytes"]["scanned_total"], "Bytes read from sources")
    counter("rows_streamed_total", snap["rows"]["streamed_total"], "Rows delivered by iter_batches")
    counter(
        "bytes_streamed_total", snap["bytes"]["streamed_total"], "Bytes delivered by iter_batches"
    )
    counter("spills_total", snap["spills"]["total"], "Operator spills to disk")
    if snap["queries"]["failed_by_error"]:
        out.append("# HELP batcher_query_errors_total Failed queries by exception type")
        out.append("# TYPE batcher_query_errors_total counter")
        for error, count in snap["queries"]["failed_by_error"].items():
            out.append(f'batcher_query_errors_total{{error="{_escape_label(error)}"}} {count}')

    out.append("# HELP batcher_query_duration_ms Query wall time in milliseconds")
    out.append("# TYPE batcher_query_duration_ms histogram")
    cumulative = 0
    for edge, count in snap["queries"]["duration_ms_buckets"].items():
        cumulative = max(cumulative, count)
        out.append(f'batcher_query_duration_ms_bucket{{le="{edge}"}} {cumulative}')
    out.append(f'batcher_query_duration_ms_bucket{{le="+Inf"}} {snap["queries"]["total"]}')
    out.append(f"batcher_query_duration_ms_sum {snap['queries']['duration_ms_total']}")
    out.append(f"batcher_query_duration_ms_count {snap['queries']['total']}")

    # The per-operator series and the process-wide work totals — CPU, spill volume, real
    # block-device I/O, faults, preemption — all of which the engine already measured per
    # operator and the exposition used to drop on the floor.
    out.extend(_collector.work.render())
    # Carbonite's envelopes: the buffer pool, the spill store's tiers, the result cache, the
    # admission queue, the shuffle session's locality and credit window.
    out.extend(_collector.resources.render())
    # Per-query streaming series, absent entirely until a stream has run a micro-batch.
    out.extend(_collector.streams.render())
    # What the job produced: files, rows and bytes on storage, overall and per sink format.
    out.extend(_collector.writes.render())

    counter(
        "partitions_done_total", snap["partitions"]["done_total"], "Distributed partitions done"
    )
    counter(
        "skipped_total",
        snap["skipped"]["total"],
        "Unreadable files or splits dropped under on_error=skip",
    )
    counter(
        "malformed_rows_total",
        snap["skipped"]["malformed_rows_total"],
        "Rows the job dropped: unparseable records (on_bad_lines) and UDF failures "
        "(max_errored_rows)",
    )
    if snap["skipped"]["malformed_rows_by_source"]:
        out.append(
            "# HELP batcher_malformed_rows_by_source_total Dropped rows by what dropped them"
        )
        out.append("# TYPE batcher_malformed_rows_by_source_total counter")
        for source, dropped in snap["skipped"]["malformed_rows_by_source"].items():
            out.append(
                f'batcher_malformed_rows_by_source_total{{source="{_escape_label(source)}"}} '
                f"{dropped}"
            )

    counter("dq_checks_total", snap["data_quality"]["checks_total"], "Data-quality checks run")
    counter(
        "dq_failed_total", snap["data_quality"]["failed_total"], "Data-quality checks that failed"
    )
    counter(
        "dq_violations_total",
        snap["data_quality"]["violations_total"],
        "Rows violating a data-quality constraint",
    )
    if snap["data_quality"]["by_constraint"]:
        out.append("# HELP batcher_dq_constraint_violations_total Violations by constraint")
        out.append("# TYPE batcher_dq_constraint_violations_total counter")
        for name, stats in snap["data_quality"]["by_constraint"].items():
            label = f'{{constraint="{_escape_label(name)}"}}'
            out.append(f"batcher_dq_constraint_violations_total{label} {stats['violations']}")

    if snap["recovery"]:
        out.append("# HELP batcher_recovery_total Fault-tolerance actions by kind")
        out.append("# TYPE batcher_recovery_total counter")
        for event_name, count in sorted(snap["recovery"].items()):
            out.append(f'batcher_recovery_total{{event="{event_name}"}} {count}')
    counter("inference_batches_total", snap["inference"]["batches_total"], "Inference batches run")
    counter("inference_rows_total", snap["inference"]["rows_total"], "Rows through inference")
    counter(
        "inference_latency_ms_total",
        snap["inference"]["latency_ms_total"],
        "Cumulative inference batch latency",
    )
    counter(
        "inference_blocked_ms_total",
        snap["inference"]["blocked_ms_total"],
        "Cumulative worker time blocked on input",
    )
    gpu_devices = snap["gpu"]["devices"]
    if gpu_devices:
        out.append("# HELP batcher_gpu_utilization_percent Current GPU utilization by device")
        out.append("# TYPE batcher_gpu_utilization_percent gauge")
        for device, stats in gpu_devices.items():
            out.append(f'batcher_gpu_utilization_percent{{device="{device}"}} {stats["util_pct"]}')
        out.append("# HELP batcher_gpu_memory_used_bytes Current GPU memory in use by device")
        out.append("# TYPE batcher_gpu_memory_used_bytes gauge")
        for device, stats in gpu_devices.items():
            out.append(
                f'batcher_gpu_memory_used_bytes{{device="{device}"}} {stats["mem_used_bytes"]}'
            )
    out.extend(device_gauges())
    # The deep per-device series: link throughput and derate, clock headroom, codec engines,
    # the memory reserve and BAR1, the integrated energy counter, and the DCGM counters where
    # they exist. Appended rather than folded into `device_gauges` because that function is
    # vendor-normalized across NVIDIA and AMD, and these are read from NVML's own detail —
    # merging them would either lose the detail or invent AMD equivalents that do not exist.
    out.extend(accelerator_gauges())
    node = snap.get("node") or {}
    for name, help_text in NODE_CONDITION_HELP.items():
        if name in node:
            out.append(f"# HELP batcher_node_{name} {help_text}")
            out.append(f"# TYPE batcher_node_{name} gauge")
            out.append(f"batcher_node_{name} {node[name]}")
    return "\n".join(out) + "\n"


def reset_metrics() -> None:
    """Zero every counter and restart the uptime clock.

    For tests, and for a service that reports per-interval numbers rather than letting a
    backend difference cumulative ones. Does not detach the collector, so counting resumes
    immediately from zero.

    Examples:
        .. doctest::

            >>> from batcher.observe import metrics_snapshot, reset_metrics
            >>> reset_metrics()
            >>> metrics_snapshot()["queries"]["total"]
            0

    Returns:
        None.
    """
    _collector.reset()
