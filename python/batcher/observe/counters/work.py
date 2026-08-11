"""Per-operator work counters — what the engine spent, not just how long it took.

The engine measures far more per operator than wall time. `bc-interp` records CPU
nanoseconds across every worker thread, the peak working set, the bytes routed to spill,
real block-device I/O, page faults, and involuntary context switches; the control plane
transcribes all of it into `plan.profile.OpProfile`. Until this module existed the
process-wide export took two fields off that record — elapsed time and output rows — and
dropped the rest, so the questions an operator actually asks of a metrics backend had no
answer: is this job CPU-bound or waiting on disk, how many bytes has it spilled, is the box
paging against it, how much of the fleet's CPU did the joins take.

Two things are worth keeping straight about the byte counters here, because they measure
different things and are easy to conflate:

- ``bytes_scanned`` is the **Arrow in-memory size** of what the scans produced. It is the
  logical volume the query pulled in, and it is what a selectivity or a cost-per-byte figure
  should be computed against.
- ``io_read_bytes`` / ``io_write_bytes`` are **block-device bytes**, page-cache hits
  excluded. A warm scan and a cold scan of the same file are identical in the first figure
  and two orders of magnitude apart in this one.

Every measured field is `0` when the platform could not report it, and the convention the
engine uses everywhere applies here too: **`0` means unmeasured, not zero.** A consumer
must not conclude from a zero `io_read_bytes` that the query read nothing from disk.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from batcher.observe.counters._series import as_number
from batcher.plan.ir_tags import Op

__all__ = ["WorkCounters"]

#: Per-kind maps are bounded by the IR's operator vocabulary, which is fixed at ~16 entries.
#: The cap exists so a malformed event carrying a per-query name cannot grow them anyway.
_MAX_KINDS = 64

#: The per-kind counters, in the order they are reported. Every one is a monotone sum, so a
#: scraper differences successive scrapes to get a rate — the same contract as every other
#: counter in the export.
_SUMMED_FIELDS = (
    "elapsed_ms",
    "cpu_ms",
    "rows_in",
    "rows_out",
    "result_bytes",
    "spill_bytes",
    "io_read_bytes",
    "io_write_bytes",
    "minor_faults",
    "major_faults",
    "vol_ctx_switches",
    "invol_ctx_switches",
)

#: The subset exported as a per-operator Prometheus series. Deliberately smaller than
#: `_SUMMED_FIELDS`: a series per kind per field is real cardinality in a scrape, and the
#: rest stay available in `metrics_snapshot()` for a consumer that wants them.
_PER_KIND_SERIES = (
    ("elapsed_ms", "Operator wall time by kind, in milliseconds"),
    ("cpu_ms", "Operator CPU time by kind, summed across worker threads"),
    ("rows_in", "Rows fed into an operator by kind"),
    ("rows_out", "Rows produced by an operator by kind"),
    ("spill_bytes", "Logical bytes an operator routed to disk, by kind"),
)

#: The whole-execution fields summed across queries. These come from the engine's
#: ``ExecMetrics.query`` block, measured once per run at the FFI boundary, and they are the
#: authority for the process-wide sections — *not* the per-operator sums above. The two would
#: double-count where both are measured, and only this one holds on the streaming tier, which
#: is where most queries run. Wall time is summed so the pair yields mean cores busy.
_USAGE_FIELDS = (
    "wall_ms",
    "cpu_ms",
    "minor_faults",
    "major_faults",
    "vol_ctx_switches",
    "invol_ctx_switches",
    "io_read_bytes",
    "io_write_bytes",
)

#: Process-wide series, as ``(usage key, metric name, help)``. Named for what they measure
#: rather than for the operator vocabulary, because these are the figures a capacity
#: dashboard plots without knowing what a `hash_join` is.
_TOTAL_SERIES = (
    ("cpu_ms", "cpu_ms_total", "CPU milliseconds consumed executing queries"),
    ("wall_ms", "execution_ms_total", "Milliseconds spent inside the engine executing"),
    ("io_read_bytes", "io_read_bytes_total", "Bytes read from block devices, page cache excluded"),
    ("io_write_bytes", "io_write_bytes_total", "Bytes written to block devices"),
    ("minor_faults", "minor_page_faults_total", "Page faults served without disk I/O"),
    ("major_faults", "major_page_faults_total", "Page faults that required disk I/O"),
    (
        "vol_ctx_switches",
        "voluntary_context_switches_total",
        "Times a query gave up a CPU to wait on I/O or a lock",
    ),
    (
        "invol_ctx_switches",
        "involuntary_context_switches_total",
        "Times the scheduler preempted a query — CPU contention",
    ),
)


class WorkCounters:
    """Cumulative per-operator and process-wide work, folded from `STAGE_END` events.

    One instance per collector, guarded by its own lock. Every counter is monotone from
    `reset`, so a monitoring backend differences successive scrapes for a rate.
    """

    # Declared here rather than assigned in `__init__`, so `reset` is the single place the
    # zero values live. Two initializers for the same field is how one of them gains a
    # counter the other does not, and a counter that only `reset` knows about reads as
    # permanently zero.
    _kinds: dict[str, dict[str, float]]
    _counts: dict[str, int]
    _spills: dict[str, int]
    _backends: dict[str, int]
    _usage: dict[str, float]
    _out_of_core_phases: int
    _out_of_core_bytes: int
    _peak_rss_bytes: int
    _threads_max: int

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Zero every counter."""
        with self._lock:
            self._kinds = {}
            self._counts = defaultdict(int)
            self._spills = defaultdict(int)
            self._backends = defaultdict(int)
            self._usage = dict.fromkeys(_USAGE_FIELDS, 0.0)
            self._out_of_core_phases = 0
            self._out_of_core_bytes = 0
            self._peak_rss_bytes = 0
            self._threads_max = 0

    def record_query(self, usage: object) -> None:
        """Fold one finished query's whole-execution reading in.

        Args:
            usage: The `plan.profile.QueryUsage.to_dict()` payload carried on `QUERY_END`,
                or anything else — a non-dict is ignored, because a bus sink must not be able
                to raise on a malformed event.
        """
        if not isinstance(usage, dict):
            return
        with self._lock:
            for name in _USAGE_FIELDS:
                self._usage[name] += as_number(usage.get(name))
            # A level, not an accumulation: summing per-query resident-set peaks would report
            # memory the process never held at one time.
            peak = int(as_number(usage.get("peak_rss_bytes")))
            self._peak_rss_bytes = max(self._peak_rss_bytes, peak)

    def record_spill_store(self, stats: object) -> None:
        """Fold one finished out-of-core phase's spill store in.

        The out-of-core path is not measured per operator — it streams the plan through
        thousands of unmetered engine dispatches — so the operator-level spill flags say
        nothing about it, and until this it was the one execution mode that spilled by
        definition and reported no spill at all. The store's lifetime `bytes_written` is
        the volume that went to disk, and it belongs in the same counter the operators'
        does: both are logical bytes routed off memory.

        Args:
            stats: The store's `stats()` reading, or anything else (ignored).
        """
        if not isinstance(stats, dict):
            return
        with self._lock:
            self._out_of_core_phases += 1
            self._out_of_core_bytes += int(as_number(stats.get("bytes_written")))

    def record_stage(self, kind: str, fields: dict[str, Any]) -> None:
        """Fold one finished operator in.

        Hot only in the sense that it runs once per operator per query — a dict lookup and
        a dozen adds, no allocation once a kind has been seen.

        An event flagged ``measured: False`` is ignored. The dashboard's timeline wants
        every stage the plan had, measured or not; a counter must not fold in an operator
        the engine never ran, or an out-of-core query — which reports its plan and measures
        none of it — would add a zero-row, zero-time entry per operator and pull every
        per-kind average toward zero.

        Args:
            kind: The operator's IR tag (``scan``, ``filter``, ``hash_join``, ...).
            fields: The `STAGE_END` event's payload.
        """
        if not fields.get("measured", True):
            return
        with self._lock:
            if kind not in self._kinds and len(self._kinds) >= _MAX_KINDS:
                kind = "other"
            entry = self._kinds.setdefault(kind, dict.fromkeys(_SUMMED_FIELDS, 0.0))
            self._counts[kind] += 1
            for name in _SUMMED_FIELDS:
                entry[name] += as_number(fields.get(name))
            if fields.get("spilled"):
                self._spills[kind] += 1
            backend = fields.get("backend")
            if backend:
                self._backends[str(backend)] += 1
            self._threads_max = max(self._threads_max, int(as_number(fields.get("threads"))))

    @property
    def spills_total(self) -> int:
        """How many operators engaged their spill path, across every kind."""
        with self._lock:
            return sum(self._spills.values())

    def scanned(self) -> tuple[int, int]:
        """Rows and Arrow bytes the source scans produced.

        The engine's answer to "how much data did this process read", and the reason it is
        derived from the scan operators rather than from progress events: the progress
        stream exists only on the `iter_batches` path, so before this a `collect()` — the
        common case — reported zero rows scanned no matter how much it read.

        Returns:
            ``(rows, bytes)``, both `0` before any scan has been measured.
        """
        with self._lock:
            scan = self._kinds.get(Op.SCAN)
            if scan is None:
                return 0, 0
            return int(scan["rows_out"]), int(scan["result_bytes"])

    def operators(self) -> dict[str, dict[str, float | int]]:
        """Per-kind totals, keyed by operator tag and sorted for a stable exposition."""
        with self._lock:
            return {
                kind: {
                    "count": self._counts[kind],
                    "spills": self._spills[kind],
                    **{f"{name}_total": entry[name] for name in _SUMMED_FIELDS},
                }
                for kind, entry in sorted(self._kinds.items())
            }

    def totals(self) -> dict[str, Any]:
        """The process-wide figures: whole-execution usage, plus what only operators know.

        The CPU, fault, context-switch and block-device figures come from the per-query
        whole-execution reading, which is sound on every tier. Spill volume and the backend
        split come from the operators, because nothing but an operator knows either.

        Returns:
            A flat dict; `cores_busy` is derived from the CPU and wall sums.
        """
        with self._lock:
            totals: dict[str, Any] = dict(self._usage)
            totals["cores_busy"] = (
                self._usage["cpu_ms"] / self._usage["wall_ms"] if self._usage["wall_ms"] else 0.0
            )
            totals["spill_bytes"] = (
                sum(entry["spill_bytes"] for entry in self._kinds.values())
                + self._out_of_core_bytes
            )
            totals["out_of_core_phases"] = self._out_of_core_phases
            totals["peak_rss_bytes_max"] = self._peak_rss_bytes
            totals["threads_max"] = self._threads_max
            totals["backends"] = dict(sorted(self._backends.items()))
            return totals

    def render(self) -> list[str]:
        """The counters as Prometheus exposition lines.

        Reads its own state rather than taking a snapshot, so the metric names here stay
        the exposition's business and the JSON snapshot stays free to name the same figures
        the way a JSON consumer expects.

        Returns:
            A list of lines. Never empty: the process-wide totals are exported as zeros
            before anything is measured, so a scrape config never has to be conditional.
        """
        operators = self.operators()
        totals = self.totals()
        out: list[str] = []
        for key, metric, help_text in _TOTAL_SERIES:
            out.append(f"# HELP batcher_{metric} {help_text}")
            out.append(f"# TYPE batcher_{metric} counter")
            out.append(f"batcher_{metric} {totals.get(key, 0)}")
        out.append("# HELP batcher_spill_bytes_total Logical bytes routed to disk")
        out.append("# TYPE batcher_spill_bytes_total counter")
        out.append(f"batcher_spill_bytes_total {totals.get('spill_bytes', 0)}")
        out.append("# HELP batcher_out_of_core_phases_total Query phases that ran out-of-core")
        out.append("# TYPE batcher_out_of_core_phases_total counter")
        out.append(f"batcher_out_of_core_phases_total {totals.get('out_of_core_phases', 0)}")
        out.append("# HELP batcher_peak_rss_bytes Largest resident-set growth in one query")
        out.append("# TYPE batcher_peak_rss_bytes gauge")
        out.append(f"batcher_peak_rss_bytes {totals.get('peak_rss_bytes_max', 0)}")
        out.append("# HELP batcher_cores_busy Mean cores kept busy across measured executions")
        out.append("# TYPE batcher_cores_busy gauge")
        out.append(f"batcher_cores_busy {totals.get('cores_busy', 0.0)}")
        if not operators:
            return out
        for field, help_text in _PER_KIND_SERIES:
            metric = f"batcher_operator_{field}_total"
            out.append(f"# HELP {metric} {help_text}")
            out.append(f"# TYPE {metric} counter")
            for kind, stats in operators.items():
                out.append(f'{metric}{{kind="{kind}"}} {stats[f"{field}_total"]}')
        out.append("# HELP batcher_operator_spills_total Operators that engaged their spill path")
        out.append("# TYPE batcher_operator_spills_total counter")
        for kind, stats in operators.items():
            out.append(f'batcher_operator_spills_total{{kind="{kind}"}} {stats["spills"]}')
        return out
