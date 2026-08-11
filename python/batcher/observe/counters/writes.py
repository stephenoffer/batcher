"""Write counters — what the job produced, not just what it consumed.

The read side of a job has always been countable and the write side never was, which is
backwards for the shape most jobs have: an ETL pipeline exists to *produce* something, and
the thing it produces was the one thing no counter reported. A run that read its inputs
correctly and wrote half of them looked, from the metrics, exactly like a healthy run.

Nothing here is newly measured. Every sink already returns a `io.WrittenFile` per file
carrying that file's row count and its size on storage, and `io.WriteManifest` already rolls
them up into `total_rows` / `total_bytes` / `num_files`. Those numbers reached whoever held
the manifest and stopped there. This folds the same three, per format, from the one commit
funnel every write branch routes through.

`bytes` here is **size on storage** — after encoding and compression — which is the figure a
capacity plan and a storage bill are made of. It is deliberately not comparable with
`bytes.scanned_total`, which is Arrow's in-memory size of what was read; a Parquet write of
a gigabyte of Arrow may be a hundred megabytes on disk, and reporting the ratio as a
"compression" figure only works because the two are labelled apart.
"""

from __future__ import annotations

import threading
from typing import Any

from batcher.observe.counters._series import as_number, escape_label

__all__ = ["WriteCounters"]

#: Formats tracked separately. Bounded by the sink registry, which is a fixed vocabulary;
#: the cap only binds a caller passing a format name built from data.
_MAX_FORMATS = 32

#: The summed fields, and the Prometheus name each is exported under. The names follow the
#: convention the read-side counters already use, so `rows_written_total` reads beside
#: `rows_scanned_total` rather than inventing a second spelling.
_FIELDS = (
    ("files", "files_written_total", "Data files a sink committed"),
    ("rows", "rows_written_total", "Rows a sink committed"),
    ("bytes", "bytes_written_total", "Bytes on storage a sink committed, after encoding"),
)


class WriteCounters:
    """Cumulative committed-write totals, overall and per format.

    One instance per collector, with its own lock. Every counter is monotone from `reset`,
    so a monitoring backend differences successive scrapes for a rate.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._commits = 0
        self._formats: dict[str, dict[str, float]] = {}

    def reset(self) -> None:
        """Zero every counter."""
        with self._lock:
            self._commits = 0
            self._formats = {}

    def record(self, fmt: str, fields: dict[str, Any]) -> None:
        """Fold one committed write in.

        Args:
            fmt: The sink's format, as the event's `name`.
            fields: The `WRITE` event's payload — ``files``, ``rows``, ``bytes``.
        """
        with self._lock:
            self._commits += 1
            if fmt not in self._formats and len(self._formats) >= _MAX_FORMATS:
                fmt = "other"
            entry = self._formats.setdefault(fmt, dict.fromkeys((f for f, _, _ in _FIELDS), 0.0))
            for field, _, _ in _FIELDS:
                entry[field] += as_number(fields.get(field))

    def snapshot(self) -> dict[str, Any]:
        """The totals plus the per-format breakdown.

        Returns:
            ``commits_total`` and the three summed fields, with ``by_format`` carrying the
            same three per sink format. All zero before any write has committed — always
            present rather than absent, because a job that writes nothing is a fact a
            scrape config should not have to be conditional about.
        """
        with self._lock:
            # Reported as ints: files, rows and bytes are counts, and a JSON consumer
            # reading `rows_total: 50000.0` has to decide whether the fraction means
            # something. It does not — the float is only an artifact of summing.
            totals = {
                f"{field}_total": int(sum(e[field] for e in self._formats.values()))
                for field, _, _ in _FIELDS
            }
            return {
                "commits_total": self._commits,
                **totals,
                "by_format": {
                    name: {field: int(value) for field, value in e.items()}
                    for name, e in sorted(self._formats.items())
                },
            }

    def render(self) -> list[str]:
        """The counters as Prometheus exposition lines, with a per-format series each.

        Returns:
            A list of lines. The roll-ups are always present; the per-format series appear
            once something has been written.
        """
        snap = self.snapshot()
        out: list[str] = [
            "# HELP batcher_writes_total Writes a sink committed",
            "# TYPE batcher_writes_total counter",
            f"batcher_writes_total {snap['commits_total']}",
        ]
        for field, metric, help_text in _FIELDS:
            name = f"batcher_{metric}"
            out.append(f"# HELP {name} {help_text}")
            out.append(f"# TYPE {name} counter")
            out.append(f"{name} {snap[f'{field}_total']}")
        by_format = snap["by_format"]
        if not by_format:
            return out
        out.append("# HELP batcher_rows_written_by_format_total Rows committed, by sink format")
        out.append("# TYPE batcher_rows_written_by_format_total counter")
        for fmt, entry in by_format.items():
            label = f'{{format="{escape_label(fmt)}"}}'
            out.append(f"batcher_rows_written_by_format_total{label} {entry['rows']}")
        return out
