"""What the console knows about one in-flight query, and how it learns it from the bus.

Split from the renderer so "what is true about this run" and "how it is drawn" are
separate: the state here is asserted on directly in tests without constructing a terminal,
and the drawing next door stays a pure function of it.

**Rate is measured over a window, not since the start.** The previous implementation
divided total rows by total elapsed and smoothed the result, then drew a sparkline of it
and described that sparkline as showing "a stall, a ramp, a stutter". A cumulative average
can show none of those: once a query has run for thirty seconds, a complete stall moves the
average by 3% per second, so the line stays flat through exactly the event a person is
watching for. The window here is short enough to fall to zero during a stall and long
enough not to flicker on morsel boundaries.
"""

from __future__ import annotations

from collections import deque

__all__ = ["RATE_WINDOW_S", "RunState"]

#: How far back the throughput window reaches. Two seconds is long enough that a
#: per-morsel arrival pattern averages out and short enough that a stall is visible within
#: about a second of starting.
RATE_WINDOW_S = 2.0
#: Sparkline history depth — one sample per repaint, so ~1.5s of throughput at 20 fps.
SPARK_N = 30
#: Smoothing on the *windowed* rate. Light, because the window has already done the
#: averaging; this only stops the last digit from jittering between frames.
RATE_ALPHA = 0.5


class RunState:
    """One in-flight query's live state — what the status line is drawn from.

    Accumulates progress, partitions, bytes, and the exceptional counts (skipped inputs,
    malformed rows, recovery actions) that the summary line reports when the query ends.
    """

    __slots__ = (
        "bytes",
        "est",
        "label",
        "malformed",
        "partitions_done",
        "partitions_total",
        "rate",
        "recoveries",
        "rows",
        "samples",
        "skipped",
        "spark",
        "spilled_bytes",
        "stage",
        "t0",
        "written_bytes",
        "written_files",
    )

    def __init__(self, label: str, est: float | None, t0: float, stage: str = "running") -> None:
        self.label = label
        # "running", not "planning": the engine reports no phase transition on the `collect`
        # path, so a hardcoded "planning" would claim, for the entire duration of every
        # query, a phase that ended in its first millisecond.
        self.stage = stage
        self.rows = 0
        self.bytes = 0
        self.est = est
        self.t0 = t0
        self.rate = 0.0
        self.partitions_done = 0
        self.partitions_total: int | None = None
        self.skipped = 0
        self.malformed = 0
        self.spilled_bytes = 0
        self.written_files = 0
        self.written_bytes = 0
        self.recoveries: dict[str, int] = {}
        #: ``(monotonic, cumulative_rows)`` samples, trimmed to `RATE_WINDOW_S`.
        self.samples: deque[tuple[float, int]] = deque()
        self.spark: deque[float] = deque(maxlen=SPARK_N)

    def observe(self, rows: int, nbytes: int = 0) -> None:
        """Fold one progress batch into the running totals."""
        self.rows += rows
        self.bytes += nbytes

    def note_partition(self, total: int | None, rows: int) -> None:
        """Record one distributed partition finishing.

        `total` is carried per event because a stage learns its partition count when it is
        scheduled, not when the query starts; the first event that knows it wins, and a
        later `None` never erases it.
        """
        self.partitions_done += 1
        self.rows += rows
        if total is not None:
            self.partitions_total = int(total)

    def tick(self, now: float) -> None:
        """Update the windowed rate and the sparkline history for one repaint."""
        self.samples.append((now, self.rows))
        cutoff = now - RATE_WINDOW_S
        while len(self.samples) > 2 and self.samples[0][0] < cutoff:
            self.samples.popleft()
        first_t, first_rows = self.samples[0]
        span = now - first_t
        # Below a tenth of a second the window has not accumulated enough to divide by;
        # holding the previous reading beats printing a number derived from one morsel.
        if span >= 0.1:
            instant = (self.rows - first_rows) / span
            self.rate = (
                instant if self.rate == 0.0 else self.rate + RATE_ALPHA * (instant - self.rate)
            )
        self.spark.append(self.rate)

    @property
    def fraction(self) -> float | None:
        """Completed fraction, from partitions when known and rows otherwise.

        Partition counts are preferred because they are *exact*: a distributed stage knows
        it has 64 buckets, while the row estimate that would otherwise drive the bar is a
        guess the query is in the middle of disproving.
        """
        if self.partitions_total:
            return self.partitions_done / self.partitions_total
        if not self.est or self.est <= 0:
            return None
        return self.rows / self.est

    @property
    def eta_s(self) -> float | None:
        """Seconds remaining at the current rate, or `None` when it cannot be known."""
        fraction = self.fraction
        if fraction is None or fraction <= 0 or fraction >= 1:
            return None
        if self.partitions_total:
            elapsed = self.samples[-1][0] - self.t0 if self.samples else 0.0
            return (elapsed / fraction) - elapsed if elapsed > 0 else None
        if self.rate <= 0 or not self.est:
            return None
        return (self.est - self.rows) / self.rate

    def note_recovery(self, event: str) -> None:
        """Count one fault-tolerance action by kind, for the end-of-run summary."""
        self.recoveries[event] = self.recoveries.get(event, 0) + 1

    def anomalies(self) -> list[str]:
        """The exceptional things that happened, as finished phrases, or an empty list.

        A query that transparently survived losing two workers and one that simply ran
        looked identical from the terminal. Every count here is already published on the
        bus and was reaching no human.
        """
        from batcher._internal.humanize import byte_size, plural

        out: list[str] = []
        if self.skipped:
            out.append(f"{plural(self.skipped, 'input')} skipped")
        if self.malformed:
            out.append(f"{plural(self.malformed, 'bad row')} dropped")
        if self.spilled_bytes:
            out.append(f"spilled {byte_size(self.spilled_bytes)}")
        for event, n in sorted(self.recoveries.items()):
            out.append(f"{n}x {event.replace('_', ' ')}")
        return out

    def written(self) -> str:
        """What the query wrote, as a phrase, or ``""`` when it wrote nothing.

        The read side of a job has always been countable and the write side never was,
        which is backwards for an ETL job: the thing it exists to produce is the thing
        nothing reported.
        """
        from batcher._internal.humanize import byte_size, plural

        if not self.written_files:
            return ""
        size = byte_size(self.written_bytes)
        return f"wrote {plural(self.written_files, 'file')}" + (
            f" ({size})" if self.written_bytes else ""
        )
