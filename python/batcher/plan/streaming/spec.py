"""Neutral streaming-query specification types — triggers, output modes, progress.

These are the immutable value types exchanged between the conductor (`api`, which
exposes them on the public surface) and the executor (`core`, which produces
progress and consumes the trigger/output-mode). Like every type under `plan`, this
module imports no subsystem, so both layers share one definition with no cross-layer
edge.

The vocabulary mirrors Spark Structured Streaming so the concepts transfer, but
batch / micro-batch / continuous are *modes of the one engine*, not separate APIs:
a `Trigger` and an `OutputMode` are optional inputs to the same `ds.write(...)`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

__all__ = [
    "OutputMode",
    "StreamingQueryProgress",
    "StreamingQueryStatus",
    "Trigger",
    "Watermark",
    "parse_interval_seconds",
]

_UNIT_SECONDS: Final[dict[str, float]] = {
    "us": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
    "ms": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}

_INTERVAL_RE: Final[re.Pattern[str]] = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*")


def parse_interval_seconds(interval: float | int | str) -> float:
    """Parse a trigger interval to seconds.

    Accepts a number (already seconds) or a Spark-style string such as
    ``"5 seconds"``, ``"1 minute"``, ``"500 milliseconds"``, ``"100ms"``. Raises
    `ValueError` for an unrecognized unit or a non-positive duration.
    """
    if isinstance(interval, (int, float)):
        seconds = float(interval)
    else:
        match = _INTERVAL_RE.fullmatch(interval)
        if match is None:
            raise ValueError(f"cannot parse interval {interval!r} (try '5 seconds', '100ms')")
        value, unit = match.group(1), match.group(2).lower()
        if unit not in _UNIT_SECONDS:
            raise ValueError(f"unknown interval unit {unit!r} in {interval!r}")
        seconds = float(value) * _UNIT_SECONDS[unit]
    if seconds < 0:
        raise ValueError(f"interval must be non-negative, got {seconds}")
    return seconds


@dataclass(frozen=True, slots=True)
class Trigger:
    """When the streaming engine fires a micro-batch (Spark `Trigger` parity).

    Build via the classmethods, never the raw constructor:

    * ``Trigger.processing_time("5 seconds")`` — fire a micro-batch on a fixed wall
      clock interval (the default streaming cadence).
    * ``Trigger.once()`` — process one micro-batch of all currently-available data,
      then stop.
    * ``Trigger.available_now()`` — drain all currently-available data across as many
      micro-batches as needed, then stop (the incremental-batch / backfill trigger).
    * ``Trigger.continuous("1 second")`` — lowest-latency processing: micro-batches
      run back-to-back with no inter-batch delay, a checkpoint epoch committed on the
      interval. Stateless pipelines only (filter / select / map_batches), as in Spark.
    """

    kind: Literal["processing_time", "once", "available_now", "continuous"]
    interval_seconds: float | None = None

    @classmethod
    def processing_time(cls, interval: float | int | str) -> Trigger:
        """Fire a micro-batch every `interval` (seconds, or a string like '5 seconds')."""
        return cls("processing_time", parse_interval_seconds(interval))

    @classmethod
    def once(cls) -> Trigger:
        """Process one micro-batch of available data, then stop."""
        return cls("once", None)

    @classmethod
    def available_now(cls) -> Trigger:
        """Drain all available data (multiple micro-batches), then stop."""
        return cls("available_now", None)

    @classmethod
    def continuous(cls, interval: float | int | str) -> Trigger:
        """Continuous processing, committing a checkpoint epoch every `interval`."""
        return cls("continuous", parse_interval_seconds(interval))


class OutputMode:
    """How each micro-batch's result is emitted to the sink (Spark `OutputMode` parity).

    * ``APPEND`` — only rows that are final and will not change again are emitted.
      For a plain (stateless) pipeline this is every row; for a windowed aggregation
      it is a window's row once the watermark has closed it.
    * ``COMPLETE`` — the full result table is emitted after every micro-batch (only
      valid for aggregations; the result must fit the sink).
    * ``UPDATE`` — only the result rows whose value changed in this micro-batch are
      emitted (keyed upsert into the sink).
    """

    APPEND: Final = "append"
    COMPLETE: Final = "complete"
    UPDATE: Final = "update"

    _ALL: Final = frozenset({APPEND, COMPLETE, UPDATE})

    @classmethod
    def validate(cls, mode: str) -> str:
        """Return `mode` if recognized, else raise `ValueError`."""
        if mode not in cls._ALL:
            raise ValueError(f"unknown output_mode {mode!r}; use one of {sorted(cls._ALL)}")
        return mode


@dataclass(frozen=True, slots=True)
class StreamingQueryProgress:
    """Metrics for one completed micro-batch (Spark `StreamingQueryProgress` parity)."""

    batch_id: int
    num_input_rows: int
    num_output_rows: int
    duration_ms: float
    timestamp: float

    @property
    def input_rows_per_second(self) -> float:
        """Throughput for this micro-batch (rows / second), 0 if it took no time."""
        return self.num_input_rows / (self.duration_ms / 1000.0) if self.duration_ms else 0.0


@dataclass(frozen=True, slots=True)
class StreamingQueryStatus:
    """A point-in-time snapshot of a running query (Spark `StreamingQueryStatus` parity)."""

    is_active: bool
    is_data_available: bool
    is_trigger_active: bool
    message: str
    batches_processed: int


@dataclass(frozen=True, slots=True)
class Watermark:
    """An event-time watermark (Spark ``withWatermark``).

    `time_col` is the event-time column; `lateness_micros` is the allowed lateness.
    The watermark advances to ``max(observed event time) - lateness``; rows older
    than it are dropped as late, and a windowed aggregation's closed windows
    (``window_end <= watermark``) are emitted and evicted so streaming state stays
    bounded. This is a driver-side annotation; it never reaches the Rust IR.
    """

    time_col: str
    lateness_micros: int
