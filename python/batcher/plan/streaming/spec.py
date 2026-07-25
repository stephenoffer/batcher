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

from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Literal

from batcher._internal.errors import PlanError, suggestion
from batcher.plan.streaming._duration import parse_interval_seconds

__all__ = [
    "OutputMode",
    "StreamingQueryProgress",
    "StreamingQueryStatus",
    "Trigger",
    "Watermark",
    "parse_interval_seconds",
]

_TRIGGER_KINDS: Final = ("processing_time", "once", "available_now", "continuous")


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

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.Trigger.processing_time("5 seconds")
            Trigger(kind='processing_time', interval_seconds=5.0)

            >>> bt.Trigger.once()
            Trigger(kind='once', interval_seconds=None)
    """

    kind: Literal["processing_time", "once", "available_now", "continuous"]
    interval_seconds: float | None = None

    def __post_init__(self) -> None:
        """Reject an unknown trigger kind (a typo in a raw construction)."""
        if self.kind not in _TRIGGER_KINDS:
            hint = suggestion(str(self.kind), _TRIGGER_KINDS)
            raise PlanError(
                f"unknown Trigger kind {self.kind!r}; build via Trigger.processing_time(), "
                ".once(), .available_now(), or .continuous()." + (f" {hint}" if hint else "")
            )

    @classmethod
    def processing_time(cls, interval: float | int | str | timedelta) -> Trigger:
        """Fire a micro-batch every `interval` (seconds, or a string like '5 seconds').

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Trigger.processing_time("5 seconds")
                Trigger(kind='processing_time', interval_seconds=5.0)

        Args:
            interval: The wall-clock cadence, as seconds or a Spark-style string
                such as ``"5 seconds"`` or ``"100ms"``.

        Returns:
            A trigger that fires on the given fixed interval.
        """
        return cls("processing_time", parse_interval_seconds(interval))

    @classmethod
    def once(cls) -> Trigger:
        """Process one micro-batch of available data, then stop.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Trigger.once()
                Trigger(kind='once', interval_seconds=None)

        Returns:
            A trigger that processes a single micro-batch and then stops.
        """
        return cls("once", None)

    @classmethod
    def available_now(cls) -> Trigger:
        """Drain all available data (multiple micro-batches), then stop.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Trigger.available_now()
                Trigger(kind='available_now', interval_seconds=None)

        Returns:
            A trigger that drains all available data, then stops.
        """
        return cls("available_now", None)

    @classmethod
    def continuous(cls, interval: float | int | str | timedelta) -> Trigger:
        """Continuous processing, committing a checkpoint epoch every `interval`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Trigger.continuous("1 second")
                Trigger(kind='continuous', interval_seconds=1.0)

        Args:
            interval: The checkpoint-epoch cadence, as seconds or a Spark-style
                string such as ``"1 second"``.

        Returns:
            A trigger that runs micro-batches back-to-back, committing an epoch on
            the interval.
        """
        return cls("continuous", parse_interval_seconds(interval))

    # Spark/Scala capitalized spellings, so a `Trigger.ProcessingTime("5 seconds")`
    # ported straight from a Spark job keeps working. Thin aliases of the snake_case
    # factories above — the concept is identical, so these are real aliases, not errors.
    @classmethod
    def ProcessingTime(cls, interval: float | int | str | timedelta) -> Trigger:
        """Spark spelling of `processing_time` — fire a micro-batch every `interval`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Trigger.ProcessingTime("5 seconds")
                Trigger(kind='processing_time', interval_seconds=5.0)

        Args:
            interval: The wall-clock cadence, as seconds or a Spark-style string.

        Returns:
            A trigger that fires on the given fixed interval.
        """
        return cls.processing_time(interval)

    @classmethod
    def Once(cls) -> Trigger:
        """Spark spelling of `once` — process one micro-batch of available data, then stop.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Trigger.Once()
                Trigger(kind='once', interval_seconds=None)

        Returns:
            A trigger that processes a single micro-batch and then stops.
        """
        return cls.once()

    @classmethod
    def AvailableNow(cls) -> Trigger:
        """Spark spelling of `available_now` — drain all available data, then stop.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Trigger.AvailableNow()
                Trigger(kind='available_now', interval_seconds=None)

        Returns:
            A trigger that drains all available data, then stops.
        """
        return cls.available_now()

    @classmethod
    def Continuous(cls, interval: float | int | str | timedelta) -> Trigger:
        """Spark spelling of `continuous` — run micro-batches back-to-back.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Trigger.Continuous("1 second")
                Trigger(kind='continuous', interval_seconds=1.0)

        Args:
            interval: The checkpoint-epoch cadence, as seconds or a Spark-style string.

        Returns:
            A trigger that runs micro-batches back-to-back, committing an epoch on
            the interval.
        """
        return cls.continuous(interval)


class OutputMode:
    """How each micro-batch's result is emitted to the sink (Spark `OutputMode` parity).

    * ``APPEND`` — only rows that are final and will not change again are emitted.
      For a plain (stateless) pipeline this is every row; for a windowed aggregation
      it is a window's row once the watermark has closed it.
    * ``COMPLETE`` — the full result table is emitted after every micro-batch (only
      valid for aggregations; the result must fit the sink).
    * ``UPDATE`` — only the result rows whose value changed in this micro-batch are
      emitted (keyed upsert into the sink).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.OutputMode.APPEND
            'append'

            >>> bt.OutputMode.validate("complete")
            'complete'
    """

    APPEND: Final = "append"
    COMPLETE: Final = "complete"
    UPDATE: Final = "update"

    _ALL: Final = frozenset({APPEND, COMPLETE, UPDATE})

    @classmethod
    def validate(cls, mode: str) -> str:
        """Return `mode` if it is a recognized (lowercase) output mode, else raise.

        Modes are canonical lowercase (``"append"``); Spark's ``"Append"`` is rejected
        with a suggestion rather than silently normalized, because downstream sink
        guards match the exact string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.OutputMode.validate("complete")
                'complete'

        Args:
            mode: The output mode to check, one of ``APPEND``, ``COMPLETE``, ``UPDATE``.

        Returns:
            The `mode` string unchanged, once validated.

        Raises:
            PlanError: If `mode` is not a recognized output mode. The message suggests
                the closest valid mode.
        """
        if not isinstance(mode, str):
            raise PlanError(
                f"output_mode must be a string, not {type(mode).__name__} ({mode!r}); "
                f"use one of {sorted(cls._ALL)}"
            )
        if mode not in cls._ALL:
            hint = suggestion(mode, cls._ALL)
            raise PlanError(
                f"unknown output_mode {mode!r}; use one of {sorted(cls._ALL)}."
                + (f" {hint}" if hint else "")
            )
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

    @property
    def output_rows_per_second(self) -> float:
        """Emission throughput for this micro-batch (output rows / second), 0 if instant.

        Distinct from `input_rows_per_second`: a filter or a windowed aggregate emits far
        fewer rows than it consumes, so this measures how fast the query *produces* results
        rather than how fast it *reads* input.
        """
        return self.num_output_rows / (self.duration_ms / 1000.0) if self.duration_ms else 0.0

    def __str__(self) -> str:
        """A one-line human summary: batch id, rows in/out, duration, throughput."""
        return (
            f"batch {self.batch_id}: {self.num_input_rows} in -> {self.num_output_rows} out "
            f"in {self.duration_ms:.0f}ms ({self.input_rows_per_second:.0f} rows/s)"
        )


@dataclass(frozen=True, slots=True)
class StreamingQueryStatus:
    """A point-in-time snapshot of a running query (Spark `StreamingQueryStatus` parity)."""

    is_active: bool
    is_data_available: bool
    is_trigger_active: bool
    message: str
    batches_processed: int

    def __str__(self) -> str:
        """A one-line human summary: liveness, the status message, and batches processed."""
        state = "active" if self.is_active else "stopped"
        return f"[{state}] {self.message} ({self.batches_processed} batches processed)"


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

    @classmethod
    def of(cls, time_col: str, delay: float | int | str | timedelta) -> Watermark:
        """Build a watermark from a human delay (Spark ``withWatermark(col, "10 minutes")``).

        Spark takes the allowed lateness as a duration string; this accepts the same,
        plus a `timedelta` or seconds, and converts to the internal microseconds.

        Examples:
            .. doctest::

                >>> from batcher.plan.streaming.spec import Watermark
                >>> Watermark.of("event_time", "10 minutes").lateness_micros
                600000000

        Args:
            time_col: The event-time column the watermark advances on.
            delay: The allowed lateness, as a duration string (``"10 minutes"``), a
                `timedelta`, or seconds.

        Returns:
            A `Watermark` with `lateness_micros` derived from `delay`.
        """
        if not time_col:
            raise PlanError("Watermark.of(): time_col must be a non-empty event-time column name")
        micros = round(parse_interval_seconds(delay) * 1_000_000)
        return cls(time_col, micros)

    @property
    def lateness_seconds(self) -> float:
        """The allowed lateness in seconds (the microseconds, humanized)."""
        return self.lateness_micros / 1_000_000
