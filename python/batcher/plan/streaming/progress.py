"""What a micro-batch reported — the progress records a streaming query publishes.

`core` produces these, `api` hands them back on `StreamingQuery.recent_progress`, and a
`StreamingQueryListener` receives them as they happen. Like everything under `plan` they
import no subsystem, so the executor and the conductor share one definition.

The vocabulary is Spark's `StreamingQueryProgress`, `StateOperatorProgress`,
`SourceProgress` and `SinkProgress`, because those are the names an operator already
knows how to read. The one addition is `behind_by_ms`, which Spark leaves the reader to
derive from `durationMs` and the trigger.

**Why the state and lateness fields exist.** A streaming query's two failure modes are
"it is falling behind" and "it is quietly dropping rows", and throughput answers neither.
Late rows in particular were dropped with no counter anywhere: a watermark closed a window
early, the rows that arrived afterwards vanished, and the only evidence was a total that
was too low by an amount nothing recorded. `num_late_rows` and
`StateOperatorProgress.num_late_inputs_dropped` are that evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "SinkProgress",
    "SourceProgress",
    "StateOperatorProgress",
    "StreamingQueryProgress",
    "StreamingQueryStatus",
]


@dataclass(frozen=True, slots=True)
class StateOperatorProgress:
    """One stateful operator's state after a micro-batch (Spark `StateOperatorProgress`).

    A streaming aggregation, a windowed aggregation, or a dedup keeps rows between
    micro-batches, and the health of the query is mostly the health of that state: how
    much of it there is, whether it is being evicted, and how many inputs arrived too late
    to be counted at all.

    Examples:
        .. doctest::

            >>> from batcher.plan.streaming import StateOperatorProgress
            >>> s = StateOperatorProgress("windowed_aggregate", num_rows_total=120)
            >>> s.num_late_inputs_dropped
            0
    """

    #: Which operator this is — ``"aggregate"``, ``"windowed_aggregate"``, ``"dedup"``.
    operator_name: str
    #: Rows the operator currently retains (open windows, live groups, seen keys).
    num_rows_total: int = 0
    #: Rows this micro-batch added to or changed in that state.
    num_rows_updated: int = 0
    #: Rows the watermark evicted this micro-batch (closed windows, expired keys).
    num_rows_removed: int = 0
    #: Bytes the retained state holds, as the memory guard measures it.
    memory_used_bytes: int = 0
    #: Input rows dropped this micro-batch for arriving below the watermark. The number
    #: a total that "looks a bit low" is explained by, and the reason to widen lateness.
    num_late_inputs_dropped: int = 0
    #: The operator's event-time watermark in microseconds since the epoch, or None
    #: before any row has set one.
    watermark_micros: int | None = None

    @property
    def num_rows_dropped_by_watermark(self) -> int:
        """Spark's name for `num_late_inputs_dropped`.

        Examples:
            .. doctest::

                >>> from batcher.plan.streaming import StateOperatorProgress
                >>> s = StateOperatorProgress("windowed_aggregate", num_late_inputs_dropped=3)
                >>> s.num_rows_dropped_by_watermark
                3

        Returns:
            Input rows this operator dropped for arriving below the watermark.
        """
        return self.num_late_inputs_dropped

    def __str__(self) -> str:
        """A one-line summary: retained rows, eviction, and anything dropped as late."""
        late = (
            f", {self.num_late_inputs_dropped} late dropped" if self.num_late_inputs_dropped else ""
        )
        return (
            f"{self.operator_name}: {self.num_rows_total} rows retained "
            f"({self.memory_used_bytes} bytes), {self.num_rows_removed} evicted{late}"
        )


@dataclass(frozen=True, slots=True)
class SourceProgress:
    """What one source contributed to a micro-batch (Spark `SourceProgress`).

    Examples:
        .. doctest::

            >>> from batcher.plan.streaming import SourceProgress
            >>> SourceProgress("kafka:events", num_input_rows=512).num_input_rows
            512
    """

    #: A human description of the source — its `identity()`, or the format and topic.
    description: str
    #: Rows this source produced for the micro-batch.
    num_input_rows: int = 0
    #: The source's position when the micro-batch began, as the checkpoint records it.
    start_offset: dict | None = None
    #: Its position when the micro-batch ended. Equal to `start_offset` for a source with
    #: no checkpointable position.
    end_offset: dict | None = None


@dataclass(frozen=True, slots=True)
class SinkProgress:
    """What the sink accepted for a micro-batch (Spark `SinkProgress`).

    Examples:
        .. doctest::

            >>> from batcher.plan.streaming import SinkProgress
            >>> SinkProgress("DeltaStreamSink", num_output_rows=7).num_output_rows
            7
    """

    #: A human description of the sink — its class name, path, or topic.
    description: str
    #: Rows written.
    num_output_rows: int = 0
    #: The opaque receipt the sink returned, as recorded in the commit log.
    token: str | None = None


@dataclass(frozen=True, slots=True)
class StreamingQueryProgress:
    """Metrics for one completed micro-batch (Spark `StreamingQueryProgress` parity).

    ``behind_by_ms`` is how much longer the micro-batch took than the trigger cadence it
    fires on — the one question a low-latency query needs answered and the one nothing here
    could answer. Throughput says how fast the batch ran; it cannot say whether that was
    fast *enough*, because "enough" is the trigger interval and the progress record did not
    carry it. A query behind by a growing amount is falling behind its source no matter how
    healthy its rows-per-second looks. ``0.0`` when the batch kept up, and for a trigger
    with no interval (``once`` / ``available_now`` / ``continuous``), where there is no
    cadence to be late for.

    Every field after `behind_by_ms` has a default, so a positional construction written
    against an earlier version still builds.

    Examples:
        .. doctest::

            >>> from batcher.plan.streaming import StreamingQueryProgress
            >>> p = StreamingQueryProgress(3, 1000, 40, 250.0, 0.0)
            >>> p.batch_id, p.num_input_rows
            (3, 1000)
    """

    batch_id: int
    num_input_rows: int
    num_output_rows: int
    duration_ms: float
    timestamp: float
    behind_by_ms: float = 0.0
    #: The query's name, so a listener receiving progress from several queries can tell
    #: them apart without holding the handles.
    name: str = ""
    #: Per-stateful-operator state metrics. Empty for a stateless pipeline.
    state_operators: tuple[StateOperatorProgress, ...] = ()
    #: Per-source input metrics.
    sources: tuple[SourceProgress, ...] = ()
    #: What the sink accepted, when the runner reports a single sink.
    sink: SinkProgress | None = None
    #: Where the micro-batch's time went, in milliseconds, keyed as Spark keys it:
    #: ``latestOffset`` (asking the source what is available), ``addBatch`` (running the
    #: plan and writing the sink), ``walCommit`` (the offset/commit log fsyncs), and
    #: ``triggerExecution`` (the whole batch). A total alone cannot distinguish a slow
    #: query from a slow *checkpoint*, which is the first thing to rule out when a stream
    #: falls behind — and the two have opposite remedies.
    duration_breakdown_ms: tuple[tuple[str, float], ...] = ()

    @property
    def duration_ms_map(self) -> dict[str, float]:
        """The `duration_breakdown_ms` pairs as a dict (Spark's ``durationMs``).

        A tuple of pairs on the record because the dataclass is frozen and hashable; a dict
        here because that is what a caller wants to index.

        Examples:
            .. doctest::

                >>> from batcher.plan.streaming import StreamingQueryProgress
                >>> p = StreamingQueryProgress(
                ...     0, 1, 1, 5.0, 0.0, duration_breakdown_ms=(("addBatch", 4.0),)
                ... )
                >>> p.duration_ms_map["addBatch"]
                4.0

        Returns:
            The phase-to-milliseconds mapping for this micro-batch.
        """
        return dict(self.duration_breakdown_ms)

    @property
    def processed_rows_per_second(self) -> float:
        """Spark's name for `output_rows_per_second`.

        Examples:
            .. doctest::

                >>> from batcher.plan.streaming import StreamingQueryProgress
                >>> StreamingQueryProgress(0, 500, 50, 250.0, 0.0).processed_rows_per_second
                200.0

        Returns:
            Output rows per second for this micro-batch.
        """
        return self.output_rows_per_second

    def to_dict(self) -> dict:
        """This record as plain JSON-encodable data (Spark's ``progress.json``).

        A progress record's destination is usually a log line or a metrics system, and both
        want data rather than a dataclass. Keyed in Spark's camelCase so a dashboard written
        against `StreamingQueryProgress` reads this one unchanged.

        Examples:
            .. doctest::

                >>> from batcher.plan.streaming import StreamingQueryProgress
                >>> StreamingQueryProgress(3, 10, 8, 5.0, 0.0).to_dict()["batchId"]
                3

        Returns:
            A JSON-encodable dict of the micro-batch's metrics.
        """
        return {
            "name": self.name,
            "batchId": self.batch_id,
            "timestamp": self.timestamp,
            "numInputRows": self.num_input_rows,
            "numOutputRows": self.num_output_rows,
            "numLateRows": self.num_late_rows,
            "inputRowsPerSecond": self.input_rows_per_second,
            "processedRowsPerSecond": self.processed_rows_per_second,
            "durationMs": {"triggerExecution": self.duration_ms, **self.duration_ms_map},
            "behindByMs": self.behind_by_ms,
            "eventTimeWatermarkMicros": self.event_time_watermark_micros,
            "stateOperators": [
                {
                    "operatorName": s.operator_name,
                    "numRowsTotal": s.num_rows_total,
                    "numRowsUpdated": s.num_rows_updated,
                    "numRowsRemoved": s.num_rows_removed,
                    "memoryUsedBytes": s.memory_used_bytes,
                    "numRowsDroppedByWatermark": s.num_late_inputs_dropped,
                    "watermarkMicros": s.watermark_micros,
                }
                for s in self.state_operators
            ],
            "sources": [
                {
                    "description": s.description,
                    "numInputRows": s.num_input_rows,
                    "startOffset": s.start_offset,
                    "endOffset": s.end_offset,
                }
                for s in self.sources
            ],
            "sink": None
            if self.sink is None
            else {
                "description": self.sink.description,
                "numOutputRows": self.sink.num_output_rows,
                "token": self.sink.token,
            },
        }

    def json(self) -> str:
        """This record as a JSON string (Spark ``StreamingQueryProgress.json``).

        Examples:
            .. doctest::

                >>> import json
                >>> from batcher.plan.streaming import StreamingQueryProgress
                >>> json.loads(StreamingQueryProgress(2, 1, 1, 1.0, 0.0).json())["batchId"]
                2

        Returns:
            The record encoded as JSON.
        """
        import json as _json

        return _json.dumps(self.to_dict())

    @property
    def input_rows_per_second(self) -> float:
        """Throughput for this micro-batch (rows / second), 0 if it took no time.

        Examples:
            .. doctest::

                >>> from batcher.plan.streaming import StreamingQueryProgress
                >>> StreamingQueryProgress(0, 500, 500, 250.0, 0.0).input_rows_per_second
                2000.0

        Returns:
            Input rows per second for this micro-batch.
        """
        return self.num_input_rows / (self.duration_ms / 1000.0) if self.duration_ms else 0.0

    @property
    def output_rows_per_second(self) -> float:
        """Emission throughput for this micro-batch (output rows / second), 0 if instant.

        Distinct from `input_rows_per_second`: a filter or a windowed aggregate emits far
        fewer rows than it consumes, so this measures how fast the query *produces* results
        rather than how fast it *reads* input.

        Examples:
            .. doctest::

                >>> from batcher.plan.streaming import StreamingQueryProgress
                >>> StreamingQueryProgress(0, 500, 50, 250.0, 0.0).output_rows_per_second
                200.0

        Returns:
            Output rows per second for this micro-batch.
        """
        return self.num_output_rows / (self.duration_ms / 1000.0) if self.duration_ms else 0.0

    @property
    def is_behind(self) -> bool:
        """Whether this micro-batch overran the trigger cadence it fires on.

        Examples:
            .. doctest::

                >>> from batcher.plan.streaming import StreamingQueryProgress
                >>> p = StreamingQueryProgress(0, 10, 10, 250.0, 0.0, behind_by_ms=150.0)
                >>> p.is_behind
                True

        Returns:
            True when the batch took longer than its trigger interval.
        """
        return self.behind_by_ms > 0.0

    @property
    def num_late_rows(self) -> int:
        """Input rows every stateful operator dropped this micro-batch for being late.

        The counter the docs used to say did not exist. A windowed aggregate whose
        watermark has closed a window discards rows that belong to it, which is correct and
        invisible: the result is simply short by an amount nothing recorded. A non-zero
        value here says the allowed lateness is too tight for the stream's real skew.

        Examples:
            .. doctest::

                >>> from batcher.plan.streaming import (
                ...     StateOperatorProgress,
                ...     StreamingQueryProgress,
                ... )
                >>> p = StreamingQueryProgress(
                ...     0, 10, 8, 5.0, 0.0,
                ...     state_operators=(
                ...         StateOperatorProgress("windowed_aggregate", num_late_inputs_dropped=2),
                ...     ),
                ... )
                >>> p.num_late_rows
                2

        Returns:
            The total late-input rows dropped across the query's stateful operators.
        """
        return sum(s.num_late_inputs_dropped for s in self.state_operators)

    @property
    def event_time_watermark_micros(self) -> int | None:
        """The query's watermark — the minimum across its stateful operators, or None.

        The *minimum*, because a query is only as caught up as its furthest-behind
        operator: taking the maximum would report a window as closed while another
        operator is still admitting rows into it.

        Examples:
            .. doctest::

                >>> from batcher.plan.streaming import (
                ...     StateOperatorProgress,
                ...     StreamingQueryProgress,
                ... )
                >>> ops = (StateOperatorProgress("windowed_aggregate", watermark_micros=90),)
                >>> StreamingQueryProgress(
                ...     0, 1, 1, 1.0, 0.0, state_operators=ops
                ... ).event_time_watermark_micros
                90

        Returns:
            Microseconds since the epoch, or None before any operator has a watermark.
        """
        marks = [s.watermark_micros for s in self.state_operators if s.watermark_micros is not None]
        return min(marks) if marks else None

    def __str__(self) -> str:
        """A one-line human summary: batch id, rows in/out, duration, throughput.

        A batch that overran its trigger says so, because a throughput figure alone reads
        as healthy right up until the query is hours behind its source. So does a batch
        that dropped late rows, for the same reason.
        """
        late = f", {self.behind_by_ms:.0f}ms behind" if self.is_behind else ""
        dropped = f", {self.num_late_rows} late rows dropped" if self.num_late_rows else ""
        return (
            f"batch {self.batch_id}: {self.num_input_rows} in -> {self.num_output_rows} out "
            f"in {self.duration_ms:.0f}ms ({self.input_rows_per_second:.0f} rows/s{late}){dropped}"
        )


@dataclass(frozen=True, slots=True)
class StreamingQueryStatus:
    """A point-in-time snapshot of a running query (Spark `StreamingQueryStatus` parity).

    Examples:
        .. doctest::

            >>> from batcher.plan.streaming import StreamingQueryStatus
            >>> print(StreamingQueryStatus(True, True, True, "Waiting for data", 12))
            [active] Waiting for data (12 batches processed)
    """

    is_active: bool
    is_data_available: bool
    is_trigger_active: bool
    message: str
    batches_processed: int
    #: Per-stateful-operator state as of the last completed micro-batch, so an operator
    #: can ask "how much state is this query holding right now" without walking history.
    state_operators: tuple[StateOperatorProgress, ...] = field(default=())

    def __str__(self) -> str:
        """A one-line human summary: liveness, the status message, and batches processed."""
        state = "active" if self.is_active else "stopped"
        return f"[{state}] {self.message} ({self.batches_processed} batches processed)"
