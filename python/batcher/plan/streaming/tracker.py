"""Per-partition event-time watermark tracking — a stream's progress is a min, not a max.

A watermark is a claim that no future row will carry an event time below it. A stream read
from several independent partitions (Kafka partitions, Kinesis shards, Pub/Sub splits) can
only make that claim as strongly as its **slowest** partition: partition 0 having reached
10:00 says nothing whatever about partition 1, which may still be replaying 09:00.

Tracking one global maximum therefore states a claim the stream cannot support. The fast
partition drags the watermark forward, and every row the slow partition subsequently
delivers is then *correctly* classified as late against a watermark that was wrong — so the
rows are dropped, the window totals are quietly short, and nothing anywhere errors. A
consumer rebalance, a partition-skewed producer, or simply one broker under load is enough
to trigger it. That silent loss is what this module exists to prevent: the watermark is the
**minimum** over per-partition maxima, which is the strongest claim that is actually true.

The price of a minimum is that one stalled partition holds the entire stream back, which is
why Flink pairs `min` with *idleness*: a partition that has delivered nothing for a while
stops contributing to the minimum, so the watermark advances again. That is a genuine trade
rather than a free fix — an idle partition that later wakes up finds its rows late — so it
is tunable (`streaming.watermark_idle_timeout_seconds`) instead of assumed, and disabling it
gives the fully conservative watermark that never advances past a silent partition.

A stream with one partition, or one whose source cannot name its partitions, has a minimum
over a single value: identical to the maximum, and identical to what this replaced.

Layer: `plan`, neutral. This is a value computation over Arrow arrays plus a small map of
scalars, needed identically by the `core` folds and the `api` streaming drivers — neither of
which may import the other.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import pyarrow as pa

__all__ = ["WatermarkTracker", "event_micros"]

#: The partition key a stream with no partition columns reports under. A stream that cannot
#: name its partitions is one partition as far as the minimum is concerned.
_UNPARTITIONED = ""

#: Separator between the parts of a composite partition key (topic + partition id). A unit
#: separator cannot occur in a topic name, so the flattened key stays injective.
_KEY_SEP = "\x1f"

#: Scratch column the per-partition max is computed under, named so it cannot collide with a
#: user column that happens to be called `max` or `t`.
_TS = "__bt_wm_ts"


def event_micros(col: Any) -> Any:
    """Event-time ticks as int64 **microseconds**, whatever the column's resolution.

    Watermarks, window widths, and allowed lateness are all microseconds. Reading the raw
    int64 ticks of a non-``us`` timestamp (a ``timestamp[ns]`` column, say) would scale the
    watermark by up to 1000x — closing windows a thousand times too early, or never. Casting
    through ``timestamp[us]`` first puts every comparison in one unit; a column already in
    ``us`` (or int64) passes through unchanged.

    Args:
        col: An Arrow array, chunked array, or scalar of event times.

    Returns:
        The same shape, as int64 microseconds.
    """
    import pyarrow.compute as pc

    return pc.cast(pc.cast(col, pa.timestamp("us")), pa.int64())


def _partition_key(values: Sequence[Any]) -> str:
    """One partition's parts flattened to a hashable, JSON-serializable key."""
    return _KEY_SEP.join("" if v is None else str(v) for v in values)


def _idle_timeout_seconds() -> float:
    """The configured processing-time idleness threshold (`streaming` config)."""
    from batcher.config import active_config

    return active_config().streaming.watermark_idle_timeout_seconds


class WatermarkTracker:
    """Per-partition event-time maxima, and the watermark that is their minimum.

    Feed it every batch a stream delivers (`observe`); read `watermark` for the current
    event-time frontier in microseconds, or ``None`` while the stream has yet to make a
    claim. The watermark is monotonic by construction: it is the running maximum of the
    minima, so a partition falling behind never rewinds event time and re-admits rows an
    earlier decision already ruled late.

    `expected_partitions` is what the source says it *will* read from, which is not the same
    as what it has read from yet. Without it, a stream whose partition 1 has not yet
    delivered its first message has a minimum over partition 0 alone — the very over-claim
    this class exists to avoid, confined to startup. Declaring the set holds the watermark
    back until every partition has spoken or gone idle.
    """

    __slots__ = ("_clock", "_expected", "_idle_after", "_lateness", "_maxima", "_seen_at", "_wm")

    def __init__(
        self,
        lateness_micros: int,
        *,
        idle_timeout_seconds: float | None = None,
        expected_partitions: Iterable[Sequence[Any]] = (),
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Track a watermark `lateness_micros` behind the slowest partition's event time.

        Args:
            lateness_micros: Allowed lateness, subtracted from each partition's maximum.
            idle_timeout_seconds: Processing-time seconds a partition may deliver nothing
                before it stops holding the minimum back. ``None`` reads the configured
                default; a non-positive value disables idleness entirely.
            expected_partitions: Partition keys the source expects to read, each a sequence
                of the values its partition columns take. A declared partition that has
                delivered nothing holds the watermark back until it goes idle.
            clock: Monotonic processing-time source, for tests.
        """
        self._lateness = lateness_micros
        timeout = _idle_timeout_seconds() if idle_timeout_seconds is None else idle_timeout_seconds
        self._idle_after: float | None = timeout if timeout > 0 else None
        self._clock = clock or time.monotonic
        #: Highest event time (micros) seen per partition key.
        self._maxima: dict[str, int] = {}
        #: Processing time each partition key last delivered a row. A declared-but-unseen
        #: partition starts the clock now, so idleness can eventually release it.
        started = self._clock()
        self._seen_at: dict[str, float] = {_partition_key(p): started for p in expected_partitions}
        self._expected: frozenset[str] = frozenset(self._seen_at)
        self._wm: int | None = None

    # --- observation ------------------------------------------------------
    def observe(
        self,
        data: pa.RecordBatch | pa.Table,
        time_col: str,
        partition_cols: Sequence[str] = (),
    ) -> None:
        """Fold one batch's per-partition event-time maxima in and recompute the watermark.

        All row-touching work is one Arrow hash aggregate; what comes back to Python is one
        row per partition, which is bounded by the source's partition count rather than by
        the batch.

        Args:
            data: The batch (or table) just read from the stream.
            time_col: The event-time column.
            partition_cols: Columns identifying which partition each row came from. Empty
                treats the batch as one partition, which is the honest answer for a source
                that cannot say.
        """
        for key, high in self._maxima_of(data, time_col, partition_cols):
            previous = self._maxima.get(key)
            self._maxima[key] = high if previous is None else max(previous, high)
        self._refresh()

    def _maxima_of(
        self, data: pa.RecordBatch | pa.Table, time_col: str, partition_cols: Sequence[str]
    ) -> list[tuple[str, int]]:
        """This batch's ``(partition key, max event time)`` pairs, marking each seen."""
        import pyarrow.compute as pc

        if data.num_rows == 0:
            return []
        table = pa.Table.from_batches([data]) if isinstance(data, pa.RecordBatch) else data
        micros = event_micros(table.column(time_col))
        present = [c for c in partition_cols if c in table.schema.names]
        now = self._clock()
        if not present:
            # The batch carries no partition attribution, so this stream *is* one partition
            # as far as anything downstream can tell. A declared partition set is then
            # unsatisfiable — nothing will ever deliver under those keys — and leaving it in
            # place would hold the watermark at nothing until every declared partition timed
            # out. Discovering the columns are absent is the moment to drop the expectation.
            self._expected = frozenset()
            high = pc.max(micros)
            if not high.is_valid:
                return []
            self._seen_at[_UNPARTITIONED] = now
            return [(_UNPARTITIONED, high.as_py())]
        keyed = pa.table(
            {**{c: table.column(c) for c in present}, _TS: micros},
        )
        grouped = keyed.group_by(present).aggregate([(_TS, "max")])
        pairs = []
        for row in grouped.to_pylist():
            high = row[f"{_TS}_max"]
            if high is None:  # a partition delivering only null event times says nothing
                continue
            key = _partition_key([row[c] for c in present])
            self._seen_at[key] = now
            pairs.append((key, high))
        return pairs

    def _refresh(self) -> None:
        """Recompute the frontier: the minimum over partitions that still have a say."""
        now = self._clock()
        active = [high for key, high in self._maxima.items() if not self._is_idle(key, now)]
        # A declared partition that has never delivered constrains the minimum to nothing
        # knowable, so the watermark simply does not advance until it speaks or goes idle.
        silent = any(
            key not in self._maxima and not self._is_idle(key, now) for key in self._expected
        )
        if silent or not active:
            return
        candidate = min(active) - self._lateness
        self._wm = candidate if self._wm is None else max(self._wm, candidate)

    def _is_idle(self, key: str, now: float) -> bool:
        """Whether `key` has been silent long enough to stop holding the minimum back."""
        if self._idle_after is None:
            return False
        last = self._seen_at.get(key)
        return last is not None and (now - last) >= self._idle_after

    # --- reading ----------------------------------------------------------
    @property
    def watermark(self) -> int | None:
        """The current event-time frontier in microseconds, or None before the first claim.

        Re-derived on read rather than only on `observe`, so a partition that goes idle
        between batches releases the minimum at the moment it expires rather than at the
        next batch that happens to arrive.
        """
        self._refresh()
        return self._wm

    @property
    def partitions_tracked(self) -> int:
        """How many partitions have delivered at least one row with an event time."""
        return len(self._maxima)

    # --- checkpointing ----------------------------------------------------
    def to_json(self) -> str:
        """The tracker's durable state — per-partition maxima and the frontier reached.

        Processing-time idleness is deliberately *not* persisted: monotonic clock readings
        mean nothing across a restart, and treating every partition as freshly seen is the
        conservative reset. A restarted query re-earns each partition's idleness rather than
        inheriting a stale claim that it had one.

        Returns:
            A JSON document `restore` accepts.
        """
        return json.dumps({"maxima": self._maxima, "watermark": self._wm})

    def restore(self, payload: str | None) -> None:
        """Resume from a `to_json` document (or a bare integer watermark).

        A bare integer is the pre-per-partition checkpoint format: one global scalar, which
        restores as one unpartitioned entry. Reading it keeps a query checkpointed by an
        earlier version resumable instead of silently rewinding its event time to whatever
        the next batch happens to carry.

        Args:
            payload: The persisted state, or None to leave the tracker empty.
        """
        if not payload:
            return
        try:
            loaded = json.loads(payload)
        except (TypeError, ValueError):
            return
        if isinstance(loaded, int):
            self._wm = loaded
            self._maxima = {_UNPARTITIONED: loaded + self._lateness}
            self._seen_at.setdefault(_UNPARTITIONED, self._clock())
            return
        if not isinstance(loaded, dict):
            return
        maxima = loaded.get("maxima")
        if isinstance(maxima, dict):
            self._maxima = {str(k): int(v) for k, v in maxima.items()}
            now = self._clock()
            for key in self._maxima:
                self._seen_at.setdefault(key, now)
        watermark = loaded.get("watermark")
        self._wm = None if watermark is None else int(watermark)
