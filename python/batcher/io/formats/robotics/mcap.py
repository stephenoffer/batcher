"""MCAP — the container robotics and ADAS logs are recorded in.

MCAP is the ROS 2 default recording format and the interchange format for autonomous-
driving stacks: one file multiplexes every sensor on the vehicle — camera frames, LiDAR
sweeps, radar returns, CAN signals, GPS/IMU, planner state — as timestamped messages on
named *topics*, in log-time order.

That shape is why this reader is not just another blob source:

* **A row is a message**, not a file. One drive log is millions of rows across a hundred
  topics, so the relation is the natural unit and `group_by("topic")` /
  `filter(col("topic") == "/lidar")` are the natural queries.
* **The container is indexed.** MCAP records a summary — per-channel message counts and
  the log-time range — so `row_count()` and `statistics()` are answered without reading a
  message, and a time-sliced or topic-filtered read *seeks* rather than scans. This reader
  pushes both down (`_pushdown.message_filters`), which is the difference between reading
  five seconds of a two-hour drive and reading the drive.
* **Messages stay encoded.** The `data` column is the raw payload (`large_binary`, since a
  LiDAR sweep or camera frame is megabytes and a batch of them overflows 32-bit offsets).
  Decoding is a downstream concern, so a query that only needs `/gps` never pays to
  deserialize `/camera`.

The ordinary multi-sensor workflow follows directly: filter to the topics of interest,
then `join_asof` them on `log_time` to align sensors sampled at different rates — the
time-alignment step every perception and ADAS pipeline starts with.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any, ClassVar

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count
from batcher._internal.optional import require
from batcher.io.base import FileSource
from batcher.io.base._readahead import ordered_readahead
from batcher.io.base.source import _ITER_READAHEAD_BYTES, _ITER_READAHEAD_FILES
from batcher.io.formats.base import SOURCES
from batcher.io.formats.robotics._pushdown import message_filters
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

__all__ = ["MCAP_SCHEMA", "MCAPSource"]

# A message is one row. `log_time` is when the recorder received it and `publish_time` when
# the publisher stamped it; they differ under transport delay, and perception pipelines
# care which one they align on, so both are kept.
MCAP_SCHEMA = pa.schema(
    [
        ("topic", pa.string()),
        ("log_time", pa.timestamp("ns")),
        ("publish_time", pa.timestamp("ns")),
        ("sequence", pa.int64()),
        ("schema_name", pa.string()),
        ("message_encoding", pa.string()),
        # `large_binary`: a single LiDAR sweep or camera frame is megabytes, so a batch of
        # them passes the 2 GB ceiling of 32-bit offsets.
        ("data", pa.large_binary()),
    ]
)

# Messages per emitted batch. A log is unbounded, so the reader streams.
_MESSAGES_PER_BATCH = 8_192


def _require_mcap() -> Any:
    """Import and return the `mcap` reader factory, or raise `BackendError`."""
    return require(
        "mcap.reader",
        "make_reader",
        feature="MCAP support",
        provides="the mcap package",
        extra="robotics",
    )


@SOURCES.register("mcap")
class MCAPSource(FileSource):
    """One or more MCAP robot/vehicle logs, one Arrow row per message.

    Produces ``{topic, log_time, publish_time, sequence, schema_name, message_encoding,
    data}``. `topics` restricts the read at the source — the reader seeks the channel
    index instead of scanning — and a pushed ``topic``/``log_time`` predicate does the
    same automatically.
    """

    suffix = ".mcap"
    format_name = "mcap"
    # `topic` equality/membership and `log_time` ranges are answered by the container's
    # index, so they are worth pushing to the reader rather than filtering after.
    supports_predicate: ClassVar[bool] = True

    __slots__ = ("_topics",)

    def __init__(
        self,
        path: str,
        *,
        topics: list[str] | None = None,
        schema_mode: str = "strict",
        files: list[str] | None = None,
        on_error: str = "raise",
    ) -> None:
        super().__init__(path, schema_mode=schema_mode, files=files, on_error=on_error)
        self._topics = list(topics) if topics is not None else None

    def _reader_kwargs(self) -> dict[str, object]:
        """`topics` changes which rows a split produces, so a worker must rebuild it."""
        base = super()._reader_kwargs()
        return {**base, "topics": list(self._topics)} if self._topics is not None else base

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (fixed schema)
        return MCAP_SCHEMA

    def schema(self) -> pa.Schema:
        """The message schema, which is fixed and needs no file access."""
        return MCAP_SCHEMA

    def identity(self) -> str:
        """The stats key, made distinct per `topics` restriction.

        A `topics=` restriction is a *different relation* from the whole log — it has a
        different row count and different `log_time` bounds (`statistics()` already reports
        the restricted count), so it MUST get its own key. Without this, a source pinned to
        ``/gps`` and one pinned to ``/lidar`` share the base `format:path` key and Kyber
        hands one topic's cardinalities to the other — the same class of silent collision
        the SQL sources fold their connection into `connection_fingerprint` to avoid.

        Returns:
            The base file identity, suffixed with a digest of the sorted topic set when
            `topics` is set.
        """
        return self._subset_identity(super().identity(), "topics", self._topics)

    # ---- reading ----------------------------------------------------------
    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        """Read every message, honoring a pushed topic/time predicate.

        Args:
            projection: Columns the scan must produce; all of them when omitted.
            predicate: Kyber's pushed filter as its IR dictionary. Its topic and
                log-time terms are answered from the container index.

        Returns:
            Every matching message's batch, in file order.
        """
        return list(self.iter_batches(projection, predicate=predicate))

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        """Stream messages, seeking past the topics and time ranges the predicate excludes.

        Args:
            projection: Columns the scan must produce; all of them when omitted.
            predicate: Kyber's pushed filter as its IR dictionary.

        Returns:
            An iterator over the matching messages' batches, in file order.
        """
        topics, (start, end) = message_filters(
            predicate, topic_column="topic", time_column="log_time"
        )
        if self._topics is not None:
            # An explicit `topics=` is the caller's own restriction; a pushed predicate can
            # only narrow it further, never widen it.
            topics = (
                self._topics if topics is None else [t for t in topics if t in set(self._topics)]
            )
        # Route through the same byte-bounded, order-preserving read-ahead the template
        # uses, so a pushed-down read is still bounded in memory and still honours
        # `on_error` — a drive-day directory holding one truncated recording is the norm.
        files = self._files()
        depth = min(len(files), max(2, min(available_cpu_count(), _ITER_READAHEAD_FILES)))
        yield from ordered_readahead(
            files,
            lambda f: self._tolerant_messages(f, projection, topics, start, end),
            depth=max(1, depth),
            max_bytes=_ITER_READAHEAD_BYTES,
        )

    def _tolerant_messages(
        self,
        path: str,
        projection: list[str] | None,
        topics: list[str] | None,
        start: int | None,
        end: int | None,
    ) -> Iterator[pa.RecordBatch]:
        """`_iter_messages`, honoring `on_error` — a truncated recording is skippable."""
        try:
            yield from self._iter_messages(path, projection, topics, start, end)
        except Exception as exc:
            self._errors.tolerate(path, exc, format_name=self.format_name)

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """The unpushed streaming path (the `FileSource` template's entry point)."""
        yield from self._iter_messages(path, projection, self._topics, None, None)

    def _iter_messages(
        self,
        path: str,
        projection: list[str] | None,
        topics: list[str] | None,
        start: int | None,
        end: int | None,
    ) -> Iterator[pa.RecordBatch]:
        with self._fs.open(path) as fh:
            reader = _require_mcap()(fh)
            yield from self._batches_from(reader, projection, topics, start, end)

    def _batches_from(
        self,
        reader: Any,
        projection: list[str] | None,
        topics: list[str] | None,
        start: int | None,
        end: int | None,
    ) -> Iterator[pa.RecordBatch]:
        """Batch a reader's messages — the one message loop every route shares.

        Shared rather than written per entry point because the row tuple has to line up
        with `MCAP_SCHEMA` positionally, and a second copy is a second thing to remember
        when a column is added. The copy that used to live in `_read_file` had already
        drifted in the way that matters: it took every message of the file into one
        unbounded list, where this batches.
        """
        if topics is not None and not topics:
            # The predicate excluded every topic; the file has nothing to contribute, and
            # asking the reader for "no topics" would mean "all topics".
            return
        kwargs: dict[str, Any] = {}
        if topics is not None:
            kwargs["topics"] = topics
        if start is not None:
            kwargs["start_time"] = start
        if end is not None:
            kwargs["end_time"] = end
        rows: list[tuple] = []
        for schema, channel, message in reader.iter_messages(**kwargs):
            rows.append(
                (
                    channel.topic,
                    message.log_time,
                    message.publish_time,
                    message.sequence,
                    schema.name if schema is not None else None,
                    channel.message_encoding,
                    message.data,
                )
            )
            if len(rows) >= _MESSAGES_PER_BATCH:
                yield _batch(rows, projection)
                rows = []
        if rows:
            yield _batch(rows, projection)

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        """Read from an open handle — the template's fallback when no path is available.

        A drive log is millions of messages carrying megabyte payloads, so this batches
        exactly as the streaming route does rather than building one batch of the file.
        """
        reader = _require_mcap()(fh)
        return list(self._batches_from(reader, projection, self._topics, None, None))

    # ---- cheap metadata, all from the summary index ------------------------
    def _summary(self, path: str) -> Any:
        make_reader = _require_mcap()
        with self._fs.open(path) as fh:
            return make_reader(fh).get_summary()

    def _file_row_count(self, path: str) -> int | None:
        """The message count from the summary — no message is read.

        None when the file carries no summary (a recording cut short mid-write) or when
        `topics` restricts the read, since the per-channel counts are the honest answer
        only if every channel is included.
        """
        summary = self._summary(path)
        if summary is None or summary.statistics is None:
            return None
        if self._topics is None:
            return int(summary.statistics.message_count)
        wanted = set(self._topics)
        counts = summary.statistics.channel_message_counts
        return sum(
            int(counts.get(cid, 0))
            for cid, channel in summary.channels.items()
            if channel.topic in wanted
        )

    def statistics(self) -> SourceStatistics:
        """Row count and an exact `log_time` range, read from the index rather than the data.

        The time bounds are exact — they are the recorded first and last message times, not
        bounds over a chunk — so a time-sliced query prunes whole files precisely.

        Returns:
            The statistics, or an empty bundle when no file carries a summary.
        """
        total = 0
        lo: int | None = None
        hi: int | None = None
        exact = True
        for path in self._files():
            summary = self._summary(path)
            stats = summary.statistics if summary is not None else None
            # `_file_row_count` already knows how to count a topic-restricted read; going
            # through it keeps the two from drifting apart, which is how `statistics()`
            # came to report the whole file for a source restricted to one topic.
            count = self._file_row_count(path)
            if stats is None or count is None:
                exact = False
                continue
            total += count
            lo = stats.message_start_time if lo is None else min(lo, stats.message_start_time)
            hi = stats.message_end_time if hi is None else max(hi, stats.message_end_time)
        columns: dict[str, ColumnStat] = {}
        if lo is not None and hi is not None:
            columns["log_time"] = ColumnStat(
                min=_timestamp(lo), max=_timestamp(hi), provenance=Provenance.EXACT
            )
        return SourceStatistics(
            row_count=total if exact else None,
            columns=columns,
            exact_rows=exact,
        )

    def topics(self) -> list[str]:
        """Every topic recorded across the source's files, sorted.

        Reads only the summary, so listing a two-hour drive's hundred topics costs no
        message decode — the discovery step before a query names the two it wants.

        Examples:
            .. doctest::

                >>> from batcher.io import MCAPSource  # doctest: +SKIP
                >>> MCAPSource("s3://drives/2026-07-18/").topics()  # doctest: +SKIP
                ['/camera/front', '/gps', '/imu', '/lidar/top']

        Returns:
            The distinct topic names.
        """
        found: set[str] = set()
        for path in self._files():
            summary = self._summary(path)
            if summary is not None:
                found.update(channel.topic for channel in summary.channels.values())
        return sorted(found)


def _timestamp(nanos: int) -> Any:
    """A nanosecond epoch count as the pyarrow scalar the zone map compares against."""
    return pa.scalar(nanos, pa.timestamp("ns"))


def _batch(rows: list[tuple], projection: list[str] | None) -> pa.RecordBatch:
    columns = list(zip(*rows, strict=True)) if rows else [()] * len(MCAP_SCHEMA)
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(list(values), field.type)
            for values, field in zip(columns, MCAP_SCHEMA, strict=True)
        ],
        schema=MCAP_SCHEMA,
    )
    return batch.select(projection) if projection is not None else batch
