"""Reading a Delta table's Change Data Feed, unbounded or over a fixed version window.

Two sources over the same feed, differing in where they stop:

`DeltaStreamSource` is the `readStream` half of a medallion pipeline: each pass reads
only the commits made since the last one and advances a cursor, so chaining bronze →
silver → gold reprocesses nothing. The cursor is a Delta version, which makes the source
checkpointable — a restarted query resumes exactly where it stopped.

`DeltaChangeFeedSource` reads a **closed** window — versions `a` through `b`, or the
commits between two timestamps — and then ends. That is the batch form of Spark's
``readChangeFeed`` and the shape incremental ETL actually asks for: "every change since
the watermark I last recorded, so I can MERGE it into the target". Expressing it as a
stream, which is all that existed, made the natural next step impossible — an unbounded
source cannot be `collect`ed, counted, or joined against a dimension, so a job that
wanted a bounded delta had nowhere to go.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher._internal.logging import note_suppressed
from batcher.io.formats.base import SOURCES
from batcher.io.formats.lakehouse._arrow import normalize_engine_types
from batcher.io.formats.lakehouse._time import normalize_timestamp
from batcher.io.formats.lakehouse.delta._snapshot import require_deltalake
from batcher.io.splits import Split, WholeSourceSplit
from batcher.plan.source_stats import SourceStatistics

__all__ = ["DeltaChangeFeedSource", "DeltaStreamSource"]

# CDF metadata columns delta-rs adds to a change feed.
_CDF_META = ("_change_type", "_commit_version", "_commit_timestamp")


def _cdf_schema(base: pa.Schema) -> pa.Schema:
    """`base` plus the three metadata columns a change feed carries."""
    extra = [
        pa.field("_change_type", pa.string()),
        pa.field("_commit_version", pa.int64()),
        pa.field("_commit_timestamp", pa.timestamp("us")),
    ]
    return pa.schema(list(base) + extra)


def _shape(batch: pa.RecordBatch, projection: list[str] | None, cdf: bool) -> pa.RecordBatch | None:
    """One change-feed batch normalized, optionally reduced to appends, and projected.

    ``cdf=True`` passes the CDC columns through; ``cdf=False`` keeps only `insert` changes
    and presents the table's own schema, which is what an append-only stream means. A batch
    left empty by the filter is dropped rather than yielded — a zero-row batch is not a
    change.
    """
    import pyarrow.compute as pc

    table = normalize_engine_types(pa.Table.from_batches([batch]))
    if not cdf:
        change_type = pc.cast(table.column("_change_type"), pa.string())
        table = table.filter(pc.equal(change_type, "insert")).drop(list(_CDF_META))
    if projection is not None:
        table = table.select(projection)
    if table.num_rows == 0:
        return None
    return _one_batch(table)


def _one_batch(table: pa.Table) -> pa.RecordBatch:
    """`table` as a single `RecordBatch`, keeping every row.

    `combine_chunks().to_batches()[0]` looks like it does this and does not: `to_batches`
    splits at the 32-bit offset limit, so a commit carrying more than 2 GiB of string or
    binary data comes back as several batches and taking the first **silently drops the
    rest**. `concat_batches` keeps them all, and raises a clear error rather than losing
    rows if the span genuinely cannot be one batch.
    """
    batches = table.combine_chunks().to_batches()
    return batches[0] if len(batches) == 1 else pa.concat_batches(batches)


@SOURCES.register("delta_stream")
class DeltaStreamSource:
    """A Delta table read incrementally as an unbounded stream (Spark ``readStream``).

    Each `iter_batches` pass reads the Change Data Feed for every commit after the
    last-processed version and advances the cursor — so chaining medallion layers
    (bronze → silver → gold) reads only new commits. ``change_feed=False`` (default)
    yields appended rows in the table's own schema (insert changes, metadata columns
    dropped); ``change_feed=True`` yields the full CDC stream including
    ``_change_type``/``_commit_version``/``_commit_timestamp`` (updates/deletes too).
    Requires ``delta.enableChangeDataFeed = true`` on the table.

    Checkpointable: the read position is the Delta version, so a streaming query
    resumes exactly-once after a restart.
    """

    bounded = False

    __slots__ = ("_cdf", "_cursor", "_max_versions", "_storage_options", "_table", "_table_uri")

    def __init__(
        self,
        table_uri: str,
        *,
        starting_version: int = 0,
        change_feed: bool = False,
        max_versions_per_trigger: int | None = None,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        self._table_uri = table_uri
        self._cdf = change_feed
        self._storage_options = storage_options
        self._cursor = starting_version - 1  # next read starts at starting_version
        if max_versions_per_trigger is not None and max_versions_per_trigger < 1:
            from batcher._internal.errors import PlanError

            raise PlanError(
                f"max_versions_per_trigger must be >= 1, got {max_versions_per_trigger}"
            )
        # Backpressure, the change-feed analogue of Auto Loader's `max_files_per_trigger`.
        # A window is *every commit since the last pass*, so a first run reads the table's
        # entire history and a resumed one reads however much accumulated while the query was
        # down. The rows already stream batch by batch, so memory is bounded — but the *epoch*
        # is not: the micro-batch does not end until the whole backlog drains, so the trigger
        # cadence collapses and no checkpoint is written until it finishes. Capping the
        # versions a pass admits drains the backlog across many bounded epochs instead.
        self._max_versions = max_versions_per_trigger
        self._table: Any = None

    def _delta_table(self) -> Any:
        """The table handle, refreshed in place rather than rebuilt.

        Constructing a `DeltaTable` parses the transaction log, and a streaming pass needed
        two of them — one to read the latest version and one to open the change feed — on
        *every trigger*. On a table with a long log that is the dominant per-trigger cost and
        it grows with the table's history. `update_incremental` reads only the commits added
        since the handle was built, which is exactly the question a stream is asking anyway;
        a delta-rs without it falls back to a rebuild.
        """
        if self._table is None:
            self._table = require_deltalake().DeltaTable(
                self._table_uri, storage_options=self._storage_options
            )
            return self._table
        refresh = getattr(self._table, "update_incremental", None)
        if refresh is None:  # pragma: no cover - older delta-rs
            self._table = require_deltalake().DeltaTable(
                self._table_uri, storage_options=self._storage_options
            )
        else:
            refresh()
        return self._table

    def schema(self) -> pa.Schema:
        # delta-rs returns an Arrow C-interface (arro3) schema; adapt to pyarrow.
        base = pa.schema(self._delta_table().schema().to_arrow())
        return _cdf_schema(base) if self._cdf else base

    def snapshot_position(self) -> dict:
        return {"version": self._cursor}

    def seek(self, position: dict) -> None:
        self._cursor = int(position["version"])

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"delta_stream:{self._table_uri}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [WholeSourceSplit(self)]

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        table = self._delta_table()
        latest = int(table.version())
        start = self._cursor + 1
        if start > latest:
            return  # no new commits since the last pass
        if self._max_versions is not None:
            # Leave the rest of the backlog for later passes, oldest-version-first, so a
            # large catch-up drains across bounded epochs each of which checkpoints.
            latest = min(latest, start + self._max_versions - 1)
        try:
            reader = table.load_cdf(starting_version=start, ending_version=latest)
            batches = pa.RecordBatchReader.from_stream(reader)
        except Exception as exc:
            raise BackendError(
                f"failed to read Delta change feed for {self._table_uri!r} "
                f"(is delta.enableChangeDataFeed set?): {exc}"
            ) from exc
        # Streamed batch by batch. This used to be `.read_all()` — the entire change-feed
        # window built into one table, filtered and projected whole, and only then
        # re-chunked into the batches it yielded. A window is *every commit since the last
        # pass*, so on a first run (`starting_version=0`) it is the table's whole history,
        # and on a resumed one it is however much accumulated while the query was down —
        # exactly the cases where memory is already tight. An unbounded source whose
        # micro-batch peak is the size of its backlog is not a stream.
        for batch in batches:
            shaped = _shape(batch, projection, self._cdf)
            if shaped is not None:
                yield shaped
        # Advance ONLY after every batch has been handed to the consumer.
        #
        # This used to run before the first `yield`, which turned the cursor into a promise
        # the stream had not kept: `snapshot_position()` reported the window as consumed
        # while the batches were still inside an unstarted generator. A consumer that
        # checkpoints its position — which is the entire reason `snapshot_position`/`seek`
        # exist — and then fails partway through the drain resumed at `latest + 1` and
        # never saw the rest. Silent, unrecoverable, and worse the larger the window.
        #
        # Placing it here makes the stream at-least-once: a consumer that dies mid-drain
        # replays the whole window, which is the correct failure mode for a change feed and
        # the one every checkpointing consumer is already built to handle. Abandoning the
        # generator early likewise leaves the cursor where it was, by construction.
        self._cursor = latest


@SOURCES.register("delta_cdf")
class DeltaChangeFeedSource:
    """A Delta table's Change Data Feed over a **closed** version window.

    The batch form of Databricks' ``readChangeFeed``: every row-level change between two
    versions (or two timestamps), carrying ``_change_type`` / ``_commit_version`` /
    ``_commit_timestamp`` beside the data columns, as an ordinary bounded relation. Unlike
    `DeltaStreamSource` it ends, so the result can be collected, counted, joined, and
    merged into a target — which is what an incremental ETL step does with it.

    Both ends default to open: with no `ending_version` the window runs to the table's
    latest commit, which is "everything new since my watermark". A version bound wins over
    a timestamp bound on the same end, matching delta-rs.

    Requires ``delta.enableChangeDataFeed = true`` on the table.
    """

    __slots__ = (
        "_ending_timestamp",
        "_ending_version",
        "_starting_timestamp",
        "_starting_version",
        "_storage_options",
        "_table",
        "_table_uri",
    )

    def __init__(
        self,
        table_uri: str,
        *,
        starting_version: int | None = None,
        ending_version: int | None = None,
        starting_timestamp: str | datetime | None = None,
        ending_timestamp: str | datetime | None = None,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        if starting_version is None and starting_timestamp is None:
            starting_version = 0
        self._table_uri = table_uri
        self._starting_version = starting_version
        self._ending_version = ending_version
        # Normalized here, at construction, so a malformed timestamp is reported against the
        # argument the caller wrote rather than surfacing from inside delta-rs on the first
        # batch — by which point the query is already running.
        self._starting_timestamp = _at(starting_timestamp, "starting_timestamp")
        self._ending_timestamp = _at(ending_timestamp, "ending_timestamp")
        self._storage_options = storage_options
        self._table: Any = None

    def _delta_table(self) -> Any:
        if self._table is None:
            self._table = require_deltalake().DeltaTable(
                self._table_uri, storage_options=self._storage_options
            )
        return self._table

    def schema(self) -> pa.Schema:
        # delta-rs returns an Arrow C-interface (arro3) schema; adapt to pyarrow.
        return _cdf_schema(pa.schema(self._delta_table().schema().to_arrow()))

    def row_count(self) -> int | None:
        """Unknown without reading: the log records commits, not changed-row counts."""
        return None

    def statistics(self) -> SourceStatistics | None:
        """An **estimated** row count for the window, read from the transaction log.

        Not cosmetic. A source that reports no cardinality at all is planned against the
        optimizer's unknown-rows placeholder, which is ~1e12 — so a change feed of four
        rows joined to a dimension estimated 97 *billion* rows, chose its join build side
        against that, and was then refused admission outright by the memory envelope
        ("plan does not fit ... no out-of-core path"). A wrong plan and, under any real
        memory pressure, a query that does not run.

        The estimate is the rows in the files the window's commits **added**, plus the rows
        in the files they **removed** — which is where inserts and update-postimages come
        from on one side, and deletes and update-preimages on the other. It runs high, since
        a rewritten file's untouched rows are counted on both sides, and running high is the
        safe direction for a memory estimate.

        `exact_rows=False`, so this can steer a plan but can never answer a `count()`. Any
        failure to read the log yields `None` and the old behavior: an estimate must never
        be the reason a readable table cannot be read.
        """
        try:
            return SourceStatistics(row_count=self._estimated_rows(), exact_rows=False)
        except Exception as exc:
            note_suppressed("io", "estimate Delta change-feed size", exc)
            return None

    def _estimated_rows(self) -> int:
        """Rows added plus rows removed across the window's commits, from the log alone."""
        from batcher.io.formats.lakehouse.delta._snapshot import open_snapshot

        end = open_snapshot(
            self._table_uri,
            version=self._ending_version,
            storage_options=self._storage_options,
        )
        after = end.rows_by_path()
        # A timestamp start names no version to diff against, and resolving one costs a
        # second log scan for a number that is only ever an estimate. The whole table is the
        # ceiling on what any window of it can report, which is the honest fallback.
        if self._starting_version is None or self._starting_version <= 0:
            return sum(after.values())
        before = open_snapshot(
            self._table_uri,
            version=self._starting_version - 1,
            storage_options=self._storage_options,
        ).rows_by_path()
        added = sum(rows for path, rows in after.items() if path not in before)
        removed = sum(rows for path, rows in before.items() if path not in after)
        return added + removed

    def identity(self) -> str:
        """Includes the window, so two different windows are two different sources.

        A CDF read is cached against this string like any other scan. Naming only the table
        would let the rows for versions 0-5 be served to a query asking for 6-10.
        """
        window = (
            self._starting_version,
            self._ending_version,
            self._starting_timestamp,
            self._ending_timestamp,
        )
        return f"delta_cdf:{self._table_uri}@{window}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [WholeSourceSplit(self)]

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream the window's changes, never materializing it.

        A window is an arbitrary span of history — a first read of a long-lived table is all
        of it — so the batches are yielded as delta-rs produces them rather than read into
        one table first.
        """
        table = self._delta_table()
        try:
            reader = table.load_cdf(
                starting_version=self._starting_version,
                ending_version=self._ending_version,
                starting_timestamp=self._starting_timestamp,
                ending_timestamp=self._ending_timestamp,
            )
            batches = pa.RecordBatchReader.from_stream(reader)
        except Exception as exc:
            raise BackendError(
                f"failed to read Delta change feed for {self._table_uri!r} "
                f"(is delta.enableChangeDataFeed set?): {exc}"
            ) from exc
        for batch in batches:
            shaped = _shape(batch, projection, cdf=True)
            if shaped is not None:
                yield shaped


def _at(value: str | datetime | None, argument: str) -> str | None:
    """A window bound normalized for delta-rs, or `None` for an open end."""
    return None if value is None else normalize_timestamp(value, argument=argument)
