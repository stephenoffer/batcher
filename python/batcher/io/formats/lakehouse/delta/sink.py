"""Writing a Delta Lake table: workers write final data files, the driver commits metadata.

`DeltaSink` is the write half of the file-skipping story. Each worker writes its shard as
a real Delta data file and records that file's column bounds; the driver then commits only
the resulting `AddAction`s (`_commit`). No data crosses the driver, and the statistics the
write leaves in the log are exactly what the next read prunes against.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError, CommitError
from batcher.io.base import FileSink
from batcher.io.formats.base import SINKS
from batcher.io.formats.lakehouse.delta._commit import (
    already_committed,
    collect_file_stats,
    commit_add_actions,
    merge_file_stats,
)
from batcher.io.formats.lakehouse.delta._snapshot import require_deltalake
from batcher.io.formats.structured.parquet.sink import ParquetSink
from batcher.io.manifest import WriteManifest, WrittenFile

__all__ = ["DeltaSink"]

#: Stands in for a NaN partition value inside a dict key. A bare NaN is not equal to itself,
#: so it cannot be looked up; this is what lets the one NaN group `_hive_partition` produces
#: be found again by the file that was written from it.
_NAN_KEY = "__BATCHER_NAN_PARTITION__"


@SINKS.register("delta")
class DeltaSink:
    """Transactional Delta writer whose commit registers files rather than rewriting them.

    Each worker writes its shard as a **final** Delta data file, straight into the
    table's own layout (Hive directories when partitioned), and collects that file's
    column statistics from the data it is already holding. The driver then commits
    nothing but the resulting `AddAction`s. The bytes move exactly once, worker →
    storage, and the driver's cost is one log write regardless of how much was written.

    This is what makes a distributed Delta write actually distributed. The previous
    design staged each shard and then streamed **every one of them back through the
    driver** into ``write_deltalake``, re-encoding the entire result in a single process:
    bounded driver memory, but unbounded driver work, with the whole dataset crossing the
    one machine the design exists to avoid.

    The statistics written here are exactly the ones `DeltaSource` prunes against on the
    next read (`io.stats.file_skipping`) — so a table this sink writes is a table that
    can be read with file skipping. Writing and skipping are two ends of one mechanism.

    Args:
        mode: ``"append"`` (default) or ``"overwrite"``.
        partition_by: Optional partition columns.
        merge_predicate: Optional SQL predicate; when set, `commit` performs an
            upsert (``when matched update / when not matched insert``) instead of an
            append/overwrite. A merge rewrites the files it matches, so it genuinely
            reads its change set back — bounded, since a change set is not a bulk load.
        replace_where: Optional predicate IR scoping an overwrite to the rows it matches
            (Delta's ``replaceWhere`` — the backfill). When the predicate is purely over
            partition columns, the commit retires just those partitions from the log and
            adds the new files, so replacing one day of a table rewrites one day. This used
            to be a copy-on-write of the *whole table*: read everything, filter out the
            replaced range, union, overwrite — which turned a one-day backfill into a
            full-table rewrite.
        merge_schema: Evolve the table to accept columns this write has and the table does
            not. Off by default: an unexpected new column is far more often a bug in the
            pipeline than an intended evolution, and the commit path cannot simply drop it —
            doing so writes the value into the data file where it stays invisible, and
            resurfaces as wrong data if the column is later added for real.
        storage_options: Optional cloud storage options passed to delta-rs.
        app_id: Optional application id for Delta `txn` idempotency. Together with
            `txn_version` the commit is recorded as an application transaction, and a
            replay of the same version commits nothing — the exactly-once contract a
            restarted streaming query relies on.
        txn_version: Optional monotonically-increasing version for `app_id`.
        table_properties: Delta table properties — Spark's ``TBLPROPERTIES``, e.g.
            ``{"delta.enableChangeDataFeed": "true"}``. Set on the table's ``metaData``
            when this write creates it, and applied as an alter when it already exists.
            Several of the protocol's most useful behaviors are reachable only this way:
            without it a table Batcher created could never have its change data feed
            turned on, so `bt.read.read_change_feed` could not read a Batcher-written
            table at all.
    """

    __slots__ = (
        "_app_id",
        "_merge_predicate",
        "_merge_schema",
        "_mode",
        "_partition_by",
        "_replace_where",
        "_storage_options",
        "_table_parts",
        "_table_properties",
        "_token",
        "_txn_version",
    )

    def __init__(
        self,
        *,
        mode: str = "append",
        partition_by: list[str] | None = None,
        merge_predicate: str | None = None,
        replace_where: dict | None = None,
        merge_schema: bool = False,
        storage_options: dict[str, str] | None = None,
        app_id: str | None = None,
        txn_version: int | None = None,
        table_properties: dict[str, str] | None = None,
    ) -> None:
        if mode not in ("append", "overwrite"):
            raise BackendError(f"unsupported Delta write mode {mode!r}; use append/overwrite")
        self._mode = mode
        self._partition_by = partition_by
        self._table_parts: dict[str, list[str]] = {}
        self._merge_predicate = merge_predicate
        self._replace_where = replace_where
        self._merge_schema = merge_schema
        self._storage_options = storage_options
        self._app_id = app_id
        self._txn_version = txn_version
        self._table_properties = table_properties
        self._token = uuid.uuid4().hex

    @property
    def _app_txn(self) -> tuple[str, int] | None:
        """The ``(app_id, version)`` this write records, if it is an idempotent one."""
        if self._app_id is None or self._txn_version is None:
            return None
        return (self._app_id, self._txn_version)

    def _data_sink(self) -> ParquetSink:
        """The Parquet writer that lays down this write's data files.

        A Delta data file *is* a Parquet file in the table directory, so `ParquetSink`
        already writes it correctly — including the Hive layout and Delta's convention
        that a partition column lives in the path, not the data.
        """
        return ParquetSink(file_token=self._token)

    def write(self, table: pa.Table, path: str, *, resume: bool = False) -> WrittenFile:  # noqa: ARG002
        """Write `table` as one final Delta data file under the table root.

        `resume` is accepted for the shared `FileSink.write` signature but ignored: a
        Delta write is made idempotent by its `txn` action, not by skipping part files.
        """
        written = self._data_sink().write_partitioned(table, path, file_index=0)[0]
        return replace(written, stats=collect_file_stats(table))

    def requires_partitioned_write(self, path: str) -> bool:
        """Whether a write to `path` must fan out by partition key rather than write one file.

        True when the target table is partitioned — by this write's `partition_by` or, for an
        existing table, by its own metadata. `write` returns a single `WrittenFile` and so
        cannot represent a shard that spans several partitions; routing such a write here is
        what keeps the partition values from being dropped.

        Args:
            path: The table root about to be written.

        Returns:
            True when the partitioned write path is required.
        """
        return bool(self._partition_columns(path))

    def write_stream(
        self,
        batches: Any,
        path: str,
        *,
        schema: pa.Schema | None = None,
        resume: bool = False,  # noqa: ARG002
    ) -> WrittenFile:
        """Stream `batches` into one final Delta data file, holding one batch at a time.

        The single-node streaming write: a breaker-free read→transform→write never
        materializes its result. Statistics are folded batch by batch (`merge_file_stats`
        is associative), so the file is still fully indexed for the next read without
        ever being held whole.
        """
        stats: dict[str, Any] = {}

        def _tracked() -> Iterator[pa.RecordBatch]:
            nonlocal stats
            for batch in batches:
                if batch.num_rows:
                    batch_stats = collect_file_stats(pa.Table.from_batches([batch]))
                    stats = merge_file_stats(stats, batch_stats)
                yield batch

        writer = self._data_sink()
        name = f"{path.rstrip('/')}/part-{0:05d}{writer.suffix}"
        written = writer.write_stream(_tracked(), name, schema=schema)
        return replace(written, stats=stats)

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,
        file_index: int = 0,
    ) -> list[WrittenFile]:
        """Write one shard as final Delta data file(s), on the worker that produced it.

        The shard never travels to the driver — only the returned locators do, carrying
        the statistics and partition values the commit registers.

        When the caller does not say how to partition, the **table's** partitioning is used.
        Delta keeps a partition value in the file's directory name rather than in the file, so
        writing an unpartitioned file into a partitioned table does not merely lay it out
        differently — it loses the value. Appending `region="us"` to a table partitioned on
        `region` without restating `partition_by` wrote the file outside every `region=`
        directory, and the row read back with ``region = NULL``: present, and silently
        stripped of the column the table is organised by. Restating the partitioning on every
        append is not something a caller should have to remember to keep their data.
        """
        if partition_by is not None:
            self._partition_by = partition_by
        parts = self._partition_columns(path) or None
        written = self._data_sink().write_partitioned(
            table, path, partition_by=parts, file_index=file_index
        )
        by_partition = self._rows_per_partition(table, parts)
        return [replace(w, stats=self._stats_for(table, w, parts, by_partition)) for w in written]

    @staticmethod
    def _rows_per_partition(
        table: pa.Table, partition_by: list[str] | None
    ) -> dict[tuple, pa.Table] | None:
        """Each partition's rows, keyed by its values — computed in **one** pass over `table`.

        Delta indexes every data file it writes, so a partitioned shard needs statistics per
        *partition*, not per shard. Selecting each partition with its own mask made that
        `O(partitions x rows)`: every distinct key rebuilt a full-table comparison and then
        filtered the whole table again. Measured against the same write without Delta's
        statistics, on 2M rows: 1.55x the plain Parquet time at 8 partitions, 2.66x at 97,
        4.45x at 400 and **6.43x at 1,000** — the overhead grows with the partition count
        while the data does not.

        `FileSink._hive_partition` answers the same question by sorting once and slicing the
        contiguous runs, which is `O(n log n)` regardless of how many partitions there are,
        and it is the very routine the writer used to lay the files out — so the rows counted
        here are by construction the rows written there, rather than a second derivation that
        could disagree with it. It also gets the two cases a mask gets wrong for free: NULL
        and NaN group as single keys.

        Args:
            table: The shard being written.
            partition_by: The partition columns, or None for an unpartitioned write.

        Returns:
            Partition key values (in `partition_by` order) to that partition's rows with the
            partition columns dropped, or None when the write is not partitioned.
        """
        if not partition_by:
            return None
        return {
            DeltaSink._partition_key(value for _, value in key_values): sub
            for key_values, sub in FileSink._hive_partition(table, list(partition_by))
        }

    @staticmethod
    def _partition_key(values: Any) -> tuple:
        """A partition's values as a dict key, with NaN made equal to itself.

        `NaN != NaN`, so a float NaN partition value cannot be looked up in a dict that a
        different NaN object was stored under — the lookup misses and the file gets the
        wrong statistics. `_hive_partition` deliberately groups NaN as one key (as
        `group_by` does), so the key that identifies that group has to as well.
        """
        return tuple(_NAN_KEY if isinstance(v, float) and v != v else v for v in values)

    @staticmethod
    def _stats_for(
        table: pa.Table,
        written: WrittenFile,
        partition_by: list[str] | None,
        by_partition: dict[tuple, pa.Table] | None,
    ) -> dict[str, Any]:
        """Statistics for one written file, over just the rows that landed in it.

        A partitioned shard fans out into one file per partition key, so each file's
        stats must describe *its* partition's rows, not the whole shard's — attributing
        the shard's bounds to every file would claim ranges those files do not contain
        and defeat the skipping it exists to enable. The partition columns themselves are
        excluded: they are constant in the file and recorded as partition values, which
        is where the reader looks for them.

        A partition that fans out into several *chunk* files (a row cap) gives each chunk
        the whole partition's bounds. That over-claims, which costs a reader some skipping
        it could have had; under-claiming would cost it rows, so this is the safe side of
        the approximation and it is what the mask-per-file form did too.
        """
        if not partition_by or not written.partition_values:
            return collect_file_stats(table)
        key = DeltaSink._partition_key(
            written.partition_values.get(column) for column in partition_by
        )
        rows = (by_partition or {}).get(key)
        if rows is None:
            # A key the one-pass grouping did not produce should be impossible — both come
            # from `_hive_partition` over this same table. Falling back to the shard's own
            # statistics keeps the file *indexed* (over-claimed, so no row is ever pruned
            # away) rather than committing a file with none.
            return collect_file_stats(table.drop_columns(list(written.partition_values)))
        return collect_file_stats(rows)

    def is_committed(self, path: str) -> bool:
        """Whether this write's `txn` transaction is already recorded in the table's log.

        A streaming query calls this *before* writing anything: a micro-batch replayed
        after a crash must not even produce a data file, let alone a second commit. With
        no `app_id`/`txn_version` configured there is no transaction to look for, so this
        is always False and the write proceeds.

        Args:
            path: The table root.

        Returns:
            True when this exact transaction was already committed.
        """
        return already_committed(path, self._app_txn, self._storage_options)

    def commit(self, manifest: WriteManifest, path: str) -> None:
        """Register the written data files as one atomic Delta transaction.

        Metadata only: the files are already in place, so this writes a single commit
        carrying their paths, sizes, partition values, and statistics. Raises
        `CommitError` on a concurrent-writer conflict. When an `app_id`/`txn_version`
        pair was configured and that transaction is already in the log, the commit is
        skipped and the replayed micro-batch does not duplicate its rows.
        """
        if self._merge_predicate is not None:
            self._commit_merge(manifest, path)
            return
        if already_committed(path, self._app_txn, self._storage_options):
            return
        mode, filters = self._overwrite_scope(path)
        commit_add_actions(
            manifest,
            path,
            mode=mode,
            partition_by=self._partition_by,
            partition_filters=filters,
            merge_schema=self._merge_schema,
            storage_options=self._storage_options,
            app_txn=self._app_txn,
            table_properties=self._table_properties,
        )

    def _overwrite_scope(self, path: str) -> tuple[str, list[tuple[str, str, str]] | None]:
        """The commit's mode and, for a `replace_where`, the partitions it is scoped to.

        A `replace_where` over partition columns becomes a partition-scoped overwrite: the
        commit retires exactly those partitions and adds the new files, moving no data. A
        predicate delta-rs cannot express as partition filters is refused rather than
        silently widened to a full overwrite — replacing a day and getting the table
        replaced instead is the worst possible failure here.

        The partition columns come from the *table* when this write does not restate them,
        which is the ordinary case: a backfill targets a table that was partitioned when it
        was created, and `partition_by=` on an overwrite is redundant. Checking only this
        call's argument refused every such write — `write(mode="overwrite",
        replace_where=col("region") == "us")` on a table whose log says
        ``partition_columns: ['region']`` raised the error below, which then told the caller
        to "partition the table on the columns you backfill by" when it already was.
        """
        if self._replace_where is None:
            return self._mode, None

        from batcher.io.formats.lakehouse.delta._predicate import to_partition_filters

        filters = to_partition_filters(self._replace_where, self._partition_columns(path))
        if filters is None:
            raise CommitError(
                "write(replace_where=...) on a Delta table needs a predicate over the "
                "table's partition columns (an AND of `partition_col == value`), so the "
                "overwrite can be scoped to those partitions. Partition the table on the "
                "columns you backfill by, or overwrite the whole table explicitly."
            )
        return "overwrite", filters

    def _partition_columns(self, path: str) -> list[str]:
        """This write's partition columns, else the existing table's.

        An unreadable or not-yet-created table falls back to what the caller passed, so a
        first write behaves as before and the error below still explains itself.

        Args:
            path: The table root.

        Returns:
            The partition column names to scope a `replace_where` against.
        """
        if self._partition_by:
            return list(self._partition_by)
        if path in self._table_parts:
            return self._table_parts[path]
        try:
            import deltalake

            found = list(
                deltalake.DeltaTable(path, storage_options=self._storage_options)
                .metadata()
                .partition_columns
            )
        except Exception:  # not a Delta table yet, or the log cannot be read here
            found = []
        # Memoized per path: a distributed write asks once per shard, and every shard reads
        # the same log and must reach the same answer.
        self._table_parts[path] = found
        return found

    def _commit_merge(self, manifest: WriteManifest, path: str) -> None:
        """Upsert the written files' rows into the table via `DeltaTable.merge`.

        Unlike an append, a merge rewrites the data files it matches, so it has to read
        the change set back. That set is a delta rather than a bulk load, so reading it
        is bounded — and the written files are removed afterwards, since it is the merge,
        not this write, that lands the rows in the table.
        """
        from batcher.io.filesystem import resolve_filesystem

        deltalake = require_deltalake()
        paths = [f.path for f in manifest.files if f.rows]
        if not paths:
            return
        fs = resolve_filesystem(path)
        # Each handle is closed as its file is read. `pq.read_table(fs.open(p))` inside a
        # comprehension never closes any of them: the change set is one file per shard, so
        # a wide write leaked a descriptor per shard on the driver — the single process
        # least able to absorb it, and the one whose fd limit fails the whole commit.
        data = pa.concat_tables([_read_change_file(fs, p) for p in paths])
        try:
            table = deltalake.DeltaTable(path, storage_options=self._storage_options)
            (
                table.merge(
                    source=data,
                    predicate=self._merge_predicate,
                    source_alias="source",
                    target_alias="target",
                )
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute()
            )
        except Exception as exc:
            raise CommitError(f"Delta merge into {path!r} failed: {exc}") from exc
        for p in paths:  # the merge landed the rows; these files were only the change set
            with contextlib.suppress(OSError, ValueError, NotImplementedError):
                fs.remove(p)


def _read_change_file(fs: Any, path: str) -> pa.Table:
    """One merge change-set file, read with its handle closed afterwards."""
    import pyarrow.parquet as pq

    with fs.open(path) as fh:
        return pq.read_table(fh)
