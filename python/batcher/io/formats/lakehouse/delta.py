"""Delta Lake format — read + transactional write via delta-rs (`deltalake`).

`DeltaSource` reads a Delta table as Arrow with projection pushdown, exact row
counts from the transaction log, time travel (by version or timestamp), and a
Change-Data-Feed incremental helper. `DeltaSink` is transactional: workers write
Parquet data files to the table directory and return a manifest; the driver makes
a single atomic delta-rs commit, so a distributed write is one Delta transaction.

All `deltalake` imports are deferred — importing this module never requires the
optional dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher-engine[delta]'`` hint; a concurrent-writer conflict raises
`CommitError`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError, CommitError
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.manifest import WriteManifest, WrittenFile
from batcher.io.splits import Split, WholeSourceSplit
from batcher.plan.source_stats import SourceStatistics

__all__ = ["DeltaSink", "DeltaSource", "DeltaStreamSource"]

# CDF metadata columns delta-rs adds to a change feed.
_CDF_META = ("_change_type", "_commit_version", "_commit_timestamp")


def _normalize_view_types(table: pa.Table) -> pa.Table:
    """Cast Arrow ``string_view``/``binary_view`` columns (delta-rs ≥ 1.x) to the
    standard ``string``/``binary`` the engine and pyarrow kernels expect."""
    fields = []
    changed = False
    for field in table.schema:
        ty = field.type
        if pa.types.is_string_view(ty):
            ty, changed = pa.string(), True
        elif pa.types.is_binary_view(ty):
            ty, changed = pa.binary(), True
        fields.append(pa.field(field.name, ty))
    return table.cast(pa.schema(fields)) if changed else table


def _require_deltalake() -> Any:
    """Import and return the `deltalake` module or raise `BackendError`."""
    try:
        import deltalake
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError(
            "Delta Lake support requires delta-rs: pip install 'batcher-engine[delta]'"
        ) from exc
    return deltalake


@dataclass(frozen=True, slots=True)
class DeltaFileSplit:
    """One Delta data file, read independently on a worker.

    Carries only locators (table URI + the file's dataset path + storage options +
    version), so it pickles cheaply and the worker reads **just this file** —
    never the whole table on the driver. The table's `_delta_log` is re-read on
    the worker to recover the dataset schema (including partition columns, which
    live in the path, not the data file). Projection + predicate are pushed into
    the per-fragment read.
    """

    table_uri: str
    file_path: str
    storage_options: dict[str, str] | None
    version: int | None

    def _fragment_table(self, projection: list[str] | None, predicate: dict | None) -> pa.Table:
        from batcher.io.splits import fragment_index

        def _build() -> Any:
            dt = _require_deltalake().DeltaTable(
                self.table_uri, version=self.version, storage_options=self.storage_options
            )
            return dt.to_pyarrow_dataset()

        # Re-read the `_delta_log` + list files ONCE per worker (cached), then O(1)
        # fragment lookup — never per read (which would be O(files^2) at scale).
        key = (
            "delta",
            self.table_uri,
            self.version,
            tuple(sorted((self.storage_options or {}).items())),
        )
        dataset, index = fragment_index(key, _build)
        flt = None
        if predicate is not None:
            from batcher.io.predicate import to_pyarrow_expression

            flt = to_pyarrow_expression(predicate)
        frag = index.get(self.file_path)
        if frag is not None:
            return frag.to_table(schema=dataset.schema, columns=projection, filter=flt)
        # File compacted/removed between planning and read: empty, schema-correct.
        empty = dataset.schema.empty_table()
        return empty.select(projection) if projection is not None else empty

    def schema(self) -> pa.Schema:
        dt = _require_deltalake().DeltaTable(
            self.table_uri, version=self.version, storage_options=self.storage_options
        )
        return dt.schema().to_pyarrow()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return self._fragment_table(projection, predicate).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        yield from self._fragment_table(projection, predicate).to_batches()

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"delta:{self.table_uri}:{self.file_path}"


@SOURCES.register("delta")
class DeltaSource:
    """A Delta Lake table read as Arrow.

    Args:
        table_uri: The table root (local path or ``s3://`` / ``az://`` / ``gs://``).
        version: Optional version number for time-travel (mutually exclusive with
            `timestamp`).
        timestamp: Optional ISO-8601 timestamp for time-travel.
        storage_options: Optional cloud storage options passed to delta-rs
            (e.g. vended Unity Catalog credentials).
    """

    # Predicate pushdown: Kyber's pushed predicate → a pyarrow dataset filter,
    # giving delta-rs partition + row-group pruning at the reader.
    supports_predicate = True

    __slots__ = ("_storage_options", "_table_uri", "_timestamp", "_version")

    def __init__(
        self,
        table_uri: str,
        *,
        version: int | None = None,
        timestamp: str | None = None,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        if version is not None and timestamp is not None:
            raise BackendError("specify at most one of version/timestamp for time travel")
        self._table_uri = table_uri
        self._version = version
        self._timestamp = timestamp
        self._storage_options = storage_options

    def _table(self) -> Any:
        deltalake = _require_deltalake()
        try:
            table = deltalake.DeltaTable(
                self._table_uri,
                version=self._version,
                storage_options=self._storage_options,
            )
            if self._timestamp is not None:
                table.load_as_version(self._timestamp)
        except Exception as exc:
            raise BackendError(f"failed to open Delta table {self._table_uri!r}: {exc}") from exc
        return table

    def schema(self) -> pa.Schema:
        # delta-rs returns an Arrow C-interface (arro3) schema; adapt to pyarrow.
        return pa.schema(self._table().schema().to_arrow())

    def _add_actions(self) -> pa.Table:
        """The table's add-action stats as a pyarrow table (delta-rs returns arro3)."""
        return pa.table(self._table().get_add_actions(flatten=True))

    @staticmethod
    def _pa_filter(predicate: dict | None) -> Any:
        if predicate is None:
            return None
        from batcher.io.predicate import to_pyarrow_expression

        return to_pyarrow_expression(predicate)

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        dataset = self._table().to_pyarrow_dataset()
        return dataset.to_table(columns=projection, filter=self._pa_filter(predicate)).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        dataset = self._table().to_pyarrow_dataset()
        yield from dataset.to_batches(columns=projection, filter=self._pa_filter(predicate))

    def row_count(self) -> int | None:
        """Exact row count summed from the transaction log's add actions."""
        import pyarrow.compute as pc

        col = self._add_actions().column("num_records")
        return int(pc.sum(col).as_py() or 0)

    def statistics(self) -> SourceStatistics | None:
        """Exact row count + per-column bounds from the add-action stats, no scan."""
        from batcher.io.stats import delta_statistics

        try:
            return delta_statistics(self._add_actions())
        except Exception:
            return None

    def read_cdf(self, starting_version: int, ending_version: int | None = None) -> pa.Table:
        """Read the Change-Data-Feed between two versions as an Arrow table.

        Requires the table to have ``delta.enableChangeDataFeed = true``. The
        returned table carries the CDF metadata columns (``_change_type``,
        ``_commit_version``, ``_commit_timestamp``).
        """
        table = self._table()
        try:
            reader = table.load_cdf(
                starting_version=starting_version,
                ending_version=ending_version,
            )
            # delta-rs returns an Arrow C-stream (arro3) reader; adapt it to pyarrow.
            return pa.RecordBatchReader.from_stream(reader).read_all()
        except Exception as exc:
            raise BackendError(f"failed to read Delta CDF for {self._table_uri!r}: {exc}") from exc

    def identity(self) -> str:
        ref = self._version if self._version is not None else (self._timestamp or "latest")
        return f"delta:{self._table_uri}@{ref}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        """One split per Delta data file — each worker reads only its files, so a
        table larger than any single node never materializes on the driver.

        Falls back to a whole-source split only if fragment enumeration fails.
        """
        try:
            dataset = self._table().to_pyarrow_dataset()
            paths = [frag.path for frag in dataset.get_fragments()]
        except Exception:
            return [WholeSourceSplit(self)]
        if not paths:
            return [WholeSourceSplit(self)]
        return [
            DeltaFileSplit(self._table_uri, path, self._storage_options, self._version)
            for path in paths
        ]


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

    __slots__ = ("_cdf", "_cursor", "_storage_options", "_table_uri")

    def __init__(
        self,
        table_uri: str,
        *,
        starting_version: int = 0,
        change_feed: bool = False,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        self._table_uri = table_uri
        self._cdf = change_feed
        self._storage_options = storage_options
        self._cursor = starting_version - 1  # next read starts at starting_version

    def _delta_table(self) -> Any:
        return _require_deltalake().DeltaTable(
            self._table_uri, storage_options=self._storage_options
        )

    def schema(self) -> pa.Schema:
        # delta-rs returns an Arrow C-interface (arro3) schema; adapt to pyarrow.
        base = pa.schema(self._delta_table().schema().to_arrow())
        if not self._cdf:
            return base
        extra = [
            pa.field("_change_type", pa.string()),
            pa.field("_commit_version", pa.int64()),
            pa.field("_commit_timestamp", pa.timestamp("us")),
        ]
        return pa.schema(list(base) + extra)

    def _latest_version(self) -> int:
        return int(self._delta_table().version())

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
        latest = self._latest_version()
        start = self._cursor + 1
        if start > latest:
            return  # no new commits since the last pass
        try:
            reader = self._delta_table().load_cdf(starting_version=start, ending_version=latest)
            table = _normalize_view_types(pa.RecordBatchReader.from_stream(reader).read_all())
        except Exception as exc:
            raise BackendError(
                f"failed to read Delta change feed for {self._table_uri!r} "
                f"(is delta.enableChangeDataFeed set?): {exc}"
            ) from exc
        self._cursor = latest
        if not self._cdf:
            # Append mode: keep insert changes, present the table's own schema.
            import pyarrow.compute as pc

            change_type = pc.cast(table.column("_change_type"), pa.string())
            table = table.filter(pc.equal(change_type, "insert")).drop(list(_CDF_META))
        if table.num_rows == 0:
            return
        out = table.select(projection) if projection is not None else table
        yield from out.to_batches()


@SINKS.register("delta")
class DeltaSink:
    """Transactional Delta Lake writer.

    Workers stage Parquet data files via `write_partitioned`; the driver performs
    one atomic commit via `commit`, so a distributed write is a single Delta
    transaction. `mode` selects ``"append"`` or ``"overwrite"``; `partition_by`
    sets the table partition columns; `merge_predicate` switches `commit` to a
    `DeltaTable.merge` upsert.

    Args:
        mode: ``"append"`` (default) or ``"overwrite"``.
        partition_by: Optional partition columns.
        merge_predicate: Optional SQL predicate; when set, `commit` performs an
            upsert (``when matched update / when not matched insert``) instead of
            an append/overwrite.
        storage_options: Optional cloud storage options passed to delta-rs.
    """

    __slots__ = ("_merge_predicate", "_mode", "_partition_by", "_pending", "_storage_options")

    def __init__(
        self,
        *,
        mode: str = "append",
        partition_by: list[str] | None = None,
        merge_predicate: str | None = None,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        if mode not in ("append", "overwrite"):
            raise BackendError(f"unsupported Delta write mode {mode!r}; use append/overwrite")
        self._mode = mode
        self._partition_by = partition_by
        self._merge_predicate = merge_predicate
        self._storage_options = storage_options
        # Staged tables held for the driver-side commit (single-process path).
        self._pending: list[pa.Table] = []

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Stage `table` for the transactional commit (no standalone file)."""
        self._pending.append(table)
        return WrittenFile(path=path, rows=table.num_rows, bytes=0)

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,  # noqa: ARG002
        *,
        partition_by: list[str] | None = None,
        file_index: int = 0,  # noqa: ARG002
    ) -> list[WrittenFile]:
        """Stage one shard's `table` for the driver-side atomic commit.

        delta-rs owns the physical Parquet layout (including partitioning) at
        commit time, so a shard is staged in-memory rather than written here; the
        returned manifest carries row counts for the driver to roll up.
        """
        if partition_by is not None:
            self._partition_by = partition_by
        self._pending.append(table)
        return [WrittenFile(path="<staged>", rows=table.num_rows, bytes=0)]

    def commit(self, manifest: WriteManifest, path: str) -> None:  # noqa: ARG002
        """Atomically commit all staged data to the Delta table at `path`.

        Performs one delta-rs transaction (append/overwrite, or a merge upsert if
        `merge_predicate` was set). Raises `CommitError` on a concurrent-writer
        conflict.
        """
        deltalake = _require_deltalake()
        if not self._pending:
            return
        data = pa.concat_tables(self._pending)
        try:
            if self._merge_predicate is not None:
                self._commit_merge(deltalake, data, path)
            else:
                deltalake.write_deltalake(
                    path,
                    data,
                    mode=self._mode,
                    partition_by=self._partition_by,
                    storage_options=self._storage_options,
                )
        except Exception as exc:
            if _is_conflict(exc):
                raise CommitError(
                    f"Delta commit to {path!r} conflicted with a concurrent writer: {exc}"
                ) from exc
            raise CommitError(f"Delta commit to {path!r} failed: {exc}") from exc
        finally:
            self._pending.clear()

    def _commit_merge(self, deltalake: Any, data: pa.Table, path: str) -> None:
        """Upsert `data` into the table via `DeltaTable.merge`."""
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


def _is_conflict(exc: Exception) -> bool:
    """Heuristically detect a delta-rs concurrency-conflict error."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "concurr" in name or "concurr" in text or "commitfailed" in name
