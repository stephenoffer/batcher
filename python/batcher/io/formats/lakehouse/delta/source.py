"""Reading a Delta Lake table: log-driven file skipping, time travel, and CDF.

`DeltaSource` reads a table as Arrow with projection and predicate pushdown, exact row
counts and column bounds straight from the transaction log, and time travel by version
or timestamp. Every metadata question is answered from one resolved `DeltaSnapshot`
(`_snapshot`) rather than by re-opening the table, and a selective predicate eliminates
whole data files *before* they are opened — see `io.stats.file_skipping`.

`DeltaFileSplit` is the worker-side unit: one data file, read independently, pinned to
the version the driver planned against.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.splits import Split, WholeSourceSplit
from batcher.plan.source_stats import SourceStatistics

__all__ = ["DeltaFileSplit", "DeltaSource", "iter_fragment", "read_fragment"]


@dataclass(frozen=True, slots=True)
class DeltaFileSplit:
    """One Delta data file, read independently on a worker.

    Carries only locators (table URI + the file's dataset path + storage options +
    version), so it pickles cheaply and the worker reads **just this file** —
    never the whole table on the driver. The version is the one the driver *resolved*
    at plan time, so every worker reads the same immutable snapshot (and shares one
    cached `_delta_log` replay per process, keyed by that version). Projection +
    predicate are pushed into the per-fragment read.

    `rows` is the file's record count, taken from the add-action the driver already
    read. It costs nothing here and lets the distributed planner bin-pack splits by
    real size instead of treating every file as weightless.
    """

    table_uri: str
    file_path: str
    # `repr=False`: these are the cloud credentials (``aws_secret_access_key``, a SAS
    # token, a vended Unity Catalog credential). A split's generated `repr` is rendered
    # into worker task logs and into every traceback that crosses the FFI or Ray boundary,
    # so leaving it on publishes the secret to wherever those are collected. The locators
    # above stay visible — redaction must not cost debuggability.
    storage_options: dict[str, str] | None = field(repr=False)
    version: int | None
    rows: int | None = None

    def _snapshot(self) -> Any:
        from batcher.io.formats.lakehouse.delta._snapshot import open_snapshot

        return open_snapshot(
            self.table_uri, version=self.version, storage_options=self.storage_options
        )

    def _dataset(self) -> tuple[Any, dict[str, Any]]:
        """The worker's cached ``(dataset, {path: fragment})`` for this table version.

        Built from the add-action manifest, **not** from delta-rs's
        ``to_pyarrow_dataset()``. That call refuses outright on any table whose protocol
        declares the ``deletionVectors`` reader feature — which is default-on for new Delta
        tables — so a split read of one raised `DeltaProtocolError` at the worker and a
        distributed read of it was impossible, whether or not the table had a single row
        deleted. Reading the files ourselves and applying the vectors (below) is what makes
        those tables readable at all.
        """
        # Replay the `_delta_log` + list files ONCE per worker, then O(1) fragment lookup —
        # never per read (which would be O(files^2) at scale). The snapshot is itself cached
        # per (uri, pinned version), so the index it memoizes is shared by every split of
        # this table on this worker.
        return self._snapshot().dataset_index(None)

    def _fragment_table(self, projection: list[str] | None, predicate: dict | None) -> pa.Table:
        dataset, index = self._dataset()
        frag = index.get(self.file_path)
        if frag is None:
            # File compacted/removed between planning and read: empty, schema-correct.
            empty = dataset.schema.empty_table()
            return empty.select(projection) if projection is not None else empty
        mask = self._snapshot().deletion_masks().get(self.file_path)
        return read_fragment(frag, dataset.schema, projection, predicate, mask)

    def schema(self) -> pa.Schema:
        from batcher.io.formats.lakehouse.delta._snapshot import open_snapshot

        return open_snapshot(
            self.table_uri, version=self.version, storage_options=self.storage_options
        ).schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return self._fragment_table(projection, predicate).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        """Stream this data file, never materializing it.

        This used to be ``self._fragment_table(...).to_batches()`` — a whole-fragment read
        into one table, then a re-chunk of something already in memory. It streamed in
        signature only: peak memory was the decoded size of the entire data file, so a
        `read → transform → write` over a table of 1 GB fragments held a gigabyte per worker
        for no reason, and a fragment larger than the worker's memory could not be read at
        all. `iter_fragment` yields the same batches in the same order without ever holding
        more than one.
        """
        dataset, index = self._dataset()
        frag = index.get(self.file_path)
        if frag is None:
            # File compacted/removed between planning and read: empty, schema-correct.
            empty = dataset.schema.empty_table()
            yield from (empty.select(projection) if projection is not None else empty).to_batches()
            return
        mask = self._snapshot().deletion_masks().get(self.file_path)
        yield from iter_fragment(frag, dataset.schema, projection, predicate, mask)

    def row_count(self) -> int | None:
        return self.rows

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

    __slots__ = ("_snap", "_storage_options", "_table_uri", "_timestamp", "_version")

    def __init__(
        self,
        table_uri: str,
        *,
        version: int | None = None,
        timestamp: str | datetime | None = None,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        if version is not None and timestamp is not None:
            raise BackendError("specify at most one of version/timestamp for time travel")
        self._table_uri = table_uri
        self._version = version
        self._timestamp = timestamp
        self._storage_options = storage_options
        self._snap: Any = None

    def _snapshot(self) -> Any:
        """The table's resolved snapshot — one `_delta_log` replay, memoized.

        Every metadata question (schema, row count, statistics, split planning, the
        deletion-vector probe) reads this one resolved state instead of re-opening the
        table, and the version it resolves is what pins the workers' caches.
        """
        from batcher.io.formats.lakehouse.delta._snapshot import open_snapshot

        if self._snap is None:
            self._snap = open_snapshot(
                self._table_uri,
                version=self._version,
                timestamp=self._timestamp,
                storage_options=self._storage_options,
            )
        return self._snap

    def _table(self) -> Any:
        return self._snapshot().table

    def schema(self) -> pa.Schema:
        return self._snapshot().schema()

    def _add_actions(self) -> pa.Table:
        """The table's add-action stats as a pyarrow table (delta-rs returns arro3)."""
        return self._snapshot().add_actions()

    def _pa_filter(self, predicate: dict | None) -> Any:
        """The pushed filter, typed against the snapshot's schema. `None` in, `None` out.

        Kept where the four wrappers like it were deleted, because this one does something:
        it reads `self._snapshot().schema()`. The `predicate is None` guard is `to_pyarrow_
        expression`'s job now.
        """
        from batcher.io.predicate import to_pyarrow_expression

        # Pass the table schema so a temporal literal is typed to its column — without it,
        # a filter on a timezone-aware timestamp column raises rather than prunes.
        return to_pyarrow_expression(predicate, self._snapshot().schema())

    def _has_deletion_vectors(self) -> bool:
        """Whether the table uses deletion vectors (delta-rs's pyarrow reader raises)."""
        return self._snapshot().has_deletion_vectors()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection, predicate))

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        snapshot = self._snapshot()
        masks = snapshot.deletion_masks()
        dataset, index = snapshot.dataset_index(predicate)
        if not masks:
            # No file carries a deletion vector: the whole scan keeps full pushdown.
            yield from dataset.to_batches(columns=projection, filter=self._pa_filter(predicate))
            return
        # Some files do. Each is read through `iter_fragment`, which applies that file's
        # vector before the predicate — and leaves the unaffected files on the fast path.
        # Streamed per fragment: the table-building variant held a whole data file in memory
        # at a time, which is the one thing a scan called `iter_batches` must not do.
        for path, fragment in index.items():
            yield from iter_fragment(
                fragment, dataset.schema, projection, predicate, masks.get(path)
            )

    def row_count(self) -> int | None:
        """Exact row count from the log: the files' records, less what the vectors delete.

        A deletion vector leaves its rows physically in the file, so the add actions
        *overcount* a table that has one. Subtracting the vectors' deleted rows makes the
        count exact again — it used to return `None` for any DV table, leaving the
        estimator to guess at a number the log states outright.

        Returns None when the log does not record `num_records` — a writer is not obliged
        to, and `_snapshot` already guards the same column for its zone maps. Reading it
        unguarded raised `KeyError` out of what is a best-effort statistic: the planner
        asks for a count it can do without, so an absent column must degrade to "unknown"
        rather than fail the query.
        """
        import pyarrow.compute as pc

        actions = self._add_actions()
        if "num_records" not in actions.column_names:
            return None
        snapshot = self._snapshot()
        physical = int(pc.sum(actions.column("num_records")).as_py() or 0)
        return physical - snapshot.deleted_rows()

    def statistics(self) -> SourceStatistics | None:
        """Exact row count + per-column bounds from the add-action stats, no scan.

        On a table with deletion vectors the row count stays exact (the vectors say how many
        rows they removed), but the column bounds must be *demoted*: a file's recorded min
        may belong to a row a vector has since deleted, so the bound is a superset of what is
        actually there. That is still perfectly sound for **pruning** — a wider range only
        keeps files it need not — but it can no longer answer a `min()`/`max()` exactly.
        """
        from batcher.io.stats import manifest_statistics

        try:
            stats = manifest_statistics(self._add_actions())
        except Exception:
            return None
        if stats is None:
            return None
        deleted = self._snapshot().deleted_rows()
        if not deleted:
            return stats
        return _demote_bounds(stats, rows=(stats.row_count or 0) - deleted)

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
        """What makes this source *this* source — including **which version** it reads.

        The identity keys the session's statistics cache, and some terminals (`count()`,
        `is_empty()`, `min()`/`max()`) are answered from those statistics without executing.
        So an identity that does not name the version lets a cached row count outlive the
        table it described: after an append, ``count()`` kept returning 3 while ``collect()``
        returned 5 — the same query, two different answers.

        Naming the resolved version fixes that at the root, and fixes it against *any*
        writer. Invalidating on our own commits only ever covered writes Batcher made; a
        table appended to by Spark, by a streaming job, or by another process went stale with
        nothing to notice. A new version is simply a new identity, so there is no stale entry
        to serve.

        Resolving ``latest`` costs one incremental log catch-up (`_snapshot`), not a full
        replay. A table that cannot be opened falls back to the unresolved form rather than
        raising — an identity is metadata about a source, not a read of it.
        """
        if self._version is not None:
            ref: Any = self._version
        elif self._timestamp is not None:
            ref = self._timestamp
        else:
            try:
                ref = self._snapshot().version
            except Exception:
                ref = "latest"
        return f"delta:{self._table_uri}@{ref}"

    def splits(
        self,
        target_size: int | None = None,  # noqa: ARG002 — protocol signature; Delta splits per file
        predicate: dict | None = None,
    ) -> list[Split]:
        """One split per *surviving* Delta data file, after transaction-log skipping.

        `predicate` is the filter Kyber pushed to this scan. Files whose add-action
        bounds prove they hold no matching row are eliminated **here, at plan time** —
        so they never become a split, are never shipped to a worker, and their footers
        are never opened. On a partitioned or naturally-clustered table this is the
        difference between one task per file in the table and one task per file that
        can actually contribute, which is where a selective distributed read gets its
        order of magnitude.

        Each split pins the *resolved* version, so every worker reads exactly the
        snapshot the driver planned against (and reuses one cached log replay), even if
        the table is committed to concurrently.

        Falls back to a whole-source split only if enumeration fails.
        """
        try:
            snapshot = self._snapshot()
            paths = snapshot.surviving_paths(predicate)
            if paths is None:
                paths = snapshot.file_paths()
            version = snapshot.version
            rows_by_path = snapshot.rows_by_path()
        except Exception:
            return [WholeSourceSplit(self)]
        if not paths:
            # Provably empty: keep one split so the scan still yields a typed, zero-row
            # result rather than an absent one.
            return [WholeSourceSplit(self)]
        return [
            DeltaFileSplit(
                self._table_uri, path, self._storage_options, version, rows_by_path.get(path)
            )
            for path in paths
        ]


def read_fragment(
    fragment: Any,
    schema: pa.Schema,
    projection: list[str] | None,
    predicate: dict | None,
    mask: Any | None,
) -> pa.Table:
    """Read one Delta data file, applying its deletion vector before any predicate.

    The ordering is the whole correctness argument. A deletion vector is indexed by the
    file's **physical row positions**, so it is only meaningful against the rows exactly as
    written. Push a predicate into the Parquet read and it drops rows first, sliding every
    position — the mask would then delete the wrong rows, silently. So a file that carries a
    vector is read *unfiltered*, masked, and only then filtered.

    A file with no vector — the overwhelming majority, since a vector attaches only to a
    file a delete actually touched — keeps the full pushdown, and pays nothing for the
    existence of deletion vectors elsewhere in the table.
    """
    flt = None
    if predicate is not None:
        from batcher.io.predicate import to_pyarrow_expression

        # Type temporal literals to the file's own columns (tz-aware timestamp columns
        # cannot be compared against a tz-naive literal).
        flt = to_pyarrow_expression(predicate, schema)

    if mask is None:
        return fragment.to_table(schema=schema, columns=projection, filter=flt)

    # Masked path: no filter in the scan, or the mask no longer lines up with the rows.
    table = fragment.to_table(schema=schema, columns=projection)
    if len(mask) != table.num_rows:
        # The vector does not describe this file's rows. Refusing beats guessing: applying a
        # misaligned mask would delete arbitrary rows and report success.
        raise BackendError(
            f"Delta deletion vector for {getattr(fragment, 'path', '?')!r} covers "
            f"{len(mask)} rows but the file holds {table.num_rows}; the table's log and its "
            "data files disagree."
        )
    table = table.filter(mask)
    return table.filter(flt) if flt is not None else table


def iter_fragment(
    fragment: Any,
    schema: pa.Schema,
    projection: list[str] | None,
    predicate: dict | None,
    mask: Any | None,
) -> Iterator[pa.RecordBatch]:
    """`read_fragment`, streamed — identical rows, in order, one batch resident at a time.

    The deletion-vector argument from `read_fragment` carries over unchanged, and is the
    only subtle part of streaming this. A vector is indexed by the file's **physical row
    positions**, so the masked path still reads unfiltered (a predicate pushed into the
    Parquet scan would slide every position and the mask would then delete the wrong rows)
    and still applies the mask before the predicate. Streaming adds one requirement: each
    batch must be masked with *its own slice* of the vector, which means tracking a running
    offset, because `to_batches` hands back successive windows of the same physical
    sequence.

    The length check has to move earlier than it was. `read_fragment` compares the vector
    against the materialized table, which a streaming read no longer has; `count_rows()`
    answers the same question from the Parquet footer, before any data page is read. Doing
    it up front also *strengthens* the guarantee — a mismatch is refused before a single
    batch is emitted, rather than after a consumer has already processed some of them.
    """
    flt = None
    if predicate is not None:
        from batcher.io.predicate import to_pyarrow_expression

        flt = to_pyarrow_expression(predicate, schema)

    if mask is None:
        yield from fragment.to_batches(schema=schema, columns=projection, filter=flt)
        return

    physical = fragment.count_rows()  # footer metadata, not a scan
    if len(mask) != physical:
        raise BackendError(
            f"Delta deletion vector for {getattr(fragment, 'path', '?')!r} covers "
            f"{len(mask)} rows but the file holds {physical}; the table's log and its "
            "data files disagree."
        )
    offset = 0
    for batch in fragment.to_batches(schema=schema, columns=projection):
        kept = batch.filter(mask[offset : offset + batch.num_rows])
        offset += batch.num_rows
        if flt is not None:
            kept = kept.filter(flt)
        if kept.num_rows:
            yield kept


def _demote_bounds(stats: SourceStatistics, *, rows: int) -> SourceStatistics:
    """The same statistics with an exact row count but bounds demoted to pruning-grade.

    A deletion vector can remove the very row that held a column's minimum, so the log's
    recorded bound is only a *superset* of the live range. Keeping it as EXACT would let a
    metadata answer report a `min()` that no longer exists in the table; demoting it keeps
    the bound (which is still sound for skipping files) without letting it answer for values.
    """
    import dataclasses

    from batcher.plan.stats import Provenance

    columns = {
        name: dataclasses.replace(stat, provenance=Provenance.DEFAULT)
        for name, stat in (stats.columns or {}).items()
    }
    return dataclasses.replace(stats, row_count=rows, columns=columns, exact_rows=True)
