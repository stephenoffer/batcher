"""One cached read of a Delta table's `_delta_log`, shared by every metadata question.

Opening a `DeltaTable` replays the transaction log: every commit JSON since the last
checkpoint, plus the checkpoint Parquet. On a busy table that is the single most
expensive control-plane operation in a read, and `DeltaSource` used to pay it five
separate times for one query — once each in `schema()`, `row_count()`, `statistics()`,
`splits()`, and the deletion-vector probe — then **once more per worker**, because a
`DeltaFileSplit` rebuilt the whole dataset just to find its own fragment.

`DeltaSnapshot` collapses that to one replay. It resolves the log once and answers
schema, protocol features, partition columns, and the add-action manifest from the
resolved state, so the planner's cost is O(log) once rather than O(log x questions).

## Why the cache is safe

A snapshot is only ever cached under a **pinned version**. The driver resolves
``latest`` to a concrete version number at plan time; that number is what travels to
the workers inside each split, and a specific version of a Delta table is immutable by
construction — a later commit creates version n+1 and cannot alter version n. So a
worker (and a re-run of the same query on the driver) can reuse a cached snapshot
without ever reading a stale table, while a *new* query re-resolves ``latest`` and
sees the new commit. An unpinned read is never cached.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher._internal.optional import require

__all__ = ["DeltaSnapshot", "open_snapshot", "require_deltalake"]

# Bounded LRU of pinned (uri, version) snapshots, so a worker reading many splits of
# the same table replays the log once, and a worker cycling across tables keeps its
# hot ones resident. Mirrors the `fragment_index` cache in `io.splits`.
_SNAPSHOT_CACHE: OrderedDict[tuple, DeltaSnapshot] = OrderedDict()
_SNAPSHOT_CACHE_MAX = 8

# One live `DeltaTable` per table, rolled forward with `update_incremental` instead of
# re-replaying the log on every query. Guarded, because two queries may refresh it at once.
_LATEST_HANDLES: OrderedDict[tuple, Any] = OrderedDict()
_HANDLE_LOCK = threading.Lock()


def require_deltalake() -> Any:
    """Import and return the `deltalake` module, or raise `BackendError`."""
    return require("deltalake", feature="Delta Lake support", provides="delta-rs", extra="delta")


@dataclass(slots=True)
class DeltaSnapshot:
    """A resolved, immutable view of one Delta table version.

    Wraps the `DeltaTable` handle whose log has already been replayed, and memoizes the
    derived metadata every caller wants. `version` is always concrete — never
    ``latest`` — which is what makes the snapshot cacheable and shippable to a worker.
    """

    table_uri: str
    version: int
    # `repr=False`: the cloud credentials, same as on `DeltaFileSplit`. A snapshot is the
    # object every metadata path holds, so it is the one most likely to appear in a
    # traceback frame — and a generated `repr` puts the secret access key in the log.
    storage_options: dict[str, str] | None = field(repr=False)
    _schema: pa.Schema
    _add_actions: pa.Table
    _partition_columns: list[str]
    _masks: dict[str, Any]
    _full_index: tuple[Any, dict[str, Any]] | None = None
    _pinned: Any = None

    @property
    def table(self) -> Any:
        """A `DeltaTable` handle pinned to *this* version.

        Built on demand, and only for the two callers that genuinely need a live delta-rs
        handle (the change-data-feed reader, and the dataset fallback). Everything else
        reads the metadata materialized at construction — which is what lets the process
        keep **one** handle per table and roll it forward with `update_incremental`, rather
        than replaying the whole `_delta_log` on every query.
        """
        if self._pinned is None:
            deltalake = require_deltalake()
            self._pinned = deltalake.DeltaTable(
                self.table_uri, version=self.version, storage_options=self.storage_options
            )
        return self._pinned

    def schema(self) -> pa.Schema:
        """The table's Arrow schema, including partition columns."""
        return self._schema

    def add_actions(self) -> pa.Table:
        """The per-file manifest: one row per data file, with stats and partition values.

        This is the add-action layout `io.stats.file_skipping` prunes against and
        `io.stats.lakehouse_manifest` aggregates — ``path``, ``num_records``, and the
        ``partition.`` / ``min.`` / ``max.`` / ``null_count.`` columns.
        """
        return self._add_actions

    def partition_columns(self) -> list[str]:
        """The table's partition columns, in their declared order."""
        return self._partition_columns

    def deletion_masks(self) -> dict[str, Any]:
        """Per-file row masks for the table's deletion vectors — ``True`` means *keep*.

        A deletion vector does not remove rows from a data file; it records, beside the
        file, which of its rows are gone. The file still physically contains them, so a
        reader that ignores the vector resurrects deleted data.

        The mask is indexed by **physical row position**, which is the constraint everything
        else here follows from: it can only be applied to the file's rows *as written*. A
        predicate pushed into the Parquet read would drop rows first and slide every
        position, so a DV'd file must be read unfiltered, masked, and only then filtered.
        Files with no vector — the vast majority, since a vector only attaches to a file a
        delete actually touched — keep full row-group pushdown.

        Empty when the table has no deletion vectors. This is also the *presence* test:
        asking the log which files have vectors is exact, where the protocol's
        ``deletionVectors`` reader feature only says the table is *allowed* to have them.
        That flag is default-on for new Delta tables, so keying off it condemned every such
        table — including the overwhelming majority with no deletions at all — to the slow
        path for nothing.
        """
        return self._masks

    @staticmethod
    def _read_deletion_masks(table: Any, relative: Any) -> dict[str, Any]:
        import pyarrow as _pa

        masks: dict[str, Any] = {}
        try:
            reader = table.deletion_vectors()
        except Exception:  # pragma: no cover - an older delta-rs without the API
            return masks
        try:
            for batch in reader:
                paths = batch.column("filepath")
                vectors = batch.column("selection_vector")
                for i in range(batch.num_rows):
                    # The mask is a plain Arrow boolean list; keep it Arrow-native (a
                    # `to_pylist()` here was measured 7x slower on a 50k-row mask).
                    masks[relative(paths[i].as_py())] = _pa.array(
                        vectors[i].values, type=_pa.bool_()
                    )
        except Exception:
            return {}
        return masks

    def _relative(self, uri: str) -> str:
        """A deletion vector's file URI as the table-relative path the add actions use."""
        return _relative_to(uri, self.table_uri)

    def has_deletion_vectors(self) -> bool:
        """Whether any data file in this version actually carries a deletion vector."""
        return bool(self.deletion_masks())

    def deleted_rows(self) -> int:
        """How many physically-present rows the deletion vectors mark as gone."""
        import pyarrow.compute as pc

        total = 0
        for mask in self.deletion_masks().values():
            kept = pc.sum(mask).as_py() or 0
            total += len(mask) - int(kept)
        return total

    def file_paths(self) -> list[str]:
        """Table-relative path of every data file in this version."""
        return self.add_actions().column("path").to_pylist()

    def rows_by_path(self) -> dict[str, int]:
        """Each data file's exact record count, from its add action.

        The count is already in the log, so a split can carry its true size without a
        footer read — which is what lets the distributed planner bin-pack files by
        weight instead of assuming they are all the same size.
        """
        manifest = self.add_actions()
        if "num_records" not in manifest.column_names:
            return {}
        paths = manifest.column("path").to_pylist()
        rows = manifest.column("num_records").to_pylist()
        return {p: r for p, r in zip(paths, rows, strict=True) if r is not None}

    def surviving_paths(self, predicate: dict | None) -> list[str] | None:
        """Paths of the files that can contain a row matching `predicate`.

        The read-side payoff of the transaction log: the add-action manifest already
        carries each file's partition values and column bounds, so a selective
        predicate is answered against the log and the files it rules out are never
        opened — no footer read, no split, no worker task. Returns `None` when the
        manifest cannot decide the predicate, which the caller reads as "scan
        everything" (see `file_skipping` on why unknown must mean keep).

        The paths are table-relative, the same form a pyarrow dataset fragment
        reports, so they join directly against `to_pyarrow_dataset()`'s fragments.
        """
        from batcher.io.stats.file_skipping import surviving_files

        return surviving_files(predicate, self.add_actions())

    def dataset(self, predicate: dict | None = None) -> Any:
        """A pyarrow dataset over only the data files that can match `predicate`.

        Built **directly from the add-action manifest**, not from delta-rs's
        `to_pyarrow_dataset()`. That matters more than it sounds: delta-rs constructs a
        fragment for every file in the table before any filter is considered, so on a
        200-file table it cost ~38 ms whether the query wanted 200 files or one — which
        was the entire cost of a selective read and would have made file skipping
        pointless. Here the manifest is pruned first and only the survivors become
        fragments, so planning scales with the files a query actually reads rather than
        with the size of the table.

        Each fragment carries its partition values as a partition expression, exactly
        as a Hive-partitioned dataset does, so partition columns (which live in the
        path, never in the Parquet file) still materialize. Falls back to delta-rs's own
        dataset whenever the fast path cannot be built — a filesystem that exposes no
        native pyarrow handle (a read-through byte cache), an unmappable partition type,
        or any other surprise — so correctness never depends on the optimization.
        """
        return self.dataset_index(predicate)[0]

    def dataset_index(self, predicate: dict | None = None) -> tuple[Any, dict[str, Any]]:
        """``(dataset, {table-relative path: fragment})`` over the surviving files.

        The index is built here, beside the fragments, because only here are both the
        add-action path and the fragment in hand. A caller that re-derives the mapping from
        ``fragment.path`` gets the *filesystem* path, which is absolute — and every other
        thing keyed by file (a split's locator, a deletion vector's mask) is keyed by the
        table-relative path the log uses. Matching those two up outside this method is
        exactly the mismatch that silently returns zero rows.
        """
        # The unpruned index is what every split of this table looks its own file up in, so
        # it is built once per snapshot — and a snapshot is already cached per (uri, version)
        # and per worker process, which is exactly the caching a split read wants.
        if predicate is None and self._full_index is not None:
            return self._full_index
        try:
            built = self._pruned_dataset(predicate)
        except Exception:
            # `table`, not `_table`: this is a slots dataclass with no such field, and the
            # live delta-rs handle is the lazily-pinned property. Written as `_table` this
            # fallback raised `AttributeError` itself — so the safety net that is supposed to
            # keep correctness independent of the pruning optimization had a hole straight
            # through it, and only the pruned path ever actually worked.
            dataset = self.table.to_pyarrow_dataset()
            built = dataset, {f.path: f for f in dataset.get_fragments()}
        if predicate is None:
            self._full_index = built
        return built

    def _pruned_dataset(self, predicate: dict | None) -> tuple[Any, dict[str, Any]]:
        import pyarrow.dataset as pds

        from batcher.io.filesystem import resolve_filesystem
        from batcher.io.stats.file_skipping import file_prune_mask

        manifest = self.add_actions()
        mask = file_prune_mask(predicate, manifest)
        if mask is not None:
            manifest = manifest.filter(mask)

        schema = self.schema()
        fmt = pds.ParquetFileFormat()
        if manifest.num_rows == 0:
            # Provably empty: a typed, zero-row dataset with not one file opened.
            return pds.FileSystemDataset([], schema, fmt), {}

        relative = manifest.column("path").to_pylist()
        fs, in_paths = self._native_paths(relative, resolve_filesystem)
        exprs = self._partition_expressions(manifest, schema, pds)
        fragments = [
            fmt.make_fragment(path, filesystem=fs, partition_expression=expr)
            for path, expr in zip(in_paths, exprs, strict=True)
        ]
        # Keyed by the table-relative path — the form the log, the splits, and the deletion
        # vectors all use. The fragment's own `.path` is the absolute filesystem path.
        index = dict(zip(relative, fragments, strict=True))
        return pds.FileSystemDataset(fragments, schema, fmt, fs), index

    def _native_paths(self, paths: list[str], resolve: Any) -> tuple[Any, list[str]]:
        """``(pyarrow filesystem, in-filesystem paths)`` for the manifest's data files."""
        absolute = [self._absolute(p) for p in paths]
        fs_handle = resolve(self.table_uri)
        targets = [fs_handle.native_read_target(p) for p in absolute]
        if any(t is None for t in targets):
            raise ValueError("filesystem exposes no native pyarrow read target")
        return targets[0][0], [t[1] for t in targets]

    def _absolute(self, path: str) -> str:
        """An add-action's table-relative path as an absolute *filesystem* URI.

        A Delta ``add.path`` is a URI: per the protocol it is URL-encoded and must be
        **decoded** to get the physical data file path. This matters the moment a
        partition value contains a URL-special character. Delta writes the value into the
        directory name Hive-encoded once (``a/b`` → ``p=a%2Fb``), and the log then encodes
        that path again (``p=a%252Fb``). Handing the raw log path straight to the
        filesystem looks for ``p=a%252Fb``, which does not exist — so every partitioned
        table whose partition value held a ``/``, a space, or a ``%`` raised
        `FileNotFoundError` at read, even though delta-rs's own reader (which decodes)
        read it fine. Decoding here restores the physical path the file was written to.
        """
        from urllib.parse import unquote

        decoded = unquote(path)
        if "://" in decoded or decoded.startswith("/"):
            return decoded
        return f"{self.table_uri.rstrip('/')}/{decoded}"

    def _partition_expressions(self, manifest: Any, schema: pa.Schema, pds: Any) -> list[Any]:
        """One partition expression per file, from its recorded partition values.

        A partition column's value is encoded in the file's *path*, not its data, so the
        fragment must carry it as an expression for the column to exist in the scan
        result at all. An unpartitioned table yields all-True expressions.
        """
        columns = [c for c in self.partition_columns() if f"partition.{c}" in manifest.column_names]
        if not columns:
            return [pds.scalar(True) for _ in range(manifest.num_rows)]
        values = {c: manifest.column(f"partition.{c}").to_pylist() for c in columns}
        types = {c: schema.field(c).type for c in columns}
        out = []
        for i in range(manifest.num_rows):
            expr = None
            for c in columns:
                term = pds.field(c) == pa.scalar(values[c][i], types[c])
                expr = term if expr is None else (expr & term)
            out.append(expr if expr is not None else pds.scalar(True))
        return out


def open_snapshot(
    table_uri: str,
    *,
    version: int | None = None,
    timestamp: str | None = None,
    storage_options: dict[str, str] | None = None,
) -> DeltaSnapshot:
    """Open (or reuse) the snapshot for a Delta table at a version/timestamp.

    A pinned `version` is served straight from the process cache — that version's state is
    immutable, so a hit can never be stale.

    An unpinned (``latest``) read is the common one, and it used to pay a full
    ``DeltaTable(path)`` on **every query**: delta-rs replays the log from the last
    checkpoint each time, which measured 6.1 ms on a 200-commit table and was, after file
    skipping, the single largest cost left in a selective read. Instead the process keeps
    **one live handle per table** and rolls it forward with ``update_incremental`` (0.58 ms
    — it reads only the commits since it last looked). If that lands on a version already in
    the cache, the query pays nothing more at all.

    Rolling a *shared* handle forward is only safe because a `DeltaSnapshot` does not hold
    it: everything version-dependent — schema, add actions, partition columns, deletion
    vectors — is materialized when the snapshot is built, while the handle still points at
    that version. A later query advancing the handle therefore cannot change what an
    already-issued snapshot reports. (The two callers that genuinely need a live handle get
    one pinned to their own version, built on demand.)

    Args:
        table_uri: The table root.
        version: Optional pinned version for time travel.
        timestamp: Optional ISO-8601 timestamp for time travel.
        storage_options: Optional cloud storage options for delta-rs.

    Returns:
        The resolved snapshot.
    """
    opts_key = tuple(sorted((storage_options or {}).items()))
    if version is not None:
        cached = _cache_get((table_uri, version, opts_key))
        if cached is not None:
            return cached
        table = _open_table(table_uri, version, None, storage_options)
        return _snapshot_from(table, table_uri, storage_options, opts_key)

    if timestamp is not None:
        # Time travel by timestamp resolves against the log every time; there is no handle
        # to roll forward, because the answer is not "the latest version".
        table = _open_table(table_uri, None, timestamp, storage_options)
        return _snapshot_from(table, table_uri, storage_options, opts_key)

    with _HANDLE_LOCK:
        handle = _LATEST_HANDLES.get((table_uri, opts_key))
        if handle is None:
            handle = _open_table(table_uri, None, None, storage_options)
            _LATEST_HANDLES[(table_uri, opts_key)] = handle
            while len(_LATEST_HANDLES) > _SNAPSHOT_CACHE_MAX:
                _LATEST_HANDLES.popitem(last=False)
        else:
            try:
                handle.update_incremental()  # catch up on any commits since we last looked
            except Exception:
                handle = _open_table(table_uri, None, None, storage_options)
                _LATEST_HANDLES[(table_uri, opts_key)] = handle
        resolved = int(handle.version())
        cached = _cache_get((table_uri, resolved, opts_key))
        if cached is not None:
            return cached
        return _snapshot_from(handle, table_uri, storage_options, opts_key)


def _open_table(
    table_uri: str,
    version: int | None,
    timestamp: str | None,
    storage_options: dict[str, str] | None,
) -> Any:
    deltalake = require_deltalake()
    try:
        table = deltalake.DeltaTable(table_uri, version=version, storage_options=storage_options)
        if timestamp is not None:
            table.load_as_version(timestamp)
    except Exception as exc:
        raise BackendError(f"failed to open Delta table {table_uri!r}: {exc}") from exc
    return table


def _snapshot_from(
    table: Any,
    table_uri: str,
    storage_options: dict[str, str] | None,
    opts_key: tuple,
) -> DeltaSnapshot:
    """Materialize every version-dependent fact `table` currently reports, and cache it.

    Reading them **now**, while the handle is known to be at this version, is what makes the
    snapshot independent of the handle — and therefore what makes rolling one shared handle
    forward across queries safe.
    """
    resolved = int(table.version())
    schema = pa.schema(table.schema().to_arrow())
    add_actions = pa.table(table.get_add_actions(flatten=True))
    try:
        partitions = list(table.metadata().partition_columns)
    except Exception:
        partitions = []
    snapshot = DeltaSnapshot(
        table_uri=table_uri,
        version=resolved,
        storage_options=storage_options,
        _schema=schema,
        _add_actions=add_actions,
        _partition_columns=partitions,
        _masks={},
    )
    snapshot._masks = DeltaSnapshot._read_deletion_masks(table, snapshot._relative)
    _cache_put((table_uri, resolved, opts_key), snapshot)
    return snapshot


def _relative_to(uri: str, table_uri: str) -> str:
    """A file URI as the table-relative path the add actions use."""
    root = table_uri.rstrip("/")
    for prefix in (f"file://{root}/", f"{root}/"):
        if uri.startswith(prefix):
            return uri[len(prefix) :]
    return uri.rsplit("/", 1)[-1] if "/" in uri and root not in uri else uri


def _cache_get(key: tuple) -> DeltaSnapshot | None:
    cached = _SNAPSHOT_CACHE.get(key)
    if cached is not None:
        _SNAPSHOT_CACHE.move_to_end(key)  # most-recently-used
    return cached


def _cache_put(key: tuple, snapshot: DeltaSnapshot) -> None:
    _SNAPSHOT_CACHE[key] = snapshot
    while len(_SNAPSHOT_CACHE) > _SNAPSHOT_CACHE_MAX:
        _SNAPSHOT_CACHE.popitem(last=False)  # evict least-recently-used
