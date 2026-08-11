"""The metadata-only Delta commit: register worker-written files, move no data.

A distributed write should move data exactly once — from the worker that produced it
to object storage — and then say so in the log. The previous Delta path moved it
twice: workers wrote Parquet into a staging area, and the driver then read **every
staged file back** and re-encoded it through ``write_deltalake``. So a 100-worker write
still funnelled 100% of the bytes through one process, and the driver's re-encode was
the whole write. Staging bought bounded driver *memory*, but not bounded driver *work*.

This module removes the second move. Workers write their shards as **final** Delta data
files, directly into the table's own layout, and collect each file's statistics while
the data is still in memory (free — no footer re-read). The driver then commits nothing
but `AddAction`s: paths, sizes, partition values, and those stats. The commit is O(files),
not O(rows), and no data crosses the driver at all.

The stats are not bookkeeping — they are the *next* query's file-skipping index
(`io.stats.file_skipping`). A write that omits them produces a table that can only be
read by brute force, so collecting them on the write path is what makes the read path
fast. Writing and skipping are two ends of one mechanism.

## Idempotent commits (`app_txn`)

Delta's `txn` action records ``(app_id, version)`` in the commit. A streaming query
passes its query name and micro-batch id, and `already_committed` checks the log before
writing: a micro-batch replayed after a crash finds its own version already there and
commits nothing. That — not the sink's atomicity — is what makes a restarted stream
exactly-once rather than at-least-once, and it is why the log ends up with exactly one
transaction per micro-batch, no matter how many times one was retried.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pyarrow as pa

from batcher._internal.errors import CommitError
from batcher._internal.logging import note_suppressed
from batcher.io.formats.lakehouse.delta._snapshot import require_deltalake
from batcher.io.manifest import WriteManifest, WrittenFile

__all__ = [
    "already_committed",
    "collect_file_stats",
    "commit_add_actions",
    "merge_file_stats",
]

# Arrow types whose min/max are meaningful and JSON-encodable for a Delta stat. Nested,
# binary, and extension columns are skipped: an absent stat is always sound (the reader
# keeps a file it cannot decide on), a *wrong* one silently loses rows.
_STATTABLE = (
    pa.types.is_integer,
    pa.types.is_floating,
    pa.types.is_boolean,
    pa.types.is_string,
    pa.types.is_large_string,
    pa.types.is_date,
    pa.types.is_timestamp,
)


def _is_stattable(dtype: pa.DataType) -> bool:
    return any(check(dtype) for check in _STATTABLE)


def collect_file_stats(table: pa.Table) -> dict[str, Any]:
    """Per-column bounds for one data file, computed from the data already in memory.

    Returned in the neutral `WrittenFile.stats` shape (``num_records`` /
    ``min_values`` / ``max_values`` / ``null_counts``). Costs one vectorized pass per
    column on data the writer is holding anyway, which is why the write path can afford
    to index every file it produces.

    Args:
        table: The rows being written to this file.

    Returns:
        The file's statistics.
    """
    import pyarrow.compute as pc

    mins: dict[str, Any] = {}
    maxs: dict[str, Any] = {}
    nulls: dict[str, int] = {}
    for field in table.schema:
        column = table.column(field.name)
        nulls[field.name] = column.null_count
        if not _is_stattable(field.type) or len(column) == column.null_count:
            continue
        try:
            bounds = pc.min_max(column)
            low, high = bounds["min"].as_py(), bounds["max"].as_py()
        except Exception as exc:
            note_suppressed("io", "compute column bounds for the commit", exc)
            continue  # a type the kernel will not order — no stat is a safe stat
        if low is not None and high is not None:
            mins[field.name] = low
            maxs[field.name] = high
    return {
        "num_records": table.num_rows,
        "min_values": mins,
        "max_values": maxs,
        "null_counts": nulls,
    }


def merge_file_stats(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Fold two files' statistics into one, for a writer streaming batch by batch.

    Associative and commutative (min of mins, max of maxs, sum of counts), so a
    bounded-memory streaming write accumulates the same statistics a single in-memory
    pass would produce.

    Args:
        left: Statistics accumulated so far.
        right: The next batch's statistics.

    Returns:
        The combined statistics.
    """
    if not left:
        return dict(right)
    if not right:
        return dict(left)
    mins = dict(left.get("min_values", {}))
    for key, value in right.get("min_values", {}).items():
        current = mins.get(key)
        mins[key] = value if current is None or _lt(value, current) else current
    maxs = dict(left.get("max_values", {}))
    for key, value in right.get("max_values", {}).items():
        current = maxs.get(key)
        maxs[key] = value if current is None or _lt(current, value) else current
    nulls = dict(left.get("null_counts", {}))
    for key, value in right.get("null_counts", {}).items():
        nulls[key] = nulls.get(key, 0) + value
    return {
        "num_records": left.get("num_records", 0) + right.get("num_records", 0),
        "min_values": mins,
        "max_values": maxs,
        "null_counts": nulls,
    }


def _lt(left: Any, right: Any) -> bool:
    """`left < right`, treating an incomparable pair as not-less (keeps the incumbent)."""
    try:
        return bool(left < right)
    except TypeError:
        return False


def _delta_stats_json(written: WrittenFile) -> str:
    """A `WrittenFile`'s neutral stats as the Delta add-action ``stats`` JSON string.

    Delta records temporal bounds as ISO-8601 text, so a `date`/`datetime` is encoded
    rather than handed over as an object json cannot serialize. A file with no collected
    stats still reports its row count, which keeps ``count(*)`` exact from the log.
    """
    stats = written.stats or {}
    payload = {
        "numRecords": stats.get("num_records", written.rows),
        "minValues": {k: _json_scalar(v) for k, v in stats.get("min_values", {}).items()},
        "maxValues": {k: _json_scalar(v) for k, v in stats.get("max_values", {}).items()},
        "nullCount": stats.get("null_counts", {}),
    }
    return json.dumps(payload)


def _json_scalar(value: Any) -> Any:
    """A stat value in the form Delta's log expects (ISO-8601 for temporals)."""
    import datetime as dt

    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    return value


def _add_action(written: WrittenFile, table_root: str) -> Any:
    """One `AddAction` for a written data file, with its path relative to the table."""
    require_deltalake()  # raises BackendError with an install hint if the dep is missing
    from deltalake.transaction import AddAction

    return AddAction(
        _relative(written.path, table_root),
        written.bytes,
        {k: _partition_str(v) for k, v in (written.partition_values or {}).items()},
        int(time.time() * 1000),
        True,  # data_change
        _delta_stats_json(written),
    )


def _partition_str(value: Any) -> str | None:
    """A partition value as the string the log stores (``None`` stays null)."""
    return None if value is None else str(value)


def _relative(path: str, table_root: str) -> str:
    """A data file's path relative to the table root — the form the log records."""
    root = table_root.rstrip("/")
    if path.startswith(root + "/"):
        return path[len(root) + 1 :]
    return path


def _schema_from_written(files: list[WrittenFile]) -> pa.Schema | None:
    """Reconstruct the table schema from the data files a write produced, or None.

    Used only when the caller did not attach one. A data file's footer gives every
    column *except* the partition columns — those live in the directory path, by Delta's
    convention, so they must be added back from the recorded partition values (whose
    Python types the writer preserved). Returns None if no file can be read, which the
    caller turns into a clear error rather than a malformed table.
    """
    import pyarrow.parquet as pq

    from batcher.io.filesystem import resolve_filesystem

    for written in files:
        try:
            fs = resolve_filesystem(written.path)
            target = fs.native_read_target(written.path)
            if target is not None:
                pafs, in_path = target
                data = pq.ParquetFile(in_path, filesystem=pafs).schema_arrow
            else:
                with fs.open(written.path) as fh:
                    data = pq.ParquetFile(fh).schema_arrow
        except Exception as exc:
            note_suppressed("io", "read a written file's schema", exc)
            continue
        partitions = [
            pa.field(name, pa.array([value]).type)
            for name, value in (written.partition_values or {}).items()
        ]
        return pa.schema(list(data) + partitions)
    return None


def already_committed(path: str, app_txn: tuple[str, int] | None, storage_options: Any) -> bool:
    """Whether ``(app_id, version)`` is already recorded in the table's log.

    The exactly-once check for a streaming write. After a crash the query replays the
    micro-batch it had not finished committing; if that batch *did* commit before the
    fault, its `txn` action is in the log and this returns True, so the replay writes
    nothing and the rows are not duplicated. A table that does not exist yet has
    committed nothing.

    Args:
        path: The table root.
        app_txn: The ``(app_id, version)`` this write would record, if any.
        storage_options: Cloud storage options for delta-rs.

    Returns:
        True when this transaction is already in the log and must not be re-applied.
    """
    if app_txn is None:
        return False
    deltalake = require_deltalake()
    app_id, version = app_txn
    try:
        if not deltalake.DeltaTable.is_deltatable(path, storage_options=storage_options):
            return False
        table = deltalake.DeltaTable(path, storage_options=storage_options)
        last = table.transaction_version(app_id)
    except Exception:
        return False  # cannot prove it was committed → commit (at-least-once, never lost)
    return last is not None and last >= version


def _set_table_properties(table: Any, properties: dict[str, str] | None) -> None:
    """Apply `properties` to an existing table, skipping the ones already in force.

    Comparing first is what keeps a repeated write from appending an identical `metaData`
    commit to the log on every run: the properties are usually passed unconditionally by a
    pipeline that just wants them set, and a no-op `ALTER` still costs a version.
    """
    if not properties:
        return
    current = table.metadata().configuration
    pending = {k: v for k, v in properties.items() if current.get(k) != v}
    if pending:
        table.alter.set_table_properties(pending)


def commit_add_actions(
    manifest: WriteManifest,
    path: str,
    *,
    mode: str,
    partition_by: list[str] | None = None,
    partition_filters: list[tuple[str, str, str]] | None = None,
    merge_schema: bool = False,
    storage_options: dict[str, str] | None = None,
    app_txn: tuple[str, int] | None = None,
    table_properties: dict[str, str] | None = None,
) -> None:
    """Commit the manifest's already-written data files as one Delta transaction.

    Registers the files; moves none of them. The driver's cost is one log write,
    independent of how much data the workers wrote.

    Args:
        manifest: The files the workers wrote, with their stats and partition values.
        path: The table root.
        mode: ``"append"`` or ``"overwrite"``.
        partition_by: The table's partition columns.
        partition_filters: Scope an overwrite to matching partitions only (Delta's
            ``replaceWhere``). delta-rs retires just those partitions' add-actions and adds
            the new ones, so a backfill of one day rewrites one day — not the table. With no
            filters an overwrite replaces everything, which is the mode's plain meaning.
        merge_schema: Evolve the table to accept columns the write has and the table does
            not. Off by default, because the safe answer to unexpected drift is to refuse.
        storage_options: Cloud storage options for delta-rs.
        app_txn: Optional ``(app_id, version)`` recorded as a Delta `txn` action, making
            the commit idempotent under replay.
        table_properties: Delta table properties (Spark's ``TBLPROPERTIES``) to set — on
            the ``metaData`` action when this commit creates the table, or as an
            ``ALTER TABLE SET TBLPROPERTIES`` when it already exists.

    Raises:
        CommitError: If the commit conflicts with a concurrent writer, or fails.
    """
    deltalake = require_deltalake()
    from deltalake import CommitProperties, Schema, Transaction
    from deltalake.transaction import create_table_with_add_actions

    files = [f for f in manifest.files if f.rows or not manifest.files]
    actions = [_add_action(f, path) for f in files]
    if not actions and mode == "append":
        return  # nothing written, nothing to say
    # The driver normally attaches the plan's output schema. A caller driving the `Sink`
    # protocol directly may not, so fall back to reconstructing it from what the workers
    # actually wrote — which is the only other place the truth exists.
    schema = manifest.schema
    if schema is None:
        schema = _schema_from_written(files)

    properties = None
    if app_txn is not None:
        properties = CommitProperties(
            app_transactions=[Transaction(app_id=app_txn[0], version=app_txn[1])]
        )

    try:
        exists = deltalake.DeltaTable.is_deltatable(path, storage_options=storage_options)
        if not exists:
            if schema is None:
                raise CommitError(
                    f"cannot create Delta table {path!r}: the write produced no schema"
                )
            create_table_with_add_actions(
                path,
                Schema.from_arrow(schema),
                actions,
                mode="error",
                partition_by=partition_by or [],
                configuration=table_properties,
                storage_options=storage_options,
                commit_properties=properties,
            )
            return
        table = deltalake.DeltaTable(path, storage_options=storage_options)
        # On an existing table the properties are a separate commit, exactly as
        # `ALTER TABLE ... SET TBLPROPERTIES` is. Applied before the data commit so a
        # property that changes how the data commit is *recorded* — enabling the change
        # data feed is the one that matters — is already in force for it, rather than
        # taking effect one write late.
        _set_table_properties(table, table_properties)
        _reconcile_schema(table, schema, merge_schema=merge_schema)
        table.create_write_transaction(
            actions,
            mode,
            Schema.from_arrow(schema) if schema is not None else table.schema(),
            partition_by=partition_by,
            partition_filters=partition_filters,
            commit_properties=properties,
        )
    except CommitError:
        raise
    except Exception as exc:
        if _is_conflict(exc):
            raise CommitError(
                f"Delta commit to {path!r} conflicted with a concurrent writer: {exc}"
            ) from exc
        raise CommitError(f"Delta commit to {path!r} failed: {exc}") from exc


def _is_conflict(exc: Exception) -> bool:
    """Heuristically detect a delta-rs concurrency-conflict error."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "concurr" in name or "concurr" in text or "commitfailed" in name


def _reconcile_schema(table: Any, schema: pa.Schema | None, *, merge_schema: bool) -> None:
    """Make the table's schema able to describe what the workers actually wrote.

    This guards a silent data-loss path. `create_write_transaction` takes a schema, but when
    that schema is *wider* than the table's, delta-rs **ignores it** and commits the add
    actions anyway. The extra column is then physically present in the data file and absent
    from the table: no error, no evolution, the value simply invisible. Worse, it is a
    delayed corruption — the moment someone legitimately adds that column later, the
    orphaned value resurfaces in a row that predates it, where the answer must be NULL.

    (delta-rs's own writer raises `SchemaMismatchError` here. Only the metadata-only commit
    path can reach the silent case, which is why the check belongs here.)

    So: a column the write has and the table does not is either **added** to the table
    (`merge_schema=True`) or **refused**. Never dropped. A column the *table* has and the
    write lacks is fine — Delta reads it back as NULL, which is what a partial write means.

    Args:
        table: The `DeltaTable` being committed to.
        schema: The write's output schema.
        merge_schema: Evolve the table rather than refusing.

    Raises:
        CommitError: On drift, when `merge_schema` is off — or on a type change, which
            evolution cannot express.
    """
    if schema is None:
        return
    from deltalake import Field, Schema

    existing = {field.name: field for field in table.schema().fields}
    incoming = Schema.from_arrow(schema).fields

    changed = [f.name for f in incoming if f.name in existing and f.type != existing[f.name].type]
    if changed:
        raise CommitError(
            f"the write changes the type of column(s) {changed} on Delta table "
            f"{table.table_uri!r}. Delta cannot evolve a column's type in place; write to a "
            "new table, or cast the column to the table's type before writing."
        )

    added = [f for f in incoming if f.name not in existing]
    if not added:
        return
    names = [f.name for f in added]
    if not merge_schema:
        raise CommitError(
            f"the write has column(s) {names} that Delta table {table.table_uri!r} does not. "
            "They would be written into the data files but stay invisible to the table — a "
            "silent loss that resurfaces as wrong data if the column is added later. Pass "
            "merge_schema=True to evolve the table, or drop the columns before writing."
        )
    try:
        table.alter.add_columns([Field(f.name, f.type, nullable=True) for f in added])
    except Exception as exc:
        raise CommitError(
            f"failed to evolve Delta table {table.table_uri!r} with column(s) {names}: {exc}"
        ) from exc
