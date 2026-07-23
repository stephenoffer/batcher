"""Native Delta ``MERGE INTO`` — the full clause set, executed as one transaction.

Batcher's `MergeBuilder` models the whole SQL statement: ordered `WHEN MATCHED` /
`WHEN NOT MATCHED` / `WHEN NOT MATCHED BY SOURCE` clauses, each with a guard and its own
set of columns to write. delta-rs's `TableMerger` models exactly the same thing, so on a
Delta target the two map one-to-one and the merge runs *natively* — delta-rs rewrites only
the data files the join touches and commits them as a single version.

That mapping is what this module is, and it closes two real holes:

* **The general merge did not work on a lakehouse table at all.** `MergeBuilder.execute`
  went straight to the copy-on-write file path, which reconstructs the target from an
  explicit file list — a thing a `DeltaSource` cannot be constructed with. So
  `merge_into` against a Delta table died with a `TypeError`: the flagship MERGE API was
  broken on the one format with a native MERGE.
* **The shorthand silently ignored half its arguments.** `write.merge` routed Delta
  through a hard-coded ``update_all`` + ``insert_all``, so ``when_matched="delete"`` and
  ``when_not_matched="ignore"`` did nothing at all — the merge quietly did something other
  than what it was asked. Both now become real clauses.

## Rendering a clause

delta-rs takes SQL text, so a clause's `Expr` guard and its value expressions are rendered
to SQL with the two sides named. Batcher spells the source side as a reserved column
prefix (`SOURCE_PREFIX`) and the target side as a bare name, so the renderer maps
``__bc_src_amount`` → ``source.amount`` and ``amount`` → ``target.amount``.

The renderer covers what a merge guard is actually made of — column and literal
references, comparisons, boolean and arithmetic operators, null tests. An expression
outside that (a UDF, a nested function call) raises rather than being approximated: a
merge that quietly ran a *different* condition than the one written would corrupt the
table, which is precisely the failure mode this module exists to remove.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError, PlanError
from batcher.api.merge.clauses import (
    MATCHED,
    NOT_MATCHED,
    NOT_MATCHED_BY_SOURCE,
    SOURCE_PREFIX,
    MergeClause,
)
from batcher.io.formats.lakehouse.delta import require_deltalake
from batcher.io.manifest import WriteManifest, WrittenFile

__all__ = ["merge_into_delta"]


def merge_into_delta(
    table_uri: str,
    source: pa.Table,
    keys: list[str],
    clauses: list[MergeClause],
    *,
    storage_options: dict[str, str] | None = None,
) -> WriteManifest:
    """Run `clauses` as one native Delta ``MERGE INTO``, keyed on `keys`.

    Args:
        table_uri: The target table's root.
        source: The change set, with its columns under their own (unprefixed) names.
        keys: The columns matching a source row to a target row.
        clauses: The ordered ``WHEN`` clauses to apply.
        storage_options: Optional cloud storage options for delta-rs.

    Returns:
        A `WriteManifest` of the data files the merge wrote — read back from the commit it
        produced, so it describes what actually landed rather than what was staged.

    Raises:
        PlanError: If a clause's expression cannot be rendered as SQL.
        BackendError: If the table cannot be opened or the merge fails.
    """
    deltalake = require_deltalake()
    predicate = " AND ".join(f"target.{k} = source.{k}" for k in keys)
    try:
        table = deltalake.DeltaTable(table_uri, storage_options=storage_options)
    except Exception as exc:
        raise BackendError(f"failed to open Delta table {table_uri!r}: {exc}") from exc

    merger = table.merge(
        source=source,
        predicate=predicate,
        source_alias="source",
        target_alias="target",
    )
    for clause in clauses:
        merger = _apply(merger, clause)
    try:
        merger.execute()
    except Exception as exc:
        raise BackendError(f"Delta merge into {table_uri!r} failed: {exc}") from exc

    merged = deltalake.DeltaTable(table_uri, storage_options=storage_options)
    return _commit_manifest(table_uri, merged.version())


def _commit_manifest(table_uri: str, version: int) -> WriteManifest:
    """The data files a merge commit added, read from that version's log entry.

    A merge rewrites the files its join touched, so what it *wrote* is exactly the `add`
    actions of the version it produced. Best-effort: a log entry we cannot read yields an
    empty manifest rather than failing a merge that already committed.
    """
    import json

    from batcher.io.filesystem import resolve_filesystem

    entry = f"{table_uri.rstrip('/')}/_delta_log/{version:020d}.json"
    files: list[WrittenFile] = []
    try:
        fs = resolve_filesystem(entry)
        with fs.open(entry) as handle:
            lines = handle.read().decode("utf-8").splitlines()
    except Exception:
        return WriteManifest()
    for line in lines:
        try:
            action = json.loads(line).get("add")
        except ValueError:
            continue
        if not action:
            continue
        stats = action.get("stats")
        rows = 0
        if isinstance(stats, str):
            with contextlib.suppress(ValueError):
                rows = int(json.loads(stats).get("numRecords", 0))
        files.append(
            WrittenFile(
                path=f"{table_uri.rstrip('/')}/{action['path']}",
                rows=rows,
                bytes=int(action.get("size", 0)),
                partition_values=dict(action.get("partitionValues") or {}),
            )
        )
    return WriteManifest(tuple(files))


def _apply(merger: Any, clause: MergeClause) -> Any:
    """Add one `MergeClause` to the delta-rs merger as its matching ``WHEN`` clause."""
    predicate = _render(clause.condition) if clause.condition is not None else None

    if clause.kind == MATCHED:
        if clause.is_delete:
            return merger.when_matched_delete(predicate=predicate)
        if clause.values is None:
            return merger.when_matched_update_all(predicate=predicate)
        return merger.when_matched_update(_updates(clause), predicate=predicate)

    if clause.kind == NOT_MATCHED:
        if clause.values is None:
            return merger.when_not_matched_insert_all(predicate=predicate)
        return merger.when_not_matched_insert(_updates(clause), predicate=predicate)

    if clause.kind == NOT_MATCHED_BY_SOURCE:
        if clause.is_delete:
            return merger.when_not_matched_by_source_delete(predicate=predicate)
        if clause.values is None:
            # There is no source row here, so "all columns" has nothing to copy from.
            raise PlanError(
                "merge(): a not-matched-by-source clause must name the columns it "
                "updates — it has no source row to take them from"
            )
        return merger.when_not_matched_by_source_update(_updates(clause), predicate=predicate)

    raise PlanError(f"merge(): unknown clause kind {clause.kind!r}")


def _updates(clause: MergeClause) -> dict[str, str]:
    """A clause's ``column -> SQL`` value map, as delta-rs wants it."""
    return {column: _render(expr) for column, expr in (clause.values or {}).items()}


def _render(expr: Any) -> str:
    """One `Expr` as Delta merge SQL, with the source and target sides named.

    The renderer is shared with the write path (`delta._predicate`); only the column
    naming differs — a merge clause has two sides to disambiguate, a `replace_where`
    predicate has one. A second copy would eventually render a merge condition differently
    from the way a write renders the same expression.
    """
    from batcher.io.formats.lakehouse.delta._predicate import to_sql

    sql = to_sql(expr.to_ir(), _merge_column)
    if sql is None:
        raise PlanError(
            "merge(): this expression cannot be expressed in a Delta MERGE. Use "
            "comparisons, AND/OR/NOT, arithmetic, and null tests over source_col()/"
            "target_col(), or merge into a plain file target instead."
        )
    return sql


def _merge_column(name: str) -> str:
    """A column reference inside a merge clause: source columns carry the reserved prefix."""
    if name.startswith(SOURCE_PREFIX):
        return f"source.{name.removeprefix(SOURCE_PREFIX)}"
    return f"target.{name}"
