"""Native Iceberg ``MERGE INTO`` via pyiceberg's `Table.upsert`.

pyiceberg performs a real upsert: it joins the change set against the table, rewrites only
the data files the join touches, and commits them as one snapshot. That is what an Iceberg
merge should be, and it was simply never called — an Iceberg target fell through to the
copy-on-write file path, which cannot reconstruct an Iceberg table from a file list, so
`merge_into` failed on it exactly as it did on Delta.

## What Iceberg's upsert can and cannot express

`Table.upsert` is the *two-clause* merge: update the matched rows, insert the unmatched
ones, each toggleable. It has no per-clause guard, no ``DELETE``, and no
``WHEN NOT MATCHED BY SOURCE``. Rather than approximate those — a merge that quietly ran a
different statement than the one written would corrupt the table — anything outside that
shape raises with the reason. The clauses Iceberg *can* run, it runs natively; the rest are
refused, not faked.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError, PlanError
from batcher._internal.logging import note_suppressed
from batcher.api.merge.clauses import MATCHED, NOT_MATCHED, MergeClause
from batcher.io.manifest import WriteManifest, WrittenFile

__all__ = ["merge_into_iceberg"]


def merge_into_iceberg(
    identifier: str,
    source: pa.Table,
    keys: list[str],
    clauses: list[MergeClause],
    opts: dict[str, Any],
) -> WriteManifest:
    """Run `clauses` as a native Iceberg upsert against the table `identifier`.

    Args:
        identifier: The table identifier (``namespace.table``).
        source: The change set.
        keys: The columns matching a source row to a target row.
        clauses: The ordered ``WHEN`` clauses to apply.
        opts: Sink options; ``catalog`` selects the catalog to resolve against.

    Returns:
        A `WriteManifest` of the data files the upsert's snapshot added.

    Raises:
        PlanError: If the clauses use a shape Iceberg's upsert cannot express.
        BackendError: If the table cannot be loaded or the upsert fails.
    """
    update_matched, insert_unmatched = _resolve(clauses)

    from batcher.io.catalog import resolve_catalog

    catalog = resolve_catalog(opts.get("catalog") or "default")
    try:
        table = catalog.load_table(identifier)
    except Exception as exc:
        raise BackendError(f"failed to load Iceberg table {identifier!r}: {exc}") from exc

    try:
        table.upsert(
            source,
            join_cols=keys,
            when_matched_update_all=update_matched,
            when_not_matched_insert_all=insert_unmatched,
        )
    except Exception as exc:
        raise BackendError(f"Iceberg upsert into {identifier!r} failed: {exc}") from exc

    return _snapshot_manifest(catalog, identifier)


def _resolve(clauses: list[MergeClause]) -> tuple[bool, bool]:
    """Reduce the clause list to Iceberg's two toggles, or say why it cannot be reduced.

    Refusing is the point. Silently dropping a ``DELETE`` clause — which is what happens if
    you map an unsupported clause to "do nothing" — leaves rows in the table the user asked
    to remove, and the merge reports success.
    """
    update_matched = False
    insert_unmatched = False
    for clause in clauses:
        if clause.condition is not None:
            raise PlanError(
                "merge(): Iceberg's upsert has no per-clause condition. Use an "
                "unconditional when_matched()/when_not_matched(), or merge into a Delta "
                "target, which supports the full MERGE."
            )
        if clause.is_delete:
            raise PlanError(
                "merge(): Iceberg's upsert cannot delete rows. Use a Delta target for a "
                "MERGE with a DELETE clause, or ds.write.iceberg(..., mode='overwrite')."
            )
        if clause.values is not None:
            raise PlanError(
                "merge(): Iceberg's upsert writes all columns; it cannot update a subset. "
                "Use update_all()/insert_all(), or a Delta target."
            )
        if clause.kind == MATCHED:
            update_matched = True
        elif clause.kind == NOT_MATCHED:
            insert_unmatched = True
        else:  # NOT_MATCHED_BY_SOURCE
            raise PlanError(
                "merge(): Iceberg's upsert has no WHEN NOT MATCHED BY SOURCE clause "
                "(the rows the change set never mentions). Use a Delta target."
            )
    return update_matched, insert_unmatched


def _snapshot_manifest(catalog: Any, identifier: str) -> WriteManifest:
    """The data files the upsert's snapshot added, from the table's own metadata.

    Best-effort: a manifest we cannot read yields an empty one rather than failing an
    upsert that already committed.
    """
    try:
        table = catalog.load_table(identifier)
        snapshot = table.current_snapshot()
        if snapshot is None:
            return WriteManifest()
        added = table.inspect.data_files(snapshot_id=snapshot.snapshot_id)
        paths = added.column("file_path").to_pylist()
        rows = added.column("record_count").to_pylist()
        sizes = added.column("file_size_in_bytes").to_pylist()
    except Exception as exc:
        note_suppressed("api", "read iceberg snapshot", exc)
        return WriteManifest()
    return WriteManifest(
        tuple(
            WrittenFile(path=p, rows=int(r or 0), bytes=int(b or 0))
            for p, r, b in zip(paths, rows, sizes, strict=False)
        )
    )
