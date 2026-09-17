"""The save modes of a table write, reduced to a backend's four primitive writes.

`ds.write.table(name, mode=...)` is the one spelling for every way a dataset reaches a
catalog table — Spark's ``saveAsTable`` and ``insertInto``, its V2 ``writeTo(...).create /
replace / createOrReplace / append / overwrite / overwritePartitions``, and Daft's and
Polars' ``write_table``. They differ in two things only, and both are parameters:

- **what to do about the table existing or not**, which is `mode`: ``"error"`` creates and
  refuses an existing table, ``"ignore"`` creates or does nothing, ``"append"`` and
  ``"overwrite"`` create when missing, ``"replace"`` requires the table, and
  ``"overwrite_partitions"`` replaces only the partitions the incoming rows cover;
- **how incoming columns meet the table's**, which is `by_name`: by name (the default,
  unlisted table columns filled with NULL) or by position (``insertInto``, SQL ``INSERT``).

A write that meets an existing table's schema is aligned to it — reordered, cast, padded —
by the same `align_insert` SQL ``INSERT`` uses, so the DataFrame and SQL spellings of an
append cannot type a column differently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.catalog.catalog import Catalog
    from batcher.api.dataset import Dataset
    from batcher.api.sql_session import Session
    from batcher.io.manifest import WriteManifest
    from batcher.plan.expr_ir import Expr

__all__ = ["plan_write", "write_to_table"]


def write_to_table(
    data: Dataset,
    name: str,
    *,
    session: Session | None,
    mode: str,
    by_name: bool,
    partition_by: list[str] | None,
    properties: dict[str, str] | None,
    replace_where: Any,
) -> WriteManifest:
    """Resolve `name` in `session` (default: the current session) and write `data` to it.

    Args:
        data: The rows to write.
        name: A session-level table name.
        session: The session whose catalogs resolve `name`; None for `bt.current_session()`.
        mode: One of `TABLE_WRITE_MODES`.
        by_name: Match incoming columns by name, else by position.
        partition_by: See `plan_write`.
        properties: See `plan_write`.
        replace_where: See `plan_write`.

    Returns:
        The write's manifest.

    Raises:
        PlanError: `name` is a session view registered with `Session.register`.
    """
    from batcher.api.session.sql import current_session

    session = current_session() if session is None else session
    if name in session._tables:
        raise PlanError(
            f"{name!r} is a session view registered with Session.register, not a catalog "
            "table, so there is no storage to write to",
            hint="Write under another name, or Session.drop the view first.",
        )
    catalog, relative = session.catalog._resolve_table(name)
    return plan_write(
        catalog,
        relative,
        data,
        mode=mode,
        by_name=by_name,
        partition_by=partition_by,
        properties=properties,
        replace_where=replace_where,
    )


def plan_write(
    catalog: Catalog,
    name: str,
    data: Dataset,
    *,
    mode: str,
    by_name: bool,
    partition_by: list[str] | None,
    properties: dict[str, str] | None,
    replace_where: Any,
) -> WriteManifest:
    """Validate a table write and run the primitive write it reduces to.

    Args:
        catalog: The catalog holding (or about to hold) the table.
        name: The catalog-relative table name.
        data: The rows to write.
        mode: One of `TABLE_WRITE_MODES`.
        by_name: Match incoming columns to the table's by name, else by position.
        partition_by: Partition columns for a created table, or the partitions
            ``"overwrite_partitions"`` scopes to.
        properties: Table properties, applied when the write creates or replaces the table.
        replace_where: A predicate scoping an ``"overwrite"`` to the rows it matches.

    Returns:
        The write's manifest; empty when nothing was written.

    Raises:
        PlanError: The mode is unknown, or the table's existence contradicts it.
    """
    from batcher.api.catalog.catalog import TABLE_WRITE_MODES
    from batcher.io.manifest import WriteManifest

    if mode not in TABLE_WRITE_MODES:
        raise PlanError(
            f"write.table(): unknown mode {mode!r}",
            available=TABLE_WRITE_MODES,
            available_label="Modes",
        )
    if replace_where is not None and mode != "overwrite":
        raise PlanError(f"write.table(): replace_where= scopes an overwrite, not mode={mode!r}")
    backend = catalog._backend
    namespace, table = catalog._locate(name)

    def primitive(rows: Dataset, kind: str, where: Any = None) -> WriteManifest:
        return backend.write_table(
            namespace,
            table,
            rows,
            mode=kind,
            partition_by=partition_by,
            properties=properties,
            replace_where=where,
        )

    if not backend.has_table(namespace, table):
        _check_creatable(catalog, name, namespace, mode=mode, by_name=by_name, scoped=replace_where)
        return primitive(data, "create")
    if mode == "error":
        raise PlanError(
            f"table {name!r} already exists in catalog {catalog.name!r} and mode='error'",
            hint="Pass mode='append' to add rows or mode='overwrite' to replace them.",
        )
    if mode == "ignore":
        return WriteManifest()
    current = backend.read_table(namespace, table)
    if mode in ("overwrite", "replace") and replace_where is None:
        # By name, an overwrite takes the incoming schema whole; by position it keeps the
        # table's (Spark ``insertInto(overwrite=True)``), so only that form is aligned.
        rows = data if by_name else _align(name, current, data, by_name)
        return primitive(rows, "overwrite")
    if properties:
        raise PlanError(
            f"write.table({name!r}, mode={mode!r}): properties= applies when a write creates "
            "or replaces a table, and this one only adds rows"
        )
    aligned = _align(name, current, data, by_name)
    if mode == "append":
        return primitive(aligned, "append")
    if mode == "overwrite":
        return primitive(aligned, "replace_where", replace_where)
    columns = partition_by or backend.partition_columns(namespace, table)
    if not columns:
        raise PlanError(
            f"write.table({name!r}, mode='overwrite_partitions') needs partition columns, and "
            "the table is not partitioned",
            hint="Pass partition_by=[...], or use mode='overwrite' to replace every row.",
        )
    # One scoped overwrite per partition the rows cover. A Delta commit can only scope an
    # overwrite to an AND of `partition == value`, so a reload of three days is three
    # commits: each atomic, the set of them not. A reader between two sees some partitions
    # reloaded and others not yet, never a partition half-written.
    files = []
    for term in _covered_partitions(aligned, columns):
        files.extend(primitive(aligned.filter(term), "replace_where", term).files)
    return WriteManifest(tuple(files))


def _check_creatable(
    catalog: Catalog, name: str, namespace: str, *, mode: str, by_name: bool, scoped: Any
) -> None:
    """Refuse a write that needs the table to exist, or whose namespace is missing."""
    if mode in ("replace", "overwrite_partitions") or scoped is not None:
        raise PlanError(
            f"write.table({name!r}, mode={mode!r}) needs an existing table, and there is no "
            f"table {name!r} in catalog {catalog.name!r}",
            hint="Create it first with mode='error', or use mode='overwrite'.",
        )
    if not by_name:
        raise PlanError(
            f"write.table({name!r}, by_name=False) matches columns by position against an "
            "existing table's schema, and the table does not exist"
        )
    if namespace not in catalog._backend.list_namespaces():
        raise PlanError(
            f"no namespace {namespace!r} in catalog {catalog.name!r} to create {name!r} in",
            hint=f"Create it first with create_namespace({namespace!r}).",
        )


def _align(name: str, current: Dataset, data: Dataset, by_name: bool) -> Dataset:
    """Reorder, cast and pad `data` to the table's schema, by name or by position."""
    from batcher._sql.dml import align_insert

    return align_insert(name, current, data, data.columns if by_name else None)


def _covered_partitions(rows: Dataset, columns: list[str]) -> list[Expr]:
    """One predicate per partition `rows` has values in, each an AND over `columns`.

    The distinct partition tuples are collected to the driver, which is the number of
    partitions touched and not the number of rows: a dynamic partition overwrite exists to
    reload a handful of days into a table of years.
    """
    from batcher.plan.expr_ir import col, lit

    terms: list[Expr] = []
    for values in rows.select(*columns).distinct().to_pylist():
        term: Expr | None = None
        for column in columns:
            value = values[column]
            match = col(column).is_null() if value is None else col(column) == lit(value)
            term = match if term is None else term & match
        if term is not None:
            terms.append(term)
    return terms
