"""``INSERT ... ON CONFLICT (k) DO NOTHING | DO UPDATE`` lowered onto the engine's merge.

An upsert *is* a merge with the inserted rows as the source: a row whose key is already in
the target is a ``WHEN MATCHED`` (left alone, or updated), and every other row is a ``WHEN
NOT MATCHED THEN INSERT``. So it goes through `compose_merge`, the function ``MERGE INTO``
and `write.merge_into` already end in, and inherits its parallel, spill and distributed
paths rather than growing its own.

Three things differ from an engine with declared constraints, and each is refused rather
than guessed:

- **The conflict target is required.** Batcher tables carry no primary key, so a bare ``ON
  CONFLICT DO NOTHING`` names no key to conflict on. DuckDB would infer the table's key.
- **Two inserted rows with the same key are an error.** Which of them wins depends on the
  order rows arrive in, which a lazy relation does not fix, so the answer would not be
  deterministic. Postgres refuses the same statement for ``DO UPDATE``. Rows whose key is
  NULL never conflict, as in SQL, and are not counted.
- **``ON CONFLICT ON CONSTRAINT`` and MySQL's ``ON DUPLICATE KEY`` are refused**: both name
  a constraint Batcher does not have.

The duplicate check is the one part that runs at statement time: a ``LIMIT 1`` probe over the
inserted rows, the same shape the SQL front end uses for an uncorrelated ``EXISTS``. It runs
before the target is rebound, so a refused upsert changes nothing.
"""

from __future__ import annotations

from typing import Any

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._sql.dml.merge import qualify_for_merge
from batcher._sql.dml.rewrite import cast_to, target_alias
from batcher._sql.parser.translator import _Translator
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import col

__all__ = ["upsert"]

_EXCLUDED = "excluded"


def upsert(
    node: Any,
    name: str,
    current: Dataset,
    rows: Dataset,
    registry: dict[str, Dataset],
    functions: dict[str, Any],
) -> Dataset:
    """The target's state after ``INSERT ... ON CONFLICT``.

    Args:
        node: The `exp.Insert` node carrying ``conflict``.
        name: The target's name, for errors.
        current: The target's current state.
        rows: The inserted rows, aligned to the target's schema.
        registry: Every visible table name and its bound `Dataset`.
        functions: Registered Python functions, for expressions in ``DO UPDATE SET``.

    Returns:
        The target's new state.

    Raises:
        PlanError: The clause names no conflict columns, names a constraint, names a column
            the target lacks, carries ``RETURNING``, or two inserted rows share a key.
    """
    from batcher.api.merge.clauses import MergeClause
    from batcher.api.merge.compose import compose_merge

    conflict = node.args["conflict"]
    keys = _conflict_keys(conflict, name, current)
    if node.args.get("returning"):
        raise PlanError(
            "INSERT ... ON CONFLICT ... RETURNING is not supported",
            hint="Query the table after the upsert.",
        )
    _refuse_duplicate_keys(rows, keys, name)

    clauses = []
    action = str(getattr(conflict.args.get("action"), "this", "")).upper()
    if action == "DO UPDATE":
        alias = target_alias(node.this)
        tr = _Translator(dict(registry), functions)
        types = {f.name: f.type for f in current.schema}
        values = {}
        for eq in conflict.expressions:
            column = eq.this.name
            if column not in types:
                raise PlanError(f"table {name!r} has no column {column!r}")
            values[column] = cast_to(tr._scalar(_rewrite(eq.expression, alias)), types[column])
        where = conflict.args.get("where")
        condition = tr._scalar(_rewrite(where.this, alias)) if where is not None else None
        clauses.append(MergeClause("matched", "update", condition, values))
    elif action != "DO NOTHING":
        raise PlanError(f"ON CONFLICT {action or '(no action)'} is not supported")
    clauses.append(MergeClause("not_matched", "insert", None, None))
    return compose_merge(rows, current, keys, clauses)


def _conflict_keys(conflict: Any, name: str, current: Dataset) -> list[str]:
    """The conflict target's column names, or a `PlanError` saying why there are none."""
    if conflict.args.get("duplicate") or conflict.args.get("constraint"):
        raise PlanError(
            "ON CONFLICT ON CONSTRAINT and ON DUPLICATE KEY name a constraint, and Batcher "
            "tables declare none",
            hint="Name the key columns: INSERT ... ON CONFLICT (id) DO UPDATE SET ...",
        )
    targets = conflict.args.get("conflict_keys") or []
    keys = [(t.this if isinstance(t, exp.Ordered) else t).name for t in targets]
    if not keys:
        raise PlanError(
            "ON CONFLICT needs a conflict target: Batcher tables have no primary key to infer "
            "one from",
            hint="Name the key columns: INSERT ... ON CONFLICT (id) DO NOTHING.",
        )
    missing = [k for k in keys if k not in current.columns]
    if missing:
        raise PlanError(f"table {name!r} has no column(s) {missing} to conflict on")
    return keys


def _refuse_duplicate_keys(rows: Dataset, keys: list[str], name: str) -> None:
    """Raise when two inserted rows share a non-NULL conflict key."""
    present = rows
    for key in keys:
        present = present.filter(col(key).is_not_null())
    counts = present.group_by(*keys).agg(__bc_n=col(keys[0]).count())
    probe = counts.filter(col("__bc_n") > 1).limit(1)
    if probe.to_arrow().num_rows:
        raise PlanError(
            f"INSERT into {name!r} ... ON CONFLICT: two inserted rows share the key {keys}, "
            "so which one wins would depend on row order",
            hint="Deduplicate the inserted rows first, e.g. with SELECT DISTINCT ON (key).",
        )


def _rewrite(node: Any, alias: str) -> Any:
    """``excluded.c`` to the inserted row's column, and the target's ``t.c``/``c`` to its own.

    A bare column in ``DO UPDATE SET`` names the existing row, as in Postgres and DuckDB.
    """
    node = node.copy()
    for column in node.find_all(exp.Column):
        if column.table.lower() == _EXCLUDED:
            column.set("table", exp.to_identifier(_EXCLUDED))
    return qualify_for_merge(node, alias, _EXCLUDED)
