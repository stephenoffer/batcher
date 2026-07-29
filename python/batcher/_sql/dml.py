"""INSERT / DELETE / UPDATE as pure plan rewrites over a session catalog.

Batcher datasets are lazy and immutable, but a `Session` catalog *is* mutable
control-plane metadata: ``register`` / ``DROP`` / ``CREATE`` all rebind a name to
a new lazy `Dataset`. DML is the same move expressed relationally — INSERT unions
new rows onto the target, DELETE keeps the rows a filter selects, UPDATE projects
a ``CASE`` over the assigned columns — and rebinds the name. Nothing executes here;
a later terminal op does. The caller (`Session`) owns the rebind.

This module is part of the neutral `_sql` frontend: it builds `Dataset`s through
the public API and the SQL translator, and imports no engine subsystem.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._sql import translate_ast
from batcher._sql.parser.translator import _Translator
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import Expr, col, lit, nullif, when

__all__ = ["apply_dml"]

_Registry = dict[str, Dataset]


def _cast_name(dtype: pa.DataType) -> str | None:
    """The engine cast target for an Arrow type, or None to leave a value as-is."""
    if pa.types.is_integer(dtype):
        return "int64"
    if pa.types.is_floating(dtype) or pa.types.is_decimal(dtype):
        return "float64"
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return "string"
    if pa.types.is_boolean(dtype):
        return "bool"
    if pa.types.is_date(dtype):
        return "date"
    if pa.types.is_timestamp(dtype):
        return "timestamp"
    return None


def _typed_null(cast_name: str | None) -> Expr:
    """A NULL literal typed to `cast_name` (`lit(None)` has no wire type)."""
    n = nullif(lit(0), lit(0))
    return n.cast(cast_name) if cast_name is not None else n


def _target_name(table_node: Any) -> str:
    """The table name a DML statement targets (unwrapping a column-list schema)."""
    if isinstance(table_node, exp.Schema):
        table_node = table_node.this
    return table_node.name


def apply_dml(node: Any, registry: _Registry, functions: dict[str, Any]) -> tuple[str, Dataset]:
    """Rewrite an INSERT / DELETE / UPDATE into ``(target_name, new_dataset)``.

    `registry` maps every visible table name to its bound `Dataset` (the session
    catalog plus per-call overrides). The returned dataset is the target table's
    new state; the caller rebinds `target_name` to it.
    """
    if isinstance(node, exp.Insert):
        return _insert(node, registry, functions)
    if isinstance(node, exp.Delete):
        return _delete(node, registry, functions)
    if isinstance(node, exp.Update):
        return _update(node, registry, functions)
    raise NotImplementedError(f"unsupported DML statement: {type(node).__name__}")


def _require_target(name: str, registry: _Registry) -> Dataset:
    if name not in registry:
        raise PlanError(f"no table {name!r} in catalog; registered: {sorted(registry)}")
    return registry[name]


def _insert(node: Any, registry: _Registry, functions: dict[str, Any]) -> tuple[str, Dataset]:
    for unsupported in ("conflict", "returning"):
        if node.args.get(unsupported):
            raise NotImplementedError(f"INSERT ... {unsupported.upper()} is not supported")
    schema = node.this
    columns = None
    if isinstance(schema, exp.Schema):
        columns = [c.name for c in schema.expressions]
    name = _target_name(schema)
    current = _require_target(name, registry)

    body = node.expression
    if body is None:
        raise PlanError("INSERT requires a VALUES or SELECT body")
    new_rows = translate_ast(body, functions=functions, **registry)
    aligned = _align_insert(name, current, new_rows, columns)
    return name, current.union(aligned, distinct=False)


def _align_insert(
    name: str, current: Dataset, new_rows: Dataset, columns: list[str] | None
) -> Dataset:
    """Reshape `new_rows` to the target schema (order, subset, and column types).

    An unnamed INSERT pairs source columns to the target positionally; a
    column-list INSERT (``INSERT INTO t (b, a) ...``) maps them by the named list,
    filling any unlisted target column with a typed NULL. Each output column is cast
    to the target type so the appended rows share the table's schema.
    """
    target_cols = current.columns
    target_types = {field.name: field.type for field in current.schema}
    source_cols = new_rows.columns

    if columns is None:
        columns = target_cols
    else:
        for c in columns:
            if c not in target_types:
                raise PlanError(f"table {name!r} has no column {c!r}")
    if len(source_cols) != len(columns):
        raise PlanError(
            f"INSERT into {name!r} supplies {len(source_cols)} column(s) but "
            f"{len(columns)} target column(s) were named"
        )
    provided = {tgt: source_cols[i] for i, tgt in enumerate(columns)}

    projections: dict[str, Expr] = {}
    for c in target_cols:
        cast_name = _cast_name(target_types[c])
        if c in provided:
            value: Expr = col(provided[c])
            if cast_name is not None:
                value = value.cast(cast_name)
        else:
            value = _typed_null(cast_name)
        projections[c] = value
    return new_rows.select(**projections)


def _delete(node: Any, registry: _Registry, functions: dict[str, Any]) -> tuple[str, Dataset]:
    for unsupported in ("using", "returning"):
        if node.args.get(unsupported):
            raise NotImplementedError(f"DELETE ... {unsupported.upper()} is not supported")
    name = _target_name(node.this)
    current = _require_target(name, registry)

    where = node.args.get("where")
    if where is None:
        # DELETE with no predicate empties the table but keeps its schema.
        return name, current.filter(lit(False))
    pred = _Translator(dict(registry), functions)._scalar(where.this)
    # DELETE removes rows where the predicate is TRUE; rows where it is FALSE *or
    # NULL* survive (SQL three-valued logic). Keep = NOT-true = (~pred) OR pred IS NULL.
    keep = (~pred) | pred.is_null()
    return name, current.filter(keep)


def _update(node: Any, registry: _Registry, functions: dict[str, Any]) -> tuple[str, Dataset]:
    name = _target_name(node.this)
    current = _require_target(name, registry)
    target_types = {field.name: field.type for field in current.schema}

    tr = _Translator(dict(registry), functions)
    where = node.args.get("where")
    pred = tr._scalar(where.this) if where is not None else None

    assignments: dict[str, Expr] = {}
    for eq in node.args.get("expressions") or []:
        target_col = eq.this.name
        if target_col not in target_types:
            raise PlanError(f"table {name!r} has no column {target_col!r}")
        assignments[target_col] = tr._scalar(eq.expression)

    projections: dict[str, Expr] = {}
    for c in current.columns:
        if c not in assignments:
            projections[c] = col(c)
            continue
        value = assignments[c]
        cast_name = _cast_name(target_types[c])
        if cast_name is not None:
            value = value.cast(cast_name)
        # A predicate restricts the update to the rows it selects; a NULL predicate
        # leaves the row unchanged (the CASE falls through to the old value).
        projections[c] = when(pred).then(value).otherwise(col(c)) if pred is not None else value
    return name, current.select(**projections)
