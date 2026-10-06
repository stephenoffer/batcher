"""INSERT / DELETE / UPDATE as pure plan rewrites over a session catalog.

Batcher datasets are lazy and immutable, but a `Session` catalog *is* mutable
control-plane metadata: ``register`` / ``DROP`` / ``CREATE`` all rebind a name to
a new lazy `Dataset`. DML is the same move expressed relationally — INSERT unions
new rows onto the target, DELETE keeps the rows a filter selects, UPDATE projects
a ``CASE`` over the assigned columns — and rebinds the name. Nothing executes here;
a later terminal op does. The caller (`Session`) owns the rebind.

Each rewrite also returns the rows the statement touched (inserted, deleted, or updated),
because that is what ``RETURNING`` projects. Those are ordinary lazy relations too: a
DELETE's are read from the target's *pre*-statement plan, which is still exactly the old
state because a plan is immutable.

This module is part of the neutral `_sql` frontend: it builds `Dataset`s through
the public API and the SQL translator, and imports no engine subsystem.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._sql import translate_ast
from batcher._sql.dml.using import delete_using
from batcher._sql.parser.translator import _Translator
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import Expr, col, lit, nullif, when
from batcher.plan.expr_ir.constructors import null_of_type
from batcher.plan.types import dtype_name

__all__ = [
    "align_insert",
    "cast_to",
    "delete",
    "inserted_rows",
    "require_target",
    "returning",
    "target_alias",
    "target_name",
    "update",
]

_Registry = dict[str, Dataset]

# The binding name `returning` gives the touched rows while it translates the projection.
_RETURNING = "__bc_returning"


def _typed_null(dtype: pa.DataType) -> Expr:
    """A NULL of the column's exact type, or the untyped (Int64) NULL for a nested one."""
    typed = null_of_type(dtype)
    return typed if typed is not None else nullif(lit(0), lit(0))


def cast_to(value: Expr, dtype: pa.DataType) -> Expr:
    """`value` cast to the column's exact type, or as-is for one the engine cannot cast to.

    Args:
        value: The expression written into the column.
        dtype: The column's type.

    Returns:
        The cast expression.
    """
    name = None if pa.types.is_null(dtype) else dtype_name(dtype)
    return value.cast(name) if name is not None else value


def target_name(table_node: Any) -> str:
    """The table name a DML statement targets (unwrapping a column-list schema).

    Args:
        table_node: The statement's target, an `exp.Table` or an `exp.Schema` around one.

    Returns:
        The bare table name.
    """
    if isinstance(table_node, exp.Schema):
        table_node = table_node.this
    return table_node.name


def target_alias(table_node: Any) -> str:
    """The name the statement's other clauses use for the target: its alias, else its name.

    Args:
        table_node: The statement's target.

    Returns:
        The alias, or the bare table name when there is none.
    """
    if isinstance(table_node, exp.Schema):
        table_node = table_node.this
    return table_node.alias or table_node.name


def require_target(name: str, registry: _Registry) -> Dataset:
    """The relation `name` is bound to, or a `PlanError` listing what is bound.

    Args:
        name: The table name.
        registry: Every visible table name and its bound `Dataset`.

    Returns:
        The bound relation.
    """
    if name not in registry:
        raise PlanError(f"no table {name!r} in catalog; registered: {sorted(registry)}")
    return registry[name]


def inserted_rows(
    node: Any, registry: _Registry, functions: dict[str, Any]
) -> tuple[str, Dataset, Dataset]:
    """An INSERT's target name, the target's current state, and the rows it adds.

    Args:
        node: The `exp.Insert` node.
        registry: Every visible table name and its bound `Dataset`.
        functions: Registered Python functions.

    Returns:
        ``(name, current, rows)``, `rows` already aligned to the target's schema.
    """
    schema = node.this
    columns = None
    if isinstance(schema, exp.Schema):
        columns = [c.name for c in schema.expressions]
    name = target_name(schema)
    current = require_target(name, registry)

    body = node.expression
    if body is None:
        raise PlanError("INSERT requires a VALUES or SELECT body")
    new_rows = translate_ast(body, functions=functions, **registry)
    return name, current, align_insert(name, current, new_rows, columns)


def align_insert(
    name: str, current: Dataset, new_rows: Dataset, columns: list[str] | None
) -> Dataset:
    """Reshape `new_rows` to the target schema (order, subset, and column types).

    An unnamed INSERT pairs source columns to the target positionally; a
    column-list INSERT (``INSERT INTO t (b, a) ...``) maps them by the named list,
    filling any unlisted target column with a typed NULL. Each output column is cast
    to the target type so the appended rows share the table's schema.

    Args:
        name: The target's name, for errors.
        current: The target's current state.
        new_rows: The relation the INSERT body produced.
        columns: The INSERT's column list, or None for a positional INSERT.

    Returns:
        `new_rows` with exactly the target's columns and types.
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
        projections[c] = (
            cast_to(col(provided[c]), target_types[c])
            if c in provided
            else _typed_null(target_types[c])
        )
    return new_rows.select(**projections)


def delete(
    node: Any, registry: _Registry, functions: dict[str, Any]
) -> tuple[str, Dataset, Dataset]:
    """A DELETE's target name, the rows that survive it, and the rows it removes.

    Args:
        node: The `exp.Delete` node.
        registry: Every visible table name and its bound `Dataset`.
        functions: Registered Python functions.

    Returns:
        ``(name, kept, deleted)``.
    """
    name = target_name(node.this)
    current = require_target(name, registry)
    if node.args.get("using"):
        kept, deleted = delete_using(node, name, current, registry, functions)
        return name, kept, deleted

    where = node.args.get("where")
    if where is None:
        # DELETE with no predicate empties the table but keeps its schema.
        return name, current.filter(lit(False)), current
    pred = _Translator(dict(registry), functions)._scalar(where.this)
    # DELETE removes rows where the predicate is TRUE; rows where it is FALSE *or
    # NULL* survive (SQL three-valued logic). Keep = NOT-true = (~pred) OR pred IS NULL.
    keep = (~pred) | pred.is_null()
    return name, current.filter(keep), current.filter(pred)


def update(
    node: Any, registry: _Registry, functions: dict[str, Any]
) -> tuple[str, Dataset, Dataset]:
    """An UPDATE's target name, the target's new state, and the updated rows' new values.

    Args:
        node: The `exp.Update` node.
        registry: Every visible table name and its bound `Dataset`.
        functions: Registered Python functions.

    Returns:
        ``(name, new_state, updated)``.
    """
    if node.args.get("from_") or node.args.get("from"):
        raise PlanError(
            "UPDATE ... FROM is not supported",
            hint="Write it as MERGE INTO t USING s ON t.k = s.k WHEN MATCHED THEN UPDATE SET ...",
        )
    name = target_name(node.this)
    current = require_target(name, registry)
    target_types = {field.name: field.type for field in current.schema}

    tr = _Translator(dict(registry), functions)
    where = node.args.get("where")
    pred = tr._scalar(where.this) if where is not None else None

    assignments: dict[str, Expr] = {}
    for eq in node.args.get("expressions") or []:
        target_col = eq.this.name
        if target_col not in target_types:
            raise PlanError(f"table {name!r} has no column {target_col!r}")
        assignments[target_col] = cast_to(tr._scalar(eq.expression), target_types[target_col])

    projections: dict[str, Expr] = {}
    for c in current.columns:
        if c not in assignments:
            projections[c] = col(c)
            continue
        # A predicate restricts the update to the rows it selects; a NULL predicate
        # leaves the row unchanged (the CASE falls through to the old value).
        value = assignments[c]
        projections[c] = when(pred).then(value).otherwise(col(c)) if pred is not None else value
    touched = current.filter(pred) if pred is not None else current
    updated = touched.select(**{c: assignments.get(c, col(c)) for c in current.columns})
    return name, current.select(**projections), updated


def returning(
    node: Any, rows: Dataset, registry: _Registry, functions: dict[str, Any]
) -> Dataset | None:
    """The statement's ``RETURNING`` projection over the rows it touched, or None.

    The projection is translated as ``SELECT <list> FROM rows AS <target alias>``, so it
    takes every expression a SELECT list does and a qualified ``t.col`` resolves.

    Args:
        node: The DML statement.
        rows: The rows the statement inserted, deleted, or updated (post-update values).
        registry: Every visible table name and its bound `Dataset`.
        functions: Registered Python functions.

    Returns:
        The projected relation, or None when the statement has no ``RETURNING``.
    """
    clause = node.args.get("returning")
    if clause is None:
        return None
    if clause.args.get("into"):
        raise PlanError("RETURNING ... INTO is not supported; the statement returns a relation")
    source = exp.Table(
        this=exp.to_identifier(_RETURNING),
        alias=exp.TableAlias(this=exp.to_identifier(target_alias(node.this))),
    )
    select = exp.select(*[e.copy() for e in clause.expressions]).from_(source)
    return translate_ast(select, functions=functions, **{**registry, _RETURNING: rows})
