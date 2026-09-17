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

__all__ = ["align_insert", "apply_dml"]

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
    if isinstance(node, exp.Merge):
        return _merge(node, registry, functions)
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
    aligned = align_insert(name, current, new_rows, columns)
    return name, current.union(aligned, distinct=False)


def align_insert(
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


def _merge(node: Any, registry: _Registry, functions: dict[str, Any]) -> tuple[str, Dataset]:
    """Rewrite ``MERGE INTO`` into the target's new state, through the engine's own merge.

    The lakehouse DML statement, and the one Delta, Snowflake and Databricks are all written
    against. Nothing here implements merge semantics: `api.merge.compose_merge` already
    composes a target's post-merge state as one lazy relation, which is what both the
    single-node and the distributed merge execute. This translates the statement into the
    clause objects that function already takes, so the SQL spelling and
    `write.merge_into(...)` are the same engine and cannot disagree.

    Args:
        node: The `exp.Merge` node.
        registry: Every visible table name and its bound `Dataset`.
        functions: Registered Python functions, for expressions in the clause bodies.

    Returns:
        The target's name and its new state.

    Raises:
        PlanError: If the `ON` condition is not a conjunction of ``target.k = source.k``
            equalities on matching column names, or a clause names an action the engine has
            no form for.
    """
    from batcher.api.merge.compose import compose_merge

    target_node, source_node = node.this, node.args.get("using")
    name = _target_name(target_node)
    target = _require_target(name, registry)
    target_alias = (target_node.alias or target_node.name) if target_node else name
    source_alias = (source_node.alias or source_node.name) if source_node else ""

    tr = _Translator(dict(registry), functions)
    source = _merge_source(source_node, tr, registry)
    keys = _merge_keys(node.args.get("on"), target_alias, source_alias)

    clauses = [
        _merge_clause(when_node, tr, target_alias, source_alias) for when_node in _merge_whens(node)
    ]
    if not clauses:
        raise PlanError("MERGE needs at least one WHEN clause.")
    return name, compose_merge(source, target, keys, clauses)


def _merge_source(source_node: Any, tr: Any, registry: _Registry) -> Dataset:
    """The relation a MERGE's ``USING`` names: a registered table or a subquery.

    A bare table arrives as `exp.Table`, which the statement translator does not serve --
    it translates *queries*, and a table reference is resolved inside a FROM clause. So the
    common form is a registry lookup and the general one falls through to the translator.
    """
    if isinstance(source_node, exp.Subquery):
        return tr.statement(source_node.this)
    if isinstance(source_node, exp.Table) and not isinstance(source_node.this, exp.Anonymous):
        return _require_target(source_node.name, registry)
    return tr.statement(source_node)


def _merge_whens(node: Any) -> list[Any]:
    """The ``WHEN`` clauses, from either sqlglot shape (a `Whens` wrapper or a bare list)."""
    whens = node.args.get("whens")
    if whens is None:
        return []
    return list(getattr(whens, "expressions", None) or whens)


def _merge_keys(on: Any, target_alias: str, source_alias: str) -> list[str]:
    """The equi-key columns in a MERGE ``ON``, or a `PlanError` naming what is unsupported.

    The engine matches a source row to a target row by column *name*, so the condition has
    to be a conjunction of ``t.k = s.k`` on the same name on both sides. A join on differing
    names, or any non-equality, has no key to be expressed as -- and quietly picking one
    would merge on a condition the user did not write.
    """
    if on is None:
        raise PlanError("MERGE requires an ON condition.")
    conjuncts, stack = [], [on]
    while stack:
        current = stack.pop()
        if isinstance(current, exp.And):
            stack.extend([current.this, current.expression])
        elif isinstance(current, exp.Paren):
            stack.append(current.this)
        else:
            conjuncts.append(current)
    keys: list[str] = []
    for conjunct in conjuncts:
        left, right = getattr(conjunct, "this", None), getattr(conjunct, "expression", None)
        if (
            not isinstance(conjunct, exp.EQ)
            or not isinstance(left, exp.Column)
            or not isinstance(right, exp.Column)
            or left.name != right.name
            or {left.table, right.table} != {target_alias, source_alias}
        ):
            raise PlanError(
                f"MERGE ON must be equalities of the form {target_alias}.k = {source_alias}.k "
                "on the same column name; this engine matches rows by key column.",
                hint="Rename the columns to match, or filter the source before merging.",
            )
        keys.append(left.name)
    return list(dict.fromkeys(keys))


def _merge_clause(when_node: Any, tr: Any, target_alias: str, source_alias: str) -> Any:
    """One sqlglot ``WHEN`` as a `MergeClause`."""
    from batcher.api.merge.clauses import MergeClause

    matched = bool(when_node.args.get("matched"))
    by_source = bool(when_node.args.get("source"))
    kind = "matched" if matched else ("not_matched_by_source" if by_source else "not_matched")
    condition_node = when_node.args.get("condition")
    condition = (
        tr._scalar(_qualify_for_merge(condition_node, target_alias, source_alias))
        if condition_node is not None
        else None
    )
    then = when_node.args.get("then")
    if isinstance(then, exp.Update):
        values = _merge_assignments(then, tr, target_alias, source_alias)
        return MergeClause(kind, "update", condition, values)
    if isinstance(then, exp.Insert):
        return MergeClause(
            kind, "insert", condition, _merge_insert(then, tr, target_alias, source_alias)
        )
    if str(getattr(then, "name", then)).upper() == "DELETE" or isinstance(then, exp.Delete):
        return MergeClause(kind, "delete", condition, None)
    raise PlanError(
        f"MERGE clause action {type(then).__name__} is not supported; "
        "use UPDATE SET, INSERT, or DELETE."
    )


def _merge_assignments(
    then: Any, tr: Any, target_alias: str, source_alias: str
) -> dict[str, Expr] | None:
    """``UPDATE SET a = x, b = y`` as target-column to expression; None for ``SET *``."""
    assignments: dict[str, Expr] = {}
    for eq in then.args.get("expressions") or []:
        if isinstance(eq, exp.Star):
            return None
        assignments[eq.this.name] = tr._scalar(
            _qualify_for_merge(eq.expression, target_alias, source_alias)
        )
    return assignments or None


def _merge_insert(
    then: Any, tr: Any, target_alias: str, source_alias: str
) -> dict[str, Expr] | None:
    """``INSERT (a, b) VALUES (x, y)`` as target-column to expression; None for ``INSERT *``."""
    columns_node = then.this
    values_node = then.expression
    if columns_node is None and values_node is None:
        return None  # INSERT * / INSERT DEFAULT VALUES: take every column from the source
    names = [c.name for c in getattr(columns_node, "expressions", [])] if columns_node else []
    values = []
    for tuple_node in getattr(values_node, "expressions", []) or []:
        values.extend(getattr(tuple_node, "expressions", []) or [tuple_node])
    if not names or len(names) != len(values):
        raise PlanError(
            "MERGE ... INSERT needs a column list and a VALUES list of the same length.",
            hint="Write INSERT (a, b) VALUES (s.a, s.b), or INSERT * to take every column.",
        )
    return {
        name: tr._scalar(_qualify_for_merge(value, target_alias, source_alias))
        for name, value in zip(names, values, strict=True)
    }


def _qualify_for_merge(node: Any, target_alias: str, source_alias: str) -> Any:
    """Rewrite `s.col` to the reserved source name and `t.col` to a bare column.

    `compose_merge` joins the two sides into one relation and moves every source column to
    a reserved name, so both sides coexist unshadowed. A clause body written in SQL names
    them by alias; this is the translation between the two, done on the sqlglot tree so the
    ordinary expression lowering below it needs to know nothing about merges.
    """
    from batcher.api.merge.clauses import source_name

    rewritten = node.copy()
    for column in rewritten.find_all(exp.Column):
        table = column.table
        if table == source_alias and source_alias:
            column.set("this", exp.to_identifier(source_name(column.name)))
            column.set("table", None)
        elif table == target_alias:
            column.set("table", None)
    if isinstance(rewritten, exp.Column):
        table = rewritten.table
        if table == source_alias and source_alias:
            rewritten.set("this", exp.to_identifier(source_name(rewritten.name)))
            rewritten.set("table", None)
        elif table == target_alias:
            rewritten.set("table", None)
    return rewritten
