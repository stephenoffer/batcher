"""``MERGE INTO`` translated into the clause objects `api.merge.compose_merge` takes.

Nothing here implements merge semantics. The statement's ``ON`` becomes key pairs, each
``WHEN`` becomes a `MergeClause`, and the engine's own merge composes the target's new
state, so the SQL spelling and `write.merge_into(...)` cannot disagree. ``INSERT ... ON
CONFLICT`` reuses the same alias rewrite (`qualify_for_merge`), with ``excluded`` as the
source alias.
"""

from __future__ import annotations

from typing import Any

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._sql.dml.rewrite import require_target, target_name
from batcher._sql.dml.using import split_conjuncts
from batcher._sql.parser.translator import _Translator
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import Expr

__all__ = ["merge", "qualify_for_merge"]

_Registry = dict[str, Dataset]


def merge(node: Any, registry: _Registry, functions: dict[str, Any]) -> tuple[str, Dataset]:
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
        PlanError: If the `ON` condition is not a conjunction of ``target.a = source.b``
            column equalities, a clause names an action the engine has no form for, or the
            statement carries ``RETURNING``.
    """
    from batcher.api.merge.compose import compose_merge

    if node.args.get("returning"):
        raise PlanError(
            "MERGE ... RETURNING is not supported: the merge composes the new table state "
            "without recording which clause produced each row.",
            hint="Query the target after the MERGE, or use INSERT/UPDATE/DELETE ... RETURNING.",
        )
    target_node, source_node = node.this, node.args.get("using")
    name = target_name(target_node)
    target = require_target(name, registry)
    target_alias = (target_node.alias or target_node.name) if target_node else name
    source_alias = (source_node.alias or source_node.name) if source_node else ""

    tr = _Translator(dict(registry), functions)
    source = _merge_source(source_node, tr, registry)
    pairs = _merge_keys(node.args.get("on"), target_alias, source_alias)

    clauses = [
        _merge_clause(when_node, tr, target_alias, source_alias) for when_node in _merge_whens(node)
    ]
    if not clauses:
        raise PlanError("MERGE needs at least one WHEN clause.")
    keys = [t for t, _ in pairs]
    return name, compose_merge(source, target, keys, clauses, source_keys=[s for _, s in pairs])


def _merge_source(source_node: Any, tr: Any, registry: _Registry) -> Dataset:
    """The relation a MERGE's ``USING`` names: a registered table or a subquery.

    A bare table arrives as `exp.Table`, which the statement translator does not serve --
    it translates *queries*, and a table reference is resolved inside a FROM clause. So the
    common form is a registry lookup and the general one falls through to the translator.
    """
    if isinstance(source_node, exp.Subquery):
        return tr.statement(source_node.this)
    if isinstance(source_node, exp.Table) and not isinstance(source_node.this, exp.Anonymous):
        return require_target(source_node.name, registry)
    return tr.statement(source_node)


def _merge_whens(node: Any) -> list[Any]:
    """The ``WHEN`` clauses, from either sqlglot shape (a `Whens` wrapper or a bare list)."""
    whens = node.args.get("whens")
    if whens is None:
        return []
    return list(getattr(whens, "expressions", None) or whens)


def _merge_keys(on: Any, target_alias: str, source_alias: str) -> list[tuple[str, str]]:
    """The ``(target column, source column)`` key pairs in a MERGE ``ON``.

    The condition has to be a conjunction of column equalities, each with one side on the
    target and the other on the source: ``t.id = s.id`` or ``t.id = s.customer_id``, in
    either order. Any non-equality, or an equality that does not span the two sides, has no
    key to be expressed as -- and quietly picking one would merge on a condition the user
    did not write.
    """
    if on is None:
        raise PlanError("MERGE requires an ON condition.")
    pairs: list[tuple[str, str]] = []
    for conjunct in split_conjuncts(on):
        left, right = getattr(conjunct, "this", None), getattr(conjunct, "expression", None)
        if (
            not isinstance(conjunct, exp.EQ)
            or not isinstance(left, exp.Column)
            or not isinstance(right, exp.Column)
            or {left.table, right.table} != {target_alias, source_alias}
        ):
            raise PlanError(
                f"MERGE ON must be equalities of the form {target_alias}.a = {source_alias}.b; "
                "this engine matches rows by key columns.",
                hint="Filter the source in a USING (SELECT ...) subquery for other conditions.",
            )
        if left.table != target_alias:
            left, right = right, left
        pairs.append((left.name, right.name))
    return list(dict.fromkeys(pairs))


def _merge_clause(when_node: Any, tr: Any, target_alias: str, source_alias: str) -> Any:
    """One sqlglot ``WHEN`` as a `MergeClause`."""
    from batcher.api.merge.clauses import MergeClause

    matched = bool(when_node.args.get("matched"))
    by_source = bool(when_node.args.get("source"))
    kind = "matched" if matched else ("not_matched_by_source" if by_source else "not_matched")
    condition_node = when_node.args.get("condition")
    condition = (
        tr._scalar(qualify_for_merge(condition_node, target_alias, source_alias))
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
            qualify_for_merge(eq.expression, target_alias, source_alias)
        )
    return assignments or None


def _merge_insert(
    then: Any, tr: Any, target_alias: str, source_alias: str
) -> dict[str, Expr] | None:
    """``INSERT (a, b) VALUES (x, y)`` as target-column to expression; None for ``INSERT *``."""
    columns_node = then.this
    values_node = then.expression
    if values_node is None and (columns_node is None or isinstance(columns_node, exp.Star)):
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
        name: tr._scalar(qualify_for_merge(value, target_alias, source_alias))
        for name, value in zip(names, values, strict=True)
    }


def qualify_for_merge(node: Any, target_alias: str, source_alias: str) -> Any:
    """Rewrite `s.col` to the reserved source name and `t.col` to a bare column.

    `compose_merge` joins the two sides into one relation and moves every source column to
    a reserved name, so both sides coexist unshadowed. A clause body written in SQL names
    them by alias; this is the translation between the two, done on the sqlglot tree so the
    ordinary expression lowering below it needs to know nothing about merges.

    Args:
        node: A clause body expression.
        target_alias: The name the statement gives the target.
        source_alias: The name the statement gives the source (``excluded`` for ``ON
            CONFLICT``).

    Returns:
        A rewritten copy of `node`.
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
