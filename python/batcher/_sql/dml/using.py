"""``DELETE ... USING`` rewritten to the ``EXISTS`` it means.

``DELETE FROM t USING s WHERE t.id = s.id`` deletes every target row for which *some*
combination of the ``USING`` relations satisfies the predicate. That is a semi-join, not a
join: a target row matched by three source rows is still deleted once. So the statement is
rewritten on the sqlglot tree to

    SELECT * FROM t WHERE NOT EXISTS (SELECT 1 FROM s WHERE t.id = s.id)

for the rows that survive, and the same ``EXISTS`` without the ``NOT`` for the rows
removed, and both go through the ordinary SELECT translator. The correlated ``EXISTS``
decorrelation there is what answers it, so a predicate it cannot decorrelate gets that
translator's precise refusal rather than a second, weaker implementation here.

A conjunct that reads only the target (``t.status = 'x'``) is kept *outside* the ``EXISTS``:
it does not correlate anything, and the decorrelation accepts only column equalities as
the correlation.
"""

from __future__ import annotations

from typing import Any

from sqlglot import expressions as exp

from batcher._sql import translate_ast
from batcher.api.dataset import Dataset

__all__ = ["delete_using", "split_conjuncts"]


def split_conjuncts(node: Any) -> list[Any]:
    """The operands of a (possibly parenthesized) chain of ``AND``s, in written order.

    Args:
        node: A sqlglot boolean expression.

    Returns:
        Its conjuncts; a single-element list when `node` is not an ``AND``.
    """
    conjuncts, stack = [], [node]
    while stack:
        current = stack.pop()
        if isinstance(current, exp.And):
            stack.extend([current.expression, current.this])
        elif isinstance(current, exp.Paren):
            stack.append(current.this)
        else:
            conjuncts.append(current)
    return conjuncts


def delete_using(
    node: Any,
    name: str,
    current: Dataset,
    registry: dict[str, Dataset],
    functions: dict[str, Any],
) -> tuple[Dataset, Dataset]:
    """The rows a ``DELETE ... USING`` keeps and the rows it removes.

    Args:
        node: The `exp.Delete` node, carrying ``using``.
        name: The target's registry name.
        current: The target's current state.
        registry: Every visible table name and its bound `Dataset`.
        functions: Registered Python functions.

    Returns:
        ``(kept, deleted)``, each with exactly the target's columns.
    """
    alias = node.this.alias or node.this.name
    using = _relations(node.args["using"])
    where = node.args.get("where")
    conjuncts = split_conjuncts(where.this.copy()) if where is not None else []

    target_cols = set(current.columns)
    using_cols = _using_columns(using, registry)
    local: list[Any] = []
    correlated: list[Any] = []
    for conjunct in conjuncts:
        if _reads_only_target(conjunct, alias, target_cols, using_cols):
            local.append(conjunct)
        else:
            _qualify_target_columns(conjunct, alias, target_cols, using_cols)
            correlated.append(conjunct)

    probe = exp.select(exp.Literal.number(1)).from_(using[0])
    for extra in using[1:]:
        probe = probe.join(extra)
    if correlated:
        probe = probe.where(exp.and_(*correlated))
    exists = exp.Exists(this=probe)
    local_pred = exp.and_(*local) if local else None

    removed = exists if local_pred is None else exp.and_(local_pred.copy(), exists.copy())
    # Kept = NOT (local AND EXISTS) under three-valued logic: a NULL local predicate does not
    # delete, so it reads as false here (COALESCE), and EXISTS itself is never NULL.
    kept = exp.Not(this=exists.copy())
    if local_pred is not None:
        settled = exp.Coalesce(this=exp.paren(local_pred), expressions=[exp.false()])
        not_local = exp.Not(this=settled)
        kept = exp.or_(not_local, kept)
    return _select(name, alias, kept, registry, functions), _select(
        name, alias, removed, registry, functions
    )


def _select(
    name: str, alias: str, condition: Any, registry: dict[str, Dataset], functions: dict
) -> Dataset:
    """``SELECT <alias>.* FROM name AS alias WHERE condition``, translated."""
    table = exp.Table(
        this=exp.to_identifier(name), alias=exp.TableAlias(this=exp.to_identifier(alias))
    )
    select = exp.select(exp.Column(this=exp.Star(), table=exp.to_identifier(alias)))
    select = select.from_(table).where(condition)
    return translate_ast(select, functions=functions, **registry)


def _relations(using: list[Any]) -> list[Any]:
    """The ``USING`` list as one flat list of relations.

    The parser hangs ``USING s, u`` off the first relation as a comma join (``s`` carrying
    ``joins=[u]``). A FROM clause states that join on the SELECT itself, so the relations
    are unhooked here and joined there.
    """
    flat: list[Any] = []
    for node in using:
        node = node.copy()
        joins = node.args.get("joins") or []
        node.set("joins", None)
        flat.append(node)
        flat.extend(j.this for j in joins)
    return flat


def _using_columns(using: list[Any], registry: dict[str, Dataset]) -> set[str] | None:
    """Every column the ``USING`` relations expose, or None when one is not a plain table.

    None makes every unqualified column ambiguous, which keeps a conjunct inside the
    ``EXISTS`` where the translator resolves it, rather than guessing it is the target's.
    """
    names: set[str] = set()
    for table in using:
        if not isinstance(table, exp.Table) or table.name not in registry:
            return None
        names.update(registry[table.name].columns)
    return names


def _reads_only_target(
    conjunct: Any, alias: str, target_cols: set[str], using_cols: set[str] | None
) -> bool:
    """Whether every column in `conjunct` is the target's, and it holds no subquery."""
    if conjunct.find(exp.Select, exp.Exists):
        return False
    columns = list(conjunct.find_all(exp.Column))
    return bool(columns) and all(
        _is_target_column(c, alias, target_cols, using_cols) for c in columns
    )


def _is_target_column(
    column: Any, alias: str, target_cols: set[str], using_cols: set[str] | None
) -> bool:
    if column.table:
        return column.table == alias
    return using_cols is not None and column.name in target_cols and column.name not in using_cols


def _qualify_target_columns(
    conjunct: Any, alias: str, target_cols: set[str], using_cols: set[str] | None
) -> None:
    """Qualify the unqualified target columns in a correlated conjunct with the target alias.

    Inside the ``EXISTS`` an unqualified name resolves against the subquery's own relations
    first, so a target-only column has to say which relation it belongs to.
    """
    for column in conjunct.find_all(exp.Column):
        if not column.table and _is_target_column(column, alias, target_cols, using_cols):
            column.set("table", exp.to_identifier(alias))
