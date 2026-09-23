"""The clauses that decide *how many rows per key* a correlated subquery yields.

Decorrelation turns `(SELECT … FROM u WHERE u.k = t.k …)` into a relation keyed on `u.k` and
joins it to the outer query. That is only sound for the clauses that act **per outer row**
once the correlation is pulled out. Four do not survive the move unchanged, and each one was
a wrong answer rather than a refusal before this module existed:

- ``ORDER BY … LIMIT n`` / ``OFFSET k`` applied to the *whole* inner relation, so
  ``(SELECT v … WHERE u.g = t.g ORDER BY v DESC LIMIT 1)`` kept one row for the entire table
  and every other key read NULL. Per key, it is a top-N: `row_number()` partitioned by the
  correlation columns. `top_n_window` builds that.
- An aggregate without ``GROUP BY`` yields exactly one row per outer row, even over no input
  rows, so ``EXISTS (SELECT count(*) … WHERE corr)`` is always TRUE and a ``HAVING`` decides
  it per key. The semi join answered "is there an input row" instead.
- The same one-row aggregate evaluated over an *empty* group — an outer key nothing matches —
  is not NULL in general: ``count(*) + 1`` is 1 and ``coalesce(sum(w), 0)`` is 0. The LEFT
  JOIN null-extends it. `empty_group_value` evaluates the expression the way SQL does.

Everything here is a pure AST reading or rewrite; the plans are built by `core` and
`scalar_sub`.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlglot import expressions as exp

from batcher._sql.parser.core_utils import _has_aggregate

__all__ = [
    "RANK_COLUMN",
    "ExistsPlan",
    "empty_group_value",
    "is_one_row_aggregate",
    "normalize_exists",
    "paging",
    "rank_predicate",
    "strip_paging",
    "top_n_window",
]

#: Hidden column the per-key top-N ranks rows under. `__bc_` keeps it out of `SELECT *`.
RANK_COLUMN = "__bc_topn_rn"

#: Aggregates whose value over zero rows is 0 rather than NULL.
_COUNT_LIKE = (exp.Count, exp.CountIf, exp.ApproxDistinct)
_COUNT_LIKE_NAMES = {"count", "count_star", "count_if", "countif", "approx_count_distinct"}


def _literal_int(node, clause: str) -> int:
    """The non-negative integer a ``LIMIT``/``OFFSET`` clause holds, refusing anything else."""
    value = node.args.get("expression") if isinstance(node, (exp.Limit, exp.Offset)) else node
    if isinstance(value, exp.Literal) and not value.is_string:
        try:
            n = int(value.this)
        except ValueError:
            n = -1
        if n >= 0:
            return n
    raise NotImplementedError(
        f"{clause} inside a correlated subquery must be a non-negative integer literal; "
        f"got {value.sql() if value is not None else 'nothing'}"
    )


def paging(inner) -> tuple[int | None, int]:
    """The ``(limit, offset)`` of a subquery's own SELECT; ``limit`` is None when absent."""
    limit = inner.args.get("limit")
    offset = inner.args.get("offset")
    return (
        _literal_int(limit, "LIMIT") if limit is not None else None,
        _literal_int(offset, "OFFSET") if offset is not None else 0,
    )


def strip_paging(inner) -> None:
    """Drop ``ORDER BY``/``LIMIT``/``OFFSET`` from `inner`, in place, once accounted for."""
    for key in ("order", "limit", "offset"):
        inner.set(key, None)


def is_one_row_aggregate(inner) -> bool:
    """Whether `inner` is an aggregate with no ``GROUP BY`` — one row per outer row, always.

    A ``HAVING`` with no aggregate in the select list still makes the query an aggregate
    (``SELECT 1 FROM u HAVING count(*) > 5``), so it counts.
    """
    if inner.args.get("group") is not None:
        return False
    return inner.args.get("having") is not None or any(_has_aggregate(e) for e in inner.expressions)


def empty_group_value(value) -> exp.Expression | None:
    """`value` evaluated over zero input rows, or None when that is simply NULL.

    SQL evaluates an ungrouped aggregate over an empty input to one row in which every
    ``count`` is 0 and every other aggregate is NULL, and then evaluates the rest of the
    expression over those. The substitution below is exactly that, so
    ``count(*) + 1`` becomes ``0 + 1`` and ``coalesce(sum(w), 0)`` becomes
    ``coalesce(NULL, 0)``; the translator evaluates what remains like any constant.

    Args:
        value: The subquery's single projected expression (unaliased).

    Returns:
        The substituted expression, or None when it reduces to a bare NULL — the common
        ``max(v)`` case, which the LEFT JOIN's null-extension already answers.
    """
    from batcher._sql.parser.expressions.aggregates import iter_agg_nodes

    out = value.copy()
    aggs = [a for a in iter_agg_nodes(out) if a.find_ancestor(exp.Window) is None]
    for agg in aggs:
        if agg is out:
            return exp.Literal.number(0) if _is_count_like(agg) else None
    for agg in aggs:
        agg.replace(exp.Literal.number(0) if _is_count_like(agg) else exp.Null())
    return None if isinstance(out, exp.Null) else out


def _is_count_like(node) -> bool:
    if isinstance(node, _COUNT_LIKE):
        return True
    return isinstance(node, exp.Anonymous) and node.name.lower() in _COUNT_LIKE_NAMES


def top_n_window(inner, partition_cols: list[str], select_items) -> exp.Expression:
    """`row_number() OVER (PARTITION BY <correlation columns> ORDER BY <inner ORDER BY>)`.

    The window the per-key top-N filters on. The inner ``ORDER BY`` may name a select-list
    alias or an ordinal, which a window cannot see, so both are replaced by the expression
    they stand for.

    Args:
        inner: The subquery SELECT, still carrying its ``ORDER BY``.
        partition_cols: The inner correlation columns.
        select_items: The subquery's original select list, for alias and ordinal lookup.

    Returns:
        The aliased window expression to add to the inner select list.
    """
    order = inner.args.get("order")
    window = exp.Window(
        this=exp.RowNumber(),
        partition_by=[exp.column(c) for c in partition_cols],
    )
    if order is not None:
        aliases = {e.alias: e.this for e in select_items if isinstance(e, exp.Alias) and e.alias}
        order = order.copy()
        for ordered in order.expressions:
            key = ordered.this
            if isinstance(key, exp.Literal) and not key.is_string:
                position = int(key.this) - 1
                if 0 <= position < len(select_items):
                    item = select_items[position]
                    ordered.set("this", (item.this if isinstance(item, exp.Alias) else item).copy())
            elif isinstance(key, exp.Column) and not key.table and key.name in aliases:
                ordered.set("this", aliases[key.name].copy())
        window.set("order", order)
    return exp.alias_(window, RANK_COLUMN)


def rank_predicate(limit: int | None, offset: int):
    """The `Expr` keeping ranks ``offset + 1 .. offset + limit`` of `top_n_window`."""
    from batcher.plan.expr_ir import col, lit

    keep = col(RANK_COLUMN) > lit(offset)
    if limit is not None:
        keep = keep & (col(RANK_COLUMN) <= lit(offset + limit))
    return keep


@dataclass(frozen=True)
class ExistsPlan:
    """How a correlated `EXISTS` is answered once its row-count clauses are read.

    Exactly one of the three fields is meaningful:

    - `constant` — the answer is the same for every outer row (an ungrouped aggregate is
      always one row; ``LIMIT 0`` is never any).
    - `predicate` — a boolean scalar-subquery AST that decides it per key (an ungrouped
      ``HAVING``, or an ``OFFSET``), for the scalar decorrelation to plan.
    - neither — the semi join applies to `inner`, now stripped of what it ignores. When
      `keep_group` is set the inner ``GROUP BY … HAVING`` must be kept and grouped by the
      correlation columns too, so a key qualifies only when one of its groups passes.
    """

    constant: bool | None = None
    predicate: exp.Expression | None = None
    keep_group: bool = False


def normalize_exists(inner, correlated: bool) -> ExistsPlan:
    """Read a correlated `EXISTS` subquery's ``LIMIT``/``OFFSET``/aggregate clauses.

    `inner` is rewritten in place: an ``ORDER BY`` never matters to `EXISTS`, and a
    ``LIMIT n`` with ``n >= 1`` does not either once the question is per key.

    Args:
        inner: The detached inner SELECT.
        correlated: Whether it references the outer query. An uncorrelated subquery is
            evaluated whole, where every clause already means what it says.

    Returns:
        The plan the caller follows.

    Raises:
        NotImplementedError: For an ``OFFSET`` over a ``DISTINCT``/``GROUP BY`` inner, whose
            per-key row count has no plan here.
    """
    if not correlated or not isinstance(inner, exp.Select):
        return ExistsPlan()
    limit, offset = paging(inner)
    strip_paging(inner)
    if limit == 0:
        return ExistsPlan(constant=False)
    one_row = is_one_row_aggregate(inner)
    if one_row and offset:
        return ExistsPlan(constant=False)
    having = inner.args.get("having")
    if one_row and having is None:
        return ExistsPlan(constant=True)
    if one_row:
        # Per key, the one aggregate row survives iff HAVING holds over that key's group —
        # including the empty group of an unmatched key, which the scalar path evaluates.
        probe = inner.copy()
        probe.set("having", None)
        probe.set("expressions", [having.this.copy()])
        subquery = exp.Subquery(this=probe)
        return ExistsPlan(predicate=exp.Coalesce(this=subquery, expressions=[exp.false()]))
    if offset:
        if inner.args.get("group") is not None or inner.args.get("distinct") is not None:
            raise NotImplementedError(
                "OFFSET inside a correlated EXISTS over DISTINCT or GROUP BY is not supported; "
                "count the groups in a scalar subquery instead: "
                "(SELECT count(*) FROM (…) WHERE …) > n"
            )
        probe = inner.copy()
        probe.set("expressions", [exp.Count(this=exp.Star())])
        subquery = exp.Subquery(this=probe)
        return ExistsPlan(predicate=exp.GT(this=subquery, expression=exp.Literal.number(offset)))
    if inner.args.get("group") is not None:
        return ExistsPlan(keep_group=having is not None)
    return ExistsPlan()
