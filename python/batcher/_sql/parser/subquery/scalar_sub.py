"""Correlated scalar subqueries as keyed LEFT JOINs.

A scalar subquery is a relation that must yield **at most one row** for each outer row, read
as a value. `(SELECT max(v) FROM u WHERE u.k = t.k)` becomes a LEFT JOIN with
`SELECT k, max(v) FROM u GROUP BY k`, keyed on the correlation, so it runs when the query
runs. An uncorrelated one is not planned here: `expressions.scalar._scalar_subquery`
evaluates it while translating and inlines the value as a literal (see the note in
`decorrelate_scalar_subqueries` for why a join is not used instead).

Three rules make the per-key relation mean what SQL says (see `shape` for the reading of the
clauses themselves):

1. **One row per key or an error.** DuckDB raises "More than one row returned by a subquery"
   rather than picking one. A relation that can hold several rows per key is checked when it
   runs (`_single_row_per_key`); the old plan deduplicated with `DISTINCT` and LEFT JOINed,
   which *multiplied* the outer rows instead.
2. **LIMIT/OFFSET/ORDER BY per key.** A top-N inside the subquery ranks rows within each
   correlation key, never across the whole inner table.
3. **The empty group has a value.** An outer key nothing matches still sees an ungrouped
   aggregate evaluate over zero rows, so `count(*) + 1` is 1 there, not NULL.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._internal.errors import ExecutionError
from batcher._sql.parser.core_utils import (
    _factor_common_conjuncts,
    _join_and,
    _split_and,
)
from batcher._sql.parser.subquery import shape
from batcher._sql.parser.subquery.correlation import (
    _correlation_pair,
    _local_columns,
    _local_tables,
    _outer_key_reducer,
    _reject_correlated,
)
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import col, lit, nullif, when

__all__ = ["MULTIPLE_ROWS_MESSAGE", "decorrelate_scalar_subqueries"]

#: DuckDB's wording, so a ported query fails with the message its author already knows.
MULTIPLE_ROWS_MESSAGE = (
    "More than one row returned by a subquery used as an expression - "
    "scalar subqueries can only return a single row."
)

_ROWS_COLUMN = "__bc_subq_rows"


def decorrelate_scalar_subqueries(tr, ds: Dataset, roots, outer_node=None) -> Dataset:
    """Replace every correlated scalar subquery under `roots` by a column joined onto `ds`.

    Args:
        tr: The translator.
        ds: The outer relation, before its WHERE residual, projection or HAVING.
        roots: The expressions to search — the SELECT list, the WHERE residual, HAVING.
        outer_node: The enclosing SELECT, for the semi-join key reducer.

    Returns:
        `ds` with one extra column per correlated subquery; each such node now names it.
    """
    for root in roots:
        if root is None:
            continue
        for sub in list(root.find_all(exp.Subquery)):
            inner = sub.this
            if not isinstance(inner, exp.Select) or _is_predicate_operand(sub):
                continue
            corr, local_preds = _split_correlation(tr, inner)
            if corr:
                ds = _correlated(tr, ds, sub, corr, local_preds, outer_node)
            # An uncorrelated one is left to `_scalar`, which evaluates it at translation time
            # and inlines the value: joining its one row on a constant key measured ~60x
            # slower than filtering on the literal (1.2 s against 20 ms over 4M rows).
    return ds


def _is_predicate_operand(sub) -> bool:
    """Whether `sub` is the set of an ``IN``/``EXISTS``/quantified test rather than a value."""
    return isinstance(sub.parent, (exp.In, exp.Exists, exp.Any, exp.All))


def _split_correlation(tr, inner) -> tuple[list[tuple[str, str]], list]:
    """The `(outer, inner)` equality pairs correlating `inner`, and its local predicates."""
    local = _local_tables(inner)
    local_cols = _local_columns(tr, inner)
    where = inner.args.get("where")
    corr, local_preds = [], []
    if where is not None:
        # A correlation repeated inside every arm of an `OR` is still a correlation;
        # factoring it back out is what lets it be seen at all (TPC-DS q41).
        for leaf in _split_and(_factor_common_conjuncts(where.this)):
            pair = _correlation_pair(leaf, local, local_cols)
            (corr if pair is not None else local_preds).append(pair or leaf)
    return corr, local_preds


def _next_names(tr, n_keys: int) -> tuple[str, list[str], str]:
    n = tr._scalar_sub_n
    tr._scalar_sub_n += 1
    return f"__bc_scalar_{n}", [f"__bc_jk_{n}_{i}" for i in range(n_keys)], f"__bc_sm_{n}"


def _correlated(tr, ds: Dataset, sub, corr, local_preds, outer_node) -> Dataset:
    """Decorrelate one equality-correlated scalar subquery into a keyed LEFT JOIN."""
    inner = sub.this
    if len(inner.expressions) != 1:
        raise NotImplementedError("scalar subquery must project one value")
    alias, jk, matched = _next_names(tr, len(corr))
    ics = [ic for (_oc, ic) in corr]

    m = inner.copy()
    items = list(m.expressions)
    value = items[0].this if isinstance(items[0], exp.Alias) else items[0]
    m.set("where", exp.Where(this=_join_and(local_preds)) if local_preds else None)
    limit, offset = shape.paging(m)
    projected = [exp.alias_(exp.column(ic), k) for k, ic in zip(jk, ics, strict=True)]
    projected.append(exp.alias_(value, alias))

    top_n = empty = passed = None
    check = False
    if shape.is_one_row_aggregate(m):
        if limit == 0 or offset:
            sub.replace(exp.Null())  # the one row is paged away for every key
            return ds
        having = m.args.get("having")
        empty = _empty_value(value, having)
        if having is not None:
            # HAVING drops a key's one row; carried as a column instead, so a key that has
            # rows but fails HAVING (NULL) stays distinguishable from a key with no rows
            # (the empty group's value).
            passed = f"{alias}_having"
            projected.append(exp.alias_(having.this.copy(), passed))
            m.set("having", None)
        m.set("group", exp.Group(expressions=[exp.column(ic) for ic in ics]))
        _reduce_keys(tr, m, outer_node, sub, corr)
    else:
        if limit == 0:
            sub.replace(exp.Null())
            return ds
        group = m.args.get("group")
        if group is not None:
            keys = [exp.column(ic) for ic in ics]
            m.set("group", exp.Group(expressions=[*keys, *group.expressions]))
        if limit is not None or offset:
            if m.args.get("order") is None:
                # Any row is a correct answer to an unordered LIMIT; ranking needs an order,
                # and ordering by the value itself at least makes the choice repeatable.
                m.set("order", exp.Order(expressions=[exp.Ordered(this=value.copy())]))
            projected.append(shape.top_n_window(m, ics, items))
            top_n = (limit, offset)
        check = not (top_n is not None and limit == 1)
    shape.strip_paging(m)
    m.set("expressions", projected)
    _reject_correlated(m)

    derived = tr.statement(m)
    if top_n is not None:
        derived = derived.filter(shape.rank_predicate(*top_n)).drop(shape.RANK_COLUMN)
    if check:
        derived = _single_row_per_key(derived, jk)
    if empty is not None:
        derived = derived.with_columns(**{matched: lit(True)})
    ds = ds.join(derived, left_on=[oc for (oc, _ic) in corr], right_on=jk, how="left")
    leftover = [k for k in jk if k in ds.columns]
    if leftover:
        ds = ds.drop(*leftover)
    if passed is not None:
        typed_null = nullif(col(alias), col(alias))
        ds = ds.with_columns(**{alias: when(col(passed)).then(col(alias)).otherwise(typed_null)})
        ds = ds.drop(passed)
    if empty is not None:
        ds = _fill_unmatched(tr, ds, derived, alias, matched, empty)
    sub.replace(exp.column(alias))
    return ds


def _empty_value(value, having) -> exp.Expression | None:
    """What the ungrouped aggregate yields for a key with no rows, or None for NULL."""
    empty = shape.empty_group_value(value)
    if empty is None or having is None:
        return empty
    passes = shape.empty_group_value(having.this)
    if passes is None:
        return None  # HAVING is NULL over the empty group, so the one row is dropped
    return exp.case().when(passes, empty)


def _reduce_keys(tr, m, outer_node, sub, corr) -> None:
    """Pre-filter a per-key aggregate to the keys the outer query can produce."""
    reducer = _outer_key_reducer(tr, outer_node, sub, corr)
    if reducer is None:
        return
    ics = [exp.column(ic) for (_oc, ic) in corr]
    lhs = ics[0] if len(ics) == 1 else exp.Tuple(expressions=ics)
    in_pred = exp.In(this=lhs, query=reducer)
    cur = m.args.get("where")
    m.set("where", exp.Where(this=exp.and_(cur.this, in_pred) if cur is not None else in_pred))


def _fill_unmatched(tr, ds: Dataset, derived: Dataset, alias: str, matched: str, empty):
    """Give an unmatched outer row the aggregate's empty-group value instead of NULL."""
    fill = tr._scalar(empty)
    schema = derived._plan.available_schema()
    dtype = None
    if schema is not None:
        dtype = next((f.type for f in schema.arrow if f.name == alias), None)
    if dtype is not None:
        fill = fill.cast(dtype)
    value = when(col(matched).is_null()).then(fill).otherwise(col(alias))
    return ds.with_columns(**{alias: value}).drop(matched)


def _single_row_per_key(derived: Dataset, keys: list[str]) -> Dataset:
    """`derived`, raising when it runs if any key holds more than one row.

    The count is a window over the keys, so the check needs no second pass and no join; the
    raise happens in a batch callback that reads one column per batch, never a row loop.
    """
    counter = lit(1).count()
    counted = derived.with_columns(
        **{_ROWS_COLUMN: counter.over(partition_by=keys) if keys else counter.over()}
    )
    return counted.map_batches(_raise_on_multiple_rows).drop(_ROWS_COLUMN)


def _raise_on_multiple_rows(batch):
    """Batch callback for `_single_row_per_key`: pass the batch through or raise."""
    import pyarrow.compute as pc

    if batch.num_rows and (pc.max(batch.column(_ROWS_COLUMN)).as_py() or 0) > 1:
        raise ExecutionError(MULTIPLE_ROWS_MESSAGE)
    return batch
