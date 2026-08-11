"""Aggregate pre-pass rewrites for the SQL translator.

Two rewrites that reshape the *input* so an aggregate the engine cannot express directly
becomes one it can: DISTINCT aggregates (dedup first) and ordered aggregates (sort first).

`<agg>(DISTINCT x)` per group is `<agg>(x)` over the rows left once duplicate `x` values
are removed *within* each group. Two shapes are handled:

* **Only distinct aggregates**, all over one expression — dedup on `(group keys, x)` once
  up front and every aggregate becomes an ordinary one.
* **Distinct mixed with plain aggregates** — a two-level aggregate. Grouping by
  `(group keys, x)` dedups `x` implicitly while pre-aggregating the plain aggregates into
  mergeable partials; the second level then aggregates `x` directly and *combines* those
  partials. Deliberately **not** implemented as two aggregates joined on the group keys:
  a join drops rows whose key is NULL, which would silently lose the NULL group that
  `GROUP BY` legitimately produces.

Split from `grouping.py` because it is a self-contained rewrite with its own correctness
argument — and because that module is at its size limit.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import AggExpr, col

__all__ = ["rewrite_distinct_aggs", "sort_for_ordered_aggs"]

# Plain aggregates that survive pre-aggregation, as {level-1 partial: level-2 combine}.
#
# The condition is that the aggregate has a **single-column mergeable partial**: the level-1
# value computed per sub-group can be combined into the group's true answer by one aggregate
# over one column. `count` is the interesting one, and the reason this is a map rather than a
# set: a group's total is the SUM of its sub-groups' counts, not their count.
#
# The rest combine with *themselves*, because level 1 partitions the group's rows — every row
# lands in exactly one sub-group — and each of these operations is associative and commutative
# over that partition. `min`/`max`/`sum`/`product` and the bitwise folds are associative
# outright; `bit_xor` needs the partition to be exact (a row counted twice would cancel) and it
# is; `bool_and`/`bool_or` are idempotent as well, so they hold regardless. `any_value` holds
# because any value of any sub-group is a value of the group, which is all it promises.
# NULL handling needs no special case: a sub-group with nothing to aggregate yields NULL and
# the level-2 aggregate skips it, exactly as the one-level form skips the same rows.
#
# Anything absent here — `mean`, `stddev`, `var`, the quantiles, `count_distinct` — has no
# single-column partial (a mean needs a sum *and* a count) and is rejected rather than
# approximated.
_DECOMPOSABLE = {
    "count": "sum",
    "count_star": "sum",
    "sum": "sum",
    "min": "min",
    "max": "max",
    "any_value": "any_value",
    "bool_and": "bool_and",
    "bool_or": "bool_or",
    "bit_and": "bit_and",
    "bit_or": "bit_or",
    "bit_xor": "bit_xor",
    "product": "product",
}


def rewrite_distinct_aggs(
    tr, ds: Dataset, group_cols, group_exprs, agg_kwargs: dict[str, AggExpr]
) -> tuple[Dataset, dict[str, AggExpr]]:
    """Rewrite a query containing DISTINCT aggregates into an equivalent plain one.

    Args:
        tr: The translator, carrying the distinct expressions collected during registration.
        ds: The dataset to aggregate.
        group_cols: Plain column group keys.
        group_exprs: Computed group keys, by output alias.
        agg_kwargs: Every aggregate in the query, by output name.

    Returns:
        The dataset to group and the aggregates to apply to it. The caller groups by
        `group_cols` plus the *aliases* of `group_exprs`, which this has materialized.
    """
    distinct_exprs = {sql for sql, _ in tr._agg_distinct.values()}
    if len(distinct_exprs) > 1:
        raise NotImplementedError(
            "aggregates with two different DISTINCT expressions in one query are not "
            f"supported (got {sorted(distinct_exprs)}); each needs its own dedup pass, "
            "so compute them in separate subqueries and join"
        )

    dv = "__distinct_agg_v"
    node = next(iter(tr._agg_distinct.values()))[1]
    keys = [*group_cols, *group_exprs]
    ds = ds.with_columns(**{dv: tr._scalar(node), **group_exprs})

    plain = {name: a for name, a in agg_kwargs.items() if name not in tr._agg_distinct}
    if not plain:
        # Every aggregate is DISTINCT over the same expression: one dedup, then each
        # aggregate reads the deduped column.
        deduped = ds.select(*keys, dv).distinct()
        return deduped, {
            name: AggExpr(a.func, col(dv), input2=a.input2, param=a.param)
            for name, a in agg_kwargs.items()
        }

    undecomposable = sorted(n for n, a in plain.items() if a.func not in _DECOMPOSABLE)
    if undecomposable:
        funcs = sorted({plain[n].func for n in undecomposable})
        raise NotImplementedError(
            f"mixing a DISTINCT aggregate with {funcs} in one query is not supported: "
            "those have no single-column mergeable partial, so they cannot be "
            "pre-aggregated alongside the DISTINCT dedup. Compute them in a separate "
            "subquery"
        )

    # Level 1: group by the keys PLUS the distinct expression. That grouping dedups `x`
    # implicitly, and each plain aggregate becomes its per-sub-group partial.
    level1 = ds.group_by(*keys, dv).agg(
        **{
            name: AggExpr(a.func, a.input, input2=a.input2, param=a.param)
            for name, a in plain.items()
        }
    )
    # Level 2 is the caller's group-by. Distinct aggregates read `dv` (now unique within a
    # group); plain ones combine their partials — every level-1 row belongs to exactly one
    # sub-group, so summing the sub-counts is the true total.
    level2 = {
        name: AggExpr(a.func, col(dv), input2=a.input2, param=a.param)
        for name, a in agg_kwargs.items()
        if name in tr._agg_distinct
    }
    level2.update({name: AggExpr(_DECOMPOSABLE[a.func], col(name)) for name, a in plain.items()})
    return level1, level2


def sort_for_ordered_aggs(tr, ds: Dataset) -> Dataset:
    """Sort the input so `string_agg(x ORDER BY y)` collects in `y`'s order.

    The list aggregate appends in input order, so ordering the input once reproduces the
    ordered aggregate exactly. Every other aggregate in the query is order-*independent*
    (`sum`/`count`/`min`/`max` do not care), so the sort cannot change their results.

    Two ordered aggregates asking for *different* orderings cannot both be served by one
    pass, so that is rejected rather than answered with whichever sort happened to win.

    Args:
        tr: The translator, carrying the orderings collected during registration.
        ds: The dataset to sort.

    Returns:
        The dataset sorted by the requested ordering.
    """
    distinct_orders = {sql for sql, _ in tr._agg_order}
    if len(distinct_orders) > 1:
        raise NotImplementedError(
            "two aggregates with different ORDER BY clauses in one query are not supported "
            f"(got {sorted(distinct_orders)}); one input ordering cannot serve both, so "
            "compute them in separate subqueries"
        )
    keys, descending = [], []
    for item in tr._agg_order[0][1]:
        if not isinstance(item.this, exp.Column):
            raise NotImplementedError(
                f"an aggregate's ORDER BY supports plain columns only; got {item.this.sql()!r}"
            )
        keys.append(item.this.name)
        descending.append(bool(item.args.get("desc")))
    return ds.sort(*keys, descending=descending)
