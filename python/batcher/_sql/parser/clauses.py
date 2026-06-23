"""SELECT / FROM / JOIN / ORDER clause building for the SQL translator.

Wires the per-theme helpers (subquery, grouping, windowing, scalar) into the
overall SELECT translation. Functions take the translator instance (`tr`) as
their first argument.
"""

from __future__ import annotations

import sys

import pyarrow as pa

from batcher._sql.parser.core_utils import _has_aggregate, _unwrap_alias
from batcher.api.dataset import Dataset
from batcher.api.session import from_arrow
from batcher.plan.expr_ir import lit


def _bool_agg_to_filter(node):
    """`bool_and(x)` / `bool_or(x)` → a count-of-filtered-rows comparison.

    `bool_and(x)` is true iff no row has `NOT x` → `COUNT(*) FILTER (WHERE NOT x) = 0`;
    `bool_or(x)` is true iff some row has `x` → `COUNT(*) FILTER (WHERE x) > 0`. The
    produced FILTER is lowered by `_filter_to_case` in the same pass.
    """
    from sqlglot import expressions as exp

    if isinstance(node, exp.LogicalAnd):
        filt = exp.Filter(
            this=exp.Count(this=exp.Star()),
            expression=exp.Where(this=exp.Not(this=node.this.copy())),
        )
        return exp.EQ(this=filt, expression=exp.Literal.number(0))
    if isinstance(node, exp.LogicalOr):
        filt = exp.Filter(
            this=exp.Count(this=exp.Star()),
            expression=exp.Where(this=node.this.copy()),
        )
        return exp.GT(this=filt, expression=exp.Literal.number(0))
    return node


def _filter_to_case(node):
    """`agg(arg) FILTER (WHERE c)` → `agg(CASE WHEN c THEN arg END)`.

    `COUNT(*) FILTER (WHERE c)` becomes `COUNT(CASE WHEN c THEN 1 END)` — counting
    the non-null CASE values is exactly counting the rows where `c` holds.
    """
    from sqlglot import expressions as exp

    if not isinstance(node, exp.Filter):
        return node
    agg = node.this.copy()
    cond = node.expression.this  # Where -> condition
    arg = agg.this
    # COUNT(*) has no argument (or a Star) — count the constant 1 where c holds.
    arg = exp.Literal.number(1) if arg is None or isinstance(arg, exp.Star) else arg.copy()
    agg.set("this", exp.case().when(cond.copy(), arg))
    return agg


def _select(tr, node) -> Dataset:
    # `bool_and`/`bool_or` → COUNT(*) FILTER comparisons; then
    # `agg(...) FILTER (WHERE c)` ≡ `agg(CASE WHEN c THEN arg END)`. Both are
    # AST rewrites done up front so the normal aggregate path handles them.
    node = node.transform(_bool_agg_to_filter)
    node = node.transform(_filter_to_case)
    # Inline `WINDOW w AS (...)` definitions into the `OVER w` references.
    tr._inline_named_windows(node)

    # ROLLUP / CUBE / GROUPING SETS expand into a UNION ALL of grouping levels.
    group = node.args.get("group")
    if group is not None and any(group.args.get(k) for k in ("rollup", "cube", "grouping_sets")):
        return tr._grouping_sets_union(node, group)

    ds = tr._from(node)

    residual = None
    where = node.args.get("where")
    if where is not None:
        ds, residual = tr._apply_subquery_predicates(ds, where.this)
    # Correlated scalar subqueries (SELECT list / HAVING / residual WHERE)
    # decorrelate into LEFT JOINs before the value expressions are built.
    ds = tr._decorrelate_scalar_subqueries(
        ds, [*node.expressions, residual, node.args.get("having")]
    )
    if residual is not None:
        ds = ds.filter(tr._scalar(residual))

    projections = node.expressions  # SELECT list
    group = node.args.get("group")
    order = node.args.get("order")
    limit = node.args.get("limit")
    offset = node.args.get("offset")
    qualify = node.args.get("qualify")
    has_agg = group is not None or any(_has_aggregate(p) for p in projections)
    has_window = any(tr._is_window(p) for p in projections)

    if has_window and not has_agg:
        ds = tr._window(ds, projections)
        # QUALIFY filters on the window-function results (named by their SELECT
        # alias) — applied after the window columns exist, before the projection
        # drops any not in the final SELECT.
        if qualify is not None:
            ds = ds.filter(tr._scalar(qualify.this))
        named = tr._projection_map(ds, projections)
        if order is not None:
            ds = ds.with_columns(**named)
            ds = tr._order(ds, order, projections)
            ds = ds.select(*named.keys())
        else:
            ds = ds.select(**named)
    elif qualify is not None:
        raise NotImplementedError(
            "QUALIFY is supported only on a query whose SELECT computes the window "
            "function(s) it filters on (reference them by their output alias)"
        )
    elif has_agg:
        ds = tr._aggregate(ds, projections, group, node.args.get("having"))
        if order is not None:
            # _agg_map is still live here, so ORDER BY can reference an
            # aggregate (e.g. ORDER BY SUM(x)) by its output column.
            ds = tr._order(ds, order, projections)
        tr._agg_map = None
    else:
        named = tr._projection_map(ds, projections)
        if order is not None:
            # ORDER BY may reference base columns not in SELECT, so keep them
            # through the sort (with_columns preserves base columns), then project.
            ds = ds.with_columns(**named)
            ds = tr._order(ds, order, projections)
            ds = ds.select(*named.keys())
        else:
            ds = ds.select(**named)

    # SELECT DISTINCT: dedup the projected rows.
    if node.args.get("distinct"):
        ds = ds.distinct()

    if limit is not None or offset is not None:
        skip = int(offset.expression.this) if offset is not None else 0
        # A bare OFFSET (no LIMIT) keeps every row after `skip`; the engine takes
        # min(n, remaining), so sys.maxsize means "all remaining".
        n = int(limit.expression.this) if limit is not None else sys.maxsize
        ds = ds.limit(n, offset=skip)
    return ds


def _from(tr, node) -> Dataset:
    from sqlglot import expressions as exp

    from_ = node.args.get("from_") or node.args.get("from")
    if from_ is None:
        # `SELECT <exprs>` with no FROM → one row of constants (e.g.
        # `SELECT 1 + 1`, `SELECT extract(year from date '2021-01-01')`).
        return from_arrow(pa.table({"__dummy": [0]}))
    ds = _table(tr, from_.this)

    for join in node.args.get("joins", []) or []:
        right = _table(tr, join.this)
        on = join.args.get("on")
        using = join.args.get("using")
        how = (join.side or "inner").lower()  # "" → inner; "LEFT" → left; "FULL" → full
        how = how if how in {"inner", "left", "right", "full"} else "inner"
        if using:
            keys = [u.name for u in using]
            ds = ds.join(right, on=keys, how=how)
        elif on is not None and isinstance(on, exp.EQ):
            left_key = on.this.name
            right_key = on.expression.name
            if left_key == right_key:
                ds = ds.join(right, on=left_key, how=how)
            else:
                ds = ds.join(right, left_on=left_key, right_on=right_key, how=how)
        elif on is None:
            # No ON/USING → cross join (cartesian product), expressed as an
            # inner join on a constant key that is then dropped.
            ck = "__cross_key"
            ds = (
                ds.with_columns(**{ck: lit(1)})
                .join(right.with_columns(**{ck: lit(1)}), on=ck)
                .drop(ck)
            )
        else:
            raise NotImplementedError("only equi-joins (ON a=b / USING) are supported")
    return ds


def _table(tr, node) -> Dataset:
    from sqlglot import expressions as exp

    # FROM (SELECT ...) AS t  → translate the inner SELECT to a Dataset.
    if isinstance(node, exp.Subquery):
        ds = tr.statement(node.this)
    elif isinstance(node, (exp.Select, exp.Union)):
        ds = tr.statement(node)
    else:
        name = node.name
        if name not in tr._registry:
            raise KeyError(f"unknown table {name!r}; registered: {list(tr._registry)}")
        ds = tr._registry[name]
    return _apply_tablesample(ds, node)


def _apply_tablesample(ds: Dataset, node) -> Dataset:
    """Apply a SQL ``TABLESAMPLE`` on a table/subquery: ``BERNOULLI(p PERCENT)`` →
    fraction sample, ``RESERVOIR(n ROWS)`` → fixed-count sample. Both lower to
    `Dataset.sample` (deterministic, partition-independent)."""
    sample = node.args.get("sample") if hasattr(node, "args") else None
    if sample is None:
        return ds

    def _num(x):
        return x.this if x is not None and hasattr(x, "this") else x

    percent = _num(sample.args.get("percent"))
    size = _num(sample.args.get("size"))
    if percent is not None:
        return ds.sample(float(percent) / 100.0)
    if size is not None:
        return ds.sample(n=int(size))
    return ds


def _order(tr, ds: Dataset, order, projections=None) -> Dataset:
    # ORDER BY accepts arbitrary expressions (columns, functions, arithmetic),
    # resolved the same way as any scalar — including aggregate outputs in a
    # grouped query (via `_scalar`'s aggregate-output resolution) — and the
    # 1-based positional form `ORDER BY <n>` referring to a SELECT item.
    from sqlglot import expressions as exp

    keys: list = []
    desc: list[bool] = []
    for o in order.expressions:
        target = o.this
        if projections is not None and isinstance(target, exp.Literal) and not target.is_string:
            target = _unwrap_alias(projections[int(target.this) - 1])
        keys.append(tr._scalar(target))
        desc.append(bool(o.args.get("desc")))
    return ds.sort(*keys, descending=desc)
