"""SELECT / FROM / JOIN / ORDER clause building for the SQL translator.

Wires the per-theme helpers (subquery, grouping, windowing, scalar) into the
overall SELECT translation. Functions take the translator instance (`tr`) as
their first argument.
"""

from __future__ import annotations

import sys

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._sql.parser import windowing
from batcher._sql.parser.core_utils import (
    _has_aggregate,
    _positional,
    _unwrap_alias,
    _within_group_to_agg,
)
from batcher.api.dataset import Dataset


def _filter_to_case(node):
    """`agg(arg) FILTER (WHERE c)` → `agg(CASE WHEN c THEN arg END)`.

    `COUNT(*) FILTER (WHERE c)` becomes `COUNT(CASE WHEN c THEN 1 END)` — counting
    the non-null CASE values is exactly counting the rows where `c` holds.
    """
    if not isinstance(node, exp.Filter):
        return node
    agg = node.this.copy()
    cond = node.expression.this  # Where -> condition
    arg = agg.this
    # `agg(DISTINCT x) FILTER (WHERE c)` must push the guard *inside* the DISTINCT:
    # `count(DISTINCT CASE WHEN c THEN x END)`. Wrapping the whole `DISTINCT x` in a
    # CASE (`count(CASE WHEN c THEN DISTINCT x END)`) is not valid SQL and would crash
    # the scalar translator on the bare `Distinct` node. NULL from a false guard is
    # dropped by the distinct set, so the guarded distinct-count matches DuckDB.
    if isinstance(arg, exp.Distinct) and len(arg.expressions) == 1:
        inner = exp.case().when(cond.copy(), arg.expressions[0].copy())
        agg.set("this", exp.Distinct(expressions=[inner]))
        return agg
    # COUNT(*) has no argument (or a Star) — count the constant 1 where c holds.
    arg = exp.Literal.number(1) if arg is None or isinstance(arg, exp.Star) else arg.copy()
    agg.set("this", exp.case().when(cond.copy(), arg))
    return agg


def _project_ordered(tr, ds: Dataset, named, order, projections) -> Dataset:
    """Apply the SELECT projection, sorting first when there is an ORDER BY.

    SQL resolves an ORDER BY term against **both** the select-list aliases and the
    columns the projection was computed from, so the sort has to happen while both are
    in scope. `with_columns` adds the aliases while keeping the input columns, which
    also gives DuckDB's precedence: an alias shadows an input column of the same name
    (verified against DuckDB — `SELECT a AS b, b AS a FROM t ORDER BY a` sorts by the
    *alias* `a`, i.e. the original `b`).

    Projecting first — which is what the aggregate path used to do — drops the input
    columns, so `SELECT i_brand_id AS brand_id ... ORDER BY i_brand_id` (TPC-DS q55/q19)
    could not resolve its own sort key.

    Args:
        tr: The translator.
        ds: The relation the projection is computed over.
        named: Output column name -> expression.
        order: The `Order` node, or None.
        projections: The SELECT list, for positional (`ORDER BY 1`) resolution.

    Returns:
        The projected (and, when asked, sorted) dataset.
    """
    if order is None:
        return ds.select(**named)
    ds = ds.with_columns(**named)
    ds = _order(tr, ds, order, projections)
    return ds.select(*named.keys())


def _select(tr, node) -> Dataset:
    # `agg(...) FILTER (WHERE c)` ≡ `agg(CASE WHEN c THEN arg END)` — an AST rewrite
    # done up front so the normal aggregate path handles it. (`bool_and`/`bool_or`
    # map directly to the native NULL-aware aggregates; see literals._AGG_FUNCS.)
    node = node.transform(_filter_to_case)
    # `agg(...) WITHIN GROUP (ORDER BY x)` → the ordinary aggregate over `x`, so the
    # ordered column is not silently dropped (the fraction was being read as it).
    #
    # `copy=False` because the tree is already exclusively ours: `transform` copies by
    # default, so the call above detached it from any enclosing AST. Letting the second
    # rewrite copy as well deep-copied the entire statement a second time, which measured
    # as a sixth of the cost of translating a SELECT — for a tree nothing else can see.
    node = node.transform(_within_group_to_agg, copy=False)
    # Sources that expose the same column name (a self-join, or two different tables
    # sharing a name) are rewritten so the alias-blind column resolver sees distinct,
    # uniquely-named columns.
    tr._disambiguate_columns(node)
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
        ds, [*node.expressions, residual, node.args.get("having")], node
    )
    if residual is not None:
        # A registered scalar function in WHERE becomes a materialized column
        # before the predicate references it.
        ds, (residual,) = tr._hoist_udfs(ds, [residual])
        ds = ds.filter(tr._scalar(residual))

    projections = node.expressions  # SELECT list
    group = node.args.get("group")
    order = node.args.get("order")
    limit = node.args.get("limit")
    offset = node.args.get("offset")
    qualify = node.args.get("qualify")

    # SELECT DISTINCT dedups the *projected* rows, and SQL orders what survives. The
    # dedup is a hash operation and does not preserve input order, so sorting first and
    # deduping after would silently discard the ORDER BY. Hold the sort back and apply it
    # to the deduped rows instead. (With DISTINCT, SQL already requires every ORDER BY
    # expression to appear in the SELECT list, so it is still resolvable after the
    # projection.)
    distinct = node.args.get("distinct")
    distinct_on = distinct.args.get("on") if distinct is not None else None
    deferred_order = order if distinct is not None and distinct_on is None else None
    if deferred_order is not None:
        order = None
    has_agg = group is not None or any(_has_aggregate(p) for p in projections)
    # A window nested inside a larger projection (`sum(x) OVER () + 1`) makes this a
    # window query just as much as a bare `sum(x) OVER ()` item does.
    has_window = any(windowing._has_window(p) for p in projections)
    if has_agg or has_window:
        _reject_udf_in_agg_window(tr, node, projections)

    if distinct_on is not None:
        # DISTINCT ON keeps one row per key set (chosen by ORDER BY); agg/window rejected.
        if has_agg or has_window:
            raise NotImplementedError(
                "SELECT DISTINCT ON (...) combined with GROUP BY / aggregates / window "
                "functions is not supported; move the dedup into a subquery"
            )
        ds = tr._distinct_on(ds, projections, order, list(distinct_on.expressions))
    elif has_window and not has_agg:
        # Windows buried inside larger expressions become synthetic columns the
        # projection then reads; whole-item windows keep their user alias directly.
        projections, nested = windowing.hoist_nested_windows(projections)
        ds = tr._window(ds, [*projections, *nested])
        # QUALIFY filters on the window-function results (named by their SELECT
        # alias) — applied after the window columns exist, before the projection
        # drops any not in the final SELECT.
        if qualify is not None:
            ds = ds.filter(tr._scalar(qualify.this))
        named = tr._projection_map(ds, projections)
        ds = _project_ordered(tr, ds, named, order, projections)
    elif qualify is not None:
        if has_agg:
            raise NotImplementedError(
                "QUALIFY combined with GROUP BY / aggregates is not supported; compute "
                "the aggregate in a subquery and QUALIFY over it"
            )
        # QUALIFY whose window function appears ONLY in the QUALIFY clause — the usual
        # idiom (`... QUALIFY row_number() OVER (...) = 1`). The window column has to
        # exist before it can be filtered on, so it is computed under a hidden alias,
        # filtered, and then dropped by the projection below.
        ds, pred = _qualify_windows(tr, ds, qualify)
        ds = ds.filter(tr._scalar(pred))
        named = tr._projection_map(ds, projections)
        ds = _project_ordered(tr, ds, named, order, projections)
    elif has_agg:
        # Any window here runs *after* the grouping, over the aggregated relation, so
        # the window items are handed to the aggregate path rather than computed first.
        windows = None
        if has_window:
            projections, nested = windowing.hoist_nested_windows(projections)
            windows = [*(p for p in projections if tr._is_window(p)), *nested]
        ds, named = tr._aggregate(ds, projections, group, node.args.get("having"), windows, order)
        # _agg_map is still live here, so ORDER BY can reference an aggregate
        # (e.g. ORDER BY SUM(x)) by its output column. It stays live until the
        # DISTINCT block below, which may still owe a deferred sort.
        ds = _project_ordered(tr, ds, named, order, projections)
    else:
        # Registered scalar functions in the SELECT list become materialized
        # columns before the projection references them.
        ds, projections = tr._hoist_udfs(ds, projections)
        named = tr._projection_map(ds, projections)
        ds = _project_ordered(tr, ds, named, order, projections)

    # SELECT DISTINCT: dedup the projected rows, then sort what survives.
    if distinct is not None and distinct_on is None:
        ds = ds.distinct()
        if deferred_order is not None:
            ds = tr._order(ds, deferred_order, projections)
    tr._agg_map = None

    if limit is not None or offset is not None:
        skip = int(offset.expression.this) if offset is not None else 0
        # A bare OFFSET (no LIMIT) keeps every row after `skip`; the engine takes
        # min(n, remaining), so sys.maxsize means "all remaining".
        n = int(limit.expression.this) if limit is not None else sys.maxsize
        ds = ds.limit(n, offset=skip)
    return ds


def _qualify_windows(tr, ds: Dataset, qualify):
    """Compute the window functions a QUALIFY filters on, as hidden columns.

    `QUALIFY row_number() OVER (...) = 1` filters on a value the SELECT list never
    mentions, so the column has to be materialized before the filter can reference it.
    Each window expression in the predicate becomes a `__qualify<n>` column and is
    replaced in the predicate by a reference to it; the projection that follows selects
    only the user's columns, so the helpers never reach the output.

    A QUALIFY with no window function is just a WHERE over the current columns and is
    returned unchanged.

    Args:
        tr: The translator.
        ds: The dataset the predicate applies to.
        qualify: The `Qualify` node.

    Returns:
        The dataset with any window columns appended, and the rewritten predicate.
    """
    pred = qualify.this.copy()
    windows = list(pred.find_all(exp.Window))
    if not windows:
        return ds, pred
    synthetic = []
    for i, win in enumerate(windows):
        alias = f"__qualify{i}"
        # Copy before replacing: `synthetic` must keep the window expression itself,
        # while `pred` keeps only the reference to its output column.
        synthetic.append(exp.alias_(win.copy(), alias))
        win.replace(exp.column(alias))
    return tr._window(ds, synthetic), pred


def _reject_udf_in_agg_window(tr, node, projections) -> None:
    """Reject a registered scalar function in an unsupported aggregate/window position."""
    from batcher._sql.parser.udf import contains_registered_scalar

    targets = [
        *projections,
        node.args.get("having"),
        node.args.get("qualify"),
        node.args.get("order"),
    ]
    if any(contains_registered_scalar(tr, t) for t in targets):
        raise PlanError(
            "a registered scalar function is not supported in an aggregate or window "
            "query's SELECT / HAVING / ORDER BY / QUALIFY; compute it in a subquery or "
            "a projected alias first, then aggregate over that column"
        )


def _order(tr, ds: Dataset, order, projections=None) -> Dataset:
    # ORDER BY accepts arbitrary expressions (columns, functions, arithmetic),
    # resolved the same way as any scalar — including aggregate outputs in a
    # grouped query (via `_scalar`'s aggregate-output resolution) — and the
    # 1-based positional form `ORDER BY <n>` referring to a SELECT item.
    keys: list = []
    desc: list[bool] = []
    nulls_first: list[bool] = []
    for o in order.expressions:
        target = o.this
        if projections is not None and isinstance(target, exp.Literal) and not target.is_string:
            target = _unwrap_alias(_positional(projections, target, "ORDER BY"))
        keys.append(tr._scalar(target))
        desc.append(bool(o.args.get("desc")))
        # sqlglot normalizes an absent NULLS clause to `nulls_first=False`, which is
        # exactly the SQL default this engine (and DuckDB) use for both ASC and DESC.
        nulls_first.append(bool(o.args.get("nulls_first")))
    return ds.sort(*keys, descending=desc, nulls_first=nulls_first)
