"""SELECT / FROM / JOIN / ORDER clause building for the SQL translator.

Wires the per-theme helpers (subquery, grouping, windowing, scalar) into the
overall SELECT translation. Functions take the translator instance (`tr`) as
their first argument.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._sql.parser import windowing
from batcher._sql.parser.core_utils import (
    _alias_of,
    _has_aggregate,
    _is_star,
    _positional,
    _positional_output,
    _row_window,
    _unwrap_alias,
    _within_group_to_agg,
)
from batcher._sql.parser.joins.lateral import select_unnest
from batcher._sql.parser.subquery import core as subquery
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import col


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
    # `with_columns` keeps the input columns in scope for the sort, so ORDER BY ALL is
    # given the projection's own columns rather than everything currently visible.
    ds = _order(tr, ds, order, projections, output=list(named))
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
        # An `EXISTS` or `IN (SELECT …)` under `OR` became a column so the predicate above
        # could read it (see `subquery.core._exists_marker` / `_in_marker`). It has served
        # its purpose now, and it is an implementation detail rather than an output column
        # — `SELECT *` must not see it.
        spent = [c for c in ds.columns if c.startswith(subquery.MARKER_PREFIXES)]
        if spent:
            ds = ds.drop(*spent)

    projections = node.expressions  # SELECT list
    # `SELECT unnest(xs)` expands the relation the projection is about to read, so it is
    # applied here — after WHERE, which SQL evaluates first, and before the projection.
    ds = select_unnest(tr, ds, projections)
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
        # `lag/lead(x, n, default)` becomes a CASE over supported window functions,
        # before hoisting, so the constructs it introduces are hoisted with the rest.
        windowing.rewrite_offset_defaults(projections)
        projections, nested = windowing.hoist_nested_windows(projections)
        # `*` expands the columns the query selects *from*, so it is captured before the
        # window pass adds its output columns. Reading them afterwards made
        # `SELECT *, sum(x) OVER (...) AS s` expand the star over `s` as well and emit it
        # twice, where DuckDB emits it once.
        star_cols = list(ds.columns)
        ds = tr._window(ds, [*projections, *nested])
        # QUALIFY filters on the window-function results (named by their SELECT
        # alias) — applied after the window columns exist, before the projection
        # drops any not in the final SELECT.
        if qualify is not None:
            ds = ds.filter(tr._scalar(qualify.this))
        named = tr._projection_map(ds, projections, star_cols)
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
            windowing.rewrite_offset_defaults(projections)
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
            deferred_order = _retarget_order_to_aliases(deferred_order, projections)
            deferred_order = _retarget_positional_order(deferred_order, projections)
            ds = tr._order(ds, deferred_order, projections)
    tr._agg_map = None

    if limit is not None or offset is not None:
        # A bare OFFSET (no LIMIT) keeps every row after the skip; the engine takes
        # min(n, remaining), so the helper's sys.maxsize means "all remaining".
        n, skip = _row_window(limit, offset)
        ds = ds.limit(n, offset=skip)
    return ds


def _qualify_windows(tr, ds: Dataset, qualify):
    """Compute the window functions a QUALIFY filters on, as hidden columns.

    `QUALIFY row_number() OVER (...) = 1` filters on a value the SELECT list never
    mentions, so the column has to be materialized before the filter can reference it.
    Each window expression in the predicate becomes a `__bc_qualify<n>` column and is
    replaced in the predicate by a reference to it; the projection that follows selects
    only the user's columns, so the helpers never reach the output.

    The `__bc_` prefix is load-bearing, not cosmetic: it is what the projection builder
    filters out of a star expansion. Under the bare `__qualify<n>` these columns *did*
    reach the output — ``SELECT * FROM t QUALIFY row_number() OVER (...) = 1`` returned
    an extra ``__qualify0`` column that no query asked for.

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
        alias = f"__bc_qualify{i}"
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


def _is_order_all(order) -> bool:
    """Whether this ORDER BY is DuckDB's ``ORDER BY ALL`` — sort by every output column.

    sqlglot leaves the bare ``ALL`` keyword as a `Var`, which the scalar path rejected as
    an unsupported expression, so the whole (valid) query failed.
    """
    items = order.expressions
    return (
        len(items) == 1
        and isinstance(items[0].this, exp.Var)
        and items[0].this.name.upper() == "ALL"
    )


def _retarget_order_to_aliases(order, projections):
    """Rewrite `ORDER BY <source column>` to the alias the projection gave it.

    With `DISTINCT` the sort runs *after* the projection (deduping is a hash operation and
    does not preserve order, so sorting first would discard the ORDER BY). That is correct,
    but it means the sort sees the projection's *output* names — and
    `SELECT DISTINCT f AS r FROM t ORDER BY f` names the input one. SQL resolves that
    perfectly well, because `f` does appear in the SELECT list; the relation simply no
    longer has a column called `f` by the time the sort runs, so it failed with
    "sort key references unknown column(s) ['f']".

    Only a *bare rename* is retargeted (`f AS r`), which is the case where the two names
    denote the same values. A computed projection is left alone: `ORDER BY f` beside
    `SELECT DISTINCT f * 2 AS r` is a genuinely unresolvable query, and it should keep
    saying so rather than silently sorting by something else.
    """
    if projections is None:
        return order
    rename: dict[str, str] = {}
    produced: set[str] = set()
    for p in projections:
        alias = _alias_of(p)
        inner = _unwrap_alias(p)
        if alias:
            produced.add(alias)
        if alias and isinstance(inner, exp.Column) and inner.name != alias:
            rename.setdefault(inner.name, alias)
    # An output column wins over a source column of the same name, which is SQL's own
    # resolution order and is not a nicety: `SELECT DISTINCT g AS f, f AS g ... ORDER BY f`
    # swaps the two names, so retargeting `f` to its source's alias would sort by the
    # *other* column. Silently.
    rename = {src: alias for src, alias in rename.items() if src not in produced}
    if not rename:
        return order
    retargeted = order.copy()
    for item in retargeted.expressions:
        target = item.this
        if isinstance(target, exp.Column) and target.name in rename:
            item.set("this", exp.column(rename[target.name]))
    return retargeted


def _retarget_positional_order(order, projections):
    """Rewrite `ORDER BY <n>` to the alias of the n-th SELECT item, for the DISTINCT path.

    `ORDER BY 1` resolves to the SELECT item's *expression*, which after `SELECT DISTINCT
    f AS r` is the source column `f` — gone from the relation by the time the deferred sort
    runs, exactly as the named form was. Resolving it to the output name instead keeps the
    positional and named spellings behaving the same way, which is the only defensible
    outcome: `ORDER BY 1` and `ORDER BY f` are the same request.
    """
    if projections is None:
        return order
    retargeted = order.copy()
    for item in retargeted.expressions:
        target = item.this
        if not (isinstance(target, exp.Literal) and not target.is_string):
            continue
        try:
            selected = _positional(projections, target, "ORDER BY")
        except Exception:  # an out-of-range position keeps its own error, raised later
            continue
        alias = _alias_of(selected)
        if alias:
            item.set("this", exp.column(alias))
    return retargeted


def _order_all(ds: Dataset, order, output: list[str]) -> Dataset:
    """``ORDER BY ALL`` — sort by each output column, left to right.

    The single ``ALL`` item carries the direction and null placement for every key, which
    is DuckDB's reading: ``ORDER BY ALL DESC`` sorts every column descending.
    """
    item = order.expressions[0]
    descending = bool(item.args.get("desc"))
    nulls_first = bool(item.args.get("nulls_first"))
    keys = [col(c) for c in output]
    if not keys:
        return ds
    return ds.sort(
        *keys, descending=[descending] * len(keys), nulls_first=[nulls_first] * len(keys)
    )


def _order(tr, ds: Dataset, order, projections=None, output: list[str] | None = None) -> Dataset:
    # ORDER BY accepts arbitrary expressions (columns, functions, arithmetic),
    # resolved the same way as any scalar — including aggregate outputs in a
    # grouped query (via `_scalar`'s aggregate-output resolution) — and the
    # 1-based positional form `ORDER BY <n>` referring to a SELECT item.
    if _is_order_all(order):
        # `output` is the projection's own column list where the caller knows it; the
        # relation's columns otherwise (a set operation, or a sort after DISTINCT, where
        # the relation *is* the projected output).
        return _order_all(ds, order, output if output is not None else list(ds.columns))
    keys: list = []
    desc: list[bool] = []
    nulls_first: list[bool] = []
    # A `SELECT *` in the list makes every position after it ambiguous against the AST,
    # so positions are resolved against the projection's own output names instead.
    by_output = output is not None and projections is not None and any(map(_is_star, projections))
    for o in order.expressions:
        target = o.this
        if projections is not None and isinstance(target, exp.Literal) and not target.is_string:
            if by_output:
                target = exp.column(_positional_output(output, target, "ORDER BY"))
            else:
                target = _unwrap_alias(_positional(projections, target, "ORDER BY"))
        keys.append(tr._scalar(target))
        desc.append(bool(o.args.get("desc")))
        # sqlglot normalizes an absent NULLS clause to `nulls_first=False`, which is
        # exactly the SQL default this engine (and DuckDB) use for both ASC and DESC.
        nulls_first.append(bool(o.args.get("nulls_first")))
    return ds.sort(*keys, descending=desc, nulls_first=nulls_first)
