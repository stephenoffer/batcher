"""Grouping, aggregation, and projection mapping for the SQL translator.

Covers ROLLUP/CUBE/GROUPING SETS expansion plus the GROUP BY / aggregate path
and the final projection map. Functions take the translator instance (`tr`) as
their first argument.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._sql.parser.agg_rewrites import rewrite_distinct_aggs, sort_for_ordered_aggs
from batcher._sql.parser.core_utils import (
    _alias_of,
    _has_aggregate,
    _positional,
    _unwrap_alias,
)
from batcher._sql.parser.expressions import _AGG_FUNCS
from batcher._sql.parser.expressions.aggregates import (
    build_anon_agg,
    build_typed_agg,
    is_agg_node,
    iter_agg_nodes,
)
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import AggExpr, Expr, col
from batcher.plan.expr_ir.selectors import expand_selectors, has_selector


def _expand_star(
    tr, star, visible: list[str], physical: dict[str, str] | None = None
) -> dict[str, Expr]:
    """Expand `SELECT *` with DuckDB's star modifiers into name -> expression.

    Supports `EXCLUDE`/`EXCEPT (cols)` (drop columns), `REPLACE (expr AS c)`
    (substitute a column's expression, keeping its position), and `RENAME (c AS d)`
    (rename in place). A modifier this translator cannot express is rejected rather
    than silently dropped — ignoring one would return the wrong columns.

    `physical` maps a name in `visible` to the column actually carrying it in the joined
    relation, for a qualified `x.*` whose columns the join disambiguator renamed.
    """
    if star.args.get("ilike") is not None:
        raise PlanError(
            "SELECT * ILIKE is not supported; list the columns explicitly or use "
            "EXCLUDE / REPLACE / RENAME"
        )

    excluded = {c.name for c in star.args.get("except_") or ()}
    replaced = {a.alias: _unwrap_alias(a) for a in star.args.get("replace") or ()}
    renamed = {a.this.name: a.alias for a in star.args.get("rename") or ()}

    for label, referenced in (
        ("EXCLUDE", excluded),
        ("REPLACE", set(replaced)),
        ("RENAME", set(renamed)),
    ):
        unknown = sorted(referenced - set(visible))
        if unknown:
            raise PlanError(
                f"SELECT * {label} names unknown column(s) {unknown}; available: {visible}"
            )

    out: dict[str, Expr] = {}
    for c in visible:
        if c in excluded:
            continue
        # REPLACE keeps the column's name and position, swapping its expression.
        # RENAME keeps the expression, swapping the output name.
        source = physical.get(c, c) if physical else c
        out[renamed.get(c, c)] = tr._scalar(replaced[c]) if c in replaced else col(source)
    return out


def _star_source(tr, projection) -> dict[str, str] | None:
    """For a qualified `x.*`, x's own columns mapped to the columns now carrying them.

    A join renames colliding columns to `alias__col` (`core_utils._disambiguate_columns`),
    which recorded per select node what each source contributed. Without that lookup a
    qualified star expanded to *every* column of the joined relation, under the
    disambiguated names — so `SELECT x.*, small.id4 AS small_id4 FROM x JOIN small
    USING (id1)` returned `x__id4` and `small__id4` where SQL requires x's `id4` alone.

    Returns `None` when the star should keep expanding to everything: an unqualified `*`,
    a single-table query (where x's columns *are* every column), or a source the join
    disambiguator could not enumerate — an inline subquery, whose columns are not known
    until it is translated. (A CTE *is* enumerable: it is in the table registry.)
    """
    if not projection.table:
        return None
    return tr._star_sources.get(id(projection.parent), {}).get(projection.table)


def _unique_name(name: str, taken: dict[str, Expr]) -> str:
    """`name`, or the first free `name_1`, `name_2`, ... if it is already claimed.

    SQL lets a SELECT list emit the same output name twice (`SELECT t.id, u.id`, or the
    `t.*, u.*` of a join on a shared key), but a `Dataset` is name-keyed and rejects a
    duplicate. Keying the projection map on the raw name instead made the *second* item
    overwrite the first, so the query silently returned one column where SQL asks for
    two. Suffixing matches what DuckDB itself does whenever it has to make result names
    unique, so a caller reading `id_1` sees the same name the oracle produces.
    """
    if name not in taken:
        return name
    i = 1
    while f"{name}_{i}" in taken:
        i += 1
    return f"{name}_{i}"


def _put(named: dict[str, Expr], name: str, expr: Expr) -> None:
    """Add one projection under a name no earlier projection has claimed."""
    named[_unique_name(name, named)] = expr


def _projection_map(tr, ds: Dataset, projections, star_cols=None) -> dict[str, Expr]:
    """Build the SELECT list's output name -> expression map.

    `star_cols` is the column list `*` expands to, when it differs from the relation's
    current columns — a window pass has already added its output columns to `ds`, and a
    star must expand what the query selects *from*, not what this same SELECT list is
    about to produce.
    """
    star_visible = list(ds.columns) if star_cols is None else list(star_cols)
    named: dict[str, Expr] = {}
    for p in projections:
        # `SELECT *` (Star) or `SELECT t.*` (a Column wrapping a Star).
        if isinstance(p, exp.Star) or (isinstance(p, exp.Column) and isinstance(p.this, exp.Star)):
            star = p if isinstance(p, exp.Star) else p.this
            # Internal columns materialized by UDF hoisting (`__bc_…`) are an
            # implementation detail and must never leak through `*`.
            visible = [c for c in star_visible if not c.startswith("__bc_")]
            physical = _star_source(tr, p) if isinstance(p, exp.Column) else None
            if physical is not None:
                visible = list(physical)
            for name, expr in _expand_star(tr, star, visible, physical).items():
                _put(named, name, expr)
            continue
        alias = _alias_of(p)
        if tr._is_window(p):
            # The window pass already materialized this column under `alias`.
            _put(named, alias, col(alias))
            continue
        expr = tr._scalar(_unwrap_alias(p))
        if has_selector(expr):
            # A `COLUMNS(*)` / `COLUMNS('regex')` item (possibly wrapped in a scalar
            # function) expands to one output per matched column, named by that column
            # — reusing the DataFrame selector engine.
            visible = [c for c in ds.columns if not c.startswith("__bc_")]
            for name, sel in expand_selectors(expr, visible, ds._plan.available_schema()):
                _put(named, name, sel)
        else:
            _put(named, alias, expr)
    return named


def _is_group_agg(a) -> bool:
    """Whether aggregate node `a` is a GROUP BY aggregate rather than a window function.

    `sum(sum(x)) OVER (...)` holds two aggregates: the outer one is the *window*
    function (its parent is the `Window` node) and runs after grouping, while the inner
    one is an ordinary group aggregate supplying its input. Registering the outer one as
    a group aggregate is what made TPC-DS q98 fail with an aggregate referencing a
    column the grouping had not produced yet.

    An aggregate inside a (scalar) subquery belongs to that inner query, not this one.
    """
    if isinstance(a.parent, exp.Window):
        return False
    return a.find_ancestor(exp.Subquery) is None


#: Aggregates a warehouse user reaches for that Batcher deliberately does not carry, and the
#: spelling that does the job. Each entry is a *design* refusal rather than a missing kernel,
#: so the message names the alternative instead of promising the function later.
#:
#: `first`/`last` are the important pair. DuckDB defines them as the first/last row in an
#: unspecified order, so two runs of the same query may disagree — and a plan that is
#: morsel-parallel and may span nodes makes that far more visible than a single-threaded scan
#: does. Batcher's own `Expr.first(order_by=...)` requires the ordering that makes the answer
#: mean something; `any_value` is the honest spelling when it genuinely does not matter.
_AGG_ADVICE: dict[str, str] = {
    "first": (
        ". Batcher has no scan-order aggregate: the row a parallel scan reaches first is not "
        "defined. Use any_value(x) when any row will do, or min_by/arg_min(x, ordering) when "
        "a particular one is meant"
    ),
    "last": (
        ". Batcher has no scan-order aggregate: the row a parallel scan reaches last is not "
        "defined. Use any_value(x) when any row will do, or max_by/arg_max(x, ordering) when "
        "a particular one is meant"
    ),
}


def _aggregate(tr, ds: Dataset, projections, group, having, windows=None, order=None) -> tuple:
    # Plain (non-ROLLUP/CUBE/GROUPING SETS) GROUP BY: no level is ever rolled up, so
    # GROUPING(...) is the constant 0. (The grouping-sets path handles it per level.)
    scopes = [*projections, having.this] if having is not None else list(projections)
    for scope in scopes:
        for g in list(scope.find_all(exp.Grouping, exp.GroupingId)):
            g.replace(exp.Paren(this=exp.Literal.number(0)))

    group_cols: list[str] = []
    group_exprs: dict[str, Expr] = {}  # internal alias -> derived key expression
    group_expr_alias: dict[str, str] = {}  # GROUP BY expr SQL text -> alias
    # SELECT output alias -> its defining expression, so a `GROUP BY <alias>` that names a
    # derived SELECT item (`extract(minute FROM EventTime) AS m … GROUP BY m`) resolves to
    # the expression rather than a non-existent column. SQL permits grouping by a select
    # alias; DuckDB/Postgres do the same.
    select_aliases: dict[str, object] = {}
    for p in projections:
        a = _alias_of(p)
        if a:
            select_aliases[a] = _unwrap_alias(p)
    input_cols = set(ds.columns)
    if group is not None:
        group_items = list(group.expressions)
        # GROUP BY ALL groups by every SELECT item that is not itself an aggregate
        # (or a window function) — DuckDB/Postgres semantics. sqlglot flags it with
        # `all=True` and an empty expression list; expand it to those items so the
        # keys actually group the rows rather than collapsing to a grand total.
        if group.args.get("all") and not group_items:
            group_items = [
                _unwrap_alias(p)
                for p in projections
                if not isinstance(_unwrap_alias(p), exp.Star)
                and not _has_aggregate(p)
                and not tr._is_window(p)
            ]
        for i, g in enumerate(group_items):
            # GROUP BY <n> refers to the n-th (1-based) SELECT item.
            if isinstance(g, exp.Literal) and not g.is_string:
                g = _unwrap_alias(_positional(projections, g, "GROUP BY"))
            # GROUP BY <select-alias> of a derived expression resolves to that expression
            # — but only as a *fallback*. GROUP BY resolves an input column first and
            # reaches for a select alias only when no input column carries the name;
            # that is the opposite precedence from ORDER BY, and it is DuckDB's
            # (`SELECT b*10 AS a FROM t GROUP BY a` is a binder error there, because
            # `a` binds to the input column, not to the alias). Preferring the alias
            # unconditionally made `SELECT SUM(p) AS k, k AS p ... GROUP BY k` try to
            # group by `SUM(p)`.
            if isinstance(g, exp.Column) and g.name in select_aliases and g.name not in input_cols:
                g = select_aliases[g.name]
            # A key repeated in the GROUP BY (`GROUP BY region, region`, or `GROUP BY
            # 1, region` where both name the same column) is redundant — DuckDB accepts
            # it and groups once. Dedup so the engine does not raise on a duplicate
            # output column (bare columns) or recompute the same derived key twice.
            if isinstance(g, exp.Column):
                if g.name not in group_cols:
                    group_cols.append(g.name)
            elif g.sql() not in group_expr_alias:
                alias = f"__gk{i}"
                group_exprs[alias] = tr._scalar(g)
                group_expr_alias[g.sql()] = alias

    # Collect every aggregate from SELECT and HAVING, assigning each a column.
    tr._agg_map = {}
    tr._agg_n = 0
    tr._agg_distinct = {}
    tr._agg_pending_distinct = []
    tr._agg_order = []
    used_aliases = set(group_cols) | set(group_exprs)
    for p in projections:
        inner = _unwrap_alias(p)
        if is_agg_node(inner):
            _register_agg(tr, inner, _alias_of(p), used_aliases)
        else:
            for a in iter_agg_nodes(inner):
                if _is_group_agg(a):
                    _register_agg(tr, a, None, used_aliases)
    # A window item's *inner* aggregates are group aggregates feeding the window, so
    # they must be registered here alongside the SELECT list's own.
    for w in windows or ():
        for a in iter_agg_nodes(_unwrap_alias(w)):
            if _is_group_agg(a):
                _register_agg(tr, a, None, used_aliases)
    if having is not None:
        for a in iter_agg_nodes(having.this):
            if _is_group_agg(a):
                _register_agg(tr, a, None, used_aliases)
    # An aggregate the query only ever names in ORDER BY (`... ORDER BY MIN(x)`) still
    # has to be computed by the grouping — it is a sort key over the grouped rows. It is
    # registered here, materialized as an internal column, and dropped by the projection
    # that follows the sort.
    for o in order.expressions if order is not None else ():
        for a in iter_agg_nodes(o):
            if _is_group_agg(a):
                _register_agg(tr, a, None, used_aliases)

    agg_kwargs = dict(tr._agg_map.values())
    if agg_kwargs and tr._agg_order:
        ds = sort_for_ordered_aggs(tr, ds)
    if agg_kwargs:
        if tr._agg_distinct:
            # The rewrite returns the relation to group and the aggregates to apply to it,
            # having already materialized the computed group keys — so group by their
            # aliases rather than recomputing them over columns that are gone.
            ds, agg_kwargs = rewrite_distinct_aggs(tr, ds, group_cols, group_exprs, agg_kwargs)
            ds = ds.group_by(*group_cols, *group_exprs).agg(**agg_kwargs)
        else:
            ds = ds.group_by(*group_cols, **group_exprs).agg(**agg_kwargs)
    else:
        # A GROUP BY with no aggregate anywhere (`SELECT k FROM t GROUP BY k`, incl.
        # with a HAVING over the keys) collapses to one row per distinct key — a
        # DISTINCT over the group columns. Calling `.agg()` with nothing errors.
        if group_exprs:
            ds = ds.with_columns(**group_exprs)
        keys = [*group_cols, *group_exprs]
        ds = ds.select(*keys).distinct() if keys else ds.limit(1)

    if having is not None:
        ds = ds.filter(tr._scalar(having.this))

    # SQL evaluates window functions *after* GROUP BY and HAVING, over the grouped
    # relation. Their aggregate arguments are now materialized columns, so point them
    # at those columns and let the ordinary window pass compute them here.
    if windows:
        from batcher._sql.parser.windowing import (
            rewrite_aggs_in_windows,
            rewrite_group_keys_in_windows,
        )

        # Derived group keys first: they are matched by SQL text, and rewriting an
        # aggregate inside one would change the text `group_expr_alias` is keyed on.
        rewrite_group_keys_in_windows(windows, group_expr_alias)
        rewrite_aggs_in_windows(tr, windows)
        ds = tr._window(ds, windows)

    # Final projection (group keys, aggregate refs, and arithmetic over them).
    # A SELECT item that *is* a GROUP BY expression resolves to that key's
    # materialized column rather than being recomputed.
    named: dict[str, Expr] = {}
    for p in projections:
        out = _alias_of(p)
        inner = _unwrap_alias(p)
        if tr._is_window(p):
            # Already materialized under `out` by the window pass above.
            _put(named, out, col(out))
        elif isinstance(inner, exp.Column) and inner.name in group_cols:
            _put(named, out, col(inner.name))
        elif inner.sql() in group_expr_alias:
            _put(named, out, col(group_expr_alias[inner.sql()]))
        else:
            _put(named, out, tr._scalar(inner))
    # The *unprojected* relation is returned with its projection map so the caller can
    # sort before projecting: SQL resolves an ORDER BY term against the select-list
    # aliases AND the columns underneath them, and projecting first destroys the latter.
    # NB: `_agg_map` stays live so an ORDER BY over an aggregate can resolve; the
    # caller (`select`) clears it once ordering is done.
    return ds, named


def _distinct_on(tr, ds: Dataset, projections, order, on_exprs) -> Dataset:
    """`SELECT DISTINCT ON (keys) ... ORDER BY ...` — keep one row per key set.

    Postgres/DuckDB semantics: for each set of rows sharing the ``DISTINCT ON`` key
    expressions, keep the first row in ``ORDER BY`` order, then order the survivors by
    the same ``ORDER BY``. Reuses ``Dataset.distinct(subset, keep="first", order_by=…)``
    — one mergeable reduction, the same operator the DataFrame spelling builds, so the two
    front-ends cannot drift into different plans for the same query. The key and sort
    expressions are materialized as internal ``__bc_`` columns first so they stay
    resolvable even when absent from the SELECT list, and never leak through ``SELECT *``.
    """
    on_cols: list[str] = []
    temp: dict[str, Expr] = {}
    for i, e in enumerate(on_exprs):
        name = f"__bc_don{i}"
        temp[name] = tr._scalar(e)
        on_cols.append(name)

    order_spec: list[tuple[str, bool]] = []  # (column, descending) for the row choice
    final_sort: list[tuple[str, bool, bool]] = []  # (column, descending, nulls_first)
    if order is not None:
        for j, o in enumerate(order.expressions):
            target = o.this
            if isinstance(target, exp.Literal) and not target.is_string:
                target = _unwrap_alias(_positional(projections, target, "ORDER BY"))
            name = f"__bc_dord{j}"
            temp[name] = tr._scalar(target)
            desc = bool(o.args.get("desc"))
            order_spec.append((name, desc))
            final_sort.append((name, desc, bool(o.args.get("nulls_first"))))

    ds = ds.with_columns(**temp)
    if order_spec:
        ds = ds.distinct(on_cols, keep="first", order_by=order_spec)
    else:
        ds = ds.distinct(on_cols)

    named = _projection_map(tr, ds, projections)
    if not final_sort:
        return ds.select(**named)
    # Sort the survivors by the (materialized) ORDER BY keys, then drop the temporaries.
    ds = ds.with_columns(**named)
    ds = ds.sort(
        *(col(k) for k, _, _ in final_sort),
        descending=[d for _, d, _ in final_sort],
        nulls_first=[nf for _, _, nf in final_sort],
    )
    return ds.select(*named.keys())


def _register_agg(tr, node, preferred: str | None, used: set) -> None:
    key = node.sql()
    if tr._agg_map is None or key in tr._agg_map:
        return
    if preferred and not preferred.startswith("__") and preferred not in used:
        alias = preferred
    else:
        alias = f"__agg{tr._agg_n}"
        tr._agg_n += 1
    used.add(alias)
    before = len(tr._agg_pending_distinct)
    tr._agg_map[key] = (alias, _agg(tr, node))
    if len(tr._agg_pending_distinct) > before:
        # `_agg` records a DISTINCT argument as it lowers; it does not know the output
        # name, so the association is made here where the alias is assigned.
        tr._agg_distinct[alias] = tr._agg_pending_distinct.pop()


def _agg(tr, node) -> AggExpr | Expr:
    if isinstance(node, exp.Anonymous):
        # A DuckDB aggregate sqlglot does not model (`product`, `sem`, `count_star`, …).
        return build_anon_agg(tr, node)
    typed = build_typed_agg(tr, node)
    if typed is not None:
        # Two-input (`corr`, `arg_max`, `regr_*`) or composite (`stddev_pop`, `sem`)
        # aggregates, which a name-to-tag table cannot express.
        return typed
    fname = type(node).__name__.lower()
    if fname == "count":
        # COUNT(*) vs COUNT(expr) vs COUNT(DISTINCT expr)
        arg = node.this
        if arg is None or isinstance(arg, exp.Star):
            return AggExpr("count_star", None)
        if isinstance(arg, exp.Distinct):
            exprs = arg.expressions
            if len(exprs) != 1:
                raise NotImplementedError("COUNT(DISTINCT ...) supports exactly one expression")
            return AggExpr("count_distinct", tr._scalar(exprs[0]))
        return AggExpr("count", tr._scalar(arg))
    # percentile_cont(x, p) / quantile_cont(x, p) → a parameterized quantile.
    if fname in ("percentilecont", "quantilecont"):
        p = node.expression
        if not isinstance(p, exp.Literal) or p.is_string:
            raise NotImplementedError("percentile_cont requires a constant fraction")
        return AggExpr("quantile", tr._scalar(node.this), param=float(p.name))
    # array_agg(x) and string_agg(x, sep) both collect into a list; the separator
    # join for string_agg happens in the projection (see scalar._scalar).
    if fname in ("arrayagg", "groupconcat"):
        if isinstance(node.this, exp.Distinct):
            # The engine's list aggregate has no per-group dedup flag; reject cleanly
            # rather than letting the bare `Distinct` node crash the scalar translator.
            raise NotImplementedError(
                "array_agg(DISTINCT x) / string_agg(DISTINCT x) is not supported; "
                "pre-aggregate the distinct values in a subquery"
            )
        arg = node.this
        if isinstance(arg, exp.Order):
            # `string_agg(x ORDER BY y)` collects x in y's order. The list aggregate appends
            # in input order, so ordering the *input* once up front gives exactly that —
            # the same shape as the DISTINCT rewrite's pre-dedup. Recorded here and applied
            # by the assembler, which checks every ordered aggregate asks for the same sort
            # (one pass cannot serve two different orderings).
            tr._agg_order.append((arg.sql(), list(arg.expressions)))
            arg = arg.this
        return AggExpr("list_agg", tr._scalar(arg))
    mapped = _AGG_FUNCS.get(fname)
    if mapped is None:
        raise NotImplementedError(f"unsupported aggregate: {fname}{_AGG_ADVICE.get(fname, '')}")
    arg = node.this
    if isinstance(arg, exp.Distinct):
        # `MIN(DISTINCT x)` / `MAX(DISTINCT x)` — dedup is a no-op for the extrema, so
        # they equal `MIN(x)` / `MAX(x)`. Other DISTINCT aggregates (SUM/AVG/…) need a
        # per-group dedup the engine's aggregate has no flag for; reject cleanly rather
        # than letting the bare `Distinct` node fall through to a confusing scalar error.
        exprs = arg.expressions
        if len(exprs) != 1:
            raise NotImplementedError(f"{fname}(DISTINCT ...) supports exactly one expression")
        if mapped in ("min", "max"):
            return AggExpr(mapped, tr._scalar(exprs[0]))
        # `SUM(DISTINCT x)` and friends need a per-group dedup the engine's aggregate has
        # no flag for. It is exactly `<agg>(x)` over rows deduped on the group keys plus
        # `x`, so record the expression and let the assembler dedup once up front; the
        # aggregate itself is then an ordinary non-distinct one. Recorded per registered
        # aggregate (not globally) so the assembler can verify every aggregate in the
        # query shares one distinct expression — the only shape a single dedup is
        # correct for.
        tr._agg_pending_distinct.append((exprs[0].sql(), exprs[0]))
        return AggExpr(mapped, tr._scalar(exprs[0]))
    return AggExpr(mapped, tr._scalar(arg))
