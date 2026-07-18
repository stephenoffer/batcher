"""Grouping, aggregation, and projection mapping for the SQL translator.

Covers ROLLUP/CUBE/GROUPING SETS expansion plus the GROUP BY / aggregate path
and the final projection map. Functions take the translator instance (`tr`) as
their first argument.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher._sql.parser.core_utils import (
    _alias_of,
    _has_aggregate,
    _positional,
    _unwrap_alias,
)
from batcher._sql.parser.literals import _AGG_FUNCS
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import AggExpr, Expr, col
from batcher.plan.expr_ir.selectors import expand_selectors, has_selector


def _grouping_key(e) -> str:
    """Identity of a grouping expression across levels.

    A bare column is identified by its (unqualified) name so a `GROUP BY a`
    key matches a `SELECT t.a` projection; any other expression is identified
    by its SQL text (`b * 10`, `extract(...)`).
    """
    from sqlglot import expressions as exp

    return e.name if isinstance(e, exp.Column) else e.sql()


def _grouping_set_members(m) -> list:
    """The grouping expressions in one GROUPING SETS member.

    `(a, b)` → `[a, b]`, `(b + 1)` → `[b + 1]`, `()` → `[]`, bare `a` → `[a]`.
    """
    from sqlglot import expressions as exp

    if isinstance(m, exp.Tuple):
        return list(m.expressions)  # `(a, b)` and `()`
    if isinstance(m, exp.Paren):
        return [m.this]  # `(a)` / `(b + 1)`
    return [m]  # a bare column / expression


def _grouping_factors(group) -> list[list[list]]:
    """The GROUP BY as a list of factors; the levels are their cross product.

    A plain item list, `ROLLUP(...)`, `CUBE(...)`, and `GROUPING SETS(...)` each
    contribute one factor — a list of alternative grouping sets. DuckDB's overall
    set of levels is the Cartesian product of the factors (so `GROUP BY ROLLUP(a),
    ROLLUP(b)` yields the product of the two rollups), which is what `itertools.
    product` over these factors produces.
    """
    import itertools

    factors: list[list[list]] = []
    if group.expressions:  # plain items — present in every level
        factors.append([list(group.expressions)])
    for r in group.args.get("rollup") or ():
        cols = list(r.expressions)
        factors.append([cols[:i] for i in range(len(cols), -1, -1)])
    for cu in group.args.get("cube") or ():
        cols = list(cu.expressions)
        factors.append(
            [list(c) for k in range(len(cols), -1, -1) for c in itertools.combinations(cols, k)]
        )
    for gs in group.args.get("grouping_sets") or ():
        factors.append([_grouping_set_members(m) for m in gs.expressions])
    return factors or [[[]]]


def _grouping_sets_union(tr, node, group) -> Dataset:
    """Expand ROLLUP/CUBE/GROUPING SETS into a UNION ALL over grouping levels.

    Each level groups by its active expressions; the inactive grouping expressions
    are projected as NULL. (Matches DuckDB's row output for non-null group keys.)
    """
    import itertools

    factors = _grouping_factors(group)
    levels = [[e for part in combo for e in part] for combo in itertools.product(*factors)]

    # Every grouping expression that appears in any level, deduped by identity.
    every: dict[str, object] = {}
    for level in levels:
        for e in level:
            every.setdefault(_grouping_key(e), e)

    datasets = [
        tr.select(_grouping_level_node(node, {_grouping_key(e): e for e in level}, every))
        for level in levels
    ]
    out = datasets[0]
    for d in datasets[1:]:
        out = out.union(d, distinct=False)
    return out


def _grouping_level_node(node, active: dict, every: dict):
    """A copy of `node` grouping only by `active`; inactive grouping expressions in
    the SELECT list become NULL so every level shares one output schema."""
    from sqlglot import expressions as exp

    m = node.copy()
    inactive = {k: v for k, v in every.items() if k not in active}

    def typed_null(e):
        # NULLIF(e, e) is a NULL *of the expression's type*; used both as a
        # (constant) group key — so it survives aggregation and the output
        # schema matches across levels — and as the projected value.
        return exp.Nullif(this=e.copy(), expression=e.copy())

    group_exprs = [e.copy() for e in active.values()]
    group_exprs += [typed_null(e) for e in inactive.values()]
    m.set("group", exp.Group(expressions=group_exprs))

    # GROUPING(x, y, ...) is a per-level constant: the integer whose bits mark which
    # of its arguments are rolled up (inactive) in this level, first argument the
    # most-significant bit (DuckDB/SQL-standard). GROUPING_ID(...) is a spelling of the
    # same bit-vector. Replace either with that literal.
    for gnode in list(m.find_all(exp.Grouping, exp.GroupingId)):
        bits = 0
        for arg in gnode.expressions:
            bits = (bits << 1) | (0 if _grouping_key(arg) in active else 1)
        # Paren-wrap so an `ORDER BY GROUPING(a)` constant is not mistaken for a
        # 1-based positional SELECT-item reference.
        gnode.replace(exp.Paren(this=exp.Literal.number(bits)))

    for proj in list(m.expressions):
        inner = proj.this if isinstance(proj, exp.Alias) else proj
        if _grouping_key(inner) in inactive:
            proj.replace(exp.alias_(typed_null(inner), proj.alias_or_name))
    return m


def _expand_star(tr, star, visible: list[str]) -> dict[str, Expr]:
    """Expand `SELECT *` with DuckDB's star modifiers into name -> expression.

    Supports `EXCLUDE`/`EXCEPT (cols)` (drop columns), `REPLACE (expr AS c)`
    (substitute a column's expression, keeping its position), and `RENAME (c AS d)`
    (rename in place). A modifier this translator cannot express is rejected rather
    than silently dropped — ignoring one would return the wrong columns.
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
        out[renamed.get(c, c)] = tr._scalar(replaced[c]) if c in replaced else col(c)
    return out


def _projection_map(tr, ds: Dataset, projections) -> dict[str, Expr]:
    from sqlglot import expressions as exp

    named: dict[str, Expr] = {}
    for p in projections:
        # `SELECT *` (Star) or `SELECT t.*` (a Column wrapping a Star) → keep
        # all current columns. (Qualified `t.*` expands to every column; in a
        # single-table query that is exactly t's columns.)
        if isinstance(p, exp.Star) or (isinstance(p, exp.Column) and isinstance(p.this, exp.Star)):
            star = p if isinstance(p, exp.Star) else p.this
            # Internal columns materialized by UDF hoisting (`__bc_…`) are an
            # implementation detail and must never leak through `*`.
            visible = [c for c in ds.columns if not c.startswith("__bc_")]
            named.update(_expand_star(tr, star, visible))
            continue
        alias = _alias_of(p)
        if tr._is_window(p):
            # The window pass already materialized this column under `alias`.
            named[alias] = col(alias)
            continue
        expr = tr._scalar(_unwrap_alias(p))
        if has_selector(expr):
            # A `COLUMNS(*)` / `COLUMNS('regex')` item (possibly wrapped in a scalar
            # function) expands to one output per matched column, named by that column
            # — reusing the DataFrame selector engine.
            visible = [c for c in ds.columns if not c.startswith("__bc_")]
            named.update(expand_selectors(expr, visible, ds._plan.available_schema()))
        else:
            named[alias] = expr
    return named


def _aggregate(tr, ds: Dataset, projections, group, having) -> Dataset:
    from sqlglot import expressions as exp

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
            # (skip when the alias just re-names a same-named column — that's a real key).
            if isinstance(g, exp.Column) and g.name in select_aliases:
                aliased = select_aliases[g.name]
                if not (isinstance(aliased, exp.Column) and aliased.name == g.name):
                    g = aliased
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
    used_aliases = set(group_cols) | set(group_exprs)
    for p in projections:
        inner = _unwrap_alias(p)
        if isinstance(inner, exp.AggFunc):
            _register_agg(tr, inner, _alias_of(p), used_aliases)
        else:
            for a in inner.find_all(exp.AggFunc):
                if a.find_ancestor(exp.Subquery) is None:
                    _register_agg(tr, a, None, used_aliases)
    if having is not None:
        for a in having.this.find_all(exp.AggFunc):
            if a.find_ancestor(exp.Subquery) is None:
                _register_agg(tr, a, None, used_aliases)

    agg_kwargs = dict(tr._agg_map.values())
    if agg_kwargs:
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

    # Final projection (group keys, aggregate refs, and arithmetic over them).
    # A SELECT item that *is* a GROUP BY expression resolves to that key's
    # materialized column rather than being recomputed.
    named: dict[str, Expr] = {}
    for p in projections:
        out = _alias_of(p)
        inner = _unwrap_alias(p)
        if isinstance(inner, exp.Column) and inner.name in group_cols:
            named[out] = col(inner.name)
        elif inner.sql() in group_expr_alias:
            named[out] = col(group_expr_alias[inner.sql()])
        else:
            named[out] = tr._scalar(inner)
    # NB: `_agg_map` stays live so an ORDER BY over an aggregate can resolve;
    # the caller (`select`) clears it once ordering is done.
    return ds.select(**named)


def _distinct_on(tr, ds: Dataset, projections, order, on_exprs) -> Dataset:
    """`SELECT DISTINCT ON (keys) ... ORDER BY ...` — keep one row per key set.

    Postgres/DuckDB semantics: for each set of rows sharing the ``DISTINCT ON`` key
    expressions, keep the first row in ``ORDER BY`` order, then order the survivors by
    the same ``ORDER BY``. Reuses ``Dataset.distinct(subset, keep="first", order_by=…)``
    (a ``row_number()`` window under the hood) rather than a full-row dedup. The key and
    sort expressions are materialized as internal ``__bc_`` columns first so they stay
    resolvable even when absent from the SELECT list, and never leak through ``SELECT *``.
    """
    from sqlglot import expressions as exp

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
    tr._agg_map[key] = (alias, _agg(tr, node))


def _agg(tr, node) -> AggExpr:
    from sqlglot import expressions as exp

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
        return AggExpr("list_agg", tr._scalar(node.this))
    mapped = _AGG_FUNCS.get(fname)
    if mapped is None:
        raise NotImplementedError(f"unsupported aggregate: {fname}")
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
        raise NotImplementedError(
            f"{fname.upper()}(DISTINCT x) is not supported (only COUNT/MIN/MAX(DISTINCT) are); "
            f"pre-aggregate the distinct values in a subquery"
        )
    return AggExpr(mapped, tr._scalar(arg))
