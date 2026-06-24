"""Grouping, aggregation, and projection mapping for the SQL translator.

Covers ROLLUP/CUBE/GROUPING SETS expansion plus the GROUP BY / aggregate path
and the final projection map. Functions take the translator instance (`tr`) as
their first argument.
"""

from __future__ import annotations

from batcher._sql.parser.core_utils import _alias_of, _unwrap_alias
from batcher._sql.parser.literals import _AGG_FUNCS
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import AggExpr, Expr, col


def _grouping_sets_union(tr, node, group) -> Dataset:
    """Expand ROLLUP/CUBE/GROUPING SETS into a UNION ALL over grouping levels.

    Each level groups by its active columns; the inactive grouping columns are
    projected as NULL. (Matches DuckDB's row output for non-null group keys.)
    """
    import itertools

    from sqlglot import expressions as exp

    base = [c.name for c in group.expressions if isinstance(c, exp.Column)]
    levels: list[list[str]]
    if group.args.get("rollup"):
        cols = [c.name for c in group.args["rollup"][0].expressions]
        levels = [cols[:i] for i in range(len(cols), -1, -1)]
    elif group.args.get("cube"):
        cols = [c.name for c in group.args["cube"][0].expressions]
        levels = [
            list(c) for r in range(len(cols), -1, -1) for c in itertools.combinations(cols, r)
        ]
    else:  # GROUPING SETS
        members = group.args["grouping_sets"][0].expressions
        levels = [_grouping_set_columns(s) for s in members]

    every = set(base) | {c for level in levels for c in level}
    datasets = [
        tr.select(_grouping_level_node(node, set(base) | set(level), every)) for level in levels
    ]
    out = datasets[0]
    for d in datasets[1:]:
        out = out.union(d, distinct=False)
    return out


def _grouping_set_columns(node) -> list[str]:
    """Column names in one GROUPING SETS member (`(a, b)` / `(a)` / `()`)."""
    from sqlglot import expressions as exp

    return [c.name for c in node.find_all(exp.Column)]


def _grouping_level_node(node, active: set[str], every: set[str]):
    """A copy of `node` grouping only by `active`; inactive grouping columns in
    the SELECT list become NULL so every level shares one output schema."""
    from sqlglot import expressions as exp

    m = node.copy()
    inactive = every - active

    def typed_null(name: str):
        # NULLIF(col, col) is a NULL *of the column's type*; used both as a
        # (constant) group key — so it survives aggregation and the output
        # schema matches across levels — and as the projected value.
        return exp.Nullif(this=exp.column(name), expression=exp.column(name))

    group_exprs = [exp.column(c) for c in sorted(active)]
    group_exprs += [typed_null(c) for c in sorted(inactive)]
    m.set("group", exp.Group(expressions=group_exprs))

    for proj in list(m.expressions):
        inner = proj.this if isinstance(proj, exp.Alias) else proj
        if isinstance(inner, exp.Column) and inner.name in inactive:
            proj.replace(exp.alias_(typed_null(inner.name), proj.alias_or_name))
    return m


def _projection_map(tr, ds: Dataset, projections) -> dict[str, Expr]:
    from sqlglot import expressions as exp

    named: dict[str, Expr] = {}
    for p in projections:
        # `SELECT *` (Star) or `SELECT t.*` (a Column wrapping a Star) → keep
        # all current columns. (Qualified `t.*` expands to every column; in a
        # single-table query that is exactly t's columns.)
        if isinstance(p, exp.Star) or (isinstance(p, exp.Column) and isinstance(p.this, exp.Star)):
            # Internal columns materialized by UDF hoisting (`__bc_…`) are an
            # implementation detail and must never leak through `*`.
            for c in ds.columns:
                if not c.startswith("__bc_"):
                    named[c] = col(c)
            continue
        alias = _alias_of(p)
        if tr._is_window(p):
            # The window pass already materialized this column under `alias`.
            named[alias] = col(alias)
        else:
            named[alias] = tr._scalar(_unwrap_alias(p))
    return named


def _aggregate(tr, ds: Dataset, projections, group, having) -> Dataset:
    from sqlglot import expressions as exp

    group_cols: list[str] = []
    group_exprs: dict[str, Expr] = {}  # internal alias -> derived key expression
    group_expr_alias: dict[str, str] = {}  # GROUP BY expr SQL text -> alias
    if group is not None:
        for i, g in enumerate(group.expressions):
            # GROUP BY <n> refers to the n-th (1-based) SELECT item.
            if isinstance(g, exp.Literal) and not g.is_string:
                g = _unwrap_alias(projections[int(g.this) - 1])
            if isinstance(g, exp.Column):
                group_cols.append(g.name)
            else:
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
    ds = ds.group_by(*group_cols, **group_exprs).agg(**agg_kwargs)

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
        return AggExpr("list_agg", tr._scalar(node.this))
    mapped = _AGG_FUNCS.get(fname)
    if mapped is None:
        raise NotImplementedError(f"unsupported aggregate: {fname}")
    return AggExpr(mapped, tr._scalar(node.this))
