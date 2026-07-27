"""ROLLUP / CUBE / GROUPING SETS expansion for the SQL translator.

A multi-level GROUP BY is not a distinct execution strategy here: each grouping
level is translated as an ordinary GROUP BY over its active expressions, with the
inactive ones projected as typed NULLs so every level shares one output schema, and
the levels are combined with UNION ALL. That keeps the aggregate path in
`grouping.py` free of level bookkeeping. Functions take the translator instance
(`tr`) as their first argument.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher.api.dataset import Dataset


def _grouping_key(e) -> str:
    """Identity of a grouping expression across levels.

    A bare column is identified by its (unqualified) name so a `GROUP BY a`
    key matches a `SELECT t.a` projection; any other expression is identified
    by its SQL text (`b * 10`, `extract(...)`).
    """
    return e.name if isinstance(e, exp.Column) else e.sql()


def _grouping_set_members(m) -> list:
    """The grouping expressions in one GROUPING SETS member.

    `(a, b)` → `[a, b]`, `(b + 1)` → `[b + 1]`, `()` → `[]`, bare `a` → `[a]`.
    """
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
