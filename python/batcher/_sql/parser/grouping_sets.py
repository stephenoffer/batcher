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

from batcher._sql.parser.clauses import _is_order_all, _order_all
from batcher._sql.parser.core_utils import _alias_of, _row_window
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import col


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

    ``ORDER BY`` / ``LIMIT`` / ``OFFSET`` sit *above* the union and are applied here,
    once, to the combined result. They are stripped from the per-level nodes first:
    each level is a copy of the whole SELECT, so leaving them in place ran the sort and
    the limit inside every branch. That is wrong twice over — the limit became
    per-level (``ROLLUP(a, b) ... LIMIT 7`` returned 7 + 5 + 1 = 13 rows), and the
    ordering was per-level, so the union's output was not sorted at all. TPC-DS q18/q22
    returned 401 rows against DuckDB's 100.
    """
    import itertools

    # The ORDER BY/LIMIT/OFFSET belong to the union, not to any one level.
    node = node.copy()
    order = node.args.pop("order", None)
    limit = node.args.pop("limit", None)
    offset = node.args.pop("offset", None)
    _pin_grouping_names(node)

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

    if order is not None:
        out = _order_union(tr, out, order, node.expressions)
    if limit is not None or offset is not None:
        n, skip = _row_window(limit, offset)
        out = out.limit(n, offset=skip)
    return out


def _pin_grouping_names(node) -> None:
    """Give every un-aliased ``GROUPING(...)`` select item an explicit output name.

    ``GROUPING(x)`` becomes a different integer literal in every level, and an un-aliased
    select item is named after the expression it holds — so the levels disagreed on the
    output name (``(0)`` against ``(1)``) and the UNION that combines them failed with
    ``union inputs must have identical columns``. Standard reporting SQL
    (``SELECT g, GROUPING(g), count(*) ... GROUP BY ROLLUP(g)``) could not run at all.

    Naming the item once, here, from the expression as *written*, pins it across levels —
    and matches what DuckDB names the same column.
    """
    for proj in list(node.expressions):
        if isinstance(proj, exp.Alias):
            continue
        if next(proj.find_all(exp.Grouping, exp.GroupingId), None) is None:
            continue
        proj.replace(exp.alias_(proj.copy(), _alias_of(proj)))


def _order_union(tr, out: Dataset, order, projections) -> Dataset:
    """Sort the unioned levels by the query's ORDER BY.

    The union carries the *projected* output columns, so an ORDER BY item is resolved
    against the SELECT list by name (its alias, or the SQL text of the item it repeats)
    rather than re-resolved against the source relation, which the union no longer has.
    The 1-based positional form is resolved the same way.

    An item that is an *expression over* the select list — TPC-DS q70's
    ``ORDER BY CASE WHEN grouping(a) + grouping(b) = 0 THEN a END`` — is rewritten term by
    term onto the output columns and then lowered normally, which is the one reading that
    works here: the source columns the expression was written against are gone, but every
    part of it is still projected under some name. An item that reaches a column the union
    does not carry is still rejected rather than silently ignored.
    """
    # `ORDER BY ALL` names no column at all, so it cannot be resolved against the SELECT
    # list the way every other term here is; it means "every output column, left to
    # right". Handled before that resolution rather than falling into it and being
    # rejected as an unresolvable term.
    if _is_order_all(order):
        return _order_all(out, order, list(out.columns))

    # Two maps, deliberately. `by_expr` holds only real expression texts and output names,
    # and is what a *sub*-expression is matched against; `by_text` adds the 1-based
    # positions, which are only ever a whole term. Folding the positions into the
    # sub-expression map would rewrite the `1` in `ORDER BY x + 1` to the first output
    # column — a silent wrong answer.
    by_expr: dict[str, str] = {}
    by_text: dict[str, str] = {}
    for i, p in enumerate(projections):
        name = p.alias_or_name
        if name:
            by_expr.setdefault(p.this.sql() if isinstance(p, exp.Alias) else p.sql(), name)
            by_expr.setdefault(name, name)
        by_text.setdefault(str(i + 1), name)
    by_text = {**by_expr, **by_text}

    columns = set(out.columns)
    keys, desc, nulls_first = [], [], []
    for o in order.expressions:
        target = o.this
        text = target.sql()
        name = by_text.get(text) or by_text.get(target.name)
        if name is None and isinstance(target, exp.Literal) and not target.is_string:
            name = by_text.get(target.this)
        if name is not None and name in columns:
            keys.append(col(name))
        else:
            keys.append(_order_expr_over_outputs(tr, target, by_expr, columns))
        desc.append(bool(o.args.get("desc")))
        nulls_first.append(bool(o.args.get("nulls_first")))
    return out.sort(*keys, descending=desc, nulls_first=nulls_first)


def _order_expr_over_outputs(tr, target, by_expr: dict[str, str], columns: set[str]):
    """Lower an ORDER BY term that computes over the union's output columns."""
    rewritten = _retarget_to_outputs(target.copy(), by_expr, columns)
    missing = sorted({c.name for c in rewritten.find_all(exp.Column)} - columns)
    if missing:
        raise NotImplementedError(
            f"ORDER BY {target.sql()} on a ROLLUP/CUBE/GROUPING SETS query reaches "
            f"{', '.join(missing)}, which the grouped result does not carry; name a column "
            "of the SELECT list, or order in an enclosing query instead"
        )
    return tr._scalar(rewritten)


def _retarget_to_outputs(node, by_expr: dict[str, str], columns: set[str]):
    """Rewrite every sub-expression of `node` that the SELECT list projects to its name.

    Outermost first, so a compound the query already names (``grouping(a) + grouping(b)``
    projected as ``lochierarchy``) becomes that one column instead of being rebuilt from
    parts the grouped relation no longer has.
    """
    name = by_expr.get(node.sql())
    if name is not None and name in columns:
        return exp.column(name)
    for child in list(node.iter_expressions()):
        replacement = _retarget_to_outputs(child, by_expr, columns)
        if replacement is not child:
            child.replace(replacement)
    return node


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
        else:
            _null_inactive_refs(inner, inactive, typed_null)
    having = m.args.get("having")
    if having is not None:
        _null_inactive_refs(having, inactive, typed_null)
    return m


def _null_inactive_refs(node, inactive: dict, typed_null) -> None:
    """NULL out, in place, every *nested* reference to a rolled-up grouping expression.

    The top-level pass above rewrites a select item that *is* an inactive grouping key.
    A reference buried inside a larger expression needs the same treatment and did not get
    it, which is what broke the reporting idiom TPC-DS q70 and q86 are built on:

        rank() OVER (PARTITION BY grouping(a) + grouping(b),
                     CASE WHEN grouping(b) = 0 THEN a END ORDER BY sum(x) DESC)

    At the level that rolls `a` up, the bare `a` inside the CASE is not a column of the
    grouped relation at all, so the window's partition key failed to resolve.

    Aggregate arguments are deliberately skipped: `sum(x)` at a level that rolls `x` up
    still sums the underlying rows. Only the *grouped* reference goes to NULL.
    """
    from batcher._sql.parser.expressions.aggregates import is_agg_node

    for child in list(node.iter_expressions()):
        if is_agg_node(child):
            continue
        if _grouping_key(child) in inactive:
            child.replace(typed_null(child))
        else:
            _null_inactive_refs(child, inactive, typed_null)
