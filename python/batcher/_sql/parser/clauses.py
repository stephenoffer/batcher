"""SELECT / FROM / JOIN / ORDER clause building for the SQL translator.

Wires the per-theme helpers (subquery, grouping, windowing, scalar) into the
overall SELECT translation. Functions take the translator instance (`tr`) as
their first argument.
"""

from __future__ import annotations

import sys

import pyarrow as pa

from batcher._internal.errors import PlanError
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
    # A self-join (same table aliased twice) is rewritten so the alias-blind column
    # resolver sees distinct, uniquely-named columns.
    tr._rewrite_self_joins(node)
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
    has_agg = group is not None or any(_has_aggregate(p) for p in projections)
    has_window = any(tr._is_window(p) for p in projections)
    if has_agg or has_window:
        _reject_udf_in_agg_window(tr, node, projections)

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
        # Registered scalar functions in the SELECT list become materialized
        # columns before the projection references them.
        ds, projections = tr._hoist_udfs(ds, projections)
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


def _from(tr, node) -> Dataset:
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
            ds = _join_on(tr, ds, right, on, how)
    return ds


def _join_on(tr, ds: Dataset, right: Dataset, on, how: str) -> Dataset:
    """Join on an ``ON`` predicate: equi-keys drive the hash join, the rest post-filters.

    A pure equi-join (``a=b`` or ``a=b AND c=d``) joins directly. A mixed predicate
    (``a=b AND x<y``) keeps the equality conjuncts as join keys and applies the
    remaining conjuncts as a filter on the joined result. A predicate with no
    equality conjunct (a pure theta join) is rejected — the engine join is equi-only.
    """
    eq_pairs, extra = _split_join_on(on)
    if not eq_pairs:
        raise NotImplementedError(
            "join needs at least one equality conjunct (ON a=b); pure non-equi/theta "
            "joins are not supported"
        )
    left_keys = [lk for lk, _ in eq_pairs]
    right_keys = [rk for _, rk in eq_pairs]
    # An ON residual on an outer join can't be a post-join filter — that would drop
    # the null-extended rows. Pre-filter the nullable side instead (or reject).
    if extra is not None and how != "inner":
        ds, right, extra = _outer_join_residual(tr, ds, right, extra, how)
    if extra is not None:
        _reject_ambiguous_residual(extra, ds, right, set(left_keys) | set(right_keys))
    if left_keys == right_keys:
        ds = ds.join(right, on=left_keys, how=how)
    else:
        ds = ds.join(right, left_on=left_keys, right_on=right_keys, how=how)
    if extra is not None:
        ds = ds.filter(tr._scalar(extra))
    return ds


def _outer_join_residual(tr, left: Dataset, right: Dataset, extra, how: str):
    """Resolve a non-equi ON residual on an outer join by pre-filtering the nullable side.

    In ``A LEFT JOIN B ON A.k = B.k AND <residual>``, the residual filters which B
    rows are eligible to match — it is *not* a predicate on the result (B columns are
    null where nothing matched, and those left rows must survive). When the residual
    references only the null-extended side, applying it to that side before the join
    is exactly correct. A residual touching the preserved side, or a FULL join (both
    sides preserved), cannot be expressed this way and is rejected rather than
    silently mis-answered. Returns ``(left, right, remaining_residual_or_None)``.
    """
    from sqlglot import expressions as exp

    refs = {c.name for c in extra.find_all(exp.Column)}
    left_cols, right_cols = set(left.columns), set(right.columns)
    if how == "left" and refs <= right_cols and not (refs & left_cols):
        return left, right.filter(tr._scalar(extra)), None
    if how == "right" and refs <= left_cols and not (refs & right_cols):
        return left.filter(tr._scalar(extra)), right, None
    raise NotImplementedError(
        f"{how} join with a non-equi ON condition that references the preserved side "
        f"(or a FULL join) is not supported; the engine join is equi-only — move the "
        f"condition to a WHERE clause or pre-filter the table"
    )


def _reject_ambiguous_residual(extra, left: Dataset, right: Dataset, keys: set[str]) -> None:
    """Reject a residual join condition that references a name present on both sides.

    The residual is applied as a post-join filter, where table qualifiers are lost
    (``a.v`` and ``b.v`` both resolve to ``v``), so a collision would be evaluated
    against the wrong column. Surface it instead of returning a wrong answer.
    """
    from sqlglot import expressions as exp

    collisions = (set(left.columns) & set(right.columns)) - keys
    referenced = {c.name for c in extra.find_all(exp.Column)}
    ambiguous = sorted(referenced & collisions)
    if ambiguous:
        raise NotImplementedError(
            f"join condition references column(s) {ambiguous} present on both sides; "
            f"rename/alias them or apply the non-equi condition as a post-join filter"
        )


def _and_conjuncts(node) -> list:
    """Flatten an ``AND`` tree (and parentheses) into its conjunct list."""
    from sqlglot import expressions as exp

    if isinstance(node, exp.And):
        return _and_conjuncts(node.this) + _and_conjuncts(node.expression)
    if isinstance(node, exp.Paren):
        return _and_conjuncts(node.this)
    return [node]


def _split_join_on(on):
    """Split an ``ON`` predicate into ``(equi key pairs, residual predicate)``."""
    from sqlglot import expressions as exp

    eq_pairs: list[tuple[str, str]] = []
    residual: list = []
    for conj in _and_conjuncts(on):
        if (
            isinstance(conj, exp.EQ)
            and isinstance(conj.this, exp.Column)
            and isinstance(conj.expression, exp.Column)
        ):
            eq_pairs.append((conj.this.name, conj.expression.name))
        else:
            residual.append(conj)
    extra = None
    for term in residual:
        extra = term if extra is None else exp.And(this=extra, expression=term)
    return eq_pairs, extra


def _table(tr, node) -> Dataset:
    from sqlglot import expressions as exp

    from batcher._sql.parser import udf

    # FROM f(t) — a registered table function (`f` wraps the relation argument).
    if isinstance(node, exp.Table) and isinstance(node.this, exp.Anonymous):
        fname = node.this.name
        rf = tr._functions.get(fname)
        if rf is None:
            raise PlanError(f"unknown table function {fname!r}; registered: {list(tr._functions)}")
        if not rf.table:
            raise PlanError(f"{fname!r} is a scalar function; call it in SELECT, not FROM")
        return _apply_tablesample(udf._apply_table_function(tr, node.this, rf), node)

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
