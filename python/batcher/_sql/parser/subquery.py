"""Subquery handling and decorrelation for the SQL translator.

Rewrites IN/EXISTS predicates into semi/anti joins and correlated scalar
subqueries into LEFT JOINs. Functions take the translator instance (`tr`) as
their first argument so they can recurse via `tr.statement` / `tr._scalar`.
"""

from __future__ import annotations

from batcher._sql.parser.core_utils import _has_aggregate
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import lit


def _apply_subquery_predicates(tr, ds: Dataset, pred):
    """Rewrite WHERE predicates that must become dataset operations.

    Handles the subquery forms that cannot be expressed as a boolean column
    expression and instead reshape the dataset:

    * ``x IN (SELECT ...)``      → semi-join
    * ``x NOT IN (SELECT ...)``  → anti-join
    * ``EXISTS (SELECT ...)``    → keep / drop all rows (uncorrelated)
    * ``NOT EXISTS (SELECT ...)``→ keep / drop all rows (uncorrelated)

    Multiple such predicates joined by AND are chained. Returns the
    (possibly transformed) dataset and the *residual* boolean predicate that
    still needs a normal ``filter`` (or ``None`` if nothing remains). Any
    subquery combined with OR (or otherwise un-foldable into a join) raises
    NotImplementedError.
    """
    from sqlglot import expressions as exp

    # Split a conjunction into its leaf predicates so each can be inspected.
    if isinstance(pred, exp.And):
        ds, left = _apply_subquery_predicates(tr, ds, pred.this)
        ds, right = _apply_subquery_predicates(tr, ds, pred.expression)
        if left is not None and right is not None:
            return ds, exp.And(this=left, expression=right)
        return ds, (left if left is not None else right)

    # A bare IN-subquery / EXISTS predicate becomes a join (no residual).
    if _is_in_subquery(pred):
        return _apply_in_subquery(tr, ds, pred, negate=False), None
    if isinstance(pred, exp.Not) and _is_in_subquery(pred.this):
        return _apply_in_subquery(tr, ds, pred.this, negate=True), None
    if isinstance(pred, exp.Exists):
        return _apply_exists(tr, ds, pred, negate=False), None
    if isinstance(pred, exp.Not) and isinstance(pred.this, exp.Exists):
        return _apply_exists(tr, ds, pred.this, negate=True), None

    # Guard: a subquery buried under OR / arbitrary boolean structure cannot
    # be folded into a join. (Scalar subqueries are fine — those resolve to a
    # literal in `_scalar` — so only reject IN/EXISTS subqueries here.)
    if any(
        _is_in_subquery(n) or isinstance(n, exp.Exists) for n in pred.find_all(exp.In, exp.Exists)
    ):
        raise NotImplementedError(
            "IN/EXISTS subquery combined with OR or other predicates "
            "in a way that cannot become a join is not supported"
        )

    return ds, pred


def _is_in_subquery(node) -> bool:
    from sqlglot import expressions as exp

    if not isinstance(node, exp.In):
        return False
    query = node.args.get("query")
    return isinstance(query, (exp.Subquery, exp.Select, exp.Union))


def _in_subquery_select(node):
    """Extract the inner SELECT/Union of an ``IN (subquery)`` node."""
    from sqlglot import expressions as exp

    query = node.args.get("query")
    if isinstance(query, exp.Subquery):
        return query.this
    if isinstance(query, (exp.Select, exp.Union)):
        return query
    raise NotImplementedError("IN (subquery) requires a SELECT subquery")


def _apply_in_subquery(tr, ds: Dataset, node, *, negate: bool) -> Dataset:
    from sqlglot import expressions as exp

    inner_select = _in_subquery_select(node).copy()  # detach from outer AST
    target = node.this
    if not _is_plain_column(target):
        raise NotImplementedError(
            "IN (subquery) supports a single plain column on the left-hand side"
        )
    left_key = target.name
    how = "anti" if negate else "semi"

    # Split the subquery WHERE into correlation equalities and local predicates.
    local = _local_tables(inner_select)
    local_cols = _local_columns(tr, inner_select)
    where = inner_select.args.get("where")
    corr, local_preds = [], []
    if where is not None:
        for leaf in _split_and(where.this):
            pair = _correlation_pair(leaf, local, local_cols)
            (corr if pair is not None else local_preds).append(pair or leaf)

    if not corr:
        _reject_correlated(inner_select)
        inner_ds = tr.statement(inner_select)
        if len(inner_ds.columns) != 1:
            raise NotImplementedError(
                "IN (subquery) requires the subquery to project exactly one column"
            )
        return ds.join(inner_ds.distinct(), left_on=left_key, right_on=inner_ds.columns[0], how=how)

    # Correlated IN: semi/anti join on (target = projected) AND the correlation
    # equalities, with local predicates applied to the inner relation.
    if len(inner_select.expressions) != 1:
        raise NotImplementedError("correlated IN subquery must project one column")
    in_col = inner_select.expressions[0]
    inner_select.set("where", exp.Where(this=_join_and(local_preds)) if local_preds else None)
    inner_select.set("group", None)
    inner_select.set("expressions", [in_col, *(exp.column(ic) for (_oc, ic) in corr)])
    _reject_correlated(inner_select)
    inner_ds = tr.statement(inner_select).distinct()
    return ds.join(
        inner_ds,
        left_on=[left_key, *(oc for (oc, _ic) in corr)],
        right_on=[inner_ds.columns[0], *(ic for (_oc, ic) in corr)],
        how=how,
    )


def _apply_exists(tr, ds: Dataset, node, *, negate: bool) -> Dataset:
    """EXISTS / NOT EXISTS, correlated or not.

    A correlated `EXISTS (SELECT … FROM b WHERE b.k = a.k AND <local>)`
    decorrelates to a SEMI join (anti for NOT EXISTS) of the outer rows with
    `b` filtered by `<local>`, keyed on the correlation equalities.

    An uncorrelated EXISTS is a whole-table keep-or-drop: collect the subquery
    eagerly to test emptiness, then keep or drop every row.
    """
    from sqlglot import expressions as exp

    inner = node.this
    if isinstance(inner, exp.Subquery):
        inner = inner.this
    inner = inner.copy()  # detach from the outer AST scope

    local = _local_tables(inner)
    local_cols = _local_columns(tr, inner)
    where = inner.args.get("where")
    corr, local_preds = [], []
    if where is not None:
        for leaf in _split_and(where.this):
            pair = _correlation_pair(leaf, local, local_cols)
            (corr if pair is not None else local_preds).append(pair or leaf)

    if not corr:
        # Uncorrelated: emptiness test → keep or drop every outer row.
        _reject_correlated(inner)
        non_empty = tr.statement(inner).limit(1).collect().num_rows > 0
        keep = non_empty if not negate else (not non_empty)
        return ds if keep else ds.filter(lit(False))

    # Correlated → semi/anti join on the correlation keys, with the local
    # (non-correlated) predicates applied to the inner relation.
    inner.set("where", exp.Where(this=_join_and(local_preds)) if local_preds else None)
    inner.set("group", None)
    inner.set("expressions", [exp.column(ic) for (_oc, ic) in corr])
    _reject_correlated(inner)  # any remaining outer ref is unsupported
    inner_ds = tr.statement(inner).distinct()
    how = "anti" if negate else "semi"
    return ds.join(
        inner_ds,
        left_on=[oc for (oc, _ic) in corr],
        right_on=[ic for (_oc, ic) in corr],
        how=how,
    )


def _split_and(pred) -> list:
    """Flatten a conjunction into its leaf predicates."""
    from sqlglot import expressions as exp

    out: list = []
    stack = [pred]
    while stack:
        p = stack.pop()
        if isinstance(p, exp.And):
            stack.extend((p.this, p.expression))
        elif isinstance(p, exp.Paren):
            stack.append(p.this)
        else:
            out.append(p)
    return out


def _join_and(preds):
    """Re-combine leaf predicates into a single AND chain."""
    from sqlglot import expressions as exp

    out = preds[0]
    for p in preds[1:]:
        out = exp.And(this=out, expression=p)
    return out


def _local_tables(select_node) -> set[str]:
    """Table names + aliases introduced by this SELECT's own FROM/JOINs."""
    from sqlglot import expressions as exp

    local: set[str] = set()
    from_ = select_node.args.get("from") or select_node.args.get("from_")
    sources = []
    if from_ is not None:
        sources.append(from_.this)
    sources += [j.this for j in select_node.args.get("joins", []) or []]
    for t in sources:
        if isinstance(t, exp.Table):
            local.add(t.name)
            if t.alias:
                local.add(t.alias)
    return local


def _local_columns(tr, select_node):
    """Column names available from this SELECT's own FROM/JOIN tables.

    Returns ``None`` when any source can't be resolved to a known relation (a derived
    table or unknown name): then unqualified references can't be classified by
    membership and correlation detection falls back to table-qualifier-only.
    """
    from sqlglot import expressions as exp

    from_ = select_node.args.get("from") or select_node.args.get("from_")
    sources = ([from_.this] if from_ is not None else []) + [
        j.this for j in select_node.args.get("joins", []) or []
    ]
    cols: set[str] = set()
    for t in sources:
        if isinstance(t, exp.Table) and t.name in tr._registry:
            cols |= set(tr._registry[t.name].columns)
        else:
            return None
    return cols


def _correlation_pair(leaf, local: set[str], local_cols: set[str] | None = None):
    """If `leaf` is `outer.col = inner.col`, return `(outer_col, inner_col)`.

    Exactly one side must be an outer reference; the other is local. A side is outer
    when it is qualified by a table outside `local`, or — for an unqualified column
    when `local_cols` is known — when its name is not among the local tables' columns
    (TPC-H references outer columns unqualified, e.g. ``l_orderkey = o_orderkey``).
    Otherwise return None (a local predicate).
    """
    from sqlglot import expressions as exp

    if not isinstance(leaf, exp.EQ):
        return None
    lhs, rhs = leaf.this, leaf.expression
    if not (isinstance(lhs, exp.Column) and isinstance(rhs, exp.Column)):
        return None

    def _is_outer(c) -> bool:
        if c.table:
            return c.table not in local
        return local_cols is not None and c.name not in local_cols

    lhs_outer, rhs_outer = _is_outer(lhs), _is_outer(rhs)
    if lhs_outer and not rhs_outer:
        return (lhs.name, rhs.name)
    if rhs_outer and not lhs_outer:
        return (rhs.name, lhs.name)
    return None


def _decorrelate_scalar_subqueries(tr, ds: Dataset, roots) -> Dataset:
    """Rewrite correlated scalar subqueries into LEFT JOINs.

    `(SELECT max(b.v) FROM b WHERE b.k = a.k)` becomes a LEFT JOIN with
    `(SELECT k, max(v) FROM b … GROUP BY k)` keyed on the correlation; the
    subquery node is replaced in place by a reference to the joined column
    (NULL where the outer row has no match — exactly scalar-subquery semantics).
    """
    from sqlglot import expressions as exp

    for root in roots:
        if root is None:
            continue
        for sub in list(root.find_all(exp.Subquery)):
            inner = sub.this
            if not isinstance(inner, exp.Select):
                continue
            local = _local_tables(inner)
            local_cols = _local_columns(tr, inner)
            where = inner.args.get("where")
            corr, local_preds = [], []
            if where is not None:
                for leaf in _split_and(where.this):
                    pair = _correlation_pair(leaf, local, local_cols)
                    (corr if pair is not None else local_preds).append(pair or leaf)
            if not corr:
                continue  # uncorrelated scalar subquery → handled eagerly in _scalar
            if len(inner.expressions) != 1:
                raise NotImplementedError("scalar subquery must project one value")

            alias = f"__scalar_{tr._scalar_sub_n}"
            jk = [f"__jk_{tr._scalar_sub_n}_{i}" for i in range(len(corr))]
            tr._scalar_sub_n += 1

            m = inner.copy()
            value = m.expressions[0]
            value = value.this if isinstance(value, exp.Alias) else value
            m.set("where", exp.Where(this=_join_and(local_preds)) if local_preds else None)
            m.set(
                "expressions",
                [exp.alias_(exp.column(ic), k) for (k, (_oc, ic)) in zip(jk, corr, strict=True)]
                + [exp.alias_(value, alias)],
            )
            if any(_has_aggregate(e) for e in m.expressions):
                m.set("group", exp.Group(expressions=[exp.column(ic) for (_oc, ic) in corr]))
            _reject_correlated(m)

            derived = tr.statement(m).distinct()
            ds = ds.join(
                derived,
                left_on=[oc for (oc, _ic) in corr],
                right_on=jk,
                how="left",
            )
            # The "COUNT bug": COUNT over an empty correlated group is 0, but
            # the LEFT JOIN yields NULL for an unmatched outer row — coalesce it.
            if isinstance(value, exp.Count):
                sub.replace(
                    exp.Coalesce(this=exp.column(alias), expressions=[exp.Literal.number(0)])
                )
            else:
                sub.replace(exp.column(alias))
    return ds


def _rewrite_self_joins(tr, select_node) -> None:
    """Rewrite a self-join so the alias-blind column resolver sees distinct columns.

    The translator resolves a column by name only, so two aliases of one table
    (``nation n1, nation n2``) would collapse onto the same physical columns. Each
    aliased instance of a table that appears more than once in this SELECT's FROM is
    wrapped in a subquery that renames its columns to flat ``alias__col`` names, and
    every ``alias.col`` reference in this scope is rewritten to match — so downstream
    translation sees uniquely-named, unqualified columns. A duplicated table whose
    columns can't be enumerated (an unknown name) is rejected rather than mis-answered.
    """
    from sqlglot import expressions as exp

    from_ = select_node.args.get("from") or select_node.args.get("from_")
    if from_ is None:
        return
    sources = [from_.this, *(j.this for j in select_node.args.get("joins", []) or [])]
    names = [t.name for t in sources if isinstance(t, exp.Table)]
    dups = {n for n in names if names.count(n) > 1}
    if not dups:
        return

    alias_map: dict[str, dict[str, str]] = {}
    for t in sources:
        if isinstance(t, exp.Table) and t.name in dups:
            if t.name not in tr._registry:
                raise NotImplementedError(
                    f"self-join on {t.name!r} is not supported (its columns can't be "
                    f"enumerated to disambiguate the aliases)"
                )
            alias = t.alias or t.name
            cols = list(tr._registry[t.name].columns)
            flat = {c: f"{alias}__{c}" for c in cols}
            alias_map[alias] = flat
            inner = exp.Select(expressions=[exp.alias_(exp.column(c), flat[c]) for c in cols])
            inner = inner.from_(exp.table_(t.name))
            t.replace(exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(alias))))

    # Preserve output names: a bare `alias.col` projected directly would otherwise be
    # named `alias__col` instead of `col`.
    for p in list(select_node.expressions):
        if isinstance(p, exp.Column) and p.table in alias_map and p.name in alias_map[p.table]:
            p.replace(exp.alias_(exp.column(alias_map[p.table][p.name]), p.name))

    # Flatten every remaining `alias.col` reference in this SELECT's own scope.
    for c in list(select_node.find_all(exp.Column)):
        if (
            c.table in alias_map
            and c.name in alias_map[c.table]
            and c.find_ancestor(exp.Select) is select_node
        ):
            c.replace(exp.column(alias_map[c.table][c.name]))


def _is_plain_column(node) -> bool:
    from sqlglot import expressions as exp

    return isinstance(node, exp.Column)


def _reject_correlated(select_node) -> None:
    """Raise if `select_node` references a table outside its own FROM/JOINs.

    A correlated subquery refers to a column qualified by an *outer* table.
    We approximate correlation by collecting the table names and aliases the
    subquery introduces and flagging any qualified column outside that set.
    Unqualified columns are assumed local (we cannot resolve them otherwise).
    """
    from sqlglot import expressions as exp

    local: set[str] = set()
    for t in select_node.find_all(exp.Table):
        local.add(t.name)
        if t.alias:
            local.add(t.alias)
    for sub in select_node.find_all(exp.Subquery):
        if sub.alias:
            local.add(sub.alias)

    for c in select_node.find_all(exp.Column):
        tbl = c.table
        if tbl and tbl not in local:
            raise NotImplementedError("correlated subqueries not supported")
