"""Subquery handling and decorrelation for the SQL translator.

Rewrites IN/EXISTS predicates into semi/anti joins and correlated scalar
subqueries into LEFT JOINs. Functions take the translator instance (`tr`) as
their first argument so they can recurse via `tr.statement` / `tr._scalar`.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._sql.parser.core_utils import _has_aggregate, _join_and, _split_and
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import col, lit


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

    from batcher._sql.parser.subquery.neq import _fuse_correlated_neq

    # Flatten the top conjunction so leaves can be co-optimized (two correlated `<>`
    # EXISTS over the same base table fuse into one group-by + join) before each
    # remaining leaf is folded individually.
    leaves = _split_and(pred)
    handled: set[int] = set()
    if len(leaves) >= 2:
        ds, handled = _fuse_correlated_neq(tr, ds, leaves)

    residual = None
    for i, leaf in enumerate(leaves):
        if i in handled:
            continue
        ds, r = _apply_single_predicate(tr, ds, leaf)
        if r is not None:
            residual = r if residual is None else exp.And(this=residual, expression=r)
    return ds, residual


def _apply_single_predicate(tr, ds: Dataset, pred):
    """Fold one WHERE leaf: an IN/EXISTS subquery becomes a join (no residual);
    anything else is returned unchanged as a residual for a normal ``filter``."""
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
    if not isinstance(node, exp.In):
        return False
    query = node.args.get("query")
    return isinstance(query, (exp.Subquery, exp.Select, exp.Union))


def _in_subquery_select(node):
    """Extract the inner SELECT/Union of an ``IN (subquery)`` node."""
    query = node.args.get("query")
    if isinstance(query, exp.Subquery):
        return query.this
    if isinstance(query, (exp.Select, exp.Union)):
        return query
    raise NotImplementedError("IN (subquery) requires a SELECT subquery")


def _apply_in_subquery(tr, ds: Dataset, node, *, negate: bool) -> Dataset:
    inner_select = _in_subquery_select(node).copy()  # detach from outer AST
    target = node.this
    # A plain column, or a row value `(a, b, …)` — a multi-column IN → multi-key semi-join.
    if _is_plain_column(target):
        left_keys = [target.name]
    elif (
        isinstance(target, exp.Tuple)
        and target.expressions
        and all(_is_plain_column(e) for e in target.expressions)
    ):
        left_keys = [e.name for e in target.expressions]
    else:
        raise NotImplementedError("IN (subquery) supports a plain column or a row value of columns")
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
        if len(inner_ds.columns) != len(left_keys):
            raise NotImplementedError("IN subquery must project one column per left-hand column")
        right_keys = list(inner_ds.columns[: len(left_keys)])
        # `x NOT IN (S)` needs SQL three-valued logic, not a plain anti-join (the
        # classic NOT-IN bug — see `_not_in_antijoin`). Handle single-key exactly.
        if negate and len(left_keys) == 1:
            return _not_in_antijoin(ds, left_keys[0], inner_ds, right_keys[0])
        return ds.join(inner_ds.distinct(), left_on=left_keys, right_on=right_keys, how=how)

    # Correlated IN: semi/anti join on (target = projected) AND the correlation
    # equalities, with local predicates applied to the inner relation.
    if len(left_keys) != 1:
        raise NotImplementedError("multi-column IN (subquery) with a correlation is unsupported")
    left_key = left_keys[0]
    if len(inner_select.expressions) != 1:
        raise NotImplementedError("correlated IN subquery must project one column")
    in_col = inner_select.expressions[0]
    inner_select.set("where", exp.Where(this=_join_and(local_preds)) if local_preds else None)
    inner_select.set("expressions", [in_col, *(exp.column(ic) for (_oc, ic) in corr)])
    # A correlated IN whose projection aggregates (`sal IN (SELECT max(sal) …
    # WHERE e2.dept = e.dept)`) is a per-correlation-key aggregate: it must GROUP
    # BY the inner correlation columns, exactly as the scalar decorrelation does.
    # Without the GROUP BY the query mixes an aggregate with a bare key column and
    # errors ("references unknown column(s) ['dept']").
    if _has_aggregate(in_col):
        inner_select.set("group", exp.Group(expressions=[exp.column(ic) for (_oc, ic) in corr]))
    else:
        inner_select.set("group", None)
    _reject_correlated(inner_select)
    inner_ds = tr.statement(inner_select).distinct()
    return ds.join(
        inner_ds,
        left_on=[left_key, *(oc for (oc, _ic) in corr)],
        right_on=[inner_ds.columns[0], *(ic for (_oc, ic) in corr)],
        how=how,
    )


def _not_in_antijoin(ds: Dataset, left_key: str, inner_ds: Dataset, right_key: str) -> Dataset:
    """`x NOT IN (uncorrelated subquery)` with correct SQL three-valued semantics.

    A plain anti-join is wrong three ways: an **empty** set makes NOT IN TRUE for
    every row (even NULL ``x``); a **NULL** anywhere in the set makes it UNKNOWN for
    all rows (none survive); otherwise a NULL ``x`` against a non-empty set is UNKNOWN
    and must drop (anti-join keeps it). NULL/emptiness are probed eagerly (uncorrelated,
    like the EXISTS path); the row-volume anti-join stays lazy.
    """
    key_only = inner_ds.select(right_key)
    if key_only.filter(col(right_key).is_null()).limit(1).collect().num_rows > 0:
        return ds.filter(lit(False))  # a NULL in the set → NOT IN is never TRUE
    if key_only.filter(col(right_key).is_not_null()).limit(1).collect().num_rows == 0:
        return ds  # empty set → NOT IN is TRUE for all rows (NULL x included)
    # Non-empty, NULL-free set: drop NULL outer keys, then anti-join the rest.
    ds = ds.filter(col(left_key).is_not_null())
    return ds.join(key_only.distinct(), left_on=[left_key], right_on=[right_key], how="anti")


def _apply_exists(tr, ds: Dataset, node, *, negate: bool) -> Dataset:
    """EXISTS / NOT EXISTS, correlated or not.

    A correlated `EXISTS (SELECT … FROM b WHERE b.k = a.k AND <local>)`
    decorrelates to a SEMI join (anti for NOT EXISTS) of the outer rows with
    `b` filtered by `<local>`, keyed on the correlation equalities.

    An uncorrelated EXISTS is a whole-table keep-or-drop: collect the subquery
    eagerly to test emptiness, then keep or drop every row.
    """
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

    # A single *inequality* correlation (`a.x < b.y`) is a range semi/anti join: not an
    # equi-key, so it never reaches `corr`, and before this it raised. See `subquery.range`.
    from batcher._sql.parser.subquery.range import decorrelate_inequality_exists

    if not corr:
        ranged = decorrelate_inequality_exists(
            tr, ds, inner, local_preds, local, local_cols, negate
        )
        if ranged is not None:
            return ranged

    if not corr:
        # Uncorrelated: emptiness test → keep or drop every outer row.
        _reject_correlated(inner)
        non_empty = tr.statement(inner).limit(1).collect().num_rows > 0
        keep = non_empty if not negate else (not non_empty)
        return ds if keep else ds.filter(lit(False))

    # A single correlated `<>` residual (`inner.c <> outer.c`) is not an equi-join and not
    # local — it correlates on a value, not a key. It decorrelates to a per-key min/max
    # bound test (`min(c) <> outer.c OR max(c) <> outer.c`), a group-by + join + filter that
    # runs single-node, streaming, and distributed — no row id. Two such subqueries over the
    # same base table fuse into one pass upstream in `_apply_subquery_predicates`. TPC-H q21
    # is exactly this shape. See `subquery_neq`.
    from batcher._sql.parser.subquery.neq import _decorrelate_neq_single, _parse_neq_exists

    spec = _parse_neq_exists(tr, node)
    if spec is not None:
        return _decorrelate_neq_single(tr, ds, spec, negate)

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


def _local_tables(select_node) -> set[str]:
    """Table names + aliases introduced by this SELECT's own FROM/JOINs."""
    local: set[str] = set()
    from_ = select_node.args.get("from") or select_node.args.get("from_")
    sources = []
    if from_ is not None:
        sources.append(from_.this)
    sources += [j.this for j in select_node.args.get("joins", []) or []]
    for t in sources:
        if isinstance(t, exp.Table):
            # An aliased table is referenceable only by its alias — SQL scoping
            # shadows the base name. Adding the base name of an *inner* aliased
            # table (`FROM emp e2`) would misclassify an outer reference that
            # happens to use that base name (`emp.dept`, an unaliased outer `emp`)
            # as local, so the correlation is missed and every group counts all rows.
            if t.alias:
                local.add(t.alias)
            else:
                local.add(t.name)
    return local


def _local_columns(tr, select_node):
    """Column names available from this SELECT's own FROM/JOIN tables.

    Returns ``None`` when any source can't be resolved to a known relation (a derived
    table or unknown name): then unqualified references can't be classified by
    membership and correlation detection falls back to table-qualifier-only.
    """
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


def _correlation_pair(leaf, local: set[str], local_cols: set[str] | None = None, *, op=None):
    """If `leaf` is `outer.col <op> inner.col`, return `(outer_col, inner_col)`.

    Exactly one side must be an outer reference; the other is local. A side is outer
    when it is qualified by a table outside `local`, or — for an unqualified column
    when `local_cols` is known — when its name is not among the local tables' columns
    (TPC-H references outer columns unqualified, e.g. ``l_orderkey = o_orderkey``).
    Otherwise return None (a local predicate).

    `op` is the sqlglot comparison class the leaf must be, defaulting to `exp.EQ`. The
    `<>` decorrelation in `subquery_neq` passes `exp.NEQ`: the outer-reference analysis is
    identical for both, and it was a verbatim copy until it was parameterized here.
    """
    if not isinstance(leaf, op or exp.EQ):
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


def _outer_key_reducer(tr, outer_node, sub, corr):
    """A cheap `SELECT k… FROM T WHERE <T's predicates>` to pre-filter a correlated aggregate
    subquery by (`(ic…) IN (…)`), or None. A per-key aggregate otherwise scans the whole fact
    table for every key, but the enclosing query only produces the keys of its own (often
    filtered) dimension `T` (the base table owning every correlation column). `T` filtered by
    the enclosing conjuncts whose *top-level* columns (not those in a nested subquery, so
    `ps_partkey IN (SELECT …)` counts) all belong to `T` — minus the conjunct carrying the sub
    (circular) — is a superset of the LEFT JOIN's keys, so it changes no surviving group."""
    if outer_node is None or not corr:
        return None
    ocs = [oc for (oc, _ic) in corr]
    from_ = outer_node.args.get("from") or outer_node.args.get("from_")
    sources = ([from_.this] if from_ is not None else []) + [
        j.this for j in outer_node.args.get("joins", []) or []
    ]
    owning = None
    for t in sources:
        if not (isinstance(t, exp.Table) and t.name in tr._registry):
            continue
        tcols = set(tr._registry[t.name].columns)
        if all(oc in tcols for oc in ocs):
            if owning is not None:
                return None  # ambiguous
            owning = t.name
    if owning is None:
        return None
    tcols = set(tr._registry[owning].columns)
    where = outer_node.args.get("where")
    if where is None:
        return None
    tpreds = []
    for leaf in _split_and(where.this):
        nested = set(leaf.find_all(exp.Subquery))
        if sub in nested:
            continue
        top_cols = {
            c.name for c in leaf.find_all(exp.Column) if c.find_ancestor(exp.Subquery) not in nested
        }
        if top_cols and top_cols <= tcols:
            tpreds.append(leaf.copy())
    if not tpreds:
        return None
    cols = [exp.column(o) for o in ocs]
    return exp.select(*cols).from_(exp.table_(owning)).where(_join_and(tpreds))


def _decorrelate_scalar_subqueries(tr, ds: Dataset, roots, outer_node=None) -> Dataset:
    """Rewrite correlated scalar subqueries into LEFT JOINs.

    `(SELECT max(b.v) FROM b WHERE b.k = a.k)` becomes a LEFT JOIN with
    `(SELECT k, max(v) FROM b … GROUP BY k)` keyed on the correlation; the
    subquery node is replaced in place by a reference to the joined column
    (NULL where the outer row has no match — exactly scalar-subquery semantics).
    """
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
            has_agg = any(_has_aggregate(e) for e in m.expressions)
            if has_agg:
                m.set("group", exp.Group(expressions=[exp.column(ic) for (_oc, ic) in corr]))
                # Semi-join reduction (see `_outer_key_reducer`).
                reducer = _outer_key_reducer(tr, outer_node, sub, corr)
                if reducer is not None:
                    ics = [exp.column(ic) for (_oc, ic) in corr]
                    lhs = ics[0] if len(ics) == 1 else exp.Tuple(expressions=ics)
                    in_pred = exp.In(this=lhs, query=reducer)
                    cur = m.args.get("where")
                    combined = exp.and_(cur.this, in_pred) if cur is not None else in_pred
                    m.set("where", exp.Where(this=combined))
            _reject_correlated(m)

            # A GROUP BY already yields one row per key, so a following DISTINCT is a
            # redundant full pass; only a non-aggregate scalar subquery needs it to dedup.
            stmt = tr.statement(m)
            derived = stmt if has_agg else stmt.distinct()
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


def _is_plain_column(node) -> bool:
    return isinstance(node, exp.Column)


def _reject_correlated(select_node) -> None:
    """Raise if `select_node` references a table outside its own FROM/JOINs.

    A correlated subquery refers to a column qualified by an *outer* table.
    We approximate correlation by collecting the table names and aliases the
    subquery introduces and flagging any qualified column outside that set.
    Unqualified columns are assumed local (we cannot resolve them otherwise).
    """
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
