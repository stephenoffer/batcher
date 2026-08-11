"""Subquery handling and decorrelation for the SQL translator.

Rewrites IN/EXISTS predicates into semi/anti joins and correlated scalar
subqueries into LEFT JOINs. Functions take the translator instance (`tr`) as
their first argument so they can recurse via `tr.statement` / `tr._scalar`.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._sql.parser.core_utils import (
    _factor_common_conjuncts,
    _has_aggregate,
    _join_and,
    _split_and,
)
from batcher._sql.parser.subquery.correlation import (
    _correlation_pair,
    _is_plain_column,
    _local_columns,
    _local_tables,
    _outer_key_reducer,
    _reject_correlated,
)
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import col, lit

#: Prefix of the synthetic boolean an `EXISTS` under `OR` is rewritten to. Leading dunders
#: make it un-typeable as a user column, the same convention `__bt_cse_` and `__jk_l` use.
#: `clauses.py` drops these once the residual predicate that reads them has been applied —
#: keying the cleanup on the prefix rather than threading a list keeps it correct for a
#: nested SELECT, whose own markers are cleared by its own pass.
EXISTS_MARKER_PREFIX = "__exists_"

#: The same, for the columns `_in_marker` adds: the probe value, the joined key and the
#: match bit. Three names rather than one because an `IN` marker has to materialize the
#: outer expression it probes with, which `EXISTS` does not.
IN_MARKER_PREFIX = "__in_"

#: Every synthesized column an under-OR subquery marker leaves on the relation. `clauses.py`
#: drops these once the residual predicate reading them has been applied — one tuple so a
#: new marker kind cannot be added without the cleanup following it.
MARKER_PREFIXES = (EXISTS_MARKER_PREFIX, IN_MARKER_PREFIX)


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

    # An `EXISTS` under `OR` becomes a marker column, and `_exists_marker` attaches it with a
    # LEFT JOIN *immediately* — against whatever `ds` is at that moment. Every other leaf here
    # only contributes to `residual`, which the caller filters with **after** this returns, so
    # for `FROM a, b, c WHERE a.k = b.k AND c.k = a.k AND (EXISTS … OR …)` that moment is the
    # bare comma-join **cross product**: the marker is joined to `a x b x c` and the equalities
    # that would have made it three ordinary joins are applied afterwards.
    #
    # It is quadratic in the FROM width and it is not theoretical. TPC-DS q10 is exactly this
    # shape, and on sf1 (371 MB) it is OOM-killed where DuckDB answers in 31.7 ms. Bisected on
    # the same data, holding the subquery fixed and adding one comma-joined table: **425 ms
    # with one table, 23,858 ms with two** (48x, same answer), dead at three.
    #
    # So apply the column-to-column equalities first, which is what turns that cross product
    # back into joins. Deliberately only `col = col`: they are the comma-join conditions, they
    # carry no subquery, no UDF and no scalar-subquery decorrelation that the caller's residual
    # path handles (`_hoist_udfs`, `_decorrelate_scalar_subqueries`), so moving them earlier
    # cannot change what any of that sees. `AND` commutes, so applying a subset sooner is the
    # same relation — this is predicate pushdown done at build time because the optimizer
    # cannot reorder past a LEFT JOIN that has already been built.
    #
    # Gated on a marker actually being needed, so every query that does not hit the pathology
    # keeps its previous plan exactly.
    if any(i not in handled and _will_markerize(leaf) for i, leaf in enumerate(leaves)):
        for i, leaf in enumerate(leaves):
            if i in handled or not _is_column_equality(leaf):
                continue
            ds = ds.filter(tr._scalar(leaf))
            handled = handled | {i}

    residual = None
    for i, leaf in enumerate(leaves):
        if i in handled:
            continue
        ds, r = _apply_single_predicate(tr, ds, leaf)
        if r is not None:
            residual = r if residual is None else exp.And(this=residual, expression=r)
    return ds, residual


def _is_column_equality(pred) -> bool:
    """Whether `pred` is `<column> = <column>` — a comma-join condition and nothing else.

    Narrow on purpose. This is the only shape `_apply_subquery_predicates` promotes ahead of
    the marker joins, and the promotion is safe precisely because such a predicate holds no
    subquery, no registered UDF and no aggregate, so none of the residual path's later
    rewrites can be looking for it.
    """
    return (
        isinstance(pred, exp.EQ)
        and isinstance(pred.this, exp.Column)
        and isinstance(pred.expression, exp.Column)
    )


def _will_markerize(pred) -> bool:
    """Whether folding `pred` will build a subquery marker (and so a LEFT JOIN on `ds`).

    True for a predicate that *contains* an `EXISTS` or an `IN (SELECT …)` without *being*
    one — the shape `_apply_single_predicate` sends to `_exists_marker` / `_in_marker`. A
    bare one becomes a semi/anti join instead and needs no reordering, which is why it is
    excluded here.
    """
    pred = _unparenthesize(pred)
    bare = pred.this if isinstance(pred, exp.Not) else pred
    bare = _unparenthesize(bare)
    if isinstance(bare, exp.Exists) or _is_in_subquery(bare):
        return False
    return any(True for _ in pred.find_all(exp.Exists)) or any(
        _is_in_subquery(n) for n in pred.find_all(exp.In)
    )


def _unparenthesize(node):
    """`node` with any purely-grouping parentheses peeled off.

    Parentheses carry no meaning of their own, but every shape test below is an
    `isinstance` on the node itself — so ``NOT (x IN (SELECT ...))`` arrived as
    `Not(Paren(In(...)))`, matched nothing, and was refused as "an IN subquery combined
    with OR or other predicates", while the identical ``NOT x IN (SELECT ...)`` and
    ``x NOT IN (SELECT ...)`` both worked. Writing the parentheses is not a different query.
    """
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _apply_single_predicate(tr, ds: Dataset, pred):
    """Fold one WHERE leaf: an IN/EXISTS subquery becomes a join (no residual);
    anything else is returned unchanged as a residual for a normal ``filter``."""
    pred = _unparenthesize(pred)
    # A bare IN-subquery / EXISTS predicate becomes a join (no residual).
    if _is_in_subquery(pred):
        return _apply_in_subquery(tr, ds, pred, negate=False), None
    inner = _unparenthesize(pred.this) if isinstance(pred, exp.Not) else None
    if inner is not None and _is_in_subquery(inner):
        return _apply_in_subquery(tr, ds, inner, negate=True), None
    if isinstance(pred, exp.Exists):
        return _apply_exists(tr, ds, pred, negate=False), None
    if inner is not None and isinstance(inner, exp.Exists):
        return _apply_exists(tr, ds, inner, negate=True), None

    # An `EXISTS` buried under OR cannot become a join — the join would drop rows the OR
    # should keep — but it *can* become a boolean column and be evaluated in place. That is
    # Spark's ExistenceJoin; see `_exists_marker`. Done before the refusal below so the
    # shapes it handles stop being refusals.
    #
    # An `IN` subquery under OR goes the same way, through `_in_marker` — which carries the
    # three-valued answer the earlier boolean-only attempt could not, and probes on a
    # synthesized column so the inner relation cannot capture the outer name. Both of those
    # were the recorded reasons `IN` was excluded here; see `_in_marker`.
    if any(True for _ in pred.find_all(exp.Exists)) or any(
        _is_in_subquery(n) for n in pred.find_all(exp.In)
    ):
        rewritten = pred.copy()
        ok = True
        for found in list(rewritten.find_all(exp.Exists, exp.In)):
            if isinstance(found, exp.In) and not _is_in_subquery(found):
                continue  # `x IN (1, 2, 3)` is an ordinary predicate
            parent, negated = found.parent, False
            if isinstance(parent, exp.Not):
                found, negated = parent, True
            subject = found.this if negated else found
            marker = _exists_marker if isinstance(subject, exp.Exists) else _in_marker
            marked = marker(tr, ds, subject, negate=negated)
            if marked is None:
                ok = False
                break
            ds, replacement = marked
            found.replace(replacement)
        if ok:
            return ds, rewritten

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


def _exists_shape(tr, node):
    """Split an `EXISTS (SELECT …)` into its inner SELECT, correlation equalities and locals.

    Shared by the join rewrite (`_apply_exists`) and the marker-column rewrite
    (`_exists_marker`) so the two cannot disagree about what correlates.

    Returns:
        `(inner, local, local_cols, corr, local_preds)` — the detached inner SELECT, the
        table names it introduces, the columns those tables offer, the `(outer, inner)`
        equality pairs that correlate it, and the predicates local to the inner relation.
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
    return inner, local, local_cols, corr, local_preds


def _exists_marker(tr, ds: Dataset, node, *, negate: bool):
    """`EXISTS (…)` as a boolean *column* on `ds`, for a predicate that cannot become a join.

    A bare `EXISTS` under `AND` folds into a semi/anti join, which is strictly better. But
    `EXISTS (…) OR <anything>` cannot: the join would drop rows the `OR` should keep. Spark
    solves this with an **ExistenceJoin** — a left join that emits, per outer row, a boolean
    saying whether the subquery matched — and then evaluates the original boolean over that
    column. This is that rewrite, spelled with the primitives already here.

    It is exact rather than approximate, and for one specific reason: the inner relation is
    reduced to its *distinct* correlation keys before the join, so a left join against it
    matches each outer row at most once and cannot multiply rows. `EXISTS` is also the one
    subquery form with no three-valued subtlety — it is TRUE or FALSE, never NULL — so the
    marker needs no null reasoning. `IN` under `OR` is deliberately *not* handled here for
    exactly that reason: `x IN (…)` is NULL when `x` is NULL or the list holds a NULL, and a
    boolean marker cannot carry that.

    Args:
        tr: The translator, used to plan the inner SELECT.
        ds: The outer relation the marker is attached to.
        node: The `EXISTS` AST node.
        negate: True for `NOT EXISTS`.

    Returns:
        `(ds, ast)` — the relation carrying the marker, and the boolean AST to substitute
        for the `EXISTS` node — or `None` when the shape is not markerizable, in which case
        the caller reports the original refusal.
    """
    inner, _local, _local_cols, corr, local_preds = _exists_shape(tr, node)

    # A counter on the translator, read defensively: `_sql/parser/translator.py` owns the
    # other `_*_n` counters, and this avoids editing that file to add one more.
    n = getattr(tr, "_exists_n", 0)
    tr._exists_n = n + 1

    if not corr:
        # Uncorrelated: a whole-relation emptiness test, so the answer is the same constant
        # for every outer row. Anything still referencing the outer query here is a range or
        # `<>` correlation, which reshapes the relation rather than yielding a column.
        try:
            _reject_correlated(inner)
        except NotImplementedError:
            return None
        non_empty = tr.statement(inner).limit(1).collect().num_rows > 0
        return ds, exp.true() if non_empty != negate else exp.false()

    # Correlated on equalities: reduce the inner relation to its distinct keys, tag it, and
    # left join. The keys are aliased to generated names first so an inner key that shares an
    # outer column's name cannot collide in the joined schema.
    keys = [f"__ex{n}_k{i}" for i in range(len(corr))]
    marker = f"{EXISTS_MARKER_PREFIX}{n}"
    inner.set("where", exp.Where(this=_join_and(local_preds)) if local_preds else None)
    inner.set("group", None)
    inner.set(
        "expressions",
        [exp.alias_(exp.column(ic), k) for k, (_oc, ic) in zip(keys, corr, strict=True)],
    )
    try:
        _reject_correlated(inner)
    except NotImplementedError:
        return None

    tagged = tr.statement(inner).distinct().with_columns(**{marker: lit(True)})
    ds = ds.join(tagged, left_on=[oc for (oc, _ic) in corr], right_on=keys, how="left")
    # Matched ⇒ the tag survives; unmatched ⇒ the left join null-extends it. That is exactly
    # the existence bit, with no coalesce needed.
    ds = ds.with_columns(**{marker: col(marker).is_not_null()})
    # The equi-join consumes the right-hand key columns, so usually there is nothing left to
    # drop; guard on what is actually present rather than assuming either behaviour.
    leftover = [k for k in keys if k in ds.columns]
    if leftover:
        ds = ds.drop(*leftover)
    ast = exp.column(marker)
    return ds, (exp.Not(this=ast) if negate else ast)


def _in_marker(tr, ds: Dataset, node, *, negate: bool):
    """`x IN (SELECT c …)` as a boolean *column*, for a predicate that cannot become a join.

    The `IN` counterpart to `_exists_marker`, and it has to answer the two objections that
    kept it from existing. Both are recorded in `_apply_single_predicate`, and neither is
    waved away here:

    * **Three-valued logic.** ``x IN (…)`` is NULL — not FALSE — when `x` is NULL, or when
      the set holds a NULL and nothing matches. A bare existence bit cannot carry that, so
      this builds the full three-way answer instead: matched ⇒ TRUE, else NULL when either
      null source applies, else FALSE. Whether the set holds a NULL is one extra probe of
      the (uncorrelated) inner relation, decided before the marker is built.
    * **Name capture.** The earlier attempt rewrote to ``EXISTS (SELECT 1 FROM S WHERE
      c = x)`` and wrote `x` as the user spelled it; for the ordinary
      ``category IN (SELECT category FROM vip)`` that unqualified name rebound to the
      *inner* relation, the predicate became ``category = category``, and every row
      matched. Here the outer value is materialized into a synthesized ``__in<n>_v``
      column and the inner key aliased to ``__in<n>_k``, so neither side can capture the
      other regardless of what the two relations call their columns.

    Only the **uncorrelated** single-column form is handled; anything else returns None and
    the caller reports the original refusal. TPC-DS q45 is the uncorrelated form.

    Args:
        tr: The translator, used to plan the inner SELECT.
        ds: The outer relation the marker is attached to.
        node: The `In` AST node.
        negate: True for `NOT IN`.

    Returns:
        `(ds, ast)` — the relation carrying the marker and the AST to substitute for the
        `IN` node — or None when the shape is not markerizable.
    """
    target = node.this
    if isinstance(target, exp.Tuple):
        return None  # a row-value IN has no single probe column
    inner = _in_subquery_select(node).copy()  # detach from the outer AST scope
    try:
        _reject_correlated(inner)
    except NotImplementedError:
        return None

    n = getattr(tr, "_in_marker_n", 0)
    tr._in_marker_n = n + 1
    probe = f"{IN_MARKER_PREFIX}{n}_v"
    key = f"{IN_MARKER_PREFIX}{n}_k"
    marker = f"{IN_MARKER_PREFIX}{n}_m"

    inner_ds = tr.statement(inner)
    if len(inner_ds.columns) != 1:
        return None
    inner_ds = inner_ds.rename({inner_ds.columns[0]: key})
    # Does the set hold a NULL? It decides whether an unmatched row is FALSE or NULL, and
    # for an uncorrelated set it is one answer for the whole query. `limit(1)` stops at the
    # first one rather than scanning the relation.
    set_has_null = inner_ds.filter(col(key).is_null()).limit(1).collect().num_rows > 0

    values = inner_ds.filter(col(key).is_not_null()).distinct().with_columns(**{marker: lit(True)})
    ds = ds.with_columns(**{probe: tr._scalar(target)})
    ds = ds.join(values, left_on=[probe], right_on=[key], how="left")
    if key in ds.columns:
        ds = ds.drop(key)
    # Matched ⇒ the tag survives; unmatched ⇒ the left join null-extends it. Collapsing that
    # to a real boolean here keeps the substituted AST a plain column reference.
    ds = ds.with_columns(**{marker: col(marker).is_not_null()})

    # `NULLIF(TRUE, TRUE)` is a *boolean* NULL. A bare `NULL` would lower to an Int64 one
    # and clash with the CASE's boolean branches.
    null_bool = exp.Nullif(this=exp.true(), expression=exp.true())
    case = exp.case().when(exp.column(marker), exp.true())
    if set_has_null:
        ast = case.else_(null_bool)
    else:
        probe_is_null = exp.Is(this=exp.column(probe), expression=exp.Null())
        ast = case.when(probe_is_null, null_bool).else_(exp.false())
    return ds, exp.Paren(this=exp.Not(this=ast) if negate else ast)


def _apply_exists(tr, ds: Dataset, node, *, negate: bool) -> Dataset:
    """EXISTS / NOT EXISTS, correlated or not.

    A correlated `EXISTS (SELECT … FROM b WHERE b.k = a.k AND <local>)`
    decorrelates to a SEMI join (anti for NOT EXISTS) of the outer rows with
    `b` filtered by `<local>`, keyed on the correlation equalities.

    An uncorrelated EXISTS is a whole-table keep-or-drop: collect the subquery
    eagerly to test emptiness, then keep or drop every row.
    """
    inner, local, local_cols, corr, local_preds = _exists_shape(tr, node)

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
                # A correlation repeated inside every arm of an `OR` is still a correlation;
                # factoring it back out is what lets it be seen at all (TPC-DS q41).
                for leaf in _split_and(_factor_common_conjuncts(where.this)):
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
