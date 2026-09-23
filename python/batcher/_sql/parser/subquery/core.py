"""Subquery handling and decorrelation for the SQL translator.

Rewrites IN/EXISTS predicates into semi/anti joins and correlated scalar
subqueries into LEFT JOINs. Functions take the translator instance (`tr`) as
their first argument so they can recurse via `tr.statement` / `tr._scalar`.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._sql.parser.core_utils import (
    _has_aggregate,
    _join_and,
    _split_and,
)
from batcher._sql.parser.subquery import shape
from batcher._sql.parser.subquery.correlation import (
    _correlation_pair,
    _is_plain_column,
    _local_columns,
    _local_tables,
    _reject_correlated,
)
from batcher._sql.parser.subquery.in_set import in_marker as _in_marker
from batcher._sql.parser.subquery.in_set import not_in_antijoin as _not_in_antijoin
from batcher._sql.parser.subquery.scalar_sub import decorrelate_scalar_subqueries
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import col, lit

#: Prefix of the synthetic boolean an `EXISTS` under `OR` is rewritten to. Leading dunders
#: make it un-typeable as a user column, the same convention `__bt_cse_` and `__jk_l` use.
#: `clauses.py` drops these once the residual predicate that reads them has been applied —
#: keying the cleanup on the prefix rather than threading a list keeps it correct for a
#: nested SELECT, whose own markers are cleared by its own pass.
EXISTS_MARKER_PREFIX = "__bc_exists_"

#: The same, for the columns `_in_marker` adds: the probe value, the joined key and the
#: match bit. Three names rather than one because an `IN` marker has to materialize the
#: outer expression it probes with, which `EXISTS` does not.
IN_MARKER_PREFIX = "__bc_in_"

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
        # An expression has no name to hand a join; `in_expr` names it and comes back here.
        from batcher._sql.parser.subquery.in_expr import in_over_expression

        return in_over_expression(tr, ds, node, negate=negate)
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
    if len(inner_select.expressions) != 1:
        raise NotImplementedError("correlated IN subquery must project one column")
    inner_ds = _correlated_in_set(tr, inner_select, corr, local_preds)
    if inner_ds is None:  # `LIMIT 0` per key: the set is empty for every outer row
        return ds if negate else ds.filter(lit(False))
    if negate:
        return _not_in_antijoin(ds, left_keys[0], inner_ds, inner_ds.columns[0], corr)
    return ds.join(
        inner_ds,
        left_on=[left_keys[0], *(oc for (oc, _ic) in corr)],
        right_on=[inner_ds.columns[0], *(ic for (_oc, ic) in corr)],
        how=how,
    )


def _correlated_in_set(tr, inner_select, corr, local_preds) -> Dataset | None:
    """The distinct `(value, correlation keys…)` rows a correlated `IN` probes, or None if empty.

    A correlated IN whose projection aggregates (`sal IN (SELECT max(sal) … WHERE e2.dept =
    e.dept)`) is a per-correlation-key aggregate: it must GROUP BY the inner correlation
    columns (plus any grouping of its own), exactly as the scalar decorrelation does. An
    ``ORDER BY … LIMIT`` inside it is a per-key top-N, never a cut of the whole inner table.
    """
    in_col = inner_select.expressions[0]
    ics = [ic for (_oc, ic) in corr]
    limit, offset = shape.paging(inner_select)
    one_row = shape.is_one_row_aggregate(inner_select)
    if limit == 0 or (one_row and offset):
        return None
    inner_select.set("where", exp.Where(this=_join_and(local_preds)) if local_preds else None)
    projected = [in_col, *(exp.column(ic) for ic in ics)]
    group = inner_select.args.get("group")
    if group is not None or one_row:
        extra = list(group.expressions) if group is not None else []
        inner_select.set("group", exp.Group(expressions=[*(exp.column(ic) for ic in ics), *extra]))
    ranked = not one_row and (limit is not None or offset > 0)
    if ranked:
        if inner_select.args.get("order") is None:
            value = in_col.this if isinstance(in_col, exp.Alias) else in_col
            inner_select.set("order", exp.Order(expressions=[exp.Ordered(this=value.copy())]))
        projected.append(shape.top_n_window(inner_select, ics, [in_col]))
    shape.strip_paging(inner_select)
    inner_select.set("expressions", projected)
    _reject_correlated(inner_select)
    inner_ds = tr.statement(inner_select)
    if ranked:
        inner_ds = inner_ds.filter(shape.rank_predicate(limit, offset)).drop(shape.RANK_COLUMN)
    return inner_ds.distinct()


def _exists_shape(tr, node):
    """Split an `EXISTS (SELECT …)` into its inner SELECT, correlation equalities and locals.

    Shared by the join rewrite (`_apply_exists`) and the marker-column rewrite
    (`_exists_marker`) so the two cannot disagree about what correlates. The inner SELECT's
    row-count clauses are read here too (`shape.normalize_exists`): a correlated `EXISTS`
    over an ungrouped aggregate, a ``HAVING``, a ``LIMIT`` or an ``OFFSET`` does not mean
    "some inner row matches", and both rewrites must see the same answer to what it means.

    Returns:
        `(inner, local, local_cols, corr, local_preds, plan)` — the detached inner SELECT,
        the table names it introduces, the columns those tables offer, the `(outer, inner)`
        equality pairs that correlate it, the predicates local to the inner relation, and
        the `shape.ExistsPlan` saying how to answer it.
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
    plan = shape.normalize_exists(inner, bool(corr) or _reaches_outside(inner))
    return inner, local, local_cols, corr, local_preds, plan


def _reaches_outside(inner) -> bool:
    try:
        _reject_correlated(inner)
    except NotImplementedError:
        return True
    return False


def _exists_keys(tr, inner, corr, local_preds, plan, aliases=None) -> Dataset:
    """The distinct inner correlation keys an equality-correlated `EXISTS` matches against."""
    inner.set("where", exp.Where(this=_join_and(local_preds)) if local_preds else None)
    keys = [exp.column(ic) for (_oc, ic) in corr]
    group = inner.args.get("group")
    if plan.keep_group and group is not None:
        # Per key, a group must survive HAVING: group by the keys *and* the inner grouping.
        inner.set("group", exp.Group(expressions=[*keys, *group.expressions]))
    else:
        inner.set("group", None)
        inner.set("having", None)
    names = aliases or [None] * len(keys)
    inner.set(
        "expressions",
        [k if a is None else exp.alias_(k.copy(), a) for k, a in zip(keys, names, strict=True)],
    )
    _reject_correlated(inner)
    return tr.statement(inner).distinct()


def _predicate_plan(tr, ds: Dataset, plan, corr, negate: bool):
    """`(ds, ast)` answering an `EXISTS` through a per-key scalar subquery (`plan.predicate`)."""
    if not corr:
        raise NotImplementedError(
            "a correlated EXISTS with HAVING or OFFSET needs an equality correlation "
            "(inner.c = outer.c); rewrite the other correlations as a join"
        )
    predicate = plan.predicate.copy()
    ds = decorrelate_scalar_subqueries(tr, ds, [predicate])
    return ds, (exp.Not(this=exp.Paren(this=predicate)) if negate else predicate)


def _exists_marker(tr, ds: Dataset, node, *, negate: bool):
    """`EXISTS (…)` as a boolean *column* on `ds`, for a predicate that cannot become a join.

    A bare `EXISTS` under `AND` folds into a semi/anti join, which is strictly better. But
    `EXISTS (…) OR <anything>` cannot: the join would drop rows the `OR` should keep. Spark
    solves this with an **ExistenceJoin** — a left join that emits, per outer row, a boolean
    saying whether the subquery matched — and then evaluates the original boolean over that
    column. This is that rewrite, spelled with the primitives already here. It also serves
    an `EXISTS` in the SELECT list, which is the same column read as a value.

    It is exact rather than approximate, and for one specific reason: the inner relation is
    reduced to its *distinct* correlation keys before the join, so a left join against it
    matches each outer row at most once and cannot multiply rows. `EXISTS` is also the one
    subquery form with no three-valued subtlety — it is TRUE or FALSE, never NULL — so the
    marker needs no null reasoning.

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
    inner, _local, _local_cols, corr, local_preds, plan = _exists_shape(tr, node)
    if plan.constant is not None:
        return ds, exp.true() if plan.constant != negate else exp.false()
    if plan.predicate is not None:
        if not corr:
            return None
        return _predicate_plan(tr, ds, plan, corr, negate)

    # A counter on the translator, read defensively: `_sql/parser/translator.py` owns the
    # other `_*_n` counters, and this avoids editing that file to add one more.
    n = getattr(tr, "_exists_n", 0)
    tr._exists_n = n + 1
    marker = f"{EXISTS_MARKER_PREFIX}{n}"

    if not corr:
        # Uncorrelated: a whole-relation emptiness test, so the answer is the same constant
        # for every outer row, probed now. Anything still referencing the outer query here is
        # a range or `<>` correlation, which reshapes the relation rather than yielding a column.
        try:
            _reject_correlated(inner)
        except NotImplementedError:
            return None
        non_empty = tr.statement(inner).limit(1).collect().num_rows > 0
        return ds, exp.true() if non_empty != negate else exp.false()

    # Correlated on equalities: reduce the inner relation to its distinct keys, tag it, and
    # left join. The keys are aliased to generated names first so an inner key that shares an
    # outer column's name cannot collide in the joined schema.
    keys = [f"__bc_ex{n}_k{i}" for i in range(len(corr))]
    try:
        tagged = _exists_keys(tr, inner, corr, local_preds, plan, keys)
    except NotImplementedError:
        return None
    tagged = tagged.with_columns(**{marker: lit(True)})
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


def exists_in_projection(tr, ds: Dataset, node) -> Dataset:
    """Turn each `EXISTS (…)` in a SELECT list into the marker column `_exists_marker` builds.

    `SELECT id, EXISTS (SELECT 1 FROM u WHERE u.k = t.id) AS f FROM t` reads the existence
    bit as a value, which is exactly the column the under-`OR` rewrite already produces.

    Args:
        tr: The translator.
        ds: The relation the SELECT list is evaluated over.
        node: The `Select` whose items are rewritten in place.

    Returns:
        `ds` carrying one marker column per `EXISTS`.

    Raises:
        NotImplementedError: The query aggregates (a per-row bit has no value per group), or
            the `EXISTS` correlates through something other than equalities.
    """
    found = [
        e
        for item in node.expressions
        for e in item.find_all(exp.Exists)
        # Spark's `exists(array, x -> …)` parses to the same node over a list, not a query.
        if e.find_ancestor(exp.Select) is node and isinstance(e.this, (exp.Select, exp.Subquery))
    ]
    if not found:
        return ds
    # An aggregate inside the EXISTS belongs to the subquery, not to this SELECT.
    outer_items = [p.copy() for p in node.expressions]
    for item in outer_items:
        for e in list(item.find_all(exp.Exists)):
            if isinstance(e.this, (exp.Select, exp.Subquery)):
                e.replace(exp.true())
    if node.args.get("group") is not None or any(_has_aggregate(p) for p in outer_items):
        raise NotImplementedError(
            "EXISTS in the SELECT list of an aggregating query is not supported; compute it "
            "in a subquery (SELECT *, EXISTS (…) AS e FROM t) and aggregate over that"
        )
    for e in found:
        marked = _exists_marker(tr, ds, e, negate=False)
        if marked is None:
            raise NotImplementedError(
                "EXISTS in the SELECT list is supported for an uncorrelated subquery or one "
                "correlated by equalities (inner.c = outer.c); rewrite this one as a LEFT JOIN"
            )
        ds, ast = marked
        e.replace(ast)
    return ds


def _apply_exists(tr, ds: Dataset, node, *, negate: bool) -> Dataset:
    """EXISTS / NOT EXISTS, correlated or not.

    A correlated `EXISTS (SELECT … FROM b WHERE b.k = a.k AND <local>)`
    decorrelates to a SEMI join (anti for NOT EXISTS) of the outer rows with
    `b` filtered by `<local>`, keyed on the correlation equalities.

    An uncorrelated EXISTS is a whole-table keep-or-drop, decided by probing the subquery for
    one row while translating.
    """
    inner, local, local_cols, corr, local_preds, plan = _exists_shape(tr, node)
    if plan.constant is not None:
        return ds if plan.constant != negate else ds.filter(lit(False))
    if plan.predicate is not None:
        ds, ast = _predicate_plan(tr, ds, plan, corr, negate)
        return ds.filter(tr._scalar(ast))

    if not plan.keep_group:
        # A pure inequality correlation, or an equality carrying one alongside it, each has
        # its own plan and neither is the semi join below. See `subquery.specialized`.
        from batcher._sql.parser.subquery.specialized import decorrelate_correlated_exists

        special = decorrelate_correlated_exists(
            tr, ds, inner, corr, local_preds, local, local_cols, negate
        )
        if special is not None:
            return special

    if not corr:
        # Uncorrelated: an emptiness test, probed now (`LIMIT 1`), keeps or drops every row.
        _reject_correlated(inner)
        non_empty = tr.statement(inner).limit(1).collect().num_rows > 0
        return ds if non_empty != negate else ds.filter(lit(False))

    # A single correlated `<>` residual (`inner.c <> outer.c`) is not an equi-join and not
    # local — it correlates on a value, not a key. It decorrelates to a per-key min/max
    # bound test (`min(c) <> outer.c OR max(c) <> outer.c`), a group-by + join + filter that
    # runs single-node, streaming, and distributed — no row id. Two such subqueries over the
    # same base table fuse into one pass upstream in `_apply_subquery_predicates`. TPC-H q21
    # is exactly this shape. See `subquery_neq`.
    from batcher._sql.parser.subquery.neq import _decorrelate_neq_single, _parse_neq_exists

    spec = _parse_neq_exists(tr, node) if not plan.keep_group else None
    if spec is not None:
        return _decorrelate_neq_single(tr, ds, spec, negate)

    # Correlated → semi/anti join on the correlation keys, with the local
    # (non-correlated) predicates applied to the inner relation.
    inner_ds = _exists_keys(tr, inner, corr, local_preds, plan)
    how = "anti" if negate else "semi"
    return ds.join(
        inner_ds,
        left_on=[oc for (oc, _ic) in corr],
        right_on=[ic for (_oc, ic) in corr],
        how=how,
    )


#: The scalar decorrelation lives in `scalar_sub`; the translator reaches it by this name.
_decorrelate_scalar_subqueries = decorrelate_scalar_subqueries
