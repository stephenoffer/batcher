"""Correlated ``<>``-residual EXISTS/NOT EXISTS decorrelation (TPC-H q21 shape).

A correlated ``EXISTS (SELECT * FROM b WHERE b.k = a.k AND b.c <> a.c [AND local])``
is not an equi-semijoin — the ``<>`` correlates on a *value*, not a key. It
decorrelates to a per-key ``min(c)``/``max(c)`` (over the locally-filtered rows)
joined back to the outer and bound-tested: a differing value exists iff
``min(c) <> a.c OR max(c) <> a.c`` (an empty group → no differing value, so NOT
EXISTS holds). Group-by + join + filter — every step runs single-node, streaming,
and distributed, with no row id (which is not streamable).

When one WHERE conjunction holds two such subqueries over the *same* base table and
correlation keys (q21's EXISTS + NOT EXISTS over ``lineitem`` by ``l_orderkey``),
they are **fused** into one group-by + one join: each member contributes its own
``min``/``max``, made conditional (``min(c) FILTER (WHERE local)``, via a value-typed
null ``nullif(c, c)`` in the else) when it carries a local filter. One pass over the
inner instead of two — the same reduction Kyber's mergeable aggregates rely on.

This is a *neutral* front-end helper on the SQL translator; it builds the same
`Dataset` operations a hand-written query would, so the optimizer and executor treat
it identically. Shared low-level parsers live in `subquery` (imported lazily by the
translator to avoid an import cycle).
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._sql.parser.core_utils import _join_and, _split_and
from batcher._sql.parser.subquery.correlation import (
    _correlation_pair,
    _local_columns,
    _local_tables,
    _reject_correlated,
)
from batcher.plan.expr_ir import col, nullif, when


def _correlation_neq(leaf, local: set[str], local_cols: set[str] | None = None):
    """If `leaf` is `outer.col <> inner.col`, return `(outer_col, inner_col)`, else None.

    The `<>` analogue of `_correlation_pair`, and nothing more: the outer-reference test is
    the same one, so this passes `exp.NEQ` rather than restating it."""
    return _correlation_pair(leaf, local, local_cols, op=exp.NEQ)


def _all_local(preds, local: set[str], local_cols: set[str] | None) -> bool:
    """Whether every column in `preds` is local to the subquery (no outer reference)."""
    for p in preds:
        for c in p.find_all(exp.Column):
            outer = (c.table and c.table not in local) or (
                not c.table and local_cols is not None and c.name not in local_cols
            )
            if outer:
                return False
    return True


def _base_tables(select_node) -> frozenset[str]:
    """Base table names in this SELECT's FROM/JOINs (ignoring aliases).

    Two correlated subqueries alias the same base table differently (q21's `l2`/`l3`),
    so fusion compatibility keys on the base name, not the alias."""
    from_ = select_node.args.get("from") or select_node.args.get("from_")
    sources = ([from_.this] if from_ is not None else []) + [
        j.this for j in select_node.args.get("joins", []) or []
    ]
    return frozenset(t.name for t in sources if isinstance(t, exp.Table))


def _parse_neq_exists(tr, exists_node):
    """Parse an EXISTS body into a correlated-`<>` spec, or None if it is a different shape.

    Returns a dict with the inner SELECT (a detached copy), its local tables, the
    correlation equi-keys `[(outer, inner), …]`, the single `<>` residual columns
    `(outer_c, inner_c)`, and the remaining purely-local predicates `true_local`."""
    inner = exists_node.this
    if isinstance(inner, exp.Subquery):
        inner = inner.this
    inner = inner.copy()  # detach from the outer AST scope
    local = _local_tables(inner)
    local_cols = _local_columns(tr, inner)
    where = inner.args.get("where")
    if where is None:
        return None
    corr, local_preds = [], []
    for leaf in _split_and(where.this):
        pair = _correlation_pair(leaf, local, local_cols)
        (corr if pair is not None else local_preds).append(pair or leaf)
    if not corr:
        return None
    neq_residuals, true_local = [], []
    for leaf in local_preds:
        npair = _correlation_neq(leaf, local, local_cols)
        (neq_residuals if npair is not None else true_local).append(npair or leaf)
    if len(neq_residuals) != 1 or not _all_local(true_local, local, local_cols):
        return None
    outer_c, inner_c = neq_residuals[0]
    return {
        "inner": inner,
        "local": local,
        "corr": corr,
        "outer_c": outer_c,
        "inner_c": inner_c,
        "true_local": true_local,
    }


def _decorrelate_neq_single(tr, ds, spec, negate: bool):
    """One correlated-`<>` subquery → one group-by (min/max over local rows) + one join."""
    inner, corr = spec["inner"], spec["corr"]
    outer_c, inner_c, true_local = spec["outer_c"], spec["inner_c"], spec["true_local"]
    inner_keys = [ic for (_oc, ic) in corr]
    inner.set("where", exp.Where(this=_join_and(true_local)) if true_local else None)
    inner.set("group", None)
    inner.set("expressions", [exp.column(c) for c in [*inner_keys, inner_c]])
    _reject_correlated(inner)
    tmp = {ik: f"__nqk{i}" for i, ik in enumerate(inner_keys)}
    agg = (
        tr.statement(inner)
        .rename(tmp)
        .group_by(*tmp.values())
        .agg(__mn=col(inner_c).min(), __mx=col(inner_c).max())
    )
    joined = ds.join(
        agg, left_on=[oc for (oc, _ic) in corr], right_on=list(tmp.values()), how="left"
    )
    bound_differs = (col("__mn") != col(outer_c)) | (col("__mx") != col(outer_c))
    cond = (col("__mn").is_null() | ~bound_differs) if negate else bound_differs.fill_null(False)
    return joined.filter(cond).drop("__mn", "__mx")


def _fuse_correlated_neq(tr, ds, leaves):
    """Fuse ≥2 correlated-`<>` EXISTS/NOT EXISTS over the same base table + keys into one pass.

    Returns `(ds, handled_indices)`; `handled_indices` is empty when no fusion applies,
    leaving every leaf for the caller's per-predicate path."""
    parsed = []  # (leaf_index, spec, negate)
    for i, leaf in enumerate(leaves):
        neg = isinstance(leaf, exp.Not) and isinstance(leaf.this, exp.Exists)
        ex = leaf.this if neg else leaf
        if not isinstance(ex, exp.Exists):
            continue
        spec = _parse_neq_exists(tr, ex)
        if spec is not None:
            parsed.append((i, spec, neg))
    if len(parsed) < 2:
        return ds, set()

    groups: dict = {}
    for item in parsed:
        _i, spec, _neg = item
        bases = _base_tables(spec["inner"])
        if len(bases) != 1:  # single-table inner only — keeps unqualified refs unambiguous
            continue
        sig = (bases, tuple(sorted(spec["corr"])))
        groups.setdefault(sig, []).append(item)
    for members in groups.values():
        if len(members) >= 2:
            return _emit_fused_group(tr, ds, members), {i for (i, _s, _n) in members}
    return ds, set()


def _emit_fused_group(tr, ds, members):
    """Emit one group-by + one join covering every member of a fusion group.

    The members share correlation keys and a single base table; each contributes its
    own `min`/`max` (conditional on its local filter, via a value-typed null else), and
    each applies its own bound test. One scan, one shuffle, instead of one per member."""
    base = members[0][1]
    corr = base["corr"]
    inner_keys = [ic for (_oc, ic) in corr]
    inner = base["inner"]  # the fused scan reuses the first member's FROM (single base table)

    proj = [exp.column(c) for c in inner_keys]
    seen = set(inner_keys)
    for _i, spec, _neg in members:
        if spec["inner_c"] not in seen:
            proj.append(exp.column(spec["inner_c"]))
            seen.add(spec["inner_c"])
    flags: dict[int, str] = {}
    for idx, (_i, spec, _neg) in enumerate(members):
        if spec["true_local"]:
            flag = f"__nqf{idx}"
            # The predicate came from a differently-aliased copy of the same base table
            # (`l3` vs the fused scan's `l2`); strip qualifiers so it resolves against
            # the single fused source, then project it as a boolean flag column.
            pred = _join_and([_strip_tables(p.copy()) for p in spec["true_local"]])
            proj.append(exp.alias_(pred, flag))
            flags[idx] = flag
    inner.set("where", None)
    inner.set("group", None)
    inner.set("expressions", proj)
    _reject_correlated(inner)

    tmp = {ik: f"__nqk{i}" for i, ik in enumerate(inner_keys)}
    aggs: dict = {}
    meta = []  # (mn_alias, mx_alias, outer_c, negate)
    for idx, (_i, spec, neg) in enumerate(members):
        ic = spec["inner_c"]
        mn, mx = f"__mn{idx}", f"__mx{idx}"
        if idx in flags:
            masked = when(col(flags[idx])).then(col(ic)).otherwise(nullif(col(ic), col(ic)))
            aggs[mn], aggs[mx] = masked.min(), masked.max()
        else:
            aggs[mn], aggs[mx] = col(ic).min(), col(ic).max()
        meta.append((mn, mx, spec["outer_c"], neg))
    agg = tr.statement(inner).rename(tmp).group_by(*tmp.values()).agg(**aggs)
    joined = ds.join(
        agg, left_on=[oc for (oc, _ic) in corr], right_on=list(tmp.values()), how="left"
    )
    drop_cols: list[str] = []
    for mn, mx, outer_c, neg in meta:
        bound_differs = (col(mn) != col(outer_c)) | (col(mx) != col(outer_c))
        cond = (col(mn).is_null() | ~bound_differs) if neg else bound_differs.fill_null(False)
        joined = joined.filter(cond)
        drop_cols += [mn, mx]
    return joined.drop(*drop_cols)


def _strip_tables(node):
    """Drop every column's table qualifier in-place (single-source scan → unambiguous)."""
    for c in node.find_all(exp.Column):
        c.set("table", None)
    return node
