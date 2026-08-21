"""Correlation analysis for SQL subqueries: which references reach out of a subquery.

Every subquery rewrite in this package — the semi/anti joins in `core`, the `<>`
decorrelation in `neq`, the range forms in `range` — first has to answer the same
question: for a predicate inside a subquery, which side names a column of the subquery's
own FROM/JOINs and which side reaches out to the enclosing query. These are the helpers
that answer it, kept together and kept free of any dependency on the rewrites themselves,
so the dependency runs one way (`core`/`neq`/`range` -> here) and cannot loop back.

They live in their own module because `core.py` outgrew the 500-line limit and this is the
seam that does not cut a concept in half: correlation *analysis* on one side, the plan
rewrites that consume it on the other.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._sql.parser.core_utils import _join_and, _split_and


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


def _is_plain_column(node) -> bool:
    return isinstance(node, exp.Column)


def _reject_correlated(select_node) -> None:
    """Raise if `select_node` references a table outside its own FROM/JOINs.

    This is the **safety net** under every rewrite in this package, and the last thing
    between an unrecognized correlation and a wrong answer. Each rewrite pulls the
    correlations it understands out of the inner `WHERE` and hands the remainder to the
    inner relation as local predicates; anything still reaching outward at that point is a
    shape nobody handled, and running it would silently resolve the outer name against the
    inner relation instead of refusing.

    Correlation is approximated by collecting the names the subquery introduces and
    flagging any qualified column outside that set. Unqualified columns are assumed local
    (they cannot be resolved otherwise).

    **An aliased table contributes only its alias.** SQL scoping shadows the base name:
    inside ``FROM orders o`` the name ``orders`` is not in scope, so a reference to it is
    an outer one. Adding the base name as well — which this did — made the net miss exactly
    the case where the inner query aliases the *same table the outer query reads*, which is
    the commonest correlated-subquery shape there is:

        SELECT k FROM a WHERE EXISTS (SELECT 1 FROM a u WHERE u.v = a.v + 1)

    `a.v` is an outer reference, and it was classified local because the inner `FROM a u`
    contributed ``a``. The predicate then became ``u.v = u.v + 1``, always false, and the
    query returned **no rows** where it should have returned five — no error, nothing in
    the plan to look at. With the outer query *aliased* (``FROM a x``) the same query was
    correctly refused, so the defect was invisible to anyone who writes aliases. This is
    the rule `_local_tables` already applies twenty lines above, for the same reason.

    Args:
        select_node: The inner SELECT, already detached from the outer AST.

    Raises:
        NotImplementedError: If any column names a table this SELECT does not introduce.
    """
    local: set[str] = set()
    for t in select_node.find_all(exp.Table):
        local.add(t.alias or t.name)
    for sub in select_node.find_all(exp.Subquery):
        if sub.alias:
            local.add(sub.alias)

    for c in select_node.find_all(exp.Column):
        tbl = c.table
        if tbl and tbl not in local:
            raise NotImplementedError(
                f"correlated subquery: {tbl}.{c.name} reaches outside the subquery, which "
                f"introduces {sorted(local) or 'no tables'}. Batcher decorrelates an "
                "equality between two plain columns (EXISTS/IN/scalar); a correlation "
                "through an expression (outer.c + 1, a function call) is not one, and is "
                "refused rather than answered wrongly. Rewrite it as an explicit join, or "
                "compute the expression as a column of the outer query first."
            )
