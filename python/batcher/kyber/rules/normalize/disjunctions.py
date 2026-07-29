"""Disjunctions of equalities folded into an `IN` list, and the range they imply.

Split out of `ranges` on the seam between *a single predicate rewritten into a range* and
*a disjunction collapsed into a set*: the two share only the literal guard, which moved
here with its users. `normalize/__init__` imports this module immediately after `ranges`,
so the rules register in the position they always did — registration order is run order.

The `_bad_range_literal` guard is the load-bearing piece and the reason the two rules live
together: a disjunct comparing against NaN must not produce a min/max bound, because the
engine's `=` matches a NaN row while `>=`/`<=` against a NaN do not.
"""

from __future__ import annotations

from batcher._internal.mathx import is_nan
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import Binary, Col, Expr, InList, Lit
from batcher.plan.expr_rewrite import combine_conjuncts, expr_key, split_conjuncts
from batcher.plan.logical import Filter, LogicalPlan

__all__ = ["or_equalities_to_in_list", "or_to_in_and_range"]


def _flat_or_equalities(expr: Expr) -> tuple[str, list] | None:
    """If `expr` is `c == v1 OR c == v2 OR …` (≥2 disjuncts, one column, literal
    values), return `(column, [values])`; else None. The shape SQL `IN (...)` and
    chained `OR` equalities lower to."""
    if not (isinstance(expr, Binary) and expr.op == "or"):
        return None
    leaves: list[tuple[str, object]] = []

    def collect(e: Expr) -> bool:
        if isinstance(e, Binary) and e.op == "or":
            return collect(e.left) and collect(e.right)
        if isinstance(e, Binary) and e.op == "eq":
            left, right = e.left, e.right
            if isinstance(left, Col) and isinstance(right, Lit):
                leaves.append((left.name, right.value))
                return True
            if isinstance(right, Col) and isinstance(left, Lit):
                leaves.append((right.name, left.value))
                return True
        return False

    if not collect(expr) or len(leaves) < 2:
        return None
    cols = {name for name, _ in leaves}
    values = [v for _, v in leaves]
    if len(cols) != 1 or any(_bad_range_literal(v) for v in values):
        return None
    return cols.pop(), values


def _in_list_values(expr: Expr) -> tuple[str, list] | None:
    """`(column, values)` for a `col IN (…)` over literals, else None.

    `or_to_in_and_range` must see this shape as well as the `OR` chain, because
    `or_equalities_to_in_list` folds the chain into an `InList` in the same phase — and
    whichever of the two runs first, the bounds this rule derives must still be derived.
    Reading only the `OR` form made the fold silently *remove* the zone-map bounds.
    """
    if not (isinstance(expr, InList) and isinstance(expr.input, Col)):
        return None
    values = list(expr.values)
    if len(values) < 2 or any(_bad_range_literal(v) for v in values):
        return None
    return expr.input.name, values


def _bad_range_literal(value: object) -> bool:
    """Whether a literal cannot participate in a min/max range bound.

    ``NULL`` and booleans are excluded (a range over them is meaningless), and so is
    **NaN**: the engine's ``=`` matches a NaN row (``col = NaN`` is TRUE where ``col`` is
    NaN), but ``NaN >= lo`` / ``NaN <= hi`` are both FALSE — so a range bound derived from a
    disjunction containing ``col = NaN`` would drop the very NaN rows that disjunct keeps,
    and Python's ``min``/``max`` over a list containing NaN is order-dependent garbage (a
    leading NaN yields ``col >= NaN``, which rejects every row). Such a disjunction gets no
    range bound at all — the original ``OR`` still selects exactly the right rows.
    """
    if value is None or isinstance(value, bool):
        return True
    return is_nan(value)


@rule(name="or_equalities_to_in_list", phase=Phase.NORMALIZE, matches=(Filter,))
def or_equalities_to_in_list(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold `c = v1 OR c = v2 OR …` into `c IN (v1, v2, …)` (DuckDB's
    `contains_to_in_clause`).

    This pays twice, and the second time is the larger one.

    At runtime the disjunction is *n* sequential comparisons over the whole column,
    while `InList` lowers to one hash-set probe per row (`bc_expr::eval::in_list`,
    which builds an `ahash` set once per operator). That is the direct win.

    The compounding one is that this repository already carries a family of rules keyed
    on `InList` — `prune_in_list_by_zonemap`, `prune_in_list_by_bloom`, `dedup_in_list`,
    `refine_in_list_by_equality`/`_comparison`/`_neq`, `intersect_in_lists`,
    `push_in_list_across_join_keys` — **none of which can fire on the `OR` form**. A user
    who writes the chain (or whose ORM does) got none of them. Producing the node they
    match is what turns eight existing rules on.

    Exactly equivalent, including NULL: `x = 1 OR x = 2` and `x IN (1, 2)` both yield
    NULL for a null `x` (verified against DuckDB, which agrees on both forms), so this is
    safe in a projection as well as under a filter — though it only matches `Filter`,
    where the pruning rules that consume it live.

    `or_to_in_and_range` is the sibling rewrite and is *not* superseded: it derives
    `min ≤ c ≤ max` bounds from the same shape, which zone maps use directly. Both fire;
    the bounds are added to the conjunction and the disjunct itself becomes the `IN`.
    """
    conjuncts = split_conjuncts(node.predicate)
    rewritten: list[Expr] = []
    changed = False
    for conj in conjuncts:
        info = _flat_or_equalities(conj)
        if info is None:
            rewritten.append(conj)
            continue
        col_name, values = info
        # First-occurrence order, deduplicated: the set semantics are the same and a
        # stable order keeps the rewrite idempotent (`dedup_in_list` would otherwise
        # rewrite it again on the next fixpoint iteration).
        deduped = tuple(dict.fromkeys(values))
        if len(deduped) < 2:
            # One distinct value is an equality, not a membership test; leave it for
            # the equality rules rather than wrapping a single value in a hash set.
            rewritten.append(conj)
            continue
        rewritten.append(InList(Col(col_name), deduped))
        changed = True
    if not changed:
        return None
    return Filter(node.input, combine_conjuncts(rewritten))


@rule(name="or_to_in_and_range", phase=Phase.NORMALIZE, matches=(Filter,))
def or_to_in_and_range(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Add `c >= min AND c <= max` alongside a `c = v1 OR c = v2 OR …` conjunct.

    A disjunction of equalities (what `IN (...)` lowers to) is opaque to range-based
    zone-map pruning. Its values imply the bound `min(vs) ≤ c ≤ max(vs)`, a superset
    that — ANDed with the original disjunction — leaves the result unchanged but gives
    `zonemap_prune_filter` a range it can use to skip whole row groups (and each
    equality is still a bloom-index probe). Idempotent: the bounds are added only if
    not already present. Skipped when the literals aren't mutually comparable.
    """
    conjuncts = split_conjuncts(node.predicate)
    # Keyed by `expr_key` (the canonical, memoized IR serialization) rather than by the
    # raw IR dict: dicts are unhashable, so the presence test used to be a linear scan of
    # a list with a full dict comparison at each step — quadratic in the conjunct count on
    # exactly the wide `OR` chains this rule exists to fold.
    existing = {expr_key(c) for c in conjuncts}
    added: list[Expr] = []
    for conj in conjuncts:
        info = _flat_or_equalities(conj) or _in_list_values(conj)
        if info is None:
            continue
        col_name, values = info
        try:
            lo, hi = min(values), max(values)
        except TypeError:
            continue  # values not mutually comparable (mixed types)
        for bound in (Binary("ge", Col(col_name), Lit(lo)), Binary("le", Col(col_name), Lit(hi))):
            key = expr_key(bound)
            if key not in existing:
                added.append(bound)
                existing.add(key)
    if not added:
        return None
    return Filter(node.input, combine_conjuncts([*conjuncts, *added]))
