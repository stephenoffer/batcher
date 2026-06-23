"""Zone-map predicate pruning — eliminate filters provably empty or always-true.

When a column's value range is known from metadata (a Parquet/ORC footer's
min/max, a lakehouse manifest's bounds), a predicate can sometimes be decided
without looking at a single row: `age < 0` over a column whose minimum is `18` is
*always false* (the result is empty); `age < 1000` over a column whose maximum is
`99` and which has no nulls is *always true* (the filter is dead). This rule
rewrites the first to an empty relation and drops the second, shrinking the plan
and — because the row count then propagates exactly — letting `count()` answer `0`
or the child's count from metadata alone.

Correctness is conservative: a rewrite fires only when the bounds *prove* the
outcome. Min/max are valid bounds regardless of provenance (a filter or limit can
only shrink a range), so they may always be used for pruning; but declaring a
predicate *always true* additionally requires a known-zero null count, since a
filter drops null rows. Anything not provable is left untouched (executed).
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.stats.selectivity import comparison_col_side
from batcher.plan.bloom_index import BloomIndex
from batcher.plan.expr_ir import Binary, Expr, IsNotNull, IsNull, Not
from batcher.plan.logical import (
    Distinct,
    Filter,
    Limit,
    LogicalPlan,
    Sample,
    Sort,
    Union,
)
from batcher.plan.stats import RelStats

__all__ = ["propagate_empty_relation", "zonemap_prune_filter"]

# A predicate's decidability against known column bounds: provably keeps every row
# (True), provably keeps none (False), or undecidable from metadata (None).
_TRUE = True
_FALSE = False
_COMPARISONS = {"lt", "le", "gt", "ge", "eq", "ne"}
# Flip a comparison when the column is on the right (`lit < col` ≡ `col > lit`).
_FLIP = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le", "eq": "eq", "ne": "ne"}


@rule(name="zonemap_prune_filter", phase=Phase.SELECTION, matches=(Filter,))
def zonemap_prune_filter(node: Filter, ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a Filter that metadata proves always-true, or replace one proved
    always-false with an empty (zero-row) relation. Returns None when undecidable."""
    stats = ctx.estimator.estimate(node.input)
    status = _predicate_status(node.predicate, stats)
    if status is _TRUE:
        return node.input  # every row passes → the filter is dead
    if status is _FALSE:
        return Limit(node.input, 0)  # no row passes → empty, schema-preserving
    return None


# Operators that pass their input through unchanged in schema and merely shrink or
# reorder rows — so an empty input produces an empty output with the same columns.
_SCHEMA_PRESERVING = (Filter, Sort, Distinct, Sample)


@rule(
    name="propagate_empty_relation",
    phase=Phase.SELECTION,
    matches=(Filter, Sort, Distinct, Sample, Union),
)
def propagate_empty_relation(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold a provably-empty subtree upward through operators that preserve it.

    An empty relation is canonically `Limit(x, 0)` — what `zonemap_prune_filter`
    emits for an always-false predicate, and what `.limit(0)` builds. This rule
    propagates that emptiness through the operators above it:

    - a schema-preserving unary parent (`Filter`/`Sort`/`Distinct`/`Sample`) over an
      empty input is itself empty — replace it with the empty input;
    - a `Union` drops its empty branches (an empty contributes no rows); if all are
      empty the union is empty, and a single surviving branch makes the union a
      pass-through (still deduplicated for a DISTINCT union).

    Registered after `zonemap_prune_filter` in the SELECTION phase so it folds the
    empties that pruning produces in the same pass; bottom-up traversal collapses a
    whole chain of empty-over-empty in one application. Returns None (no change)
    when nothing is empty, so the rule is idempotent.
    """
    if isinstance(node, Union):
        return _prune_empty_union_branches(node)
    if isinstance(node, _SCHEMA_PRESERVING) and _is_empty(node.input):
        return node.input  # empty in, empty out, identical schema
    return None


def _is_empty(node: LogicalPlan) -> bool:
    """Whether `node` provably yields zero rows (the `Limit(_, 0)` empty marker)."""
    return isinstance(node, Limit) and node.n == 0


def _prune_empty_union_branches(node: Union) -> LogicalPlan | None:
    survivors = [i for i in node.inputs if not _is_empty(i)]
    if len(survivors) == len(node.inputs):
        return None  # nothing empty → no change
    if not survivors:
        return node.inputs[0]  # all empty → the (empty) first branch keeps the schema
    if len(survivors) == 1:
        only = survivors[0]
        # A one-branch union is a pass-through; a DISTINCT union still deduplicates.
        return Distinct(only) if node.distinct else only
    return Union(tuple(survivors), node.distinct)


def _predicate_status(expr: Expr, stats: RelStats) -> bool | None:
    """Tri-state evaluation of `expr` against `stats`' column bounds."""
    if isinstance(expr, Binary):
        if expr.op == "and":
            return _and(_predicate_status(expr.left, stats), _predicate_status(expr.right, stats))
        if expr.op == "or":
            return _or(_predicate_status(expr.left, stats), _predicate_status(expr.right, stats))
        if expr.op in _COMPARISONS:
            return _comparison_status(expr, stats)
        return None
    if isinstance(expr, Not):
        inner = _predicate_status(expr.input, stats)
        return None if inner is None else (not inner)
    if isinstance(expr, IsNull):
        return _is_null_status(expr.input, stats, negate=False)
    if isinstance(expr, IsNotNull):
        return _is_null_status(expr.input, stats, negate=True)
    return None


def _and(left: bool | None, right: bool | None) -> bool | None:
    if left is _FALSE or right is _FALSE:
        return _FALSE  # any always-false conjunct → empty
    if left is _TRUE and right is _TRUE:
        return _TRUE
    return None


def _or(left: bool | None, right: bool | None) -> bool | None:
    if left is _TRUE or right is _TRUE:
        return _TRUE
    if left is _FALSE and right is _FALSE:
        return _FALSE
    return None


def _is_null_status(input_expr: Expr, stats: RelStats, *, negate: bool) -> bool | None:
    """`col IS NULL` / `IS NOT NULL` decided from a known null count.

    Only a *zero* null count is decidable here (the common case): `IS NULL` is then
    always-false (no nulls) and `IS NOT NULL` always-true.
    """
    from batcher.plan.expr_ir import Col

    if not isinstance(input_expr, Col):
        return None
    null_count = stats.column(input_expr.name).null_count
    if null_count == 0:
        return _TRUE if negate else _FALSE
    return None


def _comparison_status(expr: Binary, stats: RelStats) -> bool | None:
    """Decide a `col OP literal` comparison against the column's bounds and bloom."""
    side = comparison_col_side(expr)
    if side is None:
        return None
    name, value, col_on_left = side
    col = stats.column(name)
    op = expr.op if col_on_left else _FLIP[expr.op]
    # Bloom data-skip: for equality, absence from the column's membership index proves
    # the predicate always-false — catching point lookups *inside* [min, max] that
    # min/max can't (`id = 9700123` over a 10M-row column). `IN` reaches this via the
    # OR-of-equalities split. No false negatives, so absence is definitive.
    if op == "eq" and col.bloom is not None:
        index = BloomIndex.from_bytes(col.bloom)
        if index is not None and not index.contains(value):
            return _FALSE
    if col.min is None or col.max is None:
        return None
    no_nulls = col.null_count == 0
    try:
        return _decide(op, col.min, col.max, value, no_nulls)
    except TypeError:
        return None  # incomparable literal/bound types → undecidable


def _decide(op: str, cmin, cmax, lit, no_nulls: bool) -> bool | None:
    """The core bound comparison. `True`/`False`/`None` as defined above.

    "Empty" outcomes depend only on bounds (a filter drops nulls anyway), so they
    never need the null check; "always-true" outcomes do (a surviving null would be
    dropped), so they additionally require `no_nulls`.
    """
    if op == "lt":
        if cmin >= lit:
            return _FALSE
        return _TRUE if (cmax < lit and no_nulls) else None
    if op == "le":
        if cmin > lit:
            return _FALSE
        return _TRUE if (cmax <= lit and no_nulls) else None
    if op == "gt":
        if cmax <= lit:
            return _FALSE
        return _TRUE if (cmin > lit and no_nulls) else None
    if op == "ge":
        if cmax < lit:
            return _FALSE
        return _TRUE if (cmin >= lit and no_nulls) else None
    if op == "eq":
        if lit < cmin or lit > cmax:
            return _FALSE
        return _TRUE if (cmin == cmax == lit and no_nulls) else None
    if op == "ne":
        if cmin == cmax == lit:
            return _FALSE
        return _TRUE if ((lit < cmin or lit > cmax) and no_nulls) else None
    return None
