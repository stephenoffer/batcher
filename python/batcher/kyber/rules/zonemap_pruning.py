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
from batcher.plan.expr_ir import Binary, Expr, IsNotNull, IsNull, Lit, Not
from batcher.plan.logical import (
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Sample,
    Sort,
    Union,
)
from batcher.plan.stats import (
    ColumnStat,
    RelStats,
    ambiguous_float_bound,
    mismatched_exactness,
)

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
    matches=(Filter, Sort, Distinct, Sample, Union, Join),
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
      pass-through (still deduplicated for a DISTINCT union);
    - a `Join` folds when a side being empty decides the result — see
      `_join_over_empty_side`, where the *output schema* is what limits how far it
      can go.

    Registered after `zonemap_prune_filter` in the SELECTION phase so it folds the
    empties that pruning produces in the same pass; bottom-up traversal collapses a
    whole chain of empty-over-empty in one application. Returns None (no change)
    when nothing is empty, so the rule is idempotent.
    """
    if isinstance(node, Union):
        return _prune_empty_union_branches(node)
    if isinstance(node, Join):
        return _join_over_empty_side(node)
    if isinstance(node, _SCHEMA_PRESERVING) and _is_empty(node.input):
        return node.input  # empty in, empty out, identical schema
    return None


# Join types an empty *left* side makes empty. `right` and `full` are deliberately
# absent: they keep the right side's rows padded with nulls on the left, so an empty
# left leaves them non-empty. Only the types whose row count is bounded by the left are
# here — `inner` (nothing to match), `left` (every output row is a left row), and
# `semi`/`anti` (whose output is the left side's columns and rows).
# An empty *right* side is not the mirror image, so it is handled case by case below.
_EMPTY_LEFT_IS_EMPTY = frozenset({"inner", "left", "semi", "anti"})


def _join_over_empty_side(node: Join) -> LogicalPlan | None:
    """Fold a join whose result an empty side already decides.

    Three cases, and the difference between them is entirely about **output schema**,
    which is what stops this being one rule:

    * **An empty left is empty for every join type.** A `full` or `right` join would
      normally pad, but Batcher's `Join.output` names the surviving columns, so the
      empty relation has to keep the join's own schema — `Limit(node, 0)`, not the
      empty side, which has only half the columns.
    * **A semi or anti join outputs the left side's columns alone**, so an empty right
      collapses to something built from `node.left`: no rows for `semi` (nothing can
      match), and *all* of the left for `anti` (nothing to exclude). Both drop the
      right subtree outright, which is the case here that saves real work today.
    * **An empty right in an `inner` join** is empty, but the schema is left+right, so
      it can only be spelled `Limit(node, 0)`.

    `Limit(x, 0)` is the canonical empty marker, and it is currently a *plan-level*
    marker: `bc-interp`'s `Limit` arm executes its input and then discards every row,
    because `ops::limit` recovers the output schema from the first input batch and
    there is no Rust-side plan schema inference. So the `Limit(node, 0)` results below
    shrink the plan and let row counts propagate exactly, but do not yet skip the scan.
    A first-class `Empty { schema }` relation — which DataFusion, DuckDB and Spark all
    have — is what makes them pay off, and is tracked separately. The `anti` and `semi`
    rewrites do not wait on it: they remove a subtree rather than mark it empty.
    """
    left_empty, right_empty = _is_empty(node.left), _is_empty(node.right)
    if left_empty and node.join_type in _EMPTY_LEFT_IS_EMPTY:
        return Limit(node, 0)
    if not right_empty:
        return None
    if node.join_type == "anti":
        # Nothing on the right to exclude, so every left row survives — and an anti
        # join already outputs exactly the left side's columns, so the join goes away.
        return node.left
    if node.join_type == "semi":
        return Limit(node.left, 0)  # nothing can match; semi outputs left columns only
    if node.join_type == "inner":
        return Limit(node, 0)
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
    if isinstance(expr, Lit) and type(expr.value) is bool:
        # A boolean literal decides itself, with no bounds needed. This is not a
        # theoretical case: constant folding runs in NORMALIZE, but `filter_null_join_keys`
        # runs later, in SELECTION, and rewrites a `false` predicate under a join into
        # `false AND k IS NOT NULL`. Nothing folds after that, so without this clause the
        # conjunction reads as undecidable and a provably-empty join side ships to the
        # engine to be evaluated per row. `type(...) is bool` because `Lit(0)` is not a
        # boolean predicate and `0 == False` in Python.
        return _TRUE if expr.value else _FALSE
    if isinstance(expr, Binary):
        if expr.op == "and":
            return _and(_predicate_status(expr.left, stats), _predicate_status(expr.right, stats))
        if expr.op == "or":
            return _or(_predicate_status(expr.left, stats), _predicate_status(expr.right, stats))
        if expr.op in _COMPARISONS:
            return _comparison_status(expr, stats)
        return None
    if isinstance(expr, Not):
        return _not(_predicate_status(expr.input, stats))
    if isinstance(expr, IsNull):
        return _is_null_status(expr.input, stats, negate=False)
    if isinstance(expr, IsNotNull):
        return _is_null_status(expr.input, stats, negate=True)
    return None


def _not(inner: bool | None) -> bool | None:
    """Negate the tri-state. Sound in one direction only.

    The two states are not mirror images, so `not inner` is wrong. `_TRUE` is proven
    from bounds *and* a zero null count (see `_decide`), so every row evaluates the
    inner predicate to TRUE and negating gives FALSE for every row → `_FALSE`.

    `_FALSE` only proves no row *passes*; those rows are FALSE **or NULL**. Negating a
    NULL yields NULL, which a filter still drops — so `NOT (always-empty)` is not
    provably always-true, and claiming `_TRUE` would drop the filter and wrongly keep
    the NULL rows. Decline instead: that costs a scan, never a row.
    """
    return _FALSE if inner is _TRUE else None


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


def _bloom_domain(value: object) -> str | None:
    """The index domain `value` would be hashed in, or `None` if it is not indexable.

    Mirrors `plan.bloom_index.canonical_bytes` and `bc_py::build_column_bloom`, which index
    only Int64 (8-byte little-endian) and Utf8 (raw bytes). `bool` is excluded even though
    it is an `int` subclass, exactly as both encoders exclude it.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return "int"
    if isinstance(value, str):
        return "str"
    return None


def _same_bloom_domain(col: ColumnStat, value: object) -> bool:
    """Whether `value` lives in the same index domain the column's bloom was built over.

    The column's own `min`/`max` bounds witness its domain — they are Python values read
    from the same source that built the index. When no bound is known the domain cannot be
    established, so the bloom is not consulted: refusing to prune costs a scan, whereas
    probing across domains would delete rows that are present.
    """
    value_domain = _bloom_domain(value)
    if value_domain is None:
        return False
    witness = col.min if col.min is not None else col.max
    if witness is None:
        return False
    return _bloom_domain(witness) == value_domain


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
    #
    # This is the one place a bug **drops rows**: `contains() -> False` deletes the whole
    # relation. Absence is only a proof when the literal is encoded in the *same domain*
    # the index was built over — a bloom over Int64 values hashes 8 little-endian bytes,
    # while the string `"5"` hashes one byte, so probing one with the other reports a
    # definitive absence for a value that is present. Today a cross-domain comparison is
    # wrapped in a `Cast` (so `comparison_col_side` returns `None` and never reaches here),
    # but that is an implicit guard on a silent-wrong-answer path. Make it explicit.
    if op == "eq" and col.bloom is not None and _same_bloom_domain(col, value):
        index = BloomIndex.from_bytes(col.bloom)
        if index is not None and not index.contains(value):
            return _FALSE
    if col.min is None or col.max is None:
        return None
    if _float_order_is_ambiguous(col.min) or _float_order_is_ambiguous(col.max):
        return None
    # A decimal bound against a float literal is compared *exactly* by Python and in Float64
    # by the engine, so Python can prove a predicate empty that the engine satisfies. Decimal
    # literals arrive here as floats (the IR has no decimal literal), which puts every exact
    # money predicate on this path — see `mismatched_exactness`.
    if mismatched_exactness(col.min, value) or mismatched_exactness(col.max, value):
        return None
    no_nulls = col.null_count == 0
    try:
        return _decide(op, col.min, col.max, value, no_nulls)
    except TypeError:
        return None  # incomparable literal/bound types → undecidable


def _float_order_is_ambiguous(bound: object) -> bool:
    """Whether reasoning about this float bound could contradict what the engine computes.

    This rule is a **plan rewrite**, not a metadata answer: folding a predicate to `FALSE`
    deletes rows from the result. So it owes a stronger guarantee than "probably right" — it
    must agree with the engine's own comparison, whatever that is. On two kinds of float
    bound it currently cannot:

    * a **NaN** bound — the engine ranks NaN above every number (arrow-rs compares floats on
      their total order), while Python's `>` ranks it nowhere; and
    * a **zero** bound — the engine separates `-0.0` from `0.0` (`-0.0 < 0.0` on that same
      total order), while Python calls them equal.

    Either one lets `_decide` "prove" a predicate empty that the engine would satisfy. It is
    not hypothetical: over a column holding `-0.0`, `WHERE f < 0` was folded to `FALSE` and
    the query returned no rows, while executing the same filter returned the `-0.0` row. An
    optimizer that changes a result is not an optimizer.

    (The deeper problem is that the engine's float comparisons follow arrow-rs's total order
    rather than IEEE, so they *also* disagree with DuckDB — `WHERE f = 0.0` misses `-0.0`,
    `WHERE f > 1` matches NaN. That is a separate, engine-side bug recorded in
    `docs/internals/bug_hunt_ledger.md`. Declining here is sound under either semantics: it
    costs a scan, never a row.)
    """
    return ambiguous_float_bound(bound)


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
