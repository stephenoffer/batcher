"""Answer *filtered* counts from metadata — Kyber's EXACT-gated filter-count layer.

A terminal `count()` (or `is_empty()`/`any()`) sitting over a `Filter` can often be
answered without touching a row: `filter(col IS NULL).count()` is exactly the column's
recorded null count; `filter(col > max).count()` is provably zero. This module decides
whether such a filtered count is *provably* derivable from EXACT column statistics and,
if so, returns it. The conductor calls these before executing and falls back to a full
run on `None`, so a metadata answer is only ever an optimisation, never a risk.

The firewall is the same as the sibling `metadata_answer` module: every answer is gated
on `Provenance.EXACT` end to end (a Parquet/ORC footer's exact row count, null count,
and min/max bounds). A range/equality predicate that only *partially* overlaps the
column's range is **not** exact (it needs a histogram) — this module returns `None`
there rather than guess. Only the provably-empty side of a comparison, a null-predicate,
or an optimizer-folded constant predicate yields an exact count. A wrong count is silent
corruption; when in doubt this returns `None`.
"""

from __future__ import annotations

from batcher._internal.mathx import is_nan
from batcher.config import Config
from batcher.kyber.learning import load_learned_stats
from batcher.kyber.metadata_answer import _root_stats, exact_null_count
from batcher.kyber.stats import StatsEstimator
from batcher.metadata.hub import MetadataHub
from batcher.plan.bloom_index import BloomIndex
from batcher.plan.expr_ir import Binary, Col, IsNotNull, IsNull, Lit, Not
from batcher.plan.logical import Filter, LogicalPlan, Project, Scan
from batcher.plan.stats import ColumnStat, Provenance, RelStats, mismatched_exactness

__all__ = [
    "answer_filter_any",
    "answer_filter_count",
    "answer_filter_is_empty",
]

# Comparison operators whose provably-empty side is exact from footer min/max bounds.
_COMPARE_OPS = frozenset({"lt", "le", "gt", "ge", "eq", "ne"})
# Flip a comparison when the literal is on the left (`lit < col` == `col > lit`).
_FLIP = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le", "eq": "eq", "ne": "ne"}


def answer_filter_count(
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> int | None:
    """Exact surviving row count of a `Filter`-over-scan from metadata, or None.

    The plan's root must be a `Filter` (optionally behind row-preserving `Project`s),
    whose surviving count is provably derivable from EXACT column statistics. Handles a
    null predicate (`col IS NULL` → exact null count; `col IS NOT NULL` → rows - nulls,
    plus their `NOT`-normalised forms), an optimizer-folded constant predicate (always
    true → rows, always false → 0), a provably-empty comparison (`col > max`, `col <
    min`, `col = v` with `v` outside `[min, max]` or absent from the column bloom → 0),
    and the `col IS NULL OR col IS NOT NULL` tautology (→ rows). Any partial-overlap
    range/equality is not exact → None.
    """
    if _descend_to_filter(plan) is None:
        return None  # not a filter-count query — leave plain counts to `answer_count`
    rewritten, root = _root_stats(plan, sources, source_stats, hub, config)
    # Optimizer-folded constant predicate: `filter(TRUE)` keeps the child's EXACT rows,
    # `filter(FALSE)` is the EXACT-empty marker (0). Both surface as an EXACT root count.
    if root.rows_exact:
        return int(root.rows)
    filt = _descend_to_filter(rewritten)
    if filt is None:
        return None
    child = _child_stats(filt, sources, source_stats, hub)
    exact = _exact_surviving_count(filt.predicate, child)
    if exact is not None:
        return exact
    return _exact_predicate_count(filt, child, sources)


def answer_filter_is_empty(
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> bool | None:
    """Whether a filtered relation is empty, from metadata, or None if not provable.

    The boolean form of `answer_filter_count`: `True` iff the exact surviving count is
    zero. Fires on exactly the shapes `answer_filter_count` proves (e.g. a
    provably-empty `col > max` yields `True` with no scan).
    """
    count = answer_filter_count(plan, sources, source_stats, hub, config)
    return None if count is None else count == 0


def answer_filter_any(
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> bool | None:
    """Whether a filtered relation keeps any row, from metadata, or None if not provable.

    The negation of `answer_filter_is_empty` — `True` iff the exact surviving count is
    positive (e.g. `col IS NOT NULL` over a footer with a null count below the row count).
    """
    count = answer_filter_count(plan, sources, source_stats, hub, config)
    return None if count is None else count > 0


def _descend_to_filter(node: LogicalPlan) -> Filter | None:
    """The `Filter` at the root through any row-preserving `Project`s, else None.

    `Project` never changes the row count, so the surviving count of a `Filter` beneath a
    chain of projections is exactly the whole plan's count. Any other node breaks the
    chain (its row count is not the filter's), so we stop and report "not a filter count".
    """
    while isinstance(node, Project):
        node = node.input
    return node if isinstance(node, Filter) else None


def _child_stats(
    filt: Filter,
    sources: list,
    source_stats: list | None,
    hub: MetadataHub | None,
) -> RelStats:
    """EXACT-first stats of the `Filter`'s *input* (the relation the predicate filters).

    The filter's output count is never EXACT, so we estimate the child instead — its
    footer-EXACT null count and min/max bounds are what the predicate shapes read. Uses
    the same `exact_first` estimator as `_root_stats` so an EXACT structural stat wins
    over a weaker learned measurement.
    """
    learned = load_learned_stats(hub) if hub is not None else {}
    estimator = StatsEstimator(sources, learned, source_stats=source_stats, exact_first=True)
    return estimator.estimate(filt.input)


def _exact_surviving_count(predicate, child: RelStats) -> int | None:
    """The exact number of rows a predicate keeps over `child`'s EXACT stats, or None."""
    pred = _strip_not(predicate)
    # A null predicate is answered from the null count's **own** provenance, not the bundle's:
    # a Parquet string column has truncatable bounds (a DEFAULT bundle) and an exactly recorded
    # null count, and `WHERE name IS NULL` needs only the second.
    if isinstance(pred, IsNull) and isinstance(pred.input, Col):
        # `col IS NULL` — surviving count is exactly the recorded null count.
        return exact_null_count(child, pred.input.name)
    if isinstance(pred, IsNotNull) and isinstance(pred.input, Col):
        # `col IS NOT NULL` — rows minus the null count (needs both EXACT).
        nulls = exact_null_count(child, pred.input.name)
        if child.rows_exact and nulls is not None:
            return int(child.rows - nulls)
        return None
    if _is_null_tautology(pred) is not None and child.rows_exact:
        # `col IS NULL OR col IS NOT NULL` keeps every row (always true).
        return int(child.rows)
    if isinstance(pred, Binary) and pred.op == "and":
        # `A AND B` keeps a subset of what `A` keeps, so a provably-empty conjunct makes the
        # whole conjunction provably empty. This is the shape of every real lakehouse filter
        # (`day = 42 AND region = 'us'`), which the bare-comparison parse below could not see
        # at all — so a partition pruned to nothing still executed to discover it.
        for side in (pred.left, pred.right):
            if _exact_surviving_count(side, child) == 0:
                return 0
        return None
    parsed = _parse_comparison(pred)
    if parsed is not None:
        op, name, value = parsed
        if _comparison_empty(op, child.columns.get(name), value):
            return 0  # only the provably-empty side is exact; a partial overlap is None
    return None


def _exact_predicate_count(filt: Filter, child: RelStats, sources: list) -> int | None:
    """Exact surviving count of a bare single-column comparison (``col <op> v`` for any of
    ``= <> < <= > >=``) from a learned per-predicate count over a single in-memory source.

    The provably-empty shapes (`_exact_surviving_count`) cover the extremes from EXACT
    min/max bounds; this covers the interior — a comparison that partially overlaps the
    range. A source that can count its own predicate matches exactly
    (`InMemorySource.column_predicate_count`, one Arrow kernel pass, cached) turns a common
    filtered ``COUNT(*)`` into a metadata answer — the learned-metadata moat DuckDB's static
    engine can't match. The Arrow kernel already yields SQL semantics (a null operand is
    never true, so nulls drop from every comparison), so the match count *is* the surviving
    count directly. Sound only when the predicate sits *directly* on a bare `Scan` of the one
    source with no row-reducing or value-transforming pushdown (an intervening projection
    could redefine the column, a residual scan predicate could pre-drop rows), so a
    whole-column count matches exactly what the filter sees.
    """
    scan = filt.input
    # `isinstance`, not a name-string: a string compare silently accepts any unrelated class
    # called `Scan` and breaks silently if the class is renamed. `source_id == 0` makes the
    # single-source binding explicit rather than relying on the `len(sources) != 1` check below.
    # `getattr` for the predicate stays deliberate: a pushdown adds that attribute, a bare
    # `Scan` does not carry it at all, so a direct access raises rather than declining.
    if (
        not isinstance(scan, Scan)
        or getattr(scan, "predicate", None) is not None
        or scan.source_id != 0
    ):
        return None  # not a bare scan of the one source — a whole-source count may not match
    parsed = _parse_comparison(filt.predicate)
    if parsed is None or len(sources) != 1 or not child.rows_exact:
        return None
    op, name, value = parsed
    counter = getattr(sources[0], "column_predicate_count", None)
    if counter is None:
        return None
    matches = counter(op, name, value)
    return None if matches is None else int(matches)


def _strip_not(expr):
    """Normalise `NOT` over a null predicate (and cancel double negation), else return as-is.

    `NOT (col IS NULL)` is `col IS NOT NULL` and vice versa; `NOT NOT x` is `x`. Any other
    `NOT` (over a comparison, an AND/OR) is left untouched — those are not handled shapes.
    """
    while isinstance(expr, Not):
        inner = expr.input
        if isinstance(inner, Not):
            expr = inner.input
            continue
        if isinstance(inner, IsNull):
            return IsNotNull(inner.input)
        if isinstance(inner, IsNotNull):
            return IsNull(inner.input)
        return expr
    return expr


def _is_null_tautology(pred) -> str | None:
    """The column of a `col IS NULL OR col IS NOT NULL` tautology (either order), else None."""
    if not isinstance(pred, Binary) or pred.op != "or":
        return None

    def col_of(e, cls) -> str | None:
        return e.input.name if isinstance(e, cls) and isinstance(e.input, Col) else None

    for x, y in ((pred.left, pred.right), (pred.right, pred.left)):
        n_null = col_of(x, IsNull)
        if n_null is not None and n_null == col_of(y, IsNotNull):
            return n_null
    return None


def _parse_comparison(expr) -> tuple[str, str, object] | None:
    """Recognise `col OP lit` / `lit OP col` as `(op, column, value)` (flipped), else None."""
    if not isinstance(expr, Binary) or expr.op not in _COMPARE_OPS:
        return None
    left, right = expr.left, expr.right
    if isinstance(left, Col) and isinstance(right, Lit):
        return expr.op, left.name, right.value
    if isinstance(left, Lit) and isinstance(right, Col):
        return _FLIP[expr.op], right.name, left.value
    return None


def _comparison_empty(op: str, stat: ColumnStat | None, value) -> bool:
    """Whether `col OP value` provably keeps zero rows (nulls, being non-true, drop too).

    Range shapes read EXACT footer bounds: `col < v`/`col <= v` empty when `v` is at or
    below the min, `col > v`/`col >= v` when at or above the max, `col = v` when `v` lies
    outside `[min, max]`, `col != v` when every value equals `v` (`min == max == v`).
    Equality also consults the column bloom (a provenance-independent absence proof). A
    predicate that partially overlaps the range is *not* provably empty → False.
    """
    if stat is None:
        return False
    # Bloom absence proves `col = v` cannot match — sound in any subset, provenance-free.
    if op == "eq" and _bloom_absent(stat, value):
        return True
    if stat.provenance is not Provenance.EXACT:
        return False
    lo, hi = stat.min, stat.max
    if _is_nan(lo) or _is_nan(hi) or _is_nan(value):
        return False
    # A **zero** float bound is as unusable as a NaN one, and for the same reason: the engine
    # compares floats on their *total* order, where `-0.0 < 0.0`, while the comparison below
    # is Python's, where they are equal. A column whose minimum is `-0.0` would let us "prove"
    # that `WHERE f < 0` keeps nothing — and executing it returns the `-0.0` row we proved
    # away. Refusing costs a scan; answering costs a wrong count.
    if _is_zero_float(lo) or _is_zero_float(hi):
        return False
    # A decimal bound against a float literal is the same class of disagreement, and the one
    # that reaches every exact money predicate: the IR has no decimal literal, so
    # `price = Decimal("999999999.99")` arrives here as a float, and Python compares it to the
    # `Decimal` bound *exactly* while the engine promotes both to Float64 and finds them equal.
    # It proved `count()` was 0 for a filter whose `collect()` returned the row.
    if mismatched_exactness(lo, value) or mismatched_exactness(hi, value):
        return False
    try:
        if op == "lt":
            return lo is not None and value <= lo
        if op == "le":
            return lo is not None and value < lo
        if op == "gt":
            return hi is not None and value >= hi
        if op == "ge":
            return hi is not None and value > hi
        if op == "eq":
            return (lo is not None and value < lo) or (hi is not None and value > hi)
        if op == "ne":
            return lo is not None and hi is not None and lo == hi == value
    except TypeError:
        return False  # incomparable types (e.g. str vs int) — cannot prove emptiness
    return False


def _bloom_absent(stat: ColumnStat, value) -> bool:
    """Whether `value` is provably absent from the column's membership bloom.

    A bloom has no false negatives, so a `False` from `contains` is definitive — the
    equality cannot match. Absence in the base bloom holds over any subset, independent of
    the column's `provenance` (the bloom answers only absence, never a value).
    """
    if stat.bloom is None:
        return False
    index = BloomIndex.from_bytes(stat.bloom)
    return index is not None and not index.contains(value)


def _is_nan(value) -> bool:
    """Whether `value` is a float NaN (comparisons against which prove nothing)."""
    return is_nan(value)


def _is_zero_float(value) -> bool:
    """Whether `value` is a float zero — the bound where `-0.0` and `0.0` part company.

    Python says `-0.0 == 0.0`; the engine's float comparison (arrow-rs's total order) says
    `-0.0 < 0.0`. So a zero bound is a place where reasoning about the bound and executing the
    predicate can disagree, and the only sound move is to decline. See the note in
    `kyber.shortcuts.bounds.orderable`, which applies the identical rule.
    """
    return isinstance(value, float) and value == 0.0
