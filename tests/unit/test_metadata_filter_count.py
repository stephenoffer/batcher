"""The filtered-count shortcuts fire only from EXACT stats, and fall back otherwise.

Pins the provenance firewall and the predicate-shape matching of
`metadata_filter_count`: a null predicate answers from an EXACT null count, a
provably-empty comparison from EXACT footer bounds (or a column bloom), a tautology from
an EXACT row count — and anything weaker (a filtered/downgraded stat, a partial-overlap
range) returns `None`. Exercises the pure helpers directly (no optimizer dependence) plus
a real Parquet footer end to end.
"""

from __future__ import annotations

import batcher._native as native
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import core
from batcher.api.orchestration import collect_source_stats
from batcher.kyber.metadata_filter_count import answer_filter_count, answer_filter_is_empty
from batcher.kyber.metadata_filter_count.answers import (
    _comparison_empty,
    _exact_surviving_count,
    _parse_comparison,
    _strip_not,
)
from batcher.plan.expr_ir import Binary, Col, IsNotNull, IsNull, Lit, Not
from batcher.plan.stats import ColumnStat, Provenance, RelStats

pytestmark = pytest.mark.unit


def _exact_rel(rows, null_count=None, cmin=None, cmax=None, ndv=None, bloom=None):
    col = ColumnStat(
        min=cmin,
        max=cmax,
        null_count=null_count,
        ndv=ndv,
        provenance=Provenance.EXACT,
        bloom=bloom,
    )
    return RelStats(float(rows), Provenance.EXACT, {"x": col})


# --- null predicates over EXACT stats ---


def test_is_null_returns_exact_null_count():
    rel = _exact_rel(10, null_count=3)
    assert _exact_surviving_count(IsNull(Col("x")), rel) == 3


def test_is_not_null_returns_rows_minus_nulls():
    rel = _exact_rel(10, null_count=3)
    assert _exact_surviving_count(IsNotNull(Col("x")), rel) == 7


def test_not_normalisation_flips_null_predicates():
    rel = _exact_rel(10, null_count=3)
    assert _exact_surviving_count(Not(IsNull(Col("x"))), rel) == 7  # → IS NOT NULL
    assert _exact_surviving_count(Not(IsNotNull(Col("x"))), rel) == 3  # → IS NULL
    assert _exact_surviving_count(Not(Not(IsNull(Col("x")))), rel) == 3  # double-not


def test_tautology_returns_all_rows():
    rel = _exact_rel(10, null_count=3)
    taut = Binary("or", IsNull(Col("x")), IsNotNull(Col("x")))
    assert _exact_surviving_count(taut, rel) == 10
    assert _exact_surviving_count(Binary("or", IsNotNull(Col("x")), IsNull(Col("x"))), rel) == 10


# --- the firewall: a non-EXACT bundle never answers ---


def test_downgraded_stats_return_none():
    col = ColumnStat(null_count=3, min=1, max=5, provenance=Provenance.DEFAULT)
    rel = RelStats(10.0, Provenance.DEFAULT, {"x": col})
    assert _exact_surviving_count(IsNull(Col("x")), rel) is None
    assert _exact_surviving_count(IsNotNull(Col("x")), rel) is None
    assert _comparison_empty("gt", col, 100) is False  # non-EXACT range proves nothing


def test_missing_null_count_returns_none():
    rel = _exact_rel(10, null_count=None)
    assert _exact_surviving_count(IsNull(Col("x")), rel) is None


# --- provably-empty comparisons over EXACT footer bounds ---


@pytest.mark.parametrize(
    "op,value,expected",
    [
        ("lt", 1, True),  # v <= min
        ("lt", 2, False),  # partial overlap
        ("le", 0, True),  # v < min
        ("le", 1, False),
        ("gt", 5, True),  # v >= max
        ("gt", 4, False),
        ("ge", 6, True),  # v > max
        ("ge", 5, False),
        ("eq", 999, True),  # outside range
        ("eq", 0, True),
        ("eq", 3, False),  # inside range
        ("ne", 3, False),  # min != max, some rows survive
    ],
)
def test_comparison_empty_range(op, value, expected):
    col = ColumnStat(min=1, max=5, null_count=0, provenance=Provenance.EXACT)
    assert _comparison_empty(op, col, value) is expected


def test_ne_over_constant_column_is_empty():
    const = ColumnStat(min=7, max=7, null_count=0, provenance=Provenance.EXACT)
    assert _comparison_empty("ne", const, 7) is True  # every value == 7 → `!= 7` empty
    assert _comparison_empty("ne", const, 8) is False


def test_nan_bound_proves_nothing():
    col = ColumnStat(min=float("nan"), max=5.0, null_count=0, provenance=Provenance.EXACT)
    assert _comparison_empty("lt", col, -1.0) is False


def test_incomparable_types_return_false():
    col = ColumnStat(min=1, max=5, null_count=0, provenance=Provenance.EXACT)
    assert _comparison_empty("gt", col, "z") is False  # str vs int → cannot prove


# --- bloom absence proves equality-empty even inside the range ---


def test_bloom_absence_proves_equality_empty():
    batch = pa.record_batch({"id": pa.array([10, 20, 30, 40], type=pa.int64())})
    bloom = native.build_column_bloom([batch], 0, 4)
    col = ColumnStat(min=10, max=40, null_count=0, provenance=Provenance.EXACT, bloom=bloom)
    assert _comparison_empty("eq", col, 25) is True  # inside [10,40] but absent from bloom
    assert _comparison_empty("eq", col, 20) is False  # present


# --- parsing / normalisation helpers ---


def test_parse_comparison_both_orientations():
    assert _parse_comparison(Binary("gt", Col("x"), Lit(5))) == ("gt", "x", 5)
    assert _parse_comparison(Binary("lt", Lit(5), Col("x"))) == ("gt", "x", 5)  # flipped
    assert _parse_comparison(Binary("add", Col("x"), Lit(5))) is None
    assert _parse_comparison(Binary("eq", Col("x"), Col("y"))) is None


def test_strip_not_leaves_non_null_predicates():
    # `NOT (x > 5)` is not a handled shape — it is returned untouched (still a `Not`).
    wrapped = _strip_not(Not(Binary("gt", Col("x"), Lit(5))))
    assert isinstance(wrapped, Not)
    # `NOT (x IS NULL)` normalises to `IS NOT NULL`.
    assert isinstance(_strip_not(Not(IsNull(Col("x")))), IsNotNull)


# --- end to end over a real Parquet footer ---


@pytest.fixture
def pq_path(tmp_path):
    table = pa.table({"x": pa.array([3, 1, 2, None, 5], type=pa.int64())})
    path = str(tmp_path / "t.parquet")
    pq.write_table(table, path)
    return path


def _count(ds):
    stats = collect_source_stats(ds._sources, core.default_hub())
    return answer_filter_count(ds._plan, ds._sources, stats, core.default_hub())


def test_footer_fires_for_null_and_empty_shapes(pq_path):
    ds = bt.read.parquet(pq_path)
    assert _count(ds.filter(bt.col("x").is_null())) == 1
    assert _count(ds.filter(bt.col("x").is_not_null())) == 4
    assert _count(ds.filter(bt.col("x") > 100)) == 0
    assert _count(ds.filter(bt.col("x") < 0)) == 0
    assert (
        answer_filter_is_empty(
            (e := ds.filter(bt.col("x") > 100))._plan,
            e._sources,
            collect_source_stats(e._sources, core.default_hub()),
            core.default_hub(),
        )
        is True
    )


def test_footer_falls_back_on_partial_overlap(pq_path):
    ds = bt.read.parquet(pq_path).filter(bt.col("x") > 2)
    assert _count(ds) is None  # partial overlap needs a histogram → execute
    assert ds.count() == 2  # but execution is correct


def test_in_memory_range_pruned_from_learned_bounds(pq_path):
    ds = bt.from_arrow(pa.table({"x": pa.array([1, 2, None, 4], type=pa.int64())}))
    # `max(x) == 4 < 100`, so the range predicate is pruned to 0 from the source's EXACT
    # column bounds (learned per instance) rather than executed — and equals a full run.
    assert _count(ds.filter(bt.col("x") > 100)) == 0
    assert ds.filter(bt.col("x") > 100).count() == 0


# --- a conjunction is provably empty when either conjunct is -----------------------


def _child_with_bounds():
    from batcher.plan.stats import ColumnStat, Provenance, RelStats

    return RelStats(
        100.0,
        Provenance.EXACT,
        {
            "x": ColumnStat(min=1, max=4, null_count=0, provenance=Provenance.EXACT),
            "y": ColumnStat(min=1, max=2, null_count=0, provenance=Provenance.EXACT),
        },
    )


def test_conjunction_with_a_provably_empty_conjunct_counts_zero():
    """`A AND B` keeps a subset of `A`, so an empty `A` makes the conjunction empty.

    This is the shape of every real lakehouse filter (`day = 42 AND region = 'us'`), which
    the bare-comparison parse could not see at all — a partition pruned to nothing still
    executed to discover it.
    """
    from batcher.kyber.metadata_filter_count.answers import _exact_surviving_count
    from batcher.plan.expr_ir import col, lit

    child = _child_with_bounds()
    # `x > 100` is provably empty (max is 4) — on either side, and nested.
    assert _exact_surviving_count((col("x") > lit(100)) & (col("y") == lit(1)), child) == 0
    assert _exact_surviving_count((col("y") == lit(1)) & (col("x") > lit(100)), child) == 0
    nested = (col("y") == lit(1)) & ((col("y") == lit(2)) & (col("x") > lit(100)))
    assert _exact_surviving_count(nested, child) == 0


def test_conjunction_of_partial_overlaps_still_declines():
    """Neither conjunct is provably empty, so the count needs a histogram — execute."""
    from batcher.kyber.metadata_filter_count.answers import _exact_surviving_count
    from batcher.plan.expr_ir import col, lit

    child = _child_with_bounds()
    assert _exact_surviving_count((col("x") > lit(2)) & (col("y") == lit(1)), child) is None
