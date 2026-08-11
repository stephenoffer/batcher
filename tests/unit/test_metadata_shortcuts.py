"""The Kyber shortcut layer, tested as what it is: a pure function from facts to answers.

`kyber.shortcuts` is the *decide* layer — no plan, no engine, no I/O. It takes the facts a
relation's statistics prove and derives what follows from them, or returns `None` to mean "not
provable, go and execute". So it can be tested exactly, with hand-built facts, and every branch
of the provenance firewall pinned without a single row of data.

The differential suite (`tests/differential/test_diff_metadata_shortcuts.py`) proves the other
half — that when these functions *do* answer, the answer equals executing the query. This file
proves they answer only when they should.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

from batcher.kyber.shortcuts import bounds, checks, distinct, joins, moments, nulls, ordering, rows
from batcher.kyber.shortcuts.facts import ColumnFacts, Facts
from batcher.plan.stats import SortOrder, as_sort_orders

pytestmark = pytest.mark.unit

NAN = float("nan")


def facts(
    *,
    rows_count: int | None = 10,
    columns: dict[str, ColumnFacts] | None = None,
    sorted_by: tuple[SortOrder | str, ...] = (),
    nan_safe: bool = True,
) -> Facts:
    """A `Facts` bundle with the fields a test cares about; `rows_count=None` means not exact."""
    return Facts(
        rows=rows_count,
        estimated_rows=float(rows_count if rows_count is not None else 100),
        columns=columns or {},
        sorted_by=as_sort_orders(sorted_by),
        nan_safe=nan_safe,
    )


#: The default column type for the helper below — a module singleton, since an Arrow type is
#: built by a call and a call cannot be a default argument.
INT64 = pa.int64()


def col(name: str = "x", dtype: pa.DataType | None = None, **facets) -> ColumnFacts:
    """One column's facts — every facet defaults to unknown, so a test states only what it means."""
    return ColumnFacts(name=name, dtype=dtype or INT64, **facets)


# --- the firewall: an unprovable fact is None, never a guess -----------------------------


def test_an_exact_null_count_survives_untrustworthy_bounds():
    """A string column's bounds may be truncated (DEFAULT bundle) while its null count is exact.

    This is the whole reason `null_count_provenance` exists: a Parquet footer records a string
    column's null count exactly, but its min/max may be writer-truncated, so tying the two to one
    tag threw the exact null count away. The null answers read `null_count_is_exact`, which is the
    null count's own tag — so they fire here, where the bundle is DEFAULT.
    """
    from batcher.plan.stats import ColumnStat, Provenance

    # A string column: untrustworthy bounds (DEFAULT bundle), but an exactly recorded null count.
    stat = ColumnStat(
        min="aa",
        max="zz",
        null_count=3,
        provenance=Provenance.DEFAULT,
        null_count_provenance=Provenance.EXACT,
    )
    assert stat.null_count_is_exact
    # The distilled facts must carry the null count even though the bundle is not EXACT.
    from batcher.kyber.shortcuts.facts import _column_facts

    facts_col = _column_facts("name", stat, pa.string(), nan_safe=True)
    assert facts_col.null_count == 3
    assert facts_col.min is None  # the untrustworthy bound is *not* carried — only the null count


def test_unknown_row_count_answers_nothing_that_needs_it():
    """Without an exact row count, every count-derived shortcut must decline."""
    f = facts(rows_count=None, columns={"x": col(null_count=2)})
    assert rows.count(f) is None
    assert rows.is_empty(f) is None
    assert rows.shape(f) is None
    assert nulls.non_null_count(f, "x") is None
    assert nulls.null_fraction(f, "x") is None
    assert nulls.all_null(f, "x") is None
    # ...but a fact that needs only the column's own statistic still answers.
    assert nulls.null_count(f, "x") == 2
    assert nulls.has_nulls(f, "x") is True


def test_an_estimate_is_always_available_and_never_exact():
    """`estimated_rows` answers even when `count` cannot — and is never mistaken for it."""
    f = facts(rows_count=None)
    assert rows.count(f) is None
    assert rows.estimated_rows(f) == 100.0
    assert rows.row_count_is_exact(f) is False


def test_sketched_distinct_never_answers_the_exact_one():
    """An approximate ndv informs `approx_n_unique` and nothing else — the distinct firewall."""
    f = facts(columns={"x": col(approx_ndv=7)})  # a sketch: `ndv` (exact) is unset
    assert distinct.n_unique(f, "x") is None
    assert distinct.is_unique(f, "x") is None
    assert distinct.is_key(f, "x") is None
    assert distinct.approx_n_unique(f, "x") == 7


# --- bounds, and what follows from them --------------------------------------------------


def test_bounds_derive_range_midpoint_and_constancy():
    """A min and a max are values that *occur*, so several facts follow from the pair alone."""
    f = facts(columns={"x": col(min=0, max=100)})
    assert bounds.bounds(f, "x") == (0, 100)
    assert bounds.value_range(f, "x") == 100
    assert bounds.midpoint(f, "x") == 50.0
    assert bounds.abs_max(f, "x") == 100.0
    assert bounds.is_constant(f, "x") is False

    same = facts(columns={"x": col(min=7, max=7)})
    assert bounds.is_constant(same, "x") is True
    assert bounds.constant_value(same, "x") == 7


def test_a_column_with_no_bounds_derives_nothing():
    """An all-null column has no extremes, so no bound-derived fact is provable."""
    f = facts(columns={"x": col(null_count=10)})
    assert bounds.bounds(f, "x") is None
    assert bounds.value_range(f, "x") is None
    assert bounds.is_constant(f, "x") is None


# --- the float gate: where the engine's order and Python's part company -------------------


@pytest.mark.parametrize("bound", [NAN, 0.0, -0.0])
def test_an_ambiguous_float_bound_refuses_every_derivation(bound):
    """A NaN or a zero bound answers nothing: it is where the engine's float order and Python's
    disagree, and a derivation across that gap is a wrong answer, not a fast one."""
    f = facts(columns={"f": col("f", pa.float64(), min=bound, max=10.0, null_count=0)})
    assert bounds.orderable(f, "f") is None
    assert bounds.value_range(f, "f") is None
    assert checks.all_positive(f, "f") is None
    assert checks.any_greater_than(f, "f", 1.0) is None
    assert checks.contains(f, "f", 5.0) is None


def test_a_float_source_without_nan_aware_bounds_refuses_too():
    """A footer's max is the largest *non-NaN* value — the true one may be a NaN nobody recorded."""
    f = facts(columns={"f": col("f", pa.float64(), min=1.0, max=9.0)}, nan_safe=False)
    assert bounds.orderable(f, "f") is None
    assert checks.all_positive(f, "f") is None
    # An integer column in the same relation is unaffected — it has no NaN to hide.
    g = facts(columns={"i": col("i", pa.int64(), min=1, max=9, null_count=0)}, nan_safe=False)
    assert checks.all_positive(g, "i") is True


def test_ordinary_floats_still_answer():
    """The gate is narrow on purpose: a float column with no NaN and no zero answers as before."""
    f = facts(columns={"f": col("f", pa.float64(), min=0.5, max=8.5, null_count=0)})
    assert checks.all_positive(f, "f") is True
    assert checks.any_greater_than(f, "f", 9.0) is False
    assert bounds.value_range(f, "f") == 8.0


# --- the checks -------------------------------------------------------------------------


def test_all_checks_are_decided_by_one_bound_each():
    """`all_*` reads the near bound, `any_*` the far one — and both directions are decided."""
    f = facts(columns={"x": col(min=5, max=9, null_count=0)})
    assert checks.all_greater_than(f, "x", 4) is True
    assert checks.all_greater_than(f, "x", 5) is False
    assert checks.all_less_than(f, "x", 10) is True
    assert checks.all_less_equal(f, "x", 9) is True
    assert checks.all_between(f, "x", 5, 9) is True
    assert checks.all_between(f, "x", 6, 9) is False
    # The one that skips a scan: a max at or below the threshold proves *no* row matches.
    assert checks.any_greater_than(f, "x", 9) is False
    assert checks.any_greater_than(f, "x", 8) is True
    assert checks.any_less_than(f, "x", 5) is False
    assert checks.any_less_than(f, "x", 6) is True


def test_a_column_with_no_non_null_value_satisfies_every_all_check_and_no_any_check():
    """The vacuous rule, stated once here so a hundred call sites need not each decide it."""
    empty = facts(rows_count=0, columns={"x": col(null_count=0)})
    all_null = facts(rows_count=4, columns={"x": col(null_count=4)})
    for f in (empty, all_null):
        assert checks.all_positive(f, "x") is True
        assert checks.all_negative(f, "x") is True  # vacuously both, since there are no values
        assert checks.any_greater_than(f, "x", 0) is False
        assert checks.any_less_than(f, "x", 0) is False


def test_membership_is_asymmetric_absence_proves_presence_does_not():
    """Bounds refute a value outside the range; they cannot confirm one inside it."""
    f = facts(columns={"x": col(min=10, max=20, null_count=0)})
    assert checks.contains(f, "x", 99) is False  # provably outside → skip the scan
    assert checks.contains(f, "x", 15) is None  # inside the range proves nothing → execute
    assert checks.may_contain(f, "x", 99) is False
    assert checks.may_contain(f, "x", 15) is True

    # ...except on a constant column, where the one value it holds *is* known.
    one = facts(columns={"x": col(min=7, max=7, null_count=0)})
    assert checks.contains(one, "x", 7) is True
    assert checks.contains(one, "x", 8) is False


def test_an_incomparable_literal_declines_rather_than_raising():
    """A string against a numeric column is a question metadata cannot answer, not an error."""
    f = facts(columns={"x": col(min=1, max=9, null_count=0)})
    assert checks.all_greater_than(f, "x", "banana") is None
    assert checks.contains(f, "x", "banana") is None


# --- uniqueness --------------------------------------------------------------------------


def test_uniqueness_needs_the_distinct_count_and_the_null_count_together():
    """`is_unique` is `ndv == count(col)`, and a key is that *plus* no nulls."""
    unique_with_nulls = facts(rows_count=10, columns={"x": col(ndv=8, null_count=2)})
    assert distinct.is_unique(unique_with_nulls, "x") is True
    assert distinct.has_duplicates(unique_with_nulls, "x") is False
    assert distinct.duplicate_count(unique_with_nulls, "x") == 0
    assert distinct.is_key(unique_with_nulls, "x") is False  # unique, but nullable

    key = facts(rows_count=10, columns={"x": col(ndv=10, null_count=0)})
    assert distinct.is_key(key, "x") is True

    dupes = facts(rows_count=10, columns={"x": col(ndv=3, null_count=0)})
    assert distinct.duplicate_count(dupes, "x") == 7
    assert distinct.is_low_cardinality(dupes, "x", 3) is True
    assert distinct.is_low_cardinality(dupes, "x", 2) is False


# --- sums and averages -------------------------------------------------------------------


def test_the_average_is_derived_from_a_recorded_total_when_no_mean_is_recorded():
    """A source that records only a sum still answers `avg`, from the exact non-null count."""
    f = facts(rows_count=5, columns={"x": col(total_sum=10.0, null_count=1)})
    assert moments.total(f, "x") == 10.0
    assert moments.average(f, "x") == 2.5  # 10 / (5 rows - 1 null)


def test_sum_and_average_of_an_empty_relation_are_not_answered():
    """SQL's `SUM`/`AVG` over no non-null value is NULL — answering `0` would be wrong, not safe."""
    empty = facts(rows_count=0, columns={"x": col(total_sum=0.0, null_count=0)})
    assert moments.total(empty, "x") is None
    all_null = facts(rows_count=3, columns={"x": col(total_sum=0.0, null_count=3)})
    assert moments.average(all_null, "x") is None


# --- ordering ----------------------------------------------------------------------------


def test_sortedness_is_one_sided_a_match_proves_it_and_a_miss_proves_nothing():
    """Only a *recorded* ordering is known, so `is_sorted_by` returns True or None — never False."""
    f = facts(sorted_by=("region", "day"))
    assert ordering.is_sorted_by(f, ["region"]) is True
    assert ordering.is_sorted_by(f, ["region", "day"]) is True
    assert ordering.is_sorted_by(f, ["day"]) is None  # not a prefix — unknown, not false
    assert ordering.sort_prefix(f, ["region", "hour"]) == 1  # the work a sort can still skip


def test_a_descending_ordering_is_recorded_and_only_matches_a_descending_request():
    """Direction is part of the ordering: `ts DESC` is not satisfied by asking for `ts`."""
    f = facts(sorted_by=(SortOrder("ts", descending=True),))
    assert ordering.is_sorted_by(f, [SortOrder("ts", descending=True)]) is True
    assert ordering.is_sorted_by(f, ["ts"]) is None
    assert ordering.sorted_columns(f) == (SortOrder("ts", descending=True),)


def test_null_placement_is_ignored_only_for_a_column_proven_free_of_nulls():
    """With no null row to place, `NULLS FIRST` and `NULLS LAST` are the same row order."""
    proven = facts(
        sorted_by=(SortOrder("k"),),
        columns={"k": ColumnFacts(name="k", null_count=0)},
    )
    assert ordering.is_sorted_by(proven, [SortOrder("k", nulls_first=True)]) is True

    unproven = facts(sorted_by=(SortOrder("k"),))
    assert ordering.is_sorted_by(unproven, [SortOrder("k", nulls_first=True)]) is None


# --- joins -------------------------------------------------------------------------------


def test_disjoint_key_ranges_prove_the_join_empty():
    """Four numbers, no shuffle: if the ranges cannot overlap, the inner join has no rows."""
    left = facts(columns={"k": col("k", pa.int64(), min=1, max=10, null_count=0)})
    right = facts(columns={"k": col("k", pa.int64(), min=900, max=999, null_count=0)})
    assert joins.join_is_empty(left, right, "k", "k") is True
    assert joins.join_is_empty(right, left, "k", "k") is True  # symmetric

    overlapping = facts(columns={"k": col("k", pa.int64(), min=5, max=50, null_count=0)})
    # Overlap does not prove a *match* — the columns may interleave without sharing a value.
    assert joins.join_is_empty(left, overlapping, "k", "k") is None
    assert joins.key_overlap(left, overlapping, "k", "k") == (5, 10)


def test_an_empty_side_proves_the_join_empty_whatever_the_bounds():
    """An inner join against nothing is nothing."""
    empty = facts(rows_count=0)
    other = facts(columns={"k": col("k", pa.int64(), min=1, max=9)})
    assert joins.join_is_empty(empty, other, "k", "k") is True


def test_join_size_estimate_falls_back_to_the_cross_product_it_cannot_beat():
    """With no distinct count on either key, the honest estimate is the worst case, not a guess."""
    left = facts(rows_count=100, columns={"k": col("k")})
    right = facts(rows_count=50, columns={"k": col("k")})
    assert joins.estimated_join_rows(left, right, "k", "k") == 5000.0

    with_ndv = facts(rows_count=100, columns={"k": col("k", approx_ndv=25)})
    assert joins.estimated_join_rows(with_ndv, right, "k", "k") == 200.0  # 100*50/25


# --- nulls, all-or-nothing ---------------------------------------------------------------


def test_the_relation_wide_null_map_is_all_or_nothing():
    """A partial map reads as "the rest have none", which is a different and false statement."""
    partial = facts(columns={"a": col("a", null_count=1), "b": col("b")})  # b unknown
    assert nulls.null_counts(partial) is None
    assert nulls.columns_with_nulls(partial) is None
    assert nulls.is_complete(partial) is None

    known = facts(columns={"a": col("a", null_count=1), "b": col("b", null_count=0)})
    assert nulls.null_counts(known) == {"a": 1, "b": 0}
    assert nulls.columns_with_nulls(known) == ["a"]
    assert nulls.complete_columns(known) == ["b"]
    assert nulls.is_complete(known) is False


def test_an_empty_relation_is_not_reported_all_null():
    """It has no values to be null; calling it all-null would make `all_null` and `no_nulls` both
    true, which is a contradiction a caller would be right to trust and wrong to act on."""
    f = facts(rows_count=0, columns={"x": col(null_count=0)})
    assert nulls.all_null(f, "x") is False
    assert nulls.no_nulls(f, "x") is True
    assert nulls.null_fraction(f, "x") == 0.0  # not a division by zero
    assert not math.isnan(nulls.null_fraction(f, "x"))
