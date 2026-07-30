"""A column's exact null count must not be thrown away with the bounds it sits beside.

`InMemorySource.statistics()` falls back to a null-count-only `ColumnStat` when a column has
no trustworthy bounds — a string, a nested type, an all-null column. `column_bounds`, the
*single-column* form, returned `None` for exactly those columns while its docstring claimed
to match `statistics()`.

That is the form the conductor actually calls: every real query narrows to the columns its
predicates name (`api.source_stats._resident_subset_stats`), so the whole-relation path that
got this right was the one nothing took. The exact fact was discarded along with the inexact
one, on precisely the column types most tables are made of.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.source_stats import collect_source_stats
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.plan.stats import Provenance

pytestmark = pytest.mark.unit

_ROWS, _NULLS = 1_000, 100


def _frame(column: pa.Array):
    return bt.from_arrow(pa.table({"c": column}))


def _narrowed(ds):
    """The statistics the conductor really passes: narrowed to the predicate's columns."""
    return collect_source_stats(ds._sources, None, need_columns={"c"})


def _estimate(ds, predicate):
    plan = ds.filter(predicate)._plan
    estimator = CardinalityEstimator(ds._sources, source_stats=_narrowed(ds))
    return estimator.estimate(plan).rows


@pytest.mark.parametrize(
    ("label", "column"),
    [
        ("string", pa.array(["a"] * (_ROWS - _NULLS) + [None] * _NULLS)),
        ("all-null int", pa.array([None] * _ROWS, type=pa.int64())),
        ("list", pa.array([[1, 2]] * (_ROWS - _NULLS) + [None] * _NULLS)),
    ],
)
def test_the_narrowed_path_keeps_an_exact_null_count(label, column):
    stat = _narrowed(_frame(column))[0].columns.get("c")
    assert stat is not None, label
    assert stat.null_count_provenance is Provenance.EXACT, label


def test_an_all_null_column_matches_nothing():
    """`IS NOT NULL` over a column that is entirely null keeps zero rows.

    It estimated 47,500 of 50,000 — the `null_selectivity` prior — because the column had
    no statistic at all on the narrowed path.
    """
    ds = _frame(pa.array([None] * _ROWS, type=pa.int64()))
    assert _estimate(ds, bt.col("c").is_not_null()) == 0
    assert ds.filter(bt.col("c").is_not_null()).collect().num_rows == 0


def test_a_string_column_is_estimated_from_its_measured_nulls():
    ds = _frame(pa.array(["a"] * (_ROWS - _NULLS) + [None] * _NULLS))
    assert _estimate(ds, bt.col("c").is_null()) == pytest.approx(_NULLS)
    assert _estimate(ds, bt.col("c").is_not_null()) == pytest.approx(_ROWS - _NULLS)


def test_a_bounded_column_is_unchanged():
    """The safety property: a column that *has* bounds must still report them."""
    ds = _frame(pa.array(list(range(_ROWS))))
    stat = _narrowed(ds)[0].columns.get("c")
    assert stat is not None
    assert (stat.min, stat.max) == (0, _ROWS - 1)
    assert stat.provenance is Provenance.EXACT


def test_the_two_forms_agree():
    """The invariant that was violated: the single-column form and the whole-relation form
    must describe a column the same way."""
    for column in (
        pa.array(["a"] * 10 + [None] * 5),
        pa.array([None] * 15, type=pa.int64()),
        pa.array(list(range(15))),
    ):
        source = _frame(column)._sources[0]
        whole = source.statistics().columns.get("c")
        single = source.column_bounds("c")
        assert (whole is None) == (single is None)
        if whole is not None:
            assert whole.null_count == single.null_count
            assert (whole.min, whole.max) == (single.min, single.max)
