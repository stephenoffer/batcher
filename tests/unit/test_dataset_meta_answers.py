"""`ds.meta` answers must equal what an explicit query returns.

Eight public classes and roughly a hundred methods under `api/dataset/meta/`, and no test
mentioned any of them. The risk is structural rather than hypothetical: `ds.meta` answers from
recorded statistics wherever it can, so a wrong answer is a confident number with nothing
executing to contradict it — the same shape as the bug where a learned row count made an
unrelated HAVING return no rows.

Ground truth here is computed in Python from the input, never from another engine path, so these
tests cannot pass by two components agreeing on the same mistake.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit

_SHAPES = {
    "clean": [1, 2, 3, 4, 5],
    "with nulls": [1, None, 3, None, 5],
    "constant": [7, 7, 7, 7],
    "duplicates": [1, 1, 2, 2, 3],
    "unique": [5, 3, 1, 4, 2],
    "one row": [9],
    "negatives": [-5, -1, 0, 2, 7],
}


def _ds(values: list) -> bt.Dataset:
    return bt.from_pydict({"a": values, "b": list(range(10, 10 + len(values)))})


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_column_summaries_match_the_column(shape: str) -> None:
    values = _SHAPES[shape]
    present = [v for v in values if v is not None]
    meta = _ds(values).meta.col("a")

    assert tuple(meta.bounds()) == (min(present), max(present))
    assert meta.n_unique() == len(set(present))
    assert meta.sum() == sum(present)
    assert meta.mean() == pytest.approx(sum(present) / len(present))
    assert meta.abs_max() == max(abs(v) for v in present)
    assert meta.range() == max(present) - min(present)
    assert meta.midpoint() == pytest.approx((max(present) + min(present)) / 2)
    assert meta.is_constant() is (len(set(present)) == 1)
    assert meta.is_unique() is (len(present) == len(set(present)))
    assert meta.has_duplicates() is (len(present) != len(set(present)))
    assert meta.no_nulls() is all(v is not None for v in values)
    assert meta.null_fraction() == pytest.approx(sum(1 for v in values if v is None) / len(values))


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_column_summaries_agree_with_an_executed_aggregate(shape: str) -> None:
    """The shortcut and the query must not disagree, whichever path each one took."""
    values = _SHAPES[shape]
    ds = _ds(values)
    meta = ds.meta.col("a")
    row = ds.agg(
        lo=bt.col("a").min(),
        hi=bt.col("a").max(),
        total=bt.col("a").sum(),
        avg=bt.col("a").mean(),
        distinct=bt.col("a").n_unique(),
    ).to_pydict()

    assert tuple(meta.bounds()) == (row["lo"][0], row["hi"][0])
    assert meta.sum() == row["total"][0]
    assert meta.mean() == pytest.approx(row["avg"][0])
    assert meta.n_unique() == row["distinct"][0]


def test_constant_value_is_the_value_only_when_the_column_is_constant() -> None:
    assert _ds([7, 7, 7, 7]).meta.col("a").constant_value() == 7
    assert _ds([1, 2, 3]).meta.col("a").constant_value() is None


def test_is_binary_valued_means_at_most_two_distinct_values() -> None:
    """ "At most two", as documented — so a constant column qualifies as a degenerate flag."""
    assert _ds([0, 1, 1, 0]).meta.col("a").is_binary_valued() is True
    assert _ds([7, 7, 7]).meta.col("a").is_binary_valued() is True
    assert _ds([1, 2, 3]).meta.col("a").is_binary_valued() is False


def test_is_low_cardinality_respects_its_threshold() -> None:
    ds = _ds(list(range(50)))
    assert ds.meta.col("a").is_low_cardinality() is True
    assert ds.meta.col("a").is_low_cardinality(10) is False
    assert ds.meta.col("a").is_low_cardinality(50) is True


def test_is_key_requires_unique_and_complete() -> None:
    assert _ds([1, 2, 3]).meta.col("a").is_key() is True
    assert _ds([1, 1, 3]).meta.col("a").is_key() is False
    assert _ds([1, None, 3]).meta.col("a").is_key() is False, "a null cannot be part of a key"


# --------------------------------------------------------------------------- #
# NullsMeta
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_null_counts_and_fractions_match_the_data(shape: str) -> None:
    values = _SHAPES[shape]
    ds = _ds(values)
    nulls = ds.meta.nulls
    a_nulls = sum(1 for v in values if v is None)

    assert nulls.counts()["a"] == a_nulls
    assert nulls.counts()["b"] == 0
    assert nulls.total() == a_nulls
    assert nulls.fractions()["a"] == pytest.approx(a_nulls / len(values))
    assert nulls.any() is (a_nulls > 0)
    assert nulls.is_complete() is (a_nulls == 0)
    assert ("a" in nulls.columns_with_nulls()) is (a_nulls > 0)
    assert "b" in nulls.complete_columns()


def test_null_counts_agree_with_the_executed_null_count_table() -> None:
    """`ds.null_count()` is a per-column table, so the shortcut is checked against every column."""
    ds = _ds([1, None, 3, None, 5])
    executed = ds.null_count().to_pydict()
    counts = ds.meta.nulls.counts()
    assert counts["a"] == executed["a"][0]
    assert counts["b"] == executed["b"][0]
    assert ds.meta.nulls.total() == sum(v[0] for v in executed.values())


# --------------------------------------------------------------------------- #
# SchemaMeta
# --------------------------------------------------------------------------- #
def test_schema_classifies_each_column_family() -> None:
    ds = bt.from_pydict(
        {
            "i": [1, 2],
            "f": [1.5, 2.5],
            "s": ["x", "y"],
            "b": [True, False],
        }
    )
    schema = ds.meta.schema
    assert schema.num_columns() == 4
    assert schema.has("i") and not schema.has("missing")
    assert schema.index("f") == 1
    assert schema.is_integer("i") and not schema.is_integer("f")
    assert schema.is_float("f") and not schema.is_float("i")
    assert schema.is_numeric("i") and schema.is_numeric("f")
    assert schema.is_string("s") and not schema.is_string("i")
    assert schema.is_boolean("b") and not schema.is_boolean("s")
    assert not schema.is_nested("i")
    assert sorted(schema.numeric()) == ["f", "i"]
    assert schema.strings() == ["s"]
    assert schema.booleans() == ["b"]
    assert schema.nested() == []


def test_schema_dtype_is_the_arrow_type() -> None:
    import pyarrow as pa

    ds = bt.from_pydict({"i": [1, 2], "s": ["x", "y"]})
    assert ds.meta.schema.dtype("i") == pa.int64()
    assert (
        ds.meta.schema.dtype("s") == pa.large_string() or ds.meta.schema.dtype("s") == pa.string()
    )


# --------------------------------------------------------------------------- #
# Predicate answers must match an executed filter
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", sorted(_SHAPES))
@pytest.mark.parametrize("threshold", [-10, 0, 2, 1000])
def test_count_where_matches_an_executed_filter(shape: str, threshold: int) -> None:
    ds = _ds(_SHAPES[shape])
    predicate = bt.col("a") > bt.lit(threshold)
    executed = ds.filter(predicate).count()

    assert ds.meta.count_where(predicate) == executed
    assert ds.meta.is_empty_where(predicate) is (executed == 0)


# --------------------------------------------------------------------------- #
# ColumnChecks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_column_checks_match_the_values(shape: str) -> None:
    values = _SHAPES[shape]
    present = [v for v in values if v is not None]
    check = _ds(values).meta.col("a").check

    assert check.all_positive() is all(v > 0 for v in present)
    assert check.all_negative() is all(v < 0 for v in present)
    assert check.all_non_negative() is all(v >= 0 for v in present)
    assert check.all_zero() is all(v == 0 for v in present)
    assert check.all_between(0, 100) is all(0 <= v <= 100 for v in present)
    assert check.all_greater_than(0) is all(v > 0 for v in present)
    assert check.all_less_than(100) is all(v < 100 for v in present)


def test_a_failing_check_is_false_not_an_error() -> None:
    check = _ds([-5, 1, 2]).meta.col("a").check
    assert check.all_positive() is False
    assert check.all_non_negative() is False
    assert check.all_between(0, 10) is False
