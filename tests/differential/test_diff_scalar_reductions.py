"""The whole-dataset scalar reductions equal DuckDB.

`mean` / `sum` / `std` / `var` complete the scalar-terminal family (alongside the
existing `min` / `max` / `median`), so `ds.mean("x")` reads like `ds.median("x")`.
Each returns one scalar and MUST match DuckDB's executed answer, including the
null-ignoring and empty-input edges.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def numbers(duck):
    t = pa.table(
        {
            "i": pa.array([2, 4, 4, 4, 5, None, 7, 9], type=pa.int64()),
            "f": pa.array([1.5, 2.5, None, 4.0, 5.5, 6.0, 7.5, 8.0], type=pa.float64()),
        }
    )
    duck.register("numbers", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize("column", ["i", "f"])
@pytest.mark.parametrize(
    ("method", "sql"),
    [
        ("mean", "AVG({c})"),
        ("sum", "SUM({c})"),
        ("std", "STDDEV_SAMP({c})"),
        ("var", "VAR_SAMP({c})"),
    ],
)
def test_scalar_reduction_matches_duckdb(duck, numbers, column, method, sql):
    got = getattr(bt.from_arrow(numbers), method)(column)
    want = duck.sql(f"SELECT {sql.format(c=column)} FROM numbers").fetchone()[0]
    assert got == pytest.approx(want)


@pytest.mark.differential
def test_empty_column_reductions_are_none(numbers):
    """An empty (all-filtered) column reduces to None (SQL semantics), never 0 or a crash."""
    empty = bt.from_arrow(numbers).filter(bt.col("i") > 1000)
    assert empty.sum("i") is None
    assert empty.mean("i") is None
    assert empty.std("i") is None
    assert empty.var("i") is None


@pytest.mark.differential
def test_single_value_variance_is_none(duck):
    """Sample std/var of a single value is undefined — matches DuckDB's NULL."""
    one = bt.from_pydict({"x": [42.0]})
    assert one.std("x") is None
    assert one.var("x") is None
    assert one.mean("x") == 42.0
    assert one.sum("x") == 42.0


@pytest.mark.differential
def test_unknown_column_raises_plan_error(numbers):
    from batcher._internal.errors import PlanError

    for method in ("mean", "sum", "std", "var"):
        with pytest.raises(PlanError, match="unknown column"):
            getattr(bt.from_arrow(numbers), method)("nope")


@pytest.mark.differential
def test_reduction_family_is_complete_and_uniform(numbers):
    """min/max/sum/mean/median/std/var/n_unique all take one column and return a scalar."""
    ds = bt.from_arrow(numbers)
    for method in ("min", "max", "sum", "mean", "median", "std", "var", "n_unique"):
        value = getattr(ds, method)("i")  # no AttributeError — the family is complete
        assert not isinstance(value, bt.Dataset)  # a scalar, not a lazy frame
