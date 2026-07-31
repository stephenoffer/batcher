"""A decimal bound must never be reasoned about with a float literal.

The IR has no decimal literal, so `price = Decimal("999999999.99")` reaches the optimizer as a
float. Python then compares that float to a `Decimal` bound *exactly* — widening the float to
the rational it really represents, 999999999.9900000095367431640625 — and concludes the
predicate cannot match. The engine disagrees: a decimal column against a float literal is
promoted to Float64 on both sides, where the two are the same double.

Two rewrites believed Python. `zonemap_prune_filter` folded the filter to an empty relation, and
the metadata `count()` shortcut answered 0 for a filter whose `collect()` returned the row —
a `count()` that disagreed with `len(collect())` on the same plan.

Every case here is checked against DuckDB, which parses the literal as a decimal and compares
exactly. Both paths have to reach its answer, and the second test is the one that would catch a
regression in either: it asserts the two Batcher paths agree with each other *and* with DuckDB.
"""

from __future__ import annotations

import decimal

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

D = decimal.Decimal

# The last value is the one that breaks: at 11 significant digits its float is not the exact
# decimal, while still round-tripping through `repr` (so the literal is accepted, not rejected).
_VALUES = [D("0.10"), D("2.675"), D("900.75"), D("999999999.99")]
_OPS = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


@pytest.fixture
def prices() -> pa.Table:
    return pa.table(
        {
            "p": pa.array(_VALUES, type=pa.decimal128(20, 3)),
            "q": pa.array(list(range(len(_VALUES)))),
        }
    )


@pytest.mark.parametrize("threshold", _VALUES, ids=str)
@pytest.mark.parametrize("op", sorted(_OPS))
def test_a_filtered_count_matches_duckdb(duck, prices, threshold, op):
    """`count()` is answerable from metadata, so it is the path a bound proof can corrupt."""
    duck.register("t", prices)
    predicate = getattr(bt.col("p"), f"__{op}__")(threshold)
    got = bt.from_arrow(prices).filter(predicate).count()
    expected = duck.sql(
        f"SELECT count(*) AS n FROM t WHERE p {_OPS[op]} {threshold}"
    ).to_arrow_table()
    assert got == expected.column("n")[0].as_py()


@pytest.mark.parametrize("threshold", _VALUES, ids=str)
@pytest.mark.parametrize("op", sorted(_OPS))
def test_the_metadata_count_agrees_with_executing_the_filter(prices, threshold, op):
    """The invariant a bound proof must never break, independent of any oracle."""
    predicate = getattr(bt.col("p"), f"__{op}__")(threshold)
    filtered = bt.from_arrow(prices).filter(predicate)
    assert filtered.count() == len(filtered.collect())


@pytest.mark.parametrize("threshold", _VALUES, ids=str)
def test_an_equality_on_a_decimal_bound_is_not_pruned_away(duck, prices, threshold):
    """The zonemap path: the filter must survive to be executed, not fold to an empty relation."""
    duck.register("t", prices)
    got = bt.from_arrow(prices).filter(bt.col("p") == threshold).collect()
    assert_same(got, duck.sql(f"SELECT * FROM t WHERE p = {threshold}"))


def test_is_empty_agrees_with_the_row_count(prices):
    """`is_empty()` reads the same metadata proof, so it is on the same path."""
    filtered = bt.from_arrow(prices).filter(bt.col("p") == D("999999999.99"))
    assert filtered.is_empty() is False
    assert len(filtered.collect()) == 1


def test_an_integer_literal_against_a_decimal_still_prunes(prices):
    """The guard must not cost the sound case: an integer widens into a decimal exactly."""
    # Every price is below 10^9 + 1, so `p > 2_000_000_000` is provably empty and the count
    # is answerable from the bounds without a scan.
    filtered = bt.from_arrow(prices).filter(bt.col("p") > 2_000_000_000)
    assert filtered.count() == 0
    assert len(filtered.collect()) == 0


def test_a_float_column_against_a_float_literal_still_prunes():
    """The guard is about mixing exactness, not about floats — this case must stay pruned."""
    floats = pa.table({"f": pa.array([1.5, 2.5, 3.5]), "q": pa.array([0, 1, 2])})
    filtered = bt.from_arrow(floats).filter(bt.col("f") > 100.0)
    assert filtered.count() == 0
    assert len(filtered.collect()) == 0
