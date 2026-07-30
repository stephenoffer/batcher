"""`Decimal` is the type money is stored in, so it must work as a filter threshold.

`price > Decimal("9.99")` is the natural way, and the only exact way, to write the most
common predicate in analytics. It raised `unsupported literal type: Decimal` -- from `to_ir`,
so on a lazy API the traceback pointed at `collect()` rather than at the `filter`. The decimal
*column* worked the whole time; only the literal was missing.

The IR has no decimal literal, so the value is converted to the float that represents it. That
is exact within float64's ~15 significant digits, which covers every currency amount, rate and
price -- and stops being exact beyond it, where a decimal would silently compare equal to
nothing. DuckDB is the oracle for the first case, and the second raises rather than answering.
"""

from __future__ import annotations

import decimal

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

D = decimal.Decimal

# Values chosen so the binary representation is awkward: 0.10 and 0.30 are not exact in
# binary, 2.675 is the canonical floating-point rounding example, and the last is at the
# top of what a decimal(20,3) money column holds.
_VALUES = [D("0.10"), D("0.20"), D("0.30"), D("1.15"), D("2.675"), D("900.75"), D("999999999.99")]


@pytest.fixture
def prices():
    return pa.table(
        {
            "p": pa.array(_VALUES, type=pa.decimal128(20, 3)),
            "q": pa.array(list(range(len(_VALUES)))),
        }
    )


@pytest.mark.parametrize("threshold", _VALUES, ids=str)
@pytest.mark.parametrize("op", ["gt", "ge", "lt", "le", "eq", "ne"])
def test_every_comparison_against_a_decimal_matches_duckdb(duck, prices, threshold, op):
    duck.register("t", prices)
    sql_op = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "=", "ne": "<>"}[op]
    got = bt.from_arrow(prices).filter(getattr(bt.col("p"), f"__{op}__")(threshold))
    assert_same(got.collect(), duck.sql(f"SELECT * FROM t WHERE p {sql_op} {threshold}"))


def test_decimal_arithmetic_stays_exact_between_two_decimals(duck, prices):
    """Decimal against decimal keeps the decimal type, exactly as DuckDB does."""
    duck.register("t", prices)
    got = bt.from_arrow(prices).select(total=bt.col("p") + bt.col("p"))
    assert_same(got.collect(), duck.sql("SELECT p + p AS total FROM t"))


@pytest.mark.xfail(
    reason=(
        "Pre-existing divergence, independent of decimal literals: multiplying a decimal "
        "column by a scalar promotes to float64, so 0.10 * 1.5 is 0.15000000000000002 where "
        "DuckDB keeps decimal(22,4) and gives 0.1500. It reproduces with a plain float "
        "literal, which predates decimal literals entirely, and decimal-against-decimal "
        "arithmetic is exact. Fixing it is an engine type-promotion change, not a plan one."
    ),
    strict=True,
)
def test_a_decimal_literal_in_arithmetic_matches_duckdb(duck, prices):
    """Money arithmetic against a scalar loses exactness. Recorded, not hidden."""
    duck.register("t", prices)
    got = bt.from_arrow(prices).select(total=bt.col("p") * D("1.5"))
    assert_same(got.collect(), duck.sql("SELECT p * 1.5 AS total FROM t"))


def test_an_integral_decimal_matches_duckdb(duck, prices):
    """`Decimal("900")` carries no fractional part and must behave like the number it is."""
    duck.register("t", prices)
    got = bt.from_arrow(prices).filter(bt.col("p") > D("900"))
    assert_same(got.collect(), duck.sql("SELECT * FROM t WHERE p > 900"))


def test_a_decimal_no_float_represents_is_refused_not_answered_wrongly():
    """The safety property, and the reason this is a conversion rather than a cast.

    A 25-significant-digit decimal has no float64 that identifies it, so comparing through
    one matches nothing at all. Being told that is strictly better than being told zero rows.
    """
    from batcher._internal.errors import PlanError

    frame = bt.from_arrow(pa.table({"p": pa.array([D("1.5")], type=pa.decimal128(20, 3))}))
    with pytest.raises(PlanError, match="significant digits"):
        frame.filter(bt.col("p") > D("12345678901234567890.12345"))


def test_the_error_arrives_at_the_filter_not_at_collect():
    """It used to raise from `to_ir`, which on a lazy API is arbitrarily far from the cause."""
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError):
        bt.lit(D("12345678901234567890.12345"))


def test_a_decimal_literal_holds_a_plain_float():
    """Normalized at construction, so the plan signature and cache key read a plain value."""
    from batcher.plan.expr_ir import Lit

    assert type(Lit(D("9.99")).value) is float
    assert Lit(D("9.99")).value == 9.99


def test_a_decimal_column_aggregates_the_same_as_duckdb(duck, prices):
    """The column side was never broken, and must stay unbroken."""
    duck.register("t", prices)
    got = bt.from_arrow(prices).select(
        lo=bt.col("p").min(), hi=bt.col("p").max(), total=bt.col("p").sum()
    )
    assert_same(
        got.collect(), duck.sql("SELECT MIN(p) AS lo, MAX(p) AS hi, SUM(p) AS total FROM t")
    )
