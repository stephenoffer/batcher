"""The math-function family over a `DECIMAL` column — vs DuckDB.

Batcher handled `Decimal` well almost everywhere: arithmetic, comparison, aggregation and
negation all work and all keep the value *exact*. The math functions were the exception —
every one of them rejected a decimal column outright:

    RuntimeError: Abs expected a numeric argument, got Decimal128(10, 2)

So a Parquet money column could be summed, compared and multiplied, but not rounded,
floored, or passed to `abs`. That is not precision being protected; it is the question
being refused. They now promote to Float64, which is the same path integers already took.

**The result type is a stated trade, not an oversight.** For `sqrt`, `ln`, `exp` and the
trig family, DOUBLE is exactly what DuckDB returns, so those agree outright. For `abs`,
`floor`, `ceil`, `round` and `sign` DuckDB keeps DECIMAL, so Batcher's answer is the same
*number* in a different type, and loses exactness above 2^53. The census pinned that
divergence for `ceil`/`floor` already; it now covers the family. A decimal-preserving path
for that subset needs a scale-aware kernel per operation and is the follow-on.

This file therefore compares **values**, not types, for the DECIMAL-returning subset, and
says so — an order-independent or type-tolerant comparison hiding the difference is
exactly the failure mode worth avoiding here.

Still refused, and correctly: the bitwise family, `chr`, `factorial`. Still refused and
*not* correctly: `stddev`/`var`/`median` and the window aggregates, which reject decimals
in `bc-runtime`'s dispatch rather than this one. That is a separate change.
"""

from __future__ import annotations

import decimal

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

DECIMAL = pa.decimal128(12, 3)
VALUES = [decimal.Decimal("3.500"), decimal.Decimal("-1.250"), decimal.Decimal("2.000")]

# Functions DuckDB also returns DOUBLE for: value *and* type agree.
DOUBLE_IN_BOTH = [
    ("sqrt", lambda: col("n").abs().sqrt(), "sqrt(abs(n))"),
    ("ln", lambda: col("n").abs().ln(), "ln(abs(n))"),
    ("exp", lambda: col("n").exp(), "exp(n)"),
    ("log10", lambda: col("n").abs().log10(), "log10(abs(n))"),
    ("sin", lambda: col("n").sin(), "sin(n)"),
    ("cos", lambda: col("n").cos(), "cos(n)"),
    ("atan", lambda: col("n").atan(), "atan(n)"),
    ("tanh", lambda: col("n").tanh(), "tanh(n)"),
]

# Functions DuckDB keeps DECIMAL for: the *number* agrees, the type does not.
DECIMAL_IN_DUCKDB = [
    ("abs", lambda: col("n").abs(), "abs(n)"),
    ("floor", lambda: col("n").floor(), "floor(n)"),
    ("ceil", lambda: col("n").ceil(), "ceil(n)"),
    ("round", lambda: col("n").round(), "round(n)"),
    ("sign", lambda: col("n").sign(), "sign(n)"),
]


@pytest.fixture
def money(duck):
    t = pa.table({"k": [0, 1, 2], "n": pa.array(VALUES, type=DECIMAL)})
    duck.register("t", t)
    return t


def _duck_floats(duck, expr: str) -> list[float]:
    rows = duck.sql(f"SELECT {expr} r FROM t ORDER BY k").arrow().read_all().to_pydict()["r"]
    return [None if v is None else float(v) for v in rows]


@pytest.mark.differential
@pytest.mark.parametrize(("name", "build", "expr"), DOUBLE_IN_BOTH + DECIMAL_IN_DUCKDB)
def test_the_value_matches_duckdb(duck, money, name, build, expr):
    got = bt.from_arrow(money).select(k=col("k"), r=build()).sort("k").to_pydict()["r"]
    assert got == pytest.approx(_duck_floats(duck, expr), rel=1e-12)


@pytest.mark.differential
@pytest.mark.parametrize(("name", "build", "expr"), DOUBLE_IN_BOTH)
def test_the_type_also_matches_where_duckdb_returns_double(duck, money, name, build, expr):
    got = bt.from_arrow(money).select(r=build()).collect()
    assert got.schema.field("r").type == pa.float64()


@pytest.mark.differential
@pytest.mark.parametrize(("name", "build", "expr"), DECIMAL_IN_DUCKDB)
def test_the_pinned_divergence_is_the_type_and_only_the_type(duck, money, name, build, expr):
    """Asserted rather than described, so neither side can drift silently: Batcher answers
    DOUBLE where DuckDB answers DECIMAL, and the numbers are equal."""
    got = bt.from_arrow(money).select(r=build()).collect()
    duck_type = duck.sql(f"SELECT {expr} r FROM t").arrow().schema.field("r").type
    assert got.schema.field("r").type == pa.float64()
    assert pa.types.is_decimal(duck_type) or pa.types.is_integer(duck_type)


@pytest.mark.differential
def test_exact_decimal_paths_are_untouched(duck, money):
    """The half that already worked must keep working *and* keep being exact — the risk of
    a promotion is that it leaks into the paths that had no need of it."""
    out = bt.from_arrow(money).select(
        k=col("k"), plus=col("n") + 1, times=col("n") * 2, neg=-col("n")
    )
    schema = out.collect().schema
    assert pa.types.is_decimal(schema.field("plus").type)
    assert pa.types.is_decimal(schema.field("times").type)
    assert pa.types.is_decimal(schema.field("neg").type)
    total = bt.from_arrow(money).agg(s=col("n").sum()).to_pydict()["s"][0]
    assert total == decimal.Decimal("4.250")


@pytest.mark.differential
def test_a_decimal_column_can_be_rounded_at_all(money):
    """The headline regression, stated on its own: this raised before."""
    assert bt.from_arrow(money).select(r=col("n").round()).to_pydict()["r"] == [4.0, -1.0, 2.0]


@pytest.mark.differential
def test_a_non_numeric_column_still_errors(money):
    """Promoting decimals must not make the numeric check meaningless."""
    with pytest.raises(Exception, match=r"numeric|argument"):
        bt.from_pydict({"s": ["a", "b"]}).select(r=col("s").sqrt()).collect()
