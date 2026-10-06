"""`round` on a DECIMAL column keeps the decimal and rounds exactly — vs DuckDB (AP-274).

It used to promote to DOUBLE, which mistyped the column and broke ties on the binary
approximation: `round(2.345, 2, mode="half_to_even")` answered 2.35, because 2.345 is
stored as 2.34500000000000019984 in a double. The exact answer is 2.34.

`assert_same` is type-tolerant by design, so it could never have caught the type half of
that. Every comparison here reads the result *type* from the collected schema and the
values as exact `Decimal`s, in row order.

DuckDB's `round(DECIMAL(p,s), d)` is `DECIMAL(p, min(s, max(d, 0)))`, half away from zero,
and both are matched outright. DuckDB's `round_even` on a decimal returns DOUBLE; Batcher
keeps the decimal there too. That type difference is pinned below rather than hidden, and
the half-to-even *values* are held to Python's exact `Decimal.quantize(ROUND_HALF_EVEN)`.
"""

from __future__ import annotations

import decimal

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential

DECIMAL = pa.decimal128(10, 3)
D = decimal.Decimal
# Ties at every place, negatives, a duplicate, a null, and values that carry into a new digit.
VALUES = [
    D("2.345"),
    D("-2.345"),
    D("2.355"),
    D("2.345"),
    None,
    D("0.005"),
    D("-0.500"),
    D("99.995"),
    D("1234.500"),
]
DIGITS = [-2, -1, 0, 1, 2, 3, 5]


def _table(values: list[D | None]) -> pa.Table:
    return pa.table(
        {"k": pa.array(range(len(values)), pa.int64()), "n": pa.array(values, type=DECIMAL)}
    )


def _duck(duck, table: pa.Table, expr: str) -> pa.ChunkedArray:
    duck.register("t", table)
    return duck.sql(f"SELECT {expr} AS r FROM t ORDER BY k").arrow().read_all().column("r")


def _ours(table: pa.Table, expr) -> pa.ChunkedArray:
    return bt.from_arrow(table).select(k=col("k"), r=expr).sort("k").collect().column("r")


@pytest.mark.parametrize("digits", DIGITS)
def test_round_matches_duckdb_value_and_type(duck, digits):
    table = _table(VALUES)
    want = _duck(duck, table, f"round(n, {digits})")
    got = _ours(table, col("n").round(digits))
    assert got.type == want.type
    assert got.to_pylist() == want.to_pylist()


def test_unary_round_matches_duckdb_value_and_type(duck):
    table = _table(VALUES)
    want = _duck(duck, table, "round(n)")
    got = _ours(table, col("n").round())
    assert got.type == want.type == pa.decimal128(10, 0)
    assert got.to_pylist() == want.to_pylist()


@pytest.mark.parametrize("digits", [0, 1, 2])
def test_half_to_even_is_exact_and_stays_decimal(duck, digits):
    table = _table(VALUES)
    got = _ours(table, col("n").round(digits, mode="half_to_even"))
    quantum = D(1).scaleb(-digits)
    exact = [
        None if v is None else v.quantize(quantum, rounding=decimal.ROUND_HALF_EVEN) for v in VALUES
    ]
    assert got.type == pa.decimal128(10, digits)
    assert got.to_pylist() == exact
    # The headline case, spelled out: the double path answered 2.35.
    if digits == 2:
        assert got.to_pylist()[0] == D("2.34")
    # Pinned divergence: DuckDB's `round_even` on a decimal is DOUBLE. The values agree.
    duck_values = _duck(duck, table, f"round_even(n, {digits})")
    assert duck_values.type == pa.float64()
    assert [None if v is None else float(v) for v in exact] == duck_values.to_pylist()


@pytest.mark.parametrize(
    "values", [[], [D("7.125")], [None, None]], ids=["empty", "one-row", "all-null"]
)
def test_edge_inputs_match_duckdb(duck, values):
    table = _table(values)
    want = _duck(duck, table, "round(n, 2)")
    got = _ours(table, col("n").round(2))
    assert got.type == pa.decimal128(10, 2)
    assert got.to_pylist() == want.to_pylist()


def test_declared_schema_matches_the_result():
    """`Dataset.schema` is answered from static inference; it must name the type the engine
    then produces, for both the binary and the unary form."""
    ds = bt.from_arrow(_table(VALUES)).select(
        a=col("n").round(1), b=col("n").round(), c=col("n").round(-1, mode="half_to_even")
    )
    declared = {f.name: f.type for f in ds.schema}
    actual = ds.collect().schema
    for name in ("a", "b", "c"):
        assert declared[name] == actual.field(name).type
    assert actual.field("a").type == pa.decimal128(10, 1)
    assert actual.field("c").type == pa.decimal128(10, 0)


def test_a_carry_past_the_precision_raises_rather_than_overflowing(duck):
    """`round(999::DECIMAL(3,0), -1)` is 1000, which DECIMAL(3,0) cannot hold. DuckDB
    returns the out-of-range value anyway; Batcher refuses it with a hint, rather than
    storing a number the column's own type says is impossible."""
    table = pa.table({"k": [0], "n": pa.array([D("999")], type=pa.decimal128(3, 0))})
    with pytest.raises(Exception, match="overflows"):
        _ours(table, col("n").round(-1))


def test_dataset_round_keeps_decimal_columns():
    """`Dataset.round` rounds every numeric column through `Expr.round`, so a money column
    stays money."""
    ds = bt.from_arrow(
        pa.table({"n": pa.array([D("2.345"), D("-1.005")], type=DECIMAL), "f": [1.26, 2.5]})
    )
    out = ds.round(2).collect()
    assert out.schema.field("n").type == pa.decimal128(10, 2)
    assert out.column("n").to_pylist() == [D("2.35"), D("-1.01")]
    assert out.column("f").to_pylist() == [1.26, 2.5]
