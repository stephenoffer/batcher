"""`SUM(decimal(p, s))` is `decimal(38, s)` — the type, not only the value, against DuckDB.

The accumulators used to keep the input's precision, so a sum could leave its own declared
type: three `99999.99`s and a `1.10` in a `decimal(7, 2)` returned `300001.07` under a
seven-digit type. Every value-only comparison passed, because the `i128` held the right
number. These tests assert the **type** three ways — the collected column, the declared
`Dataset.schema`, and DuckDB's result type — across the paths that build a decimal sum state
differently: the per-call kernel (one aggregate), the fused one (several), the combine that
merges partials (a relation over many morsels), and `iter_batches`.
"""

from __future__ import annotations

import decimal as D

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

SUM38 = pa.decimal128(38, 2)


def _assert_sum_type(ds: bt.Dataset, out: pa.Table, duck_rel, column: str = "s") -> None:
    """The collected type, the declared type and DuckDB's type are all `decimal(38, 2)`."""
    duck_type = duck_rel.to_arrow_table().schema.field(column).type
    assert duck_type == SUM38, f"oracle drifted: DuckDB returned {duck_type}"
    assert out.schema.field(column).type == SUM38
    assert ds.schema.field(column).type == SUM38


@pytest.fixture
def overflowing(duck):
    """A `decimal(7, 2)` column whose sum needs eight digits, with nulls and negatives."""
    values = [D.Decimal("99999.99")] * 3 + [D.Decimal("1.10"), None, D.Decimal("-0.05")]
    tbl = pa.table(
        {"k": [1, 1, 1, 1, 2, 2], "p": pa.array(values, pa.decimal128(7, 2))},
    )
    duck.register("t", tbl)
    return tbl


def test_a_global_sum_past_the_input_precision_is_decimal_38(duck, overflowing):
    sql = "SELECT sum(p) AS s FROM t"
    ds = bt.sql(sql, t=overflowing)
    out = ds.collect()
    assert out.column("s").to_pylist() == [D.Decimal("300001.02")]
    assert_same(out, duck.sql(sql))
    _assert_sum_type(ds, out, duck.sql(sql))


def test_a_grouped_sum_with_nulls_and_negatives_is_decimal_38(duck, overflowing):
    sql = "SELECT k, sum(p) AS s FROM t GROUP BY k"
    ds = bt.sql(sql, t=overflowing)
    out = ds.collect()
    assert_same(out, duck.sql(sql))
    _assert_sum_type(ds, out, duck.sql(sql))


def test_fused_decimal_sums_are_decimal_38(duck, overflowing):
    """Two sums and a count take the fused accumulator, which carries its own precision."""
    sql = "SELECT k, sum(p) AS s, sum(-p) AS s2, count(p) AS c FROM t GROUP BY k"
    ds = bt.sql(sql, t=overflowing)
    out = ds.collect()
    assert_same(out, duck.sql(sql))
    _assert_sum_type(ds, out, duck.sql(sql))
    _assert_sum_type(ds, out, duck.sql(sql), column="s2")


@pytest.mark.parametrize("where", ["k > 99", "p IS NULL AND k = 1"])
def test_a_sum_of_no_rows_is_decimal_38(duck, overflowing, where):
    """An empty input (or one whose only rows are null) still declares and returns (38, 2)."""
    for sql in (
        f"SELECT sum(p) AS s FROM t WHERE {where}",
        f"SELECT k, sum(p) AS s FROM t WHERE {where} GROUP BY k",
    ):
        ds = bt.sql(sql, t=overflowing)
        out = ds.collect()
        assert_same(out, duck.sql(sql))
        _assert_sum_type(ds, out, duck.sql(sql))


@pytest.fixture
def sharded(duck):
    """More rows than `MIN_ROWS_TO_SHARD` (65,536), so partials merge across workers."""
    n = 200_000
    p = pa.array(
        [
            None if i % 1009 == 0 else D.Decimal(f"{(i % 99_999) - 50_000}.{i % 100:02d}")
            for i in range(n)
        ],
        pa.decimal128(7, 2),
    )
    tbl = pa.table({"k": [i % 7 for i in range(n)], "p": p})
    duck.register("big", tbl)
    return tbl


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT k, sum(p) AS s FROM big GROUP BY k",
        "SELECT k, sum(p) AS s, max(p) AS mx, count(*) AS c FROM big GROUP BY k",
        "SELECT sum(p) AS s FROM big",
    ],
)
def test_a_sharded_decimal_sum_is_decimal_38(duck, sharded, sql):
    ds = bt.sql(sql, big=sharded)
    out = ds.collect()
    assert_same(out, duck.sql(sql))
    _assert_sum_type(ds, out, duck.sql(sql))


def test_a_streamed_decimal_sum_is_decimal_38(duck, sharded):
    sql = "SELECT k, sum(p) AS s FROM big GROUP BY k"
    ds = bt.sql(sql, big=sharded)
    batches = list(ds.iter_batches())
    assert batches, "the aggregate produced no batches"
    out = pa.Table.from_batches(batches)
    assert_same(out, duck.sql(sql))
    _assert_sum_type(ds, out, duck.sql(sql))
