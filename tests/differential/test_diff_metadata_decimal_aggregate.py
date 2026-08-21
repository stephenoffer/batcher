"""The metadata shortcut must answer what an execution would — decimals included.

A keyless aggregate over a relation whose statistics are exact is answered from those
statistics without running the query (`api/terminal/metadata_answer/aggregate.py`). That is
a large win and a large risk: the shortcut *replaces* execution rather than estimating it,
so anything it gets wrong is a wrong query result with no error and no way for the caller to
tell. On a `decimal` column it got two things wrong, and the grouped path — which does
execute — was right about both, so the same expression answered differently depending on
whether a `GROUP BY` was present.

* **`avg` was rounded to the column's scale.** Arrow's decimal mean kernel returns a decimal
  at the input's own scale, so the average of `1.25`, `22.50` and `3.00` came back as `8.92`
  where DuckDB, the engine, and this engine's own grouped path all answer
  `8.9166666...`. It also returned a `Decimal` from a helper whose contract is
  `float | None`, which is how the wrong *type* then reached the result schema.
* **`min`/`max` narrowed the decimal.** The answer table was built from Python values with
  no schema, so pyarrow typed each column from its single value: `max` over a
  `decimal(10,2)` column came back as `decimal(4,2)`, the narrowest decimal holding `22.50`.
  The type therefore depended on the *data* — one more row could widen it — and everything
  downstream reads it.

Both are checked against DuckDB, and both are checked *against the engine's own grouped
answer*, because that second comparison is the one that fails if the shortcut and the
execution ever diverge again for a reason DuckDB happens to share.
"""

from __future__ import annotations

from decimal import Decimal

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


def _tbl() -> pa.Table:
    """Two groups, so the grouped answer is a real second opinion rather than the same row."""
    return pa.table(
        {
            "g": pa.array(["a", "a", "b", "b"]),
            "m": pa.array(
                [Decimal("1.25"), Decimal("22.50"), Decimal("3.00"), None],
                pa.decimal128(10, 2),
            ),
        }
    )


def test_a_global_decimal_average_is_the_double_duckdb_returns(duck):
    duck.register("t", _tbl())
    out = bt.from_arrow(_tbl()).agg(r=col("m").mean()).collect()
    assert_same(out, duck.sql("SELECT avg(m) AS r FROM t"))
    assert out.schema.field("r").type == pa.float64()
    # And it is not the scale-rounded 8.92 the decimal kernel gives.
    assert out.to_pydict()["r"][0] == pytest.approx(26.75 / 3, rel=1e-15)


def test_a_global_decimal_min_max_keeps_the_columns_own_precision(duck):
    duck.register("t", _tbl())
    out = bt.from_arrow(_tbl()).agg(lo=col("m").min(), hi=col("m").max()).collect()
    assert_same(out, duck.sql("SELECT min(m) AS lo, max(m) AS hi FROM t"))
    for name in ("lo", "hi"):
        assert out.schema.field(name).type == pa.decimal128(10, 2), (
            f"{name} narrowed to {out.schema.field(name).type}; the type must come from the "
            "column, not from the one value that happened to be extreme"
        )


def test_the_shortcut_and_an_execution_agree_on_value_and_type():
    """The comparison DuckDB cannot make: metadata answer vs the engine's own grouped one.

    A single-group `GROUP BY` executes rather than taking the shortcut, so the two paths
    answer the same question by different means and must agree exactly.
    """
    one_group = pa.table(
        {
            "g": pa.array(["a", "a", "a"]),
            "m": pa.array(
                [Decimal("1.25"), Decimal("22.50"), Decimal("3.00")], pa.decimal128(10, 2)
            ),
        }
    )
    ds = bt.from_arrow(one_group)
    for agg in (col("m").mean(), col("m").min(), col("m").max(), col("m").sum()):
        shortcut = ds.agg(r=agg).collect()
        executed = ds.group_by("g").agg(r=agg).collect()
        assert shortcut.schema.field("r").type == executed.schema.field("r").type
        assert shortcut.to_pydict()["r"] == executed.to_pydict()["r"]


def test_the_declared_schema_is_what_the_shortcut_returns():
    """`Dataset.schema` is read before the shortcut runs, so the two must not disagree."""
    ds = bt.from_arrow(_tbl())
    for agg, name in ((col("m").mean(), "mean"), (col("m").min(), "min"), (col("m").sum(), "sum")):
        q = ds.agg(r=agg)
        assert q.schema.field("r").type == q.collect().schema.field("r").type, name


def test_an_all_null_decimal_column_still_says_null(duck):
    """The mean divides by the non-null count, so an empty one must not divide by zero."""
    t = pa.table({"m": pa.array([None, None], pa.decimal128(10, 2))})
    duck.register("z", t)
    out = bt.from_arrow(t).agg(r=col("m").mean()).collect()
    assert_same(out, duck.sql("SELECT avg(m) AS r FROM z"))
    assert out.to_pydict()["r"] == [None]
