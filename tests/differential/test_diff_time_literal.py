"""A time-of-day literal works in the DataFrame API, not only in SQL.

``WHERE t > TIME '01:00:00'`` answered correctly through the SQL front-end while
``col("t") > time(1, 0)`` raised ``TypeError: unsupported literal type: time`` — the same
query, over the same engine, working through one surface and not the other. Time columns
read fine and the engine compares them fine; only the Python lowering refused.

It lowers as a cast from ISO text rather than as a tagged literal, because
`bc_expr::Literal` has `Date` and `Timestamp` variants but no `Time` one, and giving it
one would be a two-sided IR change across the FFI for something the engine already
evaluates correctly by this route. Both front-ends therefore emit the *identical* plan,
which is asserted below rather than assumed.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, duck_materialize

pytestmark = pytest.mark.differential

_TIMES = [dt.time(12, 30), dt.time(0, 0, 1, 500000), dt.time(23, 59, 59), None]


@pytest.fixture
def times():
    return pa.table({"tm": pa.array(_TIMES, type=pa.time64("us")), "v": [1, 2, 3, 4]})


@pytest.mark.parametrize(
    ("build", "where"),
    [
        (lambda: bt.col("tm") == dt.time(12, 30), "tm = TIME '12:30:00'"),
        (lambda: bt.col("tm") > dt.time(1, 0), "tm > TIME '01:00:00'"),
        (lambda: bt.col("tm") <= dt.time(12, 30), "tm <= TIME '12:30:00'"),
        (lambda: bt.col("tm") == dt.time(0, 0, 1, 500000), "tm = TIME '00:00:01.5'"),
        (lambda: bt.col("tm").is_null(), "tm IS NULL"),
    ],
)
def test_a_time_comparison_matches_duckdb(duck, times, build, where):
    duck_materialize(duck, "t", times)
    got = bt.from_arrow(times).filter(build()).collect()
    assert_same(got, duck.sql(f"SELECT * FROM t WHERE {where}"))


def test_both_front_ends_lower_a_time_literal_to_the_same_plan(times):
    """The unification this fixes: one engine, one plan, whichever surface wrote it."""
    dataset = bt.from_arrow(times)

    def predicate_of(ir):
        while "predicate" not in ir:
            ir = ir.get("input") or {}
            if not ir:
                return None
        return ir["predicate"]

    from_sql = predicate_of(
        bt.sql("SELECT * FROM t WHERE tm > TIME '01:00:00'", t=dataset)._plan.to_ir()
    )
    from_api = predicate_of(dataset.filter(bt.col("tm") > dt.time(1, 0))._plan.to_ir())
    assert from_sql == from_api


def test_a_time_literal_selects_as_a_value(times):
    got = bt.from_arrow(times).select(r=bt.lit(dt.time(9, 15))).collect()
    assert got.to_pydict()["r"] == [dt.time(9, 15)] * len(_TIMES)


def test_a_timezone_aware_time_is_refused_rather_than_silently_stripped():
    # Arrow's `time64` carries no zone, so an offset here has nowhere to go.
    with pytest.raises(TypeError, match="cannot carry a timezone"):
        bt.lit(dt.time(1, 0, tzinfo=dt.UTC)).to_ir()
