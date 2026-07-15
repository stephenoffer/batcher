"""Wave-2 consolidation: four already-diagnosed cross-area fixes pinned vs DuckDB.

Each defect states its input once, runs it through Batcher and through DuckDB, and asserts
they agree (or that both raise). Every test here fails on the pre-fix engine:

* vector-distance list ops (`cosine_similarity`/`cosine_distance`/`dot`/`l2_distance`)
  must ERROR on a dimension mismatch, not silently truncate to the shorter vector;
* `array_agg`/`list_agg` must KEEP null elements (SQL semantics), not drop them;
* `avg`/`mean` over a `Decimal128` column must return a double, not raise "unsupported";
* set operations (UNION/INTERSECT/EXCEPT) of `int64` and `float64` must coerce both branches
  to double and return a result, not error on the type mismatch.
"""

from __future__ import annotations

from decimal import Decimal

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


# --- Bug 1: vector-distance ops error on a dimension mismatch (like DuckDB) -----------
@pytest.mark.parametrize(
    ("method", "duck_fn"),
    [
        ("cosine_similarity", "list_cosine_similarity"),
        ("cosine_distance", "list_cosine_similarity"),
        ("dot", "list_dot_product"),
        ("l2_distance", "list_distance"),
    ],
)
def test_vector_distance_dimension_mismatch_errors(duck, method, duck_fn):
    t = pa.table(
        {
            "a": pa.array([[1.0, 2.0]], pa.list_(pa.float64())),
            "b": pa.array([[1.0, 2.0, 3.0]], pa.list_(pa.float64())),
        }
    )
    # DuckDB raises "list dimensions must be equal" on a mismatch; Batcher must too. The
    # error crosses the FFI as an ordinary catchable exception (never a panic), and its
    # concrete class is an engine detail — the contract under test is only that a mismatch
    # is refused, not silently truncated, so a blind `Exception` is the right assertion.
    duck.register("t", t)
    with pytest.raises(Exception):  # noqa: B017 - contract is "refuses", not a specific class
        duck.sql(f"SELECT {duck_fn}(a, b) AS r FROM t").fetchall()
    # Batcher must likewise raise, not silently truncate to a bogus ~1.0 similarity.
    expr = getattr(col("a").list, method)(col("b"))
    with pytest.raises(Exception):  # noqa: B017 - contract is "refuses", not a specific class
        bt.from_arrow(t).select(r=expr).collect()


def test_vector_distance_equal_length_still_matches(duck):
    from conftest import assert_same

    t = pa.table(
        {
            "a": pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float64())),
            "b": pa.array([[2.0, 4.0], [1.0, 0.0]], pa.list_(pa.float64())),
        }
    )
    duck.register("t", t)
    got = (
        bt.from_arrow(t)
        .select(
            d=col("a").list.dot(col("b")),
            l2=col("a").list.l2_distance(col("b")),
        )
        .collect()
    )
    assert_same(got, duck.sql("SELECT list_dot_product(a, b) d, list_distance(a, b) l2 FROM t"))


# --- Bug 2: array_agg / list_agg keeps NULL elements ----------------------------------
def test_array_agg_keeps_nulls(duck):
    from conftest import assert_same

    t = pa.table(
        {
            "g": pa.array([1, 1, 1, 1, 2], pa.int64()),
            "v": pa.array([3, None, 1, 3, None], pa.int64()),
        }
    )
    duck.register("t", t)
    # The collected list's length must include the NULLs (group 1 → 4, group 2 → 1);
    # dropping them (the bug) gave 3 and 0. Length is order-independent, so it pins the
    # null preservation without depending on array_agg's unspecified element order.
    got = (
        bt.from_arrow(t)
        .group_by("g")
        .agg(a=col("v").array_agg())
        .select(g=col("g"), n=col("a").list.len())
        .collect()
    )
    assert_same(got, duck.sql("SELECT g, len(array_agg(v)) AS n FROM t GROUP BY g"))


# --- Bug 3: avg/mean over a Decimal128 column returns a double -------------------------
def test_mean_over_decimal_returns_double(duck):
    from conftest import assert_same

    d = pa.array(
        [Decimal("1.50"), Decimal("2.50"), Decimal("3.50"), Decimal("4.50")],
        pa.decimal128(10, 2),
    )
    t = pa.table({"g": pa.array([1, 1, 2, 2], pa.int64()), "d": d})
    duck.register("t", t)
    got = bt.from_arrow(t).group_by("g").agg(m=col("d").mean()).collect()
    assert_same(got, duck.sql("SELECT g, avg(d) AS m FROM t GROUP BY g"))

    # Global (no GROUP BY) mean too.
    got_g = bt.from_arrow(t).agg(m=col("d").mean()).collect()
    assert_same(got_g, duck.sql("SELECT avg(d) AS m FROM t"))


# --- Bug 4: set ops of int64 ∪ float64 coerce to double -------------------------------
@pytest.mark.parametrize("distinct", [True, False])
def test_union_int64_float64_coerces_to_double(duck, distinct):
    from conftest import assert_same

    a = pa.table({"x": pa.array([1, 2, 3], pa.int64())})
    b = pa.table({"x": pa.array([2.5, 3.0], pa.float64())})
    duck.register("a", a)
    duck.register("b", b)
    got = bt.from_arrow(a).union(bt.from_arrow(b), distinct=distinct).collect()
    kw = "" if distinct else " ALL"
    assert_same(got, duck.sql(f"SELECT x FROM a UNION{kw} SELECT x FROM b"))


@pytest.mark.parametrize("op", ["intersect", "except_"])
def test_intersect_except_int64_float64_coerces(duck, op):
    from conftest import assert_same

    a = pa.table({"x": pa.array([1, 2, 3], pa.int64())})
    b = pa.table({"x": pa.array([2.5, 3.0], pa.float64())})
    duck.register("a", a)
    duck.register("b", b)
    got = getattr(bt.from_arrow(a), op)(bt.from_arrow(b)).collect()
    sql_op = "INTERSECT" if op == "intersect" else "EXCEPT"
    assert_same(got, duck.sql(f"SELECT x FROM a {sql_op} SELECT x FROM b"))
