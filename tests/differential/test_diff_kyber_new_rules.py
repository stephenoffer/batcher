"""Correctness vs DuckDB for the new Kyber rules (B-batch) — each rewrite must be
semantics-preserving. Plan-shape assertions live in tests/unit/test_kyber_new_rules.py."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _t(duck, name="t"):
    t = pa.table(
        {
            "g": pa.array([1, 1, 2, 2, 3, 3], pa.int64()),
            "v": pa.array([10, 20, 30, 40, 50, 60], pa.int64()),
        }
    )
    duck.register(name, t)
    return bt.from_arrow(t)


def test_collapse_adjacent_windows(duck):
    from conftest import assert_same

    ds = _t(duck, "w1")
    out = (
        ds.window(partition_by=["g"], order_by=["v"], functions={"r": "row_number"})
        .window(partition_by=["g"], order_by=["v"], functions={"mx": ("max", "v")})
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT g, v, row_number() OVER (PARTITION BY g ORDER BY v) r, "
            "max(v) OVER (PARTITION BY g ORDER BY v) mx FROM w1"
        ),
    )


def test_push_filter_through_window(duck):
    from conftest import assert_same

    ds = _t(duck, "w2")
    out = (
        ds.window(partition_by=["g"], order_by=["v"], functions={"r": "row_number"})
        .filter(col("g") == 2)
        .select("g", "v", "r")
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT g, v, r FROM ("
            "  SELECT g, v, row_number() OVER (PARTITION BY g ORDER BY v) r FROM w2"
            ") WHERE g = 2"
        ),
    )


def test_push_filter_through_window_mixed_predicate(duck):
    from conftest import assert_same

    ds = _t(duck, "w3")
    # g (partition) is pushable; v (non-partition) stays above — result must be unchanged.
    out = (
        ds.window(partition_by=["g"], order_by=["v"], functions={"r": "row_number"})
        .filter((col("g") == 2) & (col("v") > 30))
        .select("g", "v", "r")
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT g, v, r FROM ("
            "  SELECT g, v, row_number() OVER (PARTITION BY g ORDER BY v) r FROM w3"
            ") WHERE g = 2 AND v > 30"
        ),
    )


def test_distinct_over_scalar_aggregate(duck):
    from conftest import assert_same

    ds = _t(duck, "w4")
    out = ds.agg(s=col("v").sum()).distinct().collect()
    assert_same(out, duck.sql("SELECT DISTINCT (SELECT sum(v) FROM w4) AS s"))


def test_rename_into_aggregate(duck):
    from conftest import assert_same

    ds = _t(duck, "w5")
    out = ds.rename({"v": "val"}).group_by("g").agg(s=col("val").sum()).collect()
    assert_same(out, duck.sql("SELECT g, sum(v) s FROM w5 GROUP BY g"))


def test_nested_cast(duck):
    from conftest import assert_same

    ds = _t(duck, "w6")
    out = ds.with_columns(c=col("g").cast("int64").cast("int64")).select("g", "c").collect()
    assert_same(out, duck.sql("SELECT g, CAST(CAST(g AS BIGINT) AS BIGINT) c FROM w6"))


def test_or_to_in_range(duck):
    from conftest import assert_same

    ds = _t(duck, "w7")
    # In-range and absent members; result must match plain IN semantics.
    out = ds.filter(col("v").is_in([20, 50, 999])).select("g", "v").collect()
    assert_same(out, duck.sql("SELECT g, v FROM w7 WHERE v IN (20, 50, 999)"))


def test_join_to_semijoin_with_fanout(duck):
    from conftest import assert_same

    emp = pa.table(
        {"id": pa.array([1, 2, 3, 4], pa.int64()), "dept": pa.array([10, 20, 10, 30], pa.int64())}
    )
    # dept 10 appears twice → inner join fans out; semi + distinct must still match.
    dept = pa.table({"dept": pa.array([10, 10, 20], pa.int64()), "name": ["a", "a2", "b"]})
    duck.register("emp_s", emp)
    duck.register("dept_s", dept)
    out = (
        bt.from_arrow(emp)
        .join(bt.from_arrow(dept), on="dept")
        .select("id", "dept")
        .distinct()
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT DISTINCT emp_s.id, emp_s.dept "
            "FROM emp_s JOIN dept_s ON emp_s.dept = dept_s.dept"
        ),
    )


def test_transitive_predicate_inference(duck):
    from conftest import assert_same

    a = pa.table(
        {"k": pa.array([1, 2, 3, 4], pa.int64()), "av": pa.array([10, 20, 30, 40], pa.int64())}
    )
    b = pa.table(
        {"k": pa.array([1, 2, 3, 4], pa.int64()), "bv": pa.array([1, 2, 3, 4], pa.int64())}
    )
    c = pa.table(
        {"k": pa.array([1, 2, 3, 4], pa.int64()), "cv": pa.array([5, 6, 7, 8], pa.int64())}
    )
    duck.register("ta", a)
    duck.register("tb", b)
    duck.register("tc", c)
    out = (
        bt.from_arrow(a)
        .join(bt.from_arrow(b), on="k")
        .join(bt.from_arrow(c), on="k")
        .filter(col("k") > 2)
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT ta.k, av, bv, cv FROM ta JOIN tb ON ta.k=tb.k JOIN tc ON ta.k=tc.k WHERE ta.k>2"
        ),
    )


def test_pre_aggregation_through_join(duck):
    from conftest import assert_same

    fact = pa.table(
        {
            "k": pa.array([1, 1, 1, 2, 2, 3], pa.int64()),
            "amt": pa.array([10, 20, 30, 40, 50, 60], pa.int64()),
        }
    )
    dim = pa.table({"k": pa.array([1, 2, 3], pa.int64()), "region": ["e", "w", "s"]})
    duck.register("fct", fact)
    duck.register("dm", dim)
    dim_u = bt.from_arrow(dim).group_by("k").agg(region=col("region").max())
    out = (
        bt.from_arrow(fact)
        .join(dim_u, on="k")
        .group_by("region")
        .agg(s=col("amt").sum(), c=col("amt").count())
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT region, SUM(amt) s, COUNT(amt) c FROM fct "
            "JOIN (SELECT k, max(region) region FROM dm GROUP BY k) d ON fct.k=d.k GROUP BY region"
        ),
    )
