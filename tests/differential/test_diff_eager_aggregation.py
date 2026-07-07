"""Eager aggregation correctness vs DuckDB.

Pushing a `min`/`max` partial aggregate below an inner join must produce the
identical result — including when the join fans out (duplicate right keys
replicate left rows), the case where `min`/`max` idempotence is essential. The rule
is cost-gated on `ndv`, so these tests apply it via a context with a learned `ndv`
and execute the rewritten plan against DuckDB.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col
from batcher.api.dataset.frame import Dataset
from batcher.config import active_config
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rules.agg_pushdown import eager_aggregation
from batcher.kyber.stats.estimator import StatsEstimator


def _ctx(ds, ndv):
    est = StatsEstimator(ds._sources, learned={"__column_ndv__": ndv})
    return OptimizerContext(config=active_config(), sources=ds._sources, hub=None, estimator=est)


def _pushed(ds, ndv):
    """Apply eager_aggregation; assert it fired; return the rewritten Dataset."""
    out = eager_aggregation(ds._plan, _ctx(ds, ndv))
    assert out is not None, "expected eager_aggregation to fire"
    return Dataset(out, ds._sources)


def test_max_grouped_by_right_column(duck):
    from conftest import assert_same

    emp = pa.table({"dept_id": [1, 1, 2, 2, 3], "sal": [100, 200, 150, 300, 50]})
    dept = pa.table({"dept_id": [1, 2, 3], "name": ["eng", "sales", "ops"]})
    duck.register("emp", emp)
    duck.register("dept", dept)
    ds = (
        bt.from_arrow(emp)
        .join(bt.from_arrow(dept), on="dept_id")
        .group_by("name")
        .agg(top=col("sal").max())
    )
    expected = duck.sql(
        "SELECT name, max(sal) AS top FROM emp JOIN dept USING (dept_id) GROUP BY name"
    )
    assert_same(ds.collect(), expected)  # original
    assert_same(_pushed(ds, {"dept_id": 3.0}).collect(), expected)  # rewritten


def test_min_max_under_fan_out(duck):
    """The dimension has duplicate keys → left rows replicate; min/max stay correct
    (a sum would be multiplied — which is why the rule excludes it)."""
    from conftest import assert_same

    fact = pa.table({"k": [1, 1, 2, 3], "v": [10, 40, 20, 5]})
    dim = pa.table({"k": [1, 1, 2, 2, 3], "g": ["a", "a", "b", "b", "c"]})  # fan-out on k
    duck.register("fact", fact)
    duck.register("dim", dim)
    ds = (
        bt.from_arrow(fact)
        .join(bt.from_arrow(dim), on="k")
        .group_by("g")
        .agg(lo=col("v").min(), hi=col("v").max())
    )
    expected = duck.sql(
        "SELECT g, min(v) AS lo, max(v) AS hi FROM fact JOIN dim USING (k) GROUP BY g"
    )
    assert_same(ds.collect(), expected)
    assert_same(_pushed(ds, {"k": 3.0}).collect(), expected)


def test_grouped_by_left_column(duck):
    from conftest import assert_same

    fact = pa.table(
        {
            "k": [1, 1, 2, 2, 1, 2],
            "cat": ["x", "x", "y", "y", "x", "y"],
            "v": [3, 9, 4, 1, 5, 2],
        }
    )
    dim = pa.table({"k": [1, 2], "d": [100, 200]})
    duck.register("f2", fact)
    duck.register("d2", dim)
    ds = bt.from_arrow(fact).join(bt.from_arrow(dim), on="k").group_by("cat").agg(hi=col("v").max())
    expected = duck.sql("SELECT cat, max(v) AS hi FROM f2 JOIN d2 USING (k) GROUP BY cat")
    assert_same(ds.collect(), expected)
    # 6 rows, ndv(cat)*ndv(k) = 4 estimated groups < 6 → the estimator sees a reduction.
    assert_same(_pushed(ds, {"k": 2.0, "cat": 2.0}).collect(), expected)


def _pushed_measures(ds, ndv):
    """Apply pre_aggregate_join_measures; assert it fired; return the rewritten Dataset."""
    from batcher.kyber.rules.agg_pushdown import pre_aggregate_join_measures

    out = pre_aggregate_join_measures(ds._plan, _ctx(ds, ndv))
    assert out is not None, "expected pre_aggregate_join_measures to fire"
    return Dataset(out, ds._sources)


def test_measure_pushdown_left_join_count(duck):
    """Pre-aggregating the right (measure) side below a LEFT join, grouped by the left
    key, must match DuckDB — the TPC-H Q13 shape, including customers with no orders
    (an unmatched left row whose COUNT must be 0, not NULL)."""
    from conftest import assert_same

    cust = pa.table({"ck": [1, 2, 3, 4], "name": ["a", "b", "c", "d"]})
    orders = pa.table({"ok": [10, 11, 12, 13, 14], "ck": [1, 1, 1, 3, 3]})
    duck.register("cust", cust)
    duck.register("ord", orders)
    # cust 2 and 4 have no orders → COUNT(ok) must be 0 for them.
    ds = (
        bt.from_arrow(cust)
        .join(bt.from_arrow(orders), on="ck", how="left")
        .group_by("ck")
        .agg(cnt=col("ok").count())
    )
    expected = duck.sql(
        "SELECT c.ck, count(o.ok) AS cnt FROM cust c LEFT JOIN ord o USING (ck) GROUP BY c.ck"
    )
    assert_same(ds.collect(), expected)  # original
    assert_same(_pushed_measures(ds, {"ck": 3.0}).collect(), expected)  # rewritten


def test_measure_pushdown_inner_sum_min_fan_out(duck):
    """Inner join, multiple decomposable measures (SUM, MIN, MAX) on the right side with
    duplicate left keys (fan-out) — the merge of partials must equal the direct result."""
    from conftest import assert_same

    left = pa.table({"k": [1, 1, 2, 3], "g": ["p", "p", "q", "q"]})
    right = pa.table({"k": [1, 1, 2, 2, 2], "v": [10, 20, 5, 7, 9]})
    duck.register("lf", left)
    duck.register("rt", right)
    ds = (
        bt.from_arrow(left)
        .join(bt.from_arrow(right), on="k", how="inner")
        .group_by("g")
        .agg(s=col("v").sum(), lo=col("v").min(), hi=col("v").max())
    )
    expected = duck.sql(
        "SELECT g, sum(v) AS s, min(v) AS lo, max(v) AS hi FROM lf JOIN rt USING (k) GROUP BY g"
    )
    assert_same(ds.collect(), expected)
    assert_same(_pushed_measures(ds, {"k": 2.0}).collect(), expected)


def test_measure_pushdown_left_side_computed_expr(duck):
    """Measure on the *left* side via a computed expression, grouped by a *right* column
    (the operator-mix join→agg shape): pre-aggregating the left side by the join key and
    merging must match DuckDB, including duplicate join keys (fan-out)."""
    from conftest import assert_same

    line = pa.table(
        {
            "ok": [1, 1, 2, 2, 3],
            "price": [100.0, 200.0, 50.0, 70.0, 30.0],
            "disc": [0.1, 0.0, 0.2, 0.0, 0.5],
        }
    )
    order = pa.table({"ok": [1, 2, 3], "prio": ["hi", "lo", "hi"]})
    duck.register("li", line)
    duck.register("ords", order)
    ds = (
        bt.from_arrow(line)
        .join(bt.from_arrow(order), on="ok", how="inner")
        .group_by("prio")
        .agg(rev=(col("price") * (1.0 - col("disc"))).sum())
    )
    expected = duck.sql(
        "SELECT prio, sum(price * (1 - disc)) AS rev FROM li JOIN ords USING (ok) GROUP BY prio"
    )
    assert_same(ds.collect(), expected)
    assert_same(_pushed_measures(ds, {"ok": 3.0}).collect(), expected)
