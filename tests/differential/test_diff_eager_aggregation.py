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
