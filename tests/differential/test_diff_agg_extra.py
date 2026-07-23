"""`agg_extra` rewrites preserve results vs DuckDB.

Every rule that can affect a query result runs end to end through the full optimizer
(``.collect()`` uses ``DEFAULT_REGISTRY``) and is compared against DuckDB. Importing
``agg_extra`` registers the rules. Coverage includes NULL groups, duplicate rows and
empty input — the edges the rules reason about.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher.api.dataset.frame import Dataset
from batcher.kyber.rules.extra import agg_extra as _agg_extra  # noqa: F401  (registers the rules)
from batcher.plan.logical import Aggregate, Projection

# g has a NULL group, duplicate rows, and an all-NULL x within one group.
_DATA = pa.table(
    {
        "g": [1, 1, 2, 2, 2, None, None, 3],
        "x": [10, 10, 20, None, 30, 5, 5, None],
    }
)
_EMPTY = pa.table({"g": pa.array([], pa.int64()), "x": pa.array([], pa.int64())})


def _reg(duck, table=_DATA, name="t"):
    duck.register(name, table)


# --- dedupe_group_keys -------------------------------------------------------


def test_dedupe_group_keys(duck):
    _reg(duck)
    out = bt.from_arrow(_DATA).group_by(a=col("g"), b=col("g")).agg(s=col("x").sum()).collect()
    expected = duck.sql("SELECT g AS a, g AS b, sum(x) AS s FROM t GROUP BY g")
    assert_same(out, expected)


# --- drop_constant_group_key -------------------------------------------------


def test_drop_constant_group_key(duck):
    _reg(duck)
    out = bt.from_arrow(_DATA).group_by("g", c=bt.lit(5)).agg(s=col("x").sum()).collect()
    expected = duck.sql("SELECT g, 5 AS c, sum(x) AS s FROM t GROUP BY g")
    assert_same(out, expected)


# --- redundant_aggregate_of_group_key ---------------------------------------


def test_min_max_of_group_key_with_null_group(duck):
    _reg(duck)
    out = (
        bt.from_arrow(_DATA)
        .group_by("g")
        .agg(lo=col("g").min(), hi=col("g").max(), s=col("x").sum())
        .collect()
    )
    expected = duck.sql("SELECT g, min(g) AS lo, max(g) AS hi, sum(x) AS s FROM t GROUP BY g")
    assert_same(out, expected)


# --- count_distinct_of_group_key --------------------------------------------


def test_count_distinct_of_group_key_null_group(duck):
    _reg(duck)
    # The NULL group must yield COUNT(DISTINCT g) = 0.
    out = bt.from_arrow(_DATA).group_by("g").agg(n=col("g").n_unique()).collect()
    expected = duck.sql("SELECT g, count(DISTINCT g) AS n FROM t GROUP BY g")
    assert_same(out, expected)


# --- count_of_group_key ------------------------------------------------------


def test_count_of_group_key_null_group(duck):
    _reg(duck)
    # NULL group -> COUNT(g) = 0; others -> the group's row count.
    out = bt.from_arrow(_DATA).group_by("g").agg(c=col("g").count(), n=bt.count()).collect()
    expected = duck.sql("SELECT g, count(g) AS c, count(*) AS n FROM t GROUP BY g")
    assert_same(out, expected)


# --- count_constant_to_count_star -------------------------------------------


def test_count_constant_grouped(duck):
    _reg(duck)
    out = bt.from_arrow(_DATA).group_by("g").agg(c=bt.lit(1).count()).collect()
    expected = duck.sql("SELECT g, count(1) AS c FROM t GROUP BY g")
    assert_same(out, expected)


def test_count_constant_global_empty(duck):
    _reg(duck, _EMPTY)
    # Global COUNT over 0 rows is 0 (the 1-row global-aggregate case).
    out = bt.from_arrow(_EMPTY).group_by().agg(c=bt.lit(1).count()).collect()
    expected = duck.sql("SELECT count(1) AS c FROM t")
    assert_same(out, expected)


# --- sum_constant_to_count ---------------------------------------------------


def test_sum_constant_grouped(duck):
    _reg(duck)
    out = bt.from_arrow(_DATA).group_by("g").agg(s=bt.lit(2).sum()).collect()
    expected = duck.sql("SELECT g, sum(2) AS s FROM t GROUP BY g")
    assert_same(out, expected)


# --- fold_constant_grouped_aggregate ----------------------------------------


def test_fold_constant_grouped(duck):
    _reg(duck)
    out = (
        bt.from_arrow(_DATA)
        .group_by("g")
        .agg(mn=bt.lit(7).min(), mx=bt.lit(7).max(), n=bt.lit(3).n_unique())
        .collect()
    )
    expected = duck.sql(
        "SELECT g, min(7) AS mn, max(7) AS mx, count(DISTINCT 3) AS n FROM t GROUP BY g"
    )
    assert_same(out, expected)


# --- drop_distinct_before_agg ------------------------------------------------


def test_drop_distinct_before_agg(duck):
    _reg(duck)
    out = (
        bt.from_arrow(_DATA)
        .distinct()
        .group_by("g")
        .agg(lo=col("x").min(), hi=col("x").max(), nd=col("x").n_unique())
        .collect()
    )
    expected = duck.sql(
        "SELECT g, min(x) AS lo, max(x) AS hi, count(DISTINCT x) AS nd "
        "FROM (SELECT DISTINCT * FROM t) GROUP BY g"
    )
    assert_same(out, expected)


# --- aggregate_without_aggs_to_distinct (manual plan; API needs >=1 agg) -----


def test_group_only_aggregate_is_distinct(duck):
    _reg(duck)
    base = bt.from_arrow(_DATA)
    agg = Aggregate(base._plan, (Projection("g", col("g")), Projection("x", col("x"))), ())
    out = Dataset(agg, base._sources).collect()
    expected = duck.sql("SELECT DISTINCT g, x FROM t")
    assert_same(out, expected)


# --- deduplicate_aggregate_exprs --------------------------------------------


def test_deduplicate_aggregate_exprs(duck):
    _reg(duck)
    out = (
        bt.from_arrow(_DATA)
        .group_by("g")
        .agg(a=col("x").sum(), b=col("x").sum(), c=col("x").min())
        .collect()
    )
    expected = duck.sql("SELECT g, sum(x) AS a, sum(x) AS b, min(x) AS c FROM t GROUP BY g")
    assert_same(out, expected)
