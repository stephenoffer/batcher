"""Differential tests (vs DuckDB) for the `setops` rules.

Importing the module registers its rules into `DEFAULT_REGISTRY`, so every query below
runs through the FULL optimizer with the set-op rewrites active. Cases carry duplicate
rows and NULLs across branches — the exact places a bag-vs-set or NULL-comparison
mistake would surface.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
import batcher.kyber.rules.extra.setops as _setops  # noqa: F401  (registers rules into DEFAULT_REGISTRY)
from batcher import col


def _reg(duck, **tables):
    for name, tbl in tables.items():
        duck.register(name, tbl)


def test_flatten_union_all(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 2, 2]})
    b = pa.table({"x": [2, 3, None]})
    c = pa.table({"x": [3, 4, None]})
    _reg(duck, a=a, b=b, c=c)
    out = bt.from_arrow(a).union(bt.from_arrow(b)).union(bt.from_arrow(c)).collect()
    sql = "SELECT * FROM a UNION ALL SELECT * FROM b UNION ALL SELECT * FROM c"
    assert_same(out, duck.sql(sql))


def test_flatten_union_distinct(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 2, 2, None]})
    b = pa.table({"x": [2, 3, None]})
    c = pa.table({"x": [3, 4]})
    _reg(duck, a=a, b=b, c=c)
    out = (
        bt.from_arrow(a)
        .union(bt.from_arrow(b), distinct=True)
        .union(bt.from_arrow(c), distinct=True)
        .collect()
    )
    assert_same(out, duck.sql("SELECT * FROM a UNION SELECT * FROM b UNION SELECT * FROM c"))


def test_singleton_union_all(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 1, 2, None]})
    _reg(duck, a=a)
    assert_same(bt.from_arrow(a).union().collect(), duck.sql("SELECT * FROM a"))


def test_singleton_union_distinct(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 1, 2, None, None]})
    _reg(duck, a=a)
    out = bt.from_arrow(a).union(distinct=True).collect()
    assert_same(out, duck.sql("SELECT DISTINCT * FROM a"))


def test_prune_empty_branch(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 2, 2, None]})
    b = pa.table({"x": [7, 8]})
    _reg(duck, a=a)
    # b.limit(0) is a provably-empty branch, so the union collapses to `a`.
    out = bt.from_arrow(a).union(bt.from_arrow(b).limit(0)).collect()
    assert_same(out, duck.sql("SELECT * FROM a"))


def test_dedup_distinct_branches(duck):
    from batcher.api.dataset import Dataset
    from batcher.plan.logical import Union, remap_sources
    from conftest import assert_same

    a = pa.table({"x": [1, 2, 2, None]})
    b = pa.table({"x": [2, 3, None]})
    _reg(duck, a=a, b=b)
    da, db = bt.from_arrow(a), bt.from_arrow(b)
    sources = list(da._sources)
    bp = remap_sources(db._plan, len(sources))
    sources += db._sources
    # `a UNION b UNION a` — the repeated identical branch is dropped by the rule.
    plan = Union((da._plan, bp, da._plan), distinct=True)
    out = Dataset(plan, sources).collect()
    assert_same(out, duck.sql("SELECT * FROM a UNION SELECT * FROM b"))


def test_drop_branch_distinct(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 1, 2, None]})
    b = pa.table({"x": [2, 3, None]})
    _reg(duck, a=a, b=b)
    out = bt.from_arrow(a).distinct().union(bt.from_arrow(b), distinct=True).collect()
    assert_same(out, duck.sql("SELECT * FROM a UNION SELECT * FROM b"))


def test_eliminate_sort_in_distinct_branch(duck):
    from conftest import assert_same

    a = pa.table({"x": [3, 1, 2, 2, None]})
    b = pa.table({"x": [2, 4, None]})
    _reg(duck, a=a, b=b)
    out = bt.from_arrow(a).sort("x").union(bt.from_arrow(b), distinct=True).collect()
    assert_same(out, duck.sql("SELECT * FROM a UNION SELECT * FROM b"))


def test_push_project_through_union(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 2, 2], "y": [10, 20, 20]})
    b = pa.table({"x": [2, 3, None], "y": [20, 30, None]})
    _reg(duck, a=a, b=b)
    out = bt.from_arrow(a).union(bt.from_arrow(b)).select("x").collect()
    assert_same(out, duck.sql("SELECT x FROM (SELECT * FROM a UNION ALL SELECT * FROM b)"))


def test_fold_distinct_union_all(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 2, 2, None]})
    b = pa.table({"x": [2, 3, None]})
    _reg(duck, a=a, b=b)
    out = bt.from_arrow(a).union(bt.from_arrow(b)).distinct().collect()
    assert_same(out, duck.sql("SELECT * FROM a UNION SELECT * FROM b"))


def test_push_filter_through_distinct(duck):
    from conftest import assert_same

    # NULLs in x exercise three-valued logic: distinct keeps the null row, the filter
    # (x > 1) drops it — the result must match whichever order the two run.
    a = pa.table({"x": [1, 1, 2, 3, 3, None]})
    _reg(duck, a=a)
    out = bt.from_arrow(a).distinct().filter(col("x") > 1).collect()
    assert_same(out, duck.sql("SELECT * FROM (SELECT DISTINCT * FROM a) WHERE x > 1"))


def test_eliminate_sort_before_distinct(duck):
    from conftest import assert_same

    a = pa.table({"x": [3, 1, 2, 2, None, None]})
    _reg(duck, a=a)
    out = bt.from_arrow(a).sort("x").distinct().collect()
    assert_same(out, duck.sql("SELECT DISTINCT * FROM a"))


def test_prune_distinct_of_empty(duck):
    from conftest import assert_same

    a = pa.table({"x": [1, 2, 2, None]})
    _reg(duck, a=a)
    out = bt.from_arrow(a).limit(0).distinct().collect()
    assert_same(out, duck.sql("SELECT * FROM a WHERE false"))
