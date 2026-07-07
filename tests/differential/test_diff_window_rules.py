"""Differential tests (vs DuckDB) for the window_rules optimizer family.

Each redundant-window form is run through the FULL optimizer (via `.collect()`,
which uses DEFAULT_REGISTRY) and checked against DuckDB's canonical form. Importing
the rules module registers the rules so they fire in the pipeline.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.window_rules
from conftest import assert_same


@pytest.fixture
def t(duck):
    # Distinct salaries within each dept → ranking/row_number is tie-free (and thus
    # deterministic) so the multiset comparison is exact.
    tbl = pa.table(
        {
            "dept": ["a", "a", "a", "b", "b", "b"],
            "name": list("uvwxyz"),
            "salary": [100, 300, 200, 150, 250, 270],
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def t_nulls(duck):
    tbl = pa.table(
        {
            "dept": ["a", "a", None, "b", None],
            "salary": [100, 300, 200, 150, 250],
        }
    )
    duck.register("tn", tbl)
    return tbl


@pytest.fixture
def t_empty(duck):
    tbl = pa.table(
        {"dept": pa.array([], pa.string()), "salary": pa.array([], pa.int64())}
    )
    duck.register("te", tbl)
    return tbl


def test_drop_dead_window(duck, t):
    # An extra unused ranking function ("extra") must be eliminated, result unchanged.
    out = (
        bt.from_arrow(t)
        .window(
            partition_by=["dept"],
            order_by=[("salary", True)],
            functions={"rn": "row_number", "extra": "rank"},
        )
        .select("dept", "name", "salary", "rn")
        .collect()
    )
    expected = duck.sql("""
        SELECT dept, name, salary, row_number() OVER w rn
        FROM t WINDOW w AS (PARTITION BY dept ORDER BY salary DESC)
    """)
    assert_same(out, expected)


def test_dedupe_partition_keys(duck, t):
    out = (
        bt.from_arrow(t)
        .window(partition_by=["dept", "dept"], functions={"tot": ("sum", "salary")})
        .collect()
    )
    expected = duck.sql("SELECT *, sum(salary) OVER (PARTITION BY dept) tot FROM t")
    assert_same(out, expected)


def test_dedupe_partition_keys_with_nulls(duck, t_nulls):
    out = (
        bt.from_arrow(t_nulls)
        .window(partition_by=["dept", "dept"], functions={"tot": ("sum", "salary")})
        .collect()
    )
    expected = duck.sql("SELECT *, sum(salary) OVER (PARTITION BY dept) tot FROM tn")
    assert_same(out, expected)


def test_dedupe_order_keys(duck, t):
    # ORDER BY salary ASC, salary DESC → first (ASC) wins; running cumulative sum.
    out = (
        bt.from_arrow(t)
        .window(
            partition_by=["dept"],
            order_by=[("salary", False), ("salary", True)],
            functions={"csum": ("sum", "salary")},
        )
        .collect()
    )
    expected = duck.sql("""
        SELECT *, sum(salary) OVER w csum
        FROM t WINDOW w AS (PARTITION BY dept ORDER BY salary ASC)
    """)
    assert_same(out, expected)


def test_drop_constant_partition_key(duck, t):
    out = (
        bt.from_arrow(t)
        .window(partition_by=[bt.lit(1), "dept"], functions={"tot": ("sum", "salary")})
        .collect()
    )
    expected = duck.sql("SELECT *, sum(salary) OVER (PARTITION BY dept) tot FROM t")
    assert_same(out, expected)


def test_drop_constant_partition_key_all_constant(duck, t):
    # PARTITION BY <const> only → one partition over all rows == OVER ().
    out = (
        bt.from_arrow(t)
        .window(partition_by=[bt.lit(1)], functions={"tot": ("sum", "salary")})
        .collect()
    )
    expected = duck.sql("SELECT *, sum(salary) OVER () tot FROM t")
    assert_same(out, expected)


def test_drop_constant_order_key(duck, t):
    out = (
        bt.from_arrow(t)
        .window(
            partition_by=["dept"],
            order_by=["salary", bt.lit(1)],
            functions={"csum": ("sum", "salary")},
        )
        .collect()
    )
    expected = duck.sql("""
        SELECT *, sum(salary) OVER w csum
        FROM t WINDOW w AS (PARTITION BY dept ORDER BY salary)
    """)
    assert_same(out, expected)


def test_empty_input(duck, t_empty):
    out = (
        bt.from_arrow(t_empty)
        .window(partition_by=["dept", "dept"], functions={"tot": ("sum", "salary")})
        .collect()
    )
    expected = duck.sql("SELECT *, sum(salary) OVER (PARTITION BY dept) tot FROM te")
    assert_same(out, expected)
