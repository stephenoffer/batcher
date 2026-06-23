"""Transitive join-predicate inference vs DuckDB.

Mirroring a constant key constraint across an equi-join must not change the result
— the join already enforces key equality, so the inferred predicate is redundant
on the result while enabling earlier pruning. DuckDB is the oracle.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt


def _tables(duck):
    fact = pa.table({"dept_id": [10, 20, 10, 30, 20], "amt": [1, 2, 3, 4, 5]})
    dim = pa.table({"dept_id": [10, 20, 30], "region": ["EU", "US", "EU"]})
    duck.register("fact", fact)
    duck.register("dim", dim)
    return bt.from_arrow(fact), bt.from_arrow(dim)


def test_eq_constraint_on_dim_key(duck):
    from conftest import assert_same

    fact, dim = _tables(duck)
    out = fact.join(dim.filter(bt.col("dept_id") == 10), on="dept_id").collect()
    expected = duck.sql(
        "SELECT * FROM fact JOIN (SELECT * FROM dim WHERE dept_id = 10) d USING (dept_id)"
    )
    assert_same(out, expected)


def test_range_constraint_on_fact_key(duck):
    from conftest import assert_same

    fact, dim = _tables(duck)
    out = fact.filter(bt.col("dept_id") >= 20).join(dim, on="dept_id").collect()
    expected = duck.sql(
        "SELECT * FROM (SELECT * FROM fact WHERE dept_id >= 20) f JOIN dim USING (dept_id)"
    )
    assert_same(out, expected)


def test_left_join_constraint_not_transferred(duck):
    """A left join must keep its unmatched left rows — inference must not fire."""
    from conftest import assert_same

    fact, dim = _tables(duck)
    out = fact.join(dim.filter(bt.col("dept_id") == 10), on="dept_id", how="left").collect()
    expected = duck.sql(
        "SELECT * FROM fact LEFT JOIN (SELECT * FROM dim WHERE dept_id = 10) d USING (dept_id)"
    )
    assert_same(out, expected)
