"""Differential tests: spilled value-list aggregates on signed-zero float keys/values.

The bounded out-of-core aggregate paths (`median`/`quantile`/`n_unique`/`mode`/`histogram`)
flatten `(group_keys.., value)` and sort them out of core, then detect group/value
boundaries by the arrow row encoding. That encoding maps `-0.0` and `0.0` to *different*
bytes — but SQL `GROUP BY` (and the in-memory `assign_groups`) fold them to one key. So a
GROUP BY on a float key holding both `-0.0` and `0.0` used to return **two groups where
DuckDB and the in-memory path return one** — a silent wrong answer that only appears once
the aggregate spills. Likewise `n_unique` over a float value column over-counted the two
zeros as two distinct values under spill (its in-memory path dedups through `assign_groups`,
which canonicalizes). These pin both against DuckDB *and* against the non-spilling result.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")


@pytest.fixture
def signed_zero_table() -> pa.Table:
    """A float key column carrying both -0.0 and 0.0 (one SQL group), plus values."""
    return pa.table(
        {
            "k": pa.array([-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 1.5, 1.5], pa.float64()),
            "v": pa.array([1, 3, 5, 7, 9, 11, 2, 4], pa.int64()),
        }
    )


def test_spilling_median_signed_zero_key_matches_duckdb(duck, signed_zero_table):
    """`median(v) GROUP BY <float key with -0.0 and 0.0>` — one group, not two."""
    duck.register("t", signed_zero_table)
    plan = bt.from_arrow(signed_zero_table).group_by("k").agg(m=bt.col("v").median())
    out = plan.collect(spill=True)
    assert_same(out, duck.sql("SELECT k, median(v) AS m FROM t GROUP BY k"))
    # And identical to the in-memory schedule of the same plan.
    assert_tables_equal(out, plan.collect(), ordered=False)


def test_spilling_n_unique_signed_zero_key_matches_duckdb(duck, signed_zero_table):
    """`n_unique(v) GROUP BY <float key with -0.0 and 0.0>` — one group."""
    duck.register("t", signed_zero_table)
    plan = bt.from_arrow(signed_zero_table).group_by("k").agg(n=bt.col("v").n_unique())
    out = plan.collect(spill=True)
    assert_same(out, duck.sql("SELECT k, COUNT(DISTINCT v) AS n FROM t GROUP BY k"))
    assert_tables_equal(out, plan.collect(), ordered=False)


def test_spilling_n_unique_signed_zero_value_matches_duckdb(duck):
    """`n_unique(<float value with -0.0 and 0.0>)` folds the two zeros to one distinct."""
    table = pa.table(
        {
            "g": pa.array([1, 1, 1, 1, 2, 2], pa.int64()),
            "f": pa.array([-0.0, 0.0, 5.0, 5.0, -0.0, 3.0], pa.float64()),
        }
    )
    duck.register("t", table)
    plan = bt.from_arrow(table).group_by("g").agg(n=bt.col("f").n_unique())
    out = plan.collect(spill=True)
    assert_same(out, duck.sql("SELECT g, COUNT(DISTINCT f) AS n FROM t GROUP BY g"))
    assert_tables_equal(out, plan.collect(), ordered=False)


def test_spilling_mixed_agg_signed_zero_key_matches_duckdb(duck, signed_zero_table):
    """A mixed value-list + constant-state aggregate on a signed-zero float key.

    Exercises the compositional `mixed_spill` path: its value-list sub-path and its grace
    sub-path must agree on the group set (both one group for the two zeros), or the
    per-path columns misalign / the group counts mismatch.
    """
    duck.register("t", signed_zero_table)
    plan = (
        bt.from_arrow(signed_zero_table)
        .group_by("k")
        .agg(m=bt.col("v").median(), s=bt.col("v").sum())
    )
    out = plan.collect(spill=True)
    assert_same(out, duck.sql("SELECT k, median(v) AS m, SUM(v) AS s FROM t GROUP BY k"))
    assert_tables_equal(out, plan.collect(), ordered=False)
