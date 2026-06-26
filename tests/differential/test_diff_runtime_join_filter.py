"""Runtime join-filter safety vs DuckDB.

The rule pushes `probe.key BETWEEN build.min AND build.max` onto a join side. This
must be a pure superset filter: pre-filtering the probe by the other side's key
range produces the identical join result. These tests apply that same filter
explicitly (the in-memory path has no footer stats to trigger the rule itself) and
confirm the pruned plan still produces the true (DuckDB) answer.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _tables(duck):
    fact = pa.table({"k": [1, 5, 12, 15, 18, 50, 99, 12], "amt": [1, 2, 3, 4, 5, 6, 7, 8]})
    dim = pa.table({"k": [12, 15, 18], "region": ["a", "b", "c"]})
    duck.register("fact", fact)
    duck.register("dim", dim)
    return bt.from_arrow(fact), bt.from_arrow(dim)


def test_range_filtered_probe_matches_unfiltered(duck):
    from conftest import assert_same

    fact, dim = _tables(duck)
    expected = duck.sql("SELECT * FROM fact JOIN dim USING (k)")
    # dim.k spans [12, 18]; pre-filtering the fact to that range is what the rule does.
    pruned = fact.filter((col("k") >= 12) & (col("k") <= 18)).join(dim, on="k").collect()
    assert_same(pruned, expected)
    assert_same(fact.join(dim, on="k").collect(), expected)


def test_multi_key_range_filter_matches_unfiltered(duck):
    """Composite-key join: pruning the probe by the build's range on *every* key
    (`k1 BETWEEN .. AND k2 BETWEEN ..`) is still a pure superset — identical result."""
    from conftest import assert_same

    fact = pa.table(
        {
            "k1": [1, 5, 12, 12, 18, 50, 12, 15],
            "k2": [9, 3, 5, 6, 6, 9, 7, 5],
            "amt": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    dim = pa.table({"k1": [12, 12, 15], "k2": [5, 6, 5], "region": ["a", "b", "c"]})
    duck.register("fact2", fact)
    duck.register("dim2", dim)
    bfact, bdim = bt.from_arrow(fact), bt.from_arrow(dim)
    expected = duck.sql("SELECT * FROM fact2 JOIN dim2 USING (k1, k2)")
    # dim spans k1∈[12,15], k2∈[5,6]; pre-filtering the fact to both ranges is the
    # multi-key form of what the rule pushes.
    pruned = (
        bfact.filter((col("k1") >= 12) & (col("k1") <= 15) & (col("k2") >= 5) & (col("k2") <= 6))
        .join(bdim, on=["k1", "k2"])
        .collect()
    )
    assert_same(pruned, expected)
    assert_same(bfact.join(bdim, on=["k1", "k2"]).collect(), expected)


def test_semi_join_range_filter_safe(duck):
    from conftest import assert_same

    fact, dim = _tables(duck)
    expected = duck.sql("SELECT fact.* FROM fact SEMI JOIN dim USING (k)")
    narrowed = fact.filter((col("k") >= 12) & (col("k") <= 18))
    pruned = narrowed.join(dim, on="k", how="semi").collect()
    assert_same(pruned, expected)


def test_left_join_filters_right_side_safely(duck):
    """For a left join only the right (non-preserved) side may be pruned; pruning the
    right by the left's key range keeps every preserved left row."""
    from conftest import assert_same

    fact, dim = _tables(duck)
    expected = duck.sql("SELECT * FROM fact LEFT JOIN dim USING (k)")
    pruned = fact.join(dim.filter((col("k") >= 1) & (col("k") <= 99)), on="k", how="left").collect()
    assert_same(pruned, expected)
