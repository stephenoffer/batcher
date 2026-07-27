"""`.map.len()` and `.map.contains(key)` — vs DuckDB.

The `.map` namespace had three methods against DuckDB's eleven. These are the two whose
answer the key list already determines, so neither needs a kernel: a map's key list has
one element per entry by construction, so its length *is* the cardinality, and membership
in the map is membership in that list.

Composing them rather than writing kernels is only correct if the composition carries
nullness the same way, which is what this file checks. The fixture therefore separates the
three cases a naive implementation collapses:

* a **null map** — the answer is null, not 0 / false;
* an **empty map** — the answer is 0 / false, not null;
* a **present** and an **absent** key in a non-empty map.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered
from batcher import col


@pytest.fixture
def maps(duck):
    t = pa.table(
        {
            "k": [0, 1, 2, 3],
            "m": pa.array(
                [[("a", 1), ("b", 2)], [("c", 3)], None, []],
                type=pa.map_(pa.string(), pa.int64()),
            ),
        }
    )
    duck.register("maps", t)
    return t


@pytest.mark.differential
def test_cardinality_matches_duckdb(duck, maps):
    out = bt.from_arrow(maps).select(k=col("k"), r=col("m").map.len()).sort("k").collect()
    assert_same_ordered(out, duck.sql("SELECT k, cardinality(m) r FROM maps ORDER BY k"))


@pytest.mark.differential
def test_contains_matches_duckdb(duck, maps):
    out = bt.from_arrow(maps).select(k=col("k"), r=col("m").map.contains("a")).sort("k").collect()
    assert_same_ordered(out, duck.sql("SELECT k, map_contains(m, 'a') r FROM maps ORDER BY k"))


@pytest.mark.differential
def test_a_null_map_is_null_and_an_empty_map_is_not(maps):
    """The distinction a composition gets wrong if nullness stops propagating."""
    out = bt.from_arrow(maps).select(
        k=col("k"), n=col("m").map.len(), has=col("m").map.contains("a")
    )
    rows = out.sort("k").to_pydict()
    assert rows["n"] == [2, 1, None, 0]
    assert rows["has"] == [True, False, None, False]


@pytest.mark.differential
@pytest.mark.parametrize(
    "query",
    [
        "SELECT k, cardinality(m) r FROM maps ORDER BY k",
        "SELECT k, map_contains(m, 'a') r FROM maps ORDER BY k",
    ],
)
def test_reachable_from_sql(duck, maps, query):
    assert_same_ordered(bt.sql(query, maps=maps).collect(), duck.sql(query))
