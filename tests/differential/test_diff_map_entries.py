"""Differential tests for ``.map.entries()`` against DuckDB's ``map_entries``.

``map_keys`` and ``map_values`` each return a list, so pairing a key back with its value
means trusting that the two share an order. ``map_entries`` makes the pairing structural
instead, which is what a caller needs before an ``explode`` turns one map row into one
row per entry.

The interesting cases are the ones where a list-of-structs can go wrong in a way the
values alone do not show: an empty map (an empty list, not a null), a null map row (a
null list, not an empty one), and a duplicate key (a map may carry one, and the entry
list must keep both rather than deduplicating).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


def _map_table(rows: list[list[tuple[str, int]] | None]) -> pa.Table:
    col = pa.array(rows, type=pa.map_(pa.string(), pa.int64()))
    return pa.table({"m": col, "p": pa.array(range(len(rows)), type=pa.int64())})


@pytest.mark.differential
def test_map_entries_matches_duckdb(duck):
    t = _map_table([[("a", 1), ("b", 2)], [("c", 3)], []])
    duck.register("t", t)
    out = bt.from_arrow(t).select(bt.col("m").map.entries().alias("e")).collect()
    assert_same(out, duck.sql("SELECT map_entries(m) AS e FROM t"))


@pytest.mark.differential
def test_map_entries_pairs_keys_with_their_own_values():
    """The pairing is the contract, so assert it directly rather than through DuckDB.

    Values ascend with their keys here, so a mispaired entry shows up as a key whose
    value is not its index.
    """
    rows = [[(f"k{i}", i) for i in range(n)] for n in (0, 1, 5)]
    t = _map_table(rows)
    got = bt.from_arrow(t).select(bt.col("m").map.entries().alias("e")).to_pydict()["e"]
    assert got == [[{"key": f"k{i}", "value": i} for i in range(n)] for n in (0, 1, 5)]


@pytest.mark.differential
def test_map_entries_distinguishes_a_null_row_from_an_empty_map():
    """A null map is a null list; an empty map is an empty list. Collapsing the two is
    the classic list-column bug, and it is invisible in any per-element assertion."""
    t = _map_table([[("a", 1)], [], None])
    got = bt.from_arrow(t).select(bt.col("m").map.entries().alias("e")).to_pydict()["e"]
    assert got == [[{"key": "a", "value": 1}], [], None]


@pytest.mark.differential
def test_map_entries_keeps_a_duplicate_key():
    """Arrow's Map permits a repeated key, and the entry list must not deduplicate it.

    **Deliberately not compared against DuckDB**, because DuckDB cannot represent the
    input: converting this table raises ``Invalid Input Error: Arrow map contains
    duplicate key, which isn't supported by DuckDB map type``. That is a real divergence
    in the *type systems* rather than in either engine's ``map_entries`` -- Arrow's
    ``Map`` is a list of key/value pairs with no uniqueness constraint, DuckDB's ``MAP``
    enforces one -- so there is no oracle for this input, and asserting Batcher's own
    behaviour is the strongest check available. Recorded here rather than dropped,
    because a silently missing case is how a dedupe would later slip in unnoticed.
    """
    t = _map_table([[("a", 1), ("a", 2)]])
    got = bt.from_arrow(t).select(bt.col("m").map.entries().alias("e")).to_pydict()["e"]
    assert got == [[{"key": "a", "value": 1}, {"key": "a", "value": 2}]]


@pytest.mark.differential
def test_map_entries_agrees_with_keys_and_values(duck):
    """The three accessors must describe the same map.

    This is the assertion that would catch an offsets bug: `entries` re-wraps the map's
    entries child under its own offsets, so a wrong offset buffer would misalign the
    entry list against the key and value lists while each stayed internally plausible.
    """
    t = _map_table([[("a", 1), ("b", 2)], [], [("c", 3)], None])
    ds = bt.from_arrow(t)
    out = ds.select(
        bt.col("m").map.entries().alias("e"),
        bt.col("m").map.keys().alias("k"),
        bt.col("m").map.values().alias("v"),
    ).to_pydict()
    for entries, keys, values in zip(out["e"], out["k"], out["v"], strict=True):
        if entries is None:
            assert keys is None and values is None
            continue
        assert [x["key"] for x in entries] == keys
        assert [x["value"] for x in entries] == values


@pytest.mark.differential
def test_map_entries_explodes_to_one_row_per_entry(duck):
    """The use case the function exists for: one map row becomes one row per pair."""
    t = _map_table([[("a", 1), ("b", 2)], [("c", 3)]])
    duck.register("t", t)
    out = (
        bt.from_arrow(t)
        .select(bt.col("m").map.entries().alias("e"))
        .explode("e")
        .select(
            bt.col("e").struct.get("key").alias("k"),
            bt.col("e").struct.get("value").alias("v"),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT e.key AS k, e.value AS v FROM (SELECT unnest(map_entries(m)) AS e FROM t)"
        ),
    )
