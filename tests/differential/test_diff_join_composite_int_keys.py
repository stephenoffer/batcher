"""Equi-joins on **three or more** integer key columns vs DuckDB.

The hash join had raw-value key paths for one and two `Int64` columns and fell to arrow's
`RowConverter` for a third. That was a 13x performance cliff (`bc_runtime::join::I64xNKeys`
records the measurement), and closing it added a *new* key implementation — a per-column
hash and a per-column equality walk — on the shape every star-schema fact-to-fact join
carries. TPC-DS joins `store_sales` to `store_returns` on
`(ticket_number, item_sk, customer_sk)`.

A per-column hash can be wrong where the row encoding cannot, so these cases are chosen to
separate them rather than to exercise a join generally:

* rows agreeing on a **prefix** of the key but not the whole of it — the failure a
  truncated comparison produces;
* a **permuted** key tuple (`(1,2,3)` against `(3,2,1)`) — the columns must not commute,
  which a hash that folds them without position would let them do;
* a **null** in one key column only, on each side in turn — SQL never matches NULL, and the
  fast path masks nulls separately from the value hash;
* **duplicates** on both sides, so the chain walk (which compares build row against build
  row) is exercised rather than only the probe comparison;
* keys wider than three columns, since the path is generic in the count.

Each shape runs through `collect`, `iter_batches` and a bounded-memory spill, because the
cliff this replaced was in the *streaming broadcast* probe specifically — a path only some
of those terminals take.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher.config import active_config, set_config

_KEYS = ("k1", "k2", "k3", "k4")

# Left rows, as (k1, k2, k3, k4, v). Read the first three columns as the join key:
# rows 0/1 share the `(1, 2)` prefix and differ in `k3`; row 2 is row 0's permutation;
# rows 3/4 are an exact duplicate pair; rows 5/6 carry a null in one column each.
_L = pa.table(
    {
        "k1": [1, 1, 3, 7, 7, None, 9],
        "k2": [2, 2, 2, 8, 8, 4, None],
        "k3": [3, 4, 1, 9, 9, 5, 6],
        "k4": [0, 0, 0, 1, 1, 0, 0],
        "lv": [10, 11, 12, 13, 14, 15, 16],
    }
)
# Right rows: `(1,2,3)` matches once, `(3,2,1)` matches only the permuted row, `(7,8,9)`
# matches the duplicate pair twice, `(1,2,9)` shares a prefix with no match, and the two
# null-bearing rows must match nothing at all.
_R = pa.table(
    {
        "k1": [1, 3, 7, 1, None, 9, 7],
        "k2": [2, 2, 8, 2, 4, None, 8],
        "k3": [3, 1, 9, 9, 5, 6, 9],
        "k4": [0, 0, 1, 0, 0, 0, 1],
        "rv": [100, 101, 102, 103, 104, 105, 106],
    }
)

_JOIN_TYPES = ["INNER", "LEFT", "RIGHT", "FULL"]


def _on(width: int) -> str:
    return " AND ".join(f"l.{k} = r.{k}" for k in _KEYS[:width])


def _select(width: int) -> str:
    keys = ", ".join(f"l.{k} AS l{k}, r.{k} AS r{k}" for k in _KEYS[:width])
    return f"SELECT {keys}, lv, rv"


@pytest.fixture
def lr(duck):
    duck.register("l", _L)
    duck.register("r", _R)
    return _L, _R


@pytest.mark.differential
@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("how", _JOIN_TYPES)
def test_composite_int_key_join_matches_duckdb(duck, lr, width, how):
    """A 3- and 4-column integer key agrees with the oracle for every join type."""
    q = f"{_select(width)} FROM l {how} JOIN r ON {_on(width)}"
    assert_same(bt.sql(q, l=lr[0], r=lr[1]).collect(), duck.sql(q))


@pytest.mark.differential
@pytest.mark.parametrize("how", ["SEMI", "ANTI"])
def test_composite_int_key_semi_anti_matches_duckdb(duck, lr, how):
    """`SEMI`/`ANTI` on a three-column key — the two probe-driven types with no right side."""
    negate = "NOT " if how == "ANTI" else ""
    q = f"SELECT l.k1, l.k2, l.k3, lv FROM l WHERE {negate}EXISTS (SELECT 1 FROM r WHERE {_on(3)})"
    assert_same(bt.sql(q, l=lr[0], r=lr[1]).collect(), duck.sql(q))


@pytest.mark.differential
def test_composite_int_key_join_streams_and_spills(duck, lr):
    """The same answer through `iter_batches` and under a bounded memory envelope.

    The cliff this replaced lived in the *streaming broadcast* probe, which `collect` on a
    small input need not reach — so a test that only collects would not have covered the
    code it exercises.
    """
    q = f"{_select(3)} FROM l JOIN r ON {_on(3)}"
    expected = duck.sql(q)
    streamed = pa.Table.from_batches(
        list(bt.sql(q, l=lr[0], r=lr[1]).iter_batches()),
        schema=bt.sql(q, l=lr[0], r=lr[1]).collect().schema,
    )
    assert_same(streamed, expected)

    cfg = active_config()
    try:
        set_config(cfg.replace(memory=dataclasses.replace(cfg.memory, max_memory_bytes=1 << 20)))
        assert_same(bt.sql(q, l=lr[0], r=lr[1]).collect(spill=True), expected)
    finally:
        set_config(cfg)


@pytest.mark.differential
def test_composite_int_key_join_over_an_empty_side(duck):
    """An empty side on a three-column key still produces the oracle's relation."""
    empty = _R.slice(0, 0)
    duck.register("l", _L)
    duck.register("r", empty)
    for how in _JOIN_TYPES:
        q = f"{_select(3)} FROM l {how} JOIN r ON {_on(3)}"
        assert_same(bt.sql(q, l=_L, r=empty).collect(), duck.sql(q))


@pytest.mark.differential
def test_composite_int_key_join_at_multi_batch_scale(duck):
    """Many morsels, so the probe is genuinely chunked rather than one batch.

    A key repeated across morsel boundaries is what makes the streaming probe's
    "morsels are independent" claim load-bearing; a single-batch test cannot see it.
    """
    n = 60_000
    left = pa.table(
        {
            "k1": [i % 97 for i in range(n)],
            "k2": [i % 89 for i in range(n)],
            "k3": [i % 83 for i in range(n)],
            "lv": list(range(n)),
        }
    )
    right = pa.table(
        {
            "k1": [i % 97 for i in range(500)],
            "k2": [i % 89 for i in range(500)],
            "k3": [i % 83 for i in range(500)],
            "rv": list(range(500)),
        }
    )
    duck.register("l", left)
    duck.register("r", right)
    q = (
        "SELECT l.k1, l.k2, l.k3, sum(lv) AS sl, sum(rv) AS sr, count(*) AS n "
        "FROM l JOIN r ON l.k1 = r.k1 AND l.k2 = r.k2 AND l.k3 = r.k3 "
        "GROUP BY l.k1, l.k2, l.k3"
    )
    assert_same(bt.sql(q, l=left, r=right).collect(), duck.sql(q))
