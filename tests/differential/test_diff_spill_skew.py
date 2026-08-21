"""Out-of-core joins under **key skew** and under a memory envelope, vs DuckDB.

`test_diff_spill_paths.py` pins that the spilling operators compute the right relation on
well-behaved input. This file pins the input they are actually deployed against: a fact
table whose join key is one sentinel value for a large fraction of its rows.

Skew is the interesting case because it is the one a grace join's *partitioning* cannot
help with. Bucket counts are sized from the build side's average bytes per bucket, so a
hot key leaves one probe bucket orders of magnitude over the envelope even when every
build bucket fits — and re-partitioning is a re-hash, so those rows land together again at
every level. Bounded memory has to come from not holding the probe bucket at all, which
means the join emits its result while consuming the probe side rather than after it. That
is a different code path from the in-memory join for every outer flavor, since an
unmatched *build* row is a property of the whole probe side and not of any one morsel of
it — so every flavor is checked here, not just the inner join.

The range join is here for the same reason at the other end: it holds its right side for a
global sort order and streams its left side past it in envelope-sized chunks, so `RIGHT`
and `FULL` have the same across-chunk bookkeeping to get wrong.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal
from batcher.config import Config, MemoryConfig, config_context

pytestmark = pytest.mark.differential

#: Every join flavor the engine and DuckDB both spell the same way.
FLAVORS = ["inner", "left", "right", "full", "semi", "anti"]

#: Rows sharing the sentinel key, spread over many morsels. Large enough that the probe
#: bucket holding them dwarfs the four-row build side it joins against.
HOT_ROWS = 40_000


@pytest.fixture
def skewed() -> pa.Table:
    """A fact table where one key value dominates, plus cold keys, nulls and a non-match.

    The shape is the ordinary one: `-1` is what an ETL writes when the dimension lookup
    failed, so it is both the most common value and the one with no match on the other
    side.
    """
    keys = [-1] * HOT_ROWS
    vals = list(range(HOT_ROWS))
    # Cold keys: 1 and 2 match the dimension, 77 does not, None never matches anything.
    keys += [1, 2, 77, None, 1]
    vals += [-10, -20, -30, -40, -50]
    return pa.table({"k": pa.array(keys, pa.int64()), "v": pa.array(vals, pa.int64())})


@pytest.fixture
def dim() -> pa.Table:
    """The build side: two keys that are probed, and one that never is (so `right`/`full`
    have a genuine unmatched remainder no probe morsel can supply)."""
    return pa.table(
        {
            "k": pa.array([1, 2, 999, None], pa.int64()),
            "w": pa.array(["one", "two", "orphan", "nullkey"]),
        }
    )


@pytest.mark.parametrize("how", FLAVORS)
def test_skewed_spilling_join_matches_duckdb(duck, skewed, dim, how):
    """A spilled join over a hot key equals DuckDB's, for every flavor.

    The hot key's rows exceed any bucket the partitioner can build for them, so the probe
    side is consumed morsel by morsel. `right` and `full` are the load-bearing cases: their
    unmatched build rows have to be withheld until the last probe morsel has been seen, and
    emitted exactly once — a version that emitted per morsel would return `HOT_ROWS` copies
    of the orphan row.
    """
    duck.register("f", skewed)
    duck.register("d", dim)
    out = (
        bt.from_arrow(skewed)
        .join(bt.from_arrow(dim), left_on="k", right_on="k", how=how)
        .collect(spill=True)
    )
    # `USING (k)` rather than `ON f.k = d.k`, because both engines then emit a single
    # coalesced key column. Under `ON`, a right/full join's unmatched build rows carry a NULL
    # `f.k` while Batcher's join keeps the key it matched on, and the two disagree about the
    # projection rather than about the join.
    sql = {
        "semi": "SELECT f.k, f.v FROM f WHERE EXISTS (SELECT 1 FROM d WHERE d.k = f.k)",
        "anti": "SELECT f.k, f.v FROM f WHERE NOT EXISTS (SELECT 1 FROM d WHERE d.k = f.k)",
    }.get(how, f"SELECT k, v, w FROM f {how.upper()} JOIN d USING (k)")
    assert_same(out, duck.sql(sql))


@pytest.mark.parametrize("how", FLAVORS)
def test_skewed_spilling_join_equals_the_in_memory_join(skewed, dim, how):
    """Spilled and in-memory are one operator under two memory regimes, not two operators."""
    plan = bt.from_arrow(skewed).join(bt.from_arrow(dim), left_on="k", right_on="k", how=how)
    assert_tables_equal(plan.collect(spill=True), plan.collect())


def test_a_hot_key_on_both_sides_still_matches_duckdb(duck):
    """The pathological case: the same key is hot on *both* sides.

    No partitioning can separate these rows and the output is quadratic in the group, so
    this is where a grace join is at its least comfortable. It still has to be right.
    """
    left = pa.table({"k": pa.array([5] * 900, pa.int64()), "a": pa.array(range(900), pa.int64())})
    right = pa.table({"k": pa.array([5] * 40, pa.int64()), "b": pa.array(range(40), pa.int64())})
    duck.register("l", left)
    duck.register("r", right)
    out = (
        bt.from_arrow(left)
        .join(bt.from_arrow(right), left_on="k", right_on="k")
        .collect(spill=True)
    )
    assert_same(out, duck.sql("SELECT l.k, l.a, r.b FROM l JOIN r ON l.k = r.k"))


def test_a_range_join_under_a_tight_envelope_matches_duckdb(duck):
    """A range join whose left side exceeds the envelope streams it, and still matches.

    The left side is decomposable — a left row's matches depend on the whole right side and
    on nothing else about the left — so it is consumed in envelope-sized chunks with the
    right side resident. The envelope here is far below the left side's footprint, so the
    chunking actually runs rather than degenerating to the single sweep.
    """
    n = 60_000
    events = pa.table(
        {"t": pa.array(range(n), pa.int64()), "e": pa.array([i % 7 for i in range(n)], pa.int64())}
    )
    windows = pa.table({"lo": pa.array([10, 5_000, 59_000], pa.int64())})
    duck.register("events", events)
    duck.register("windows", windows)
    sql = "SELECT events.t, windows.lo FROM events, windows WHERE events.t < windows.lo"
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=2_000_000))
    with config_context(cfg):
        out = bt.sql(sql, events=events, windows=windows).collect()
    assert_same(out, duck.sql(sql))
