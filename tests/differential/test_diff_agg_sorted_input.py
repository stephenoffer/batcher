"""Grouping a *sorted* key matches DuckDB, on every path that groups.

``bc_runtime::agg::group::runs`` assigns group ids by scanning runs of equal adjacent values
instead of hashing, whenever it can first prove the key is monotonic. That is a second
implementation of the engine's single most-shared primitive — ``assign_groups`` backs every
``GROUP BY``, every ``DISTINCT`` and every partitioned window — so it is checked here against
the oracle rather than only against the hash path it short-circuits.

What makes these cases worth their runtime is that sortedness is a property of the *input*, so
an ordinary aggregate test does not reach the new code at all: the data has to arrive ordered.
Each case below therefore sorts its input first, and the shapes are chosen for where runs and
hashing can disagree — a run that straddles a morsel boundary, a key that is ordered within
each morsel but not across them, ``-0.0``/NaN, nulls, and descending order.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same
from batcher import col

# More than one 16,384-row morsel, so every run-based case has runs that straddle a boundary --
# the seam where a per-morsel run assignment has to hand off to the mergeable `combine`.
_N = 50_000


def _sorted_int(n: int = _N, groups: int = 500) -> pa.Table:
    """Ascending group key, so equal keys are adjacent."""
    g = [i * groups // n for i in range(n)]
    return pa.table({"g": g, "x": [float(i % 97) for i in range(n)], "i": list(range(n))})


def test_sorted_key_group_by(duck):
    t = _sorted_int()
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("g").agg(s=col("x").sum(), c=col("g").count()).collect()
    assert_same(out, duck.sql("SELECT g, sum(x) AS s, count(g) AS c FROM t GROUP BY g"))


def test_descending_key_group_by(duck):
    """Descending input clusters equal keys just as well, and is accepted for that reason."""
    t = _sorted_int()
    t = t.take(list(range(t.num_rows - 1, -1, -1)))
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("g").agg(s=col("x").sum()).collect()
    assert_same(out, duck.sql("SELECT g, sum(x) AS s FROM t GROUP BY g"))


def test_every_row_its_own_group(duck):
    """The shape the run path wins most on: a sorted key with no repeats at all."""
    t = pa.table({"g": list(range(_N)), "x": [float(i) for i in range(_N)]})
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("g").agg(s=col("x").sum()).collect()
    assert_same(out, duck.sql("SELECT g, sum(x) AS s FROM t GROUP BY g"))


def test_one_group_over_the_whole_input(duck):
    """A single run spanning every morsel: the carry case, if runs were ever carried."""
    t = pa.table({"g": [7] * _N, "x": [float(i % 13) for i in range(_N)]})
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("g").agg(s=col("x").sum(), n=col("x").count()).collect()
    assert_same(out, duck.sql("SELECT g, sum(x) AS s, count(x) AS n FROM t GROUP BY g"))


def test_sorted_within_each_morsel_but_not_across(duck):
    """The trap: every morsel is monotonic on its own, the relation is not.

    A per-morsel run assignment accepts each batch, and the same key therefore appears in
    several batches' outputs. Only the mergeable `combine` puts them back together, so this is
    the case that fails if runs were ever treated as final groups rather than as partials.
    """
    per = 8_000
    g = []
    for _ in range(_N // per + 1):
        g.extend(sorted(i % 250 for i in range(per)))
    g = g[:_N]
    t = pa.table({"g": g, "x": [float(i % 31) for i in range(_N)]})
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("g").agg(s=col("x").sum(), n=col("x").count()).collect()
    assert_same(out, duck.sql("SELECT g, sum(x) AS s, count(x) AS n FROM t GROUP BY g"))


def test_cyclic_key_is_rejected_and_still_correct(duck):
    """Ordered across any short prefix, not ordered at all -- the shape a sampled gate passes."""
    t = pa.table({"g": [i % 100 for i in range(_N)], "x": [float(i % 7) for i in range(_N)]})
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("g").agg(s=col("x").sum()).collect()
    assert_same(out, duck.sql("SELECT g, sum(x) AS s FROM t GROUP BY g"))


def test_sorted_string_key(duck):
    t = pa.table(
        {
            "g": [f"cat-{i * 300 // _N:04d}" for i in range(_N)],
            "x": [float(i % 11) for i in range(_N)],
        }
    )
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("g").agg(s=col("x").sum()).collect()
    assert_same(out, duck.sql("SELECT g, sum(x) AS s FROM t GROUP BY g"))


def test_sorted_composite_key(duck):
    """Lexicographic order: the minor key may descend inside a major run."""
    a = [i * 40 // _N for i in range(_N)]
    b = [(i * 900 // _N) % 25 for i in range(_N)]
    pairs = sorted(zip(a, b, strict=True))
    t = pa.table(
        {
            "a": [p[0] for p in pairs],
            "b": [p[1] for p in pairs],
            "x": [float(i % 5) for i in range(_N)],
        }
    )
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("a", "b").agg(s=col("x").sum()).collect()
    assert_same(out, duck.sql("SELECT a, b, sum(x) AS s FROM t GROUP BY a, b"))


def test_sorted_key_with_nulls(duck):
    """Nulls group together under GROUP BY, whichever path assigns the ids."""
    g = [None] * 5_000 + [i * 100 // _N for i in range(_N - 5_000)]
    t = pa.table({"g": pa.array(g, pa.int64()), "x": [float(i % 3) for i in range(_N)]})
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("g").agg(s=col("x").sum(), n=col("x").count()).collect()
    assert_same(out, duck.sql("SELECT g, sum(x) AS s, count(x) AS n FROM t GROUP BY g"))


def test_sorted_float_key_zero_and_nan(duck):
    """-0.0 groups with 0.0 and every NaN groups together, as SQL requires.

    Arrow orders these three apart, so a run scan that compared raw values would split them.
    """
    g = [-0.0, -0.0, 0.0, 0.0, 1.5, 1.5, 2.5, float("nan"), float("nan")]
    t = pa.table({"g": pa.array(g, pa.float64()), "x": [float(i) for i in range(len(g))]})
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("g").agg(s=col("x").sum()).collect()
    assert_same(out, duck.sql("SELECT g, sum(x) AS s FROM t GROUP BY g"))


def test_sorted_key_single_row_and_empty(duck):
    one = pa.table({"g": [1], "x": [2.0]})
    duck.register("one", one)
    assert_same(
        bt.from_arrow(one).group_by("g").agg(s=col("x").sum()).collect(),
        duck.sql("SELECT g, sum(x) AS s FROM one GROUP BY g"),
    )
    empty = pa.table({"g": pa.array([], pa.int64()), "x": pa.array([], pa.float64())})
    duck.register("empty", empty)
    assert_same(
        bt.from_arrow(empty).group_by("g").agg(s=col("x").sum()).collect(),
        duck.sql("SELECT g, sum(x) AS s FROM empty GROUP BY g"),
    )


def test_sorted_key_distinct(duck):
    """DISTINCT shares `assign_groups`, so it takes the run path too."""
    t = _sorted_int()
    duck.register("t", t)
    out = bt.from_arrow(t).select("g").distinct().collect()
    assert_same(out, duck.sql("SELECT DISTINCT g FROM t"))


def test_sorted_key_window_partition(duck):
    """A partitioned window shares `assign_groups` as well."""
    t = _sorted_int(n=20_000, groups=200)
    duck.register("t", t)
    out = (
        bt.from_arrow(t)
        .with_columns(s=col("x").sum().over(partition_by="g"))
        .select("i", "g", "s")
        .collect()
    )
    assert_same(out, duck.sql("SELECT i, g, sum(x) OVER (PARTITION BY g) AS s FROM t"))


def test_sorted_key_streaming_matches_collect(duck):
    """The streaming path folds morsel by morsel, so runs straddle its batch boundaries."""
    t = _sorted_int()
    duck.register("t", t)
    ds = bt.from_arrow(t).group_by("g").agg(s=col("x").sum())
    streamed = pa.Table.from_batches(list(ds.iter_batches()), schema=None)
    assert_same(streamed, duck.sql("SELECT g, sum(x) AS s FROM t GROUP BY g"))


def test_sorted_key_group_by_after_an_explicit_sort(duck):
    """The ordering the engine itself produced, which is where this path is most reachable."""
    t = pa.table({"g": [i % 400 for i in range(_N)], "x": [float(i % 17) for i in range(_N)]})
    duck.register("t", t)
    out = bt.from_arrow(t).sort("g").group_by("g").agg(s=col("x").sum()).collect()
    assert_same(out, duck.sql("SELECT g, sum(x) AS s FROM t GROUP BY g"))


def test_sorted_key_single_node_equals_distributed(duck):
    """The run assignment must stay a *partial*, so many machines answer as one does.

    This is the invariant a sorted-input specialization is most likely to break. Runs are
    found per batch, and a distributed run finds them per partition, so the same key can be
    the last run of one partition and the first of another. Only `combine` may merge them --
    if runs were ever treated as final groups, this is where the extra rows would appear.
    """
    t = _sorted_int()
    duck.register("t", t)
    ds = bt.from_arrow(t).group_by("g").agg(s=col("x").sum(), n=col("x").count())
    expected = duck.sql("SELECT g, sum(x) AS s, count(x) AS n FROM t GROUP BY g")
    assert_same(ds.collect(), expected)
    assert_same(ds.collect(distributed=True), expected)
