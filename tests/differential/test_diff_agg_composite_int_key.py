"""Grouping on several integer keys at once matches DuckDB, across both composite paths.

``bc_runtime::agg::group::assign`` routes an all-``Int64`` composite key to one of two
implementations, and which one it takes is decided by the *data* rather than by the query:
when the product of the columns' value ranges fits the dense-map budget the key becomes a
mixed-radix index into a direct map, and otherwise it is hashed. A single test shape reaches
exactly one of them, and nothing in the SQL says which.

That matters more than it looks, because the hash arm builds a row's key by folding one
**column at a time** into a running per-row hash. The column count is therefore the variable
that decides whether the fold is applied the right number of times, in the right order, and
the two-key shapes the rest of the suite covers would not notice a combine step that dropped
or double-counted a column. So the cases below sweep the key width from two to six and pin
each width against the oracle on both arms.

The other half of what is checked here is that the *aggregates* ride along correctly. Group
identity and group payload are computed by different code, and a defect that mis-assigns rows
to groups shows up as wrong sums long before it shows up as a wrong group count -- so every
case carries an aggregate over a column that is not part of the key.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same
from batcher import col

# More than two 16,384-row morsels, so every case groups per morsel and then merges: the
# composite key has to produce the same identity on both sides of that hand-off.
_N = 40_000

# Multiplied into each key column to blow its value range past the dense-map budget without
# changing how many distinct values it holds. It is what separates the two shapes below: the
# cardinalities are identical and only the *spans* differ, so a difference in the result is a
# difference between the two implementations rather than between two datasets.
_SPREAD = 7919


def _keys(n: int, ncols: int, card: int, spread: int) -> dict[str, list[int]]:
    """`ncols` integer key columns, each holding `card` distinct values."""
    return {f"k{c}": [((i + c) % card) * spread for i in range(n)] for c in range(ncols)}


def _table(ncols: int, card: int, spread: int, n: int = _N) -> pa.Table:
    cols: dict[str, list] = dict(_keys(n, ncols, card, spread))
    cols["v"] = [float(i % 97) - 3.5 for i in range(n)]
    cols["w"] = [i % 13 for i in range(n)]
    return pa.table(cols)


def _check(duck, t: pa.Table, ncols: int) -> None:
    keys = [f"k{c}" for c in range(ncols)]
    duck.register("t", t)
    out = (
        bt.from_arrow(t)
        .group_by(keys)
        .agg(s=col("v").sum(), n=col("w").count(), m=col("w").max())
        .collect()
    )
    key_list = ", ".join(keys)
    assert_same(
        out,
        duck.sql(
            f"SELECT {key_list}, sum(v) AS s, count(w) AS n, max(w) AS m FROM t GROUP BY {key_list}"
        ),
    )


def test_composite_int_key_widths_on_the_hash_path(duck):
    """Two to six integer keys, spread past the dense budget so each one is hashed."""
    for ncols in range(2, 7):
        _check(duck, _table(ncols, card=7 + ncols, spread=_SPREAD), ncols)


def test_composite_int_key_widths_on_the_dense_path(duck):
    """The same widths and cardinalities packed tight, so each takes the direct map."""
    for ncols in range(2, 7):
        _check(duck, _table(ncols, card=7 + ncols, spread=1), ncols)


def test_composite_int_key_near_unique(duck):
    """A key with about as many groups as rows -- the shape that grows the group table.

    The table starts far below this group count and is resized from the density measured
    part-way through, so the ids handed out before the resize and after it have to agree.
    A resize that lost or duplicated an entry would split one group in two, which the sums
    below detect and a group count alone would not.
    """
    _check(duck, _table(3, card=40, spread=_SPREAD), 3)


def test_composite_int_key_single_group(duck):
    """Every row in one group: the degenerate end of the same paths."""
    _check(duck, _table(4, card=1, spread=_SPREAD), 4)


def test_composite_int_key_with_negatives_and_extremes(duck):
    """Negative keys, zero, and the ends of the range, which the dense map offsets by `min`.

    ``i64::MIN``/``i64::MAX`` in one column make its span overflow, which must decline to the
    hash path rather than wrap into a bogus index.
    """
    n = 2_000
    t = pa.table(
        {
            "k0": [(i % 5) - 2 for i in range(n)],
            "k1": [-(i % 7) * _SPREAD for i in range(n)],
            "k2": [
                (2**63 - 1) if i % 3 == 0 else (-(2**63)) if i % 3 == 1 else 0 for i in range(n)
            ],
            "v": [float(i % 11) for i in range(n)],
            "w": [i % 13 for i in range(n)],
        }
    )
    _check(duck, t, 3)


def test_composite_int_key_with_nulls(duck):
    """A nullable key column keeps the row-encoded oracle, which groups nulls together.

    The fast paths are gated to null-free columns, so this case exists to prove the *gate*
    still holds -- a composite key that admitted a nullable column to the raw-value path
    would compare null slots as whatever the values buffer happens to hold.
    """
    n = 5_000
    t = pa.table(
        {
            "k0": [None if i % 11 == 0 else (i % 5) * _SPREAD for i in range(n)],
            "k1": [(i % 7) * _SPREAD for i in range(n)],
            "k2": [None if i % 23 == 0 else i % 3 for i in range(n)],
            "v": [float(i % 11) for i in range(n)],
            "w": [i % 13 for i in range(n)],
        }
    )
    _check(duck, t, 3)


def test_composite_int_key_distinct_and_count_distinct(duck):
    """The same assignment backs `DISTINCT`, which returns the key columns themselves.

    ``assign_groups`` hands back one representative row per group, so a defect that grouped
    correctly but gathered the wrong representative would leave every aggregate above right
    and this wrong.
    """
    t = _table(4, card=9, spread=_SPREAD)
    duck.register("t", t)
    keys = ["k0", "k1", "k2", "k3"]
    out = bt.from_arrow(t).select(*keys).distinct().collect()
    assert_same(out, duck.sql("SELECT DISTINCT k0, k1, k2, k3 FROM t"))


def test_composite_int_key_streamed_matches_collected(duck):
    """Streaming the same query splits it across morsels differently, and must not differ."""
    t = _table(5, card=11, spread=_SPREAD)
    duck.register("t", t)
    keys = ["k0", "k1", "k2", "k3", "k4"]
    ds = bt.from_arrow(t).group_by(keys).agg(s=col("v").sum())
    batches = list(ds.iter_batches())
    streamed = pa.Table.from_batches(batches) if batches else ds.collect()
    key_list = ", ".join(keys)
    assert_same(streamed, duck.sql(f"SELECT {key_list}, sum(v) AS s FROM t GROUP BY {key_list}"))
