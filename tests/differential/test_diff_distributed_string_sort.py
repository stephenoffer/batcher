"""A distributed ``ORDER BY <string column>`` must reproduce the single-node lexical order.

The distributed sort routes rows against sampled quantile boundaries. A numeric key is
compared as ``f64``; a string key cannot be, because arrow would read ``"12"`` as ``12.0``
and order the buckets numerically. So the string case was refused outright — which was a
harmless performance limit only while the refusal could fall back to a single node. Once an
earlier stage leaves its result on the workers every source is splittable, the fallback is
withdrawn, and the query fails: four of the 22 TPC-H queries (q4, q9, q12, q22) end in a
string ``ORDER BY`` over a materialized aggregate and did exactly that.

Every assertion here compares an **ordered** sequence. ``assert_same`` is order-independent
by design and cannot see a sort bug, which is precisely the bug a range-partition change
risks: the right rows in the wrong order.
"""

from __future__ import annotations

import random
import string

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_RNG = random.Random(20260731)
# Duplicates (equal keys must never straddle a bucket boundary), nulls at both ends, an
# empty string, and values whose *numeric* reading would order differently from their
# lexical one ("12" < "9" lexically, the other way round as floats).
_WORDS: list[str | None] = [
    *("".join(_RNG.choices(string.ascii_lowercase, k=3)) for _ in range(600)),
    *(["dup"] * 120),
    *["9", "12", "100", "8", ""],
    *([None] * 40),
]
_RNG.shuffle(_WORDS)
_T = pa.table({"w": _WORDS, "v": list(range(len(_WORDS)))})


def _keys(table) -> list:
    return table.to_pydict()["w"]


@pytest.fixture(scope="module")
def splittable(cluster_tmp_dir):
    """The same rows as a parquet file, which is a **splittable** source.

    This is load-bearing, not incidental. With an in-memory `from_arrow` source the
    dispatcher's refusal falls back to one node, so every assertion below would pass just as
    well with the string key still refused — the tests would pin nothing. A splittable
    source is what withdraws the fallback, which is both the reason these queries used to
    fail and the only way to prove they no longer take that route.
    """
    path = cluster_tmp_dir / "strsort_rows.parquet"
    pq.write_table(_T, path, row_group_size=128)
    return bt.read.parquet(str(path))


# One bucket (the partitioner's early return) and several (the routing proper), each in
# both directions — the null bucket moves to the opposite end for a descending sort, and
# a sort that reversed the ranges but not the rows would still pass an ascending check.
# Every case is a real distributed collect, so the grid stays small on purpose.
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("workers", [1, 3])
def test_distributed_string_sort_matches_single_node_order(
    splittable, workers: int, descending: bool
):
    ds = splittable.sort("w", descending=descending)
    single = _keys(ds.collect())
    multi = _keys(ds.collect(distributed=True, num_workers=workers))
    assert multi == single, f"workers={workers} descending={descending}"


def test_a_string_sort_over_a_shuffled_intermediate_distributes(splittable):
    """The shape that failed: ``ORDER BY <string>`` above an aggregate.

    The aggregate is a pipeline breaker, so the sort reads an intermediate rather than the
    original source. That is the staged plan whose sources are all splittable, where
    declining to distribute raises instead of falling back.
    """
    ds = splittable.group_by("w").agg(n=bt.col("v").count()).sort("w")
    single = _keys(ds.collect())
    multi = _keys(ds.collect(distributed=True, num_workers=4))
    assert multi == single


def test_equal_string_keys_do_not_straddle_a_boundary(splittable):
    """A duplicated key routed to two buckets would sort correctly *within* each and
    still emerge interleaved with the neighbouring range.

    `"dup"` is in the data 120 times for this: a key split across a boundary comes back as
    two separate runs, which the concatenation cannot merge because it never merges.
    """
    rows = splittable.sort("w").collect(distributed=True, num_workers=4).to_pydict()["w"]
    runs = [key for i, key in enumerate(rows) if i == 0 or rows[i - 1] != key]
    split = [k for k in runs if runs.count(k) > 1]
    assert not split, f"these keys came back as more than one run: {sorted(set(split), key=str)}"


def _tuples(table) -> list[tuple]:
    return [tuple(r) for r in zip(*table.to_pydict().values(), strict=True)]


def test_a_secondary_string_key_still_sorts_by_the_full_key_list(splittable):
    """Only the LEADING key drives the partitioning; the rest are the reducer's local sort."""
    ds = splittable.sort("w", "v", descending=[False, True])
    assert _tuples(ds.collect(distributed=True, num_workers=3)) == _tuples(ds.collect())


def test_an_all_null_string_key_still_distributes(cluster_tmp_path):
    """No sample means no boundaries; every row must still land somewhere and come back."""
    t = pa.table({"w": pa.array([None] * 50, type=pa.string()), "v": list(range(50))})
    pq.write_table(t, cluster_tmp_path / "nulls.parquet")
    ds = bt.read.parquet(str(cluster_tmp_path / "nulls.parquet")).sort("w")
    assert _keys(ds.collect(distributed=True, num_workers=3)) == _keys(ds.collect())
