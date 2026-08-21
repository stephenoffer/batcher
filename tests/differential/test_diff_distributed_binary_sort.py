"""A distributed ``ORDER BY <binary column>`` must reproduce the single-node byte order.

The distributed sort routes rows against sampled quantile boundaries. A numeric key is
compared as ``f64`` and a text key byte-lexically; a **binary** key was refused outright,
because neither the sampler nor the range partitioner would take it. That refusal is a
harmless performance limit only while it can fall back to a single node — and a fixed-width
key over a wide payload is precisely the shape that does not fit one, so the refusal capped
the canonical large-sort workload at whatever a single machine holds.

Every assertion here compares an **ordered** sequence. ``assert_same`` is order-independent
by design and cannot see a sort bug, which is precisely the bug a range-partition change
risks: the right rows in the wrong order.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_RNG = random.Random(20260818)
# The CloudSort record shape in miniature: a short fixed-width key over a wide payload.
# Duplicates (equal keys must never straddle a bucket boundary), nulls, an empty value, and
# keys whose leading byte is `0` — the byte a padded key pack could confuse with the end of
# a shorter value, and the one a text-only sampler could never have carried.
_KEYS: list[bytes | None] = [
    *(bytes(_RNG.randrange(256) for _ in range(10)) for _ in range(600)),
    *([b"\x00" * 9 + b"\x01"] * 120),
    *[b"\x00" * 10, b"\xff" * 10, b"\x00\x01" + b"\x00" * 8],
    *([None] * 40),
]
_RNG.shuffle(_KEYS)
_T = pa.table(
    {
        "k": pa.array(_KEYS, type=pa.binary()),
        "v": list(range(len(_KEYS))),
        "payload": pa.array([bytes(90) for _ in _KEYS], type=pa.binary()),
    }
)


def _keys(table) -> list:
    return table.to_pydict()["k"]


@pytest.fixture(scope="module")
def splittable(cluster_tmp_dir):
    """The same rows as a parquet file, which is a **splittable** source.

    This is load-bearing, not incidental. With an in-memory `from_arrow` source the
    dispatcher's refusal falls back to one node, so every assertion below would pass just as
    well with the binary key still refused — the tests would pin nothing. A splittable
    source is what withdraws the fallback, which is both the reason this shape used to fail
    and the only way to prove it no longer takes that route.
    """
    path = cluster_tmp_dir / "binsort_rows.parquet"
    pq.write_table(_T, path, row_group_size=128)
    return bt.read.parquet(str(path))


# One bucket (the partitioner's early return) and several (the routing proper), each in
# both directions — the null bucket moves to the opposite end for a descending sort, and
# a sort that reversed the ranges but not the rows would still pass an ascending check.
# Every case is a real distributed collect, so the grid stays small on purpose.
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("workers", [1, 3])
def test_distributed_binary_sort_matches_single_node_order(
    splittable, workers: int, descending: bool
):
    ds = splittable.sort("k", descending=descending)
    single = _keys(ds.collect())
    multi = _keys(ds.collect(distributed=True, num_workers=workers))
    assert multi == single, f"workers={workers} descending={descending}"


def test_a_binary_sort_over_a_shuffled_intermediate_distributes(splittable):
    """The shape that failed: ``ORDER BY <binary>`` above an aggregate.

    The aggregate is a pipeline breaker, so the sort reads an intermediate rather than the
    original source. That is the staged plan whose sources are all splittable, where
    declining to distribute raises instead of falling back.
    """
    ds = splittable.group_by("k").agg(n=bt.col("v").count()).sort("k")
    single = _keys(ds.collect())
    multi = _keys(ds.collect(distributed=True, num_workers=4))
    assert multi == single


def test_equal_binary_keys_do_not_straddle_a_boundary(splittable):
    """A duplicated key routed to two buckets would sort correctly *within* each and
    still emerge interleaved with the neighbouring range.

    One key is in the data 120 times for this: a key split across a boundary comes back as
    two separate runs, which the concatenation cannot merge because it never merges.
    """
    rows = splittable.sort("k").collect(distributed=True, num_workers=4).to_pydict()["k"]
    runs = [key for i, key in enumerate(rows) if i == 0 or rows[i - 1] != key]
    split = [k for k in runs if runs.count(k) > 1]
    assert not split, f"these keys came back as more than one run: {sorted(set(split), key=str)}"


def _tuples(table) -> list[tuple]:
    return [tuple(r) for r in zip(*table.to_pydict().values(), strict=True)]


def test_a_secondary_key_under_a_binary_leading_key_still_sorts_fully(splittable):
    """Only the LEADING key drives the partitioning; the rest are the reducer's local sort."""
    ds = splittable.sort("k", "v", descending=[False, True])
    assert _tuples(ds.collect(distributed=True, num_workers=3)) == _tuples(ds.collect())


def test_a_fixed_size_binary_key_distributes(cluster_tmp_path):
    """`FixedSizeBinary` is the type a fixed-layout record key actually arrives as, and it
    routes through the same sampler and partitioner as the variable-length spellings."""
    keys = [bytes(_RNG.randrange(256) for _ in range(10)) for _ in range(500)]
    t = pa.table({"k": pa.array(keys, type=pa.binary(10)), "v": list(range(len(keys)))})
    pq.write_table(t, cluster_tmp_path / "fixed.parquet", row_group_size=64)
    ds = bt.read.parquet(str(cluster_tmp_path / "fixed.parquet")).sort("k")
    assert _keys(ds.collect(distributed=True, num_workers=3)) == _keys(ds.collect())


def test_an_all_null_binary_key_still_distributes(cluster_tmp_path):
    """No sample means no boundaries; every row must still land somewhere and come back."""
    t = pa.table({"k": pa.array([None] * 50, type=pa.binary()), "v": list(range(50))})
    pq.write_table(t, cluster_tmp_path / "binnulls.parquet")
    ds = bt.read.parquet(str(cluster_tmp_path / "binnulls.parquet")).sort("k")
    assert _keys(ds.collect(distributed=True, num_workers=3)) == _keys(ds.collect())


def test_a_dictionary_encoded_key_still_routes_by_its_values(cluster_tmp_path):
    """A dictionary-encoded column must route by the values, never by the dictionary indices.

    The reader decodes dictionaries at the read boundary, so by the time the range partitioner
    sees the column it is a plain byte key — which is exactly why `bucketize` may dispatch on
    the *runtime* batch type. This pins that seam: were a dictionary column ever to reach the
    partitioner still encoded, `grid_kind_of` would call it numeric and route every row by its
    integer index instead of its bytes, which orders the relation by dictionary-insertion order
    and looks like a plausible sort.
    """
    values = [f"cat{i % 7}" for i in range(20_000)]
    t = pa.table({"k": pa.array(values).dictionary_encode(), "v": list(range(len(values)))})
    pq.write_table(t, cluster_tmp_path / "dict.parquet", row_group_size=2_000)
    ds = bt.read.parquet(str(cluster_tmp_path / "dict.parquet")).sort("k")
    assert ds.collect().column("k").to_pylist() == sorted(values)
    assert ds.collect(distributed=True, num_workers=3).column("k").to_pylist() == sorted(values)
