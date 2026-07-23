"""Distributed aggregate equivalence: single-node == multi-partition, across many
aggregates and edge-case group keys.

The mergeable contract (invariant #7) is that
``combine_finalize(partition(partial(p_k)))`` over all partitions equals the
single-node aggregate for *every* stateful operator. The Rust composition tests in
``bc-interp::dist`` pin this for the primitives directly; these pin it end-to-end
through the Python distributed executor, and cross-check the numeric shapes against
DuckDB.

The edge keys are the dangerous ones for a hash shuffle: ``-0.0``/``0.0`` (group-equal
but bit-different), ``NaN`` (all NaN are one group), and ``NULL`` (its own group). If
the shuffle routed any of these differently than single-node grouping, a distributed
run would return more groups than single-node — a silent split.
"""

from __future__ import annotations

import collections

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential

# An integer group key with duplicates spanning many morsels, plus values with nulls.
_INT_KEYS = pa.table(
    {
        "k": pa.array(([1, 2, 3, 1, 2, 5, 3, 4, 2, 1] * 20), pa.int64()),
        "v": pa.array(list(range(200)), pa.int64()),
        "o": pa.array([((i * 7) % 13) for i in range(200)], pa.int64()),
    }
)

# A float group key exercising the edge equivalence classes.
_FLOAT_KEYS = pa.table(
    {
        "k": pa.array(
            [0.0, -0.0, float("nan"), None, 1.0, -0.0, 0.0, float("nan"), None, 2.5] * 6,
            pa.float64(),
        ),
        "v": pa.array([float(i) for i in range(60)], pa.float64()),
    }
)


def _canon(x: object) -> object:
    # NaN != NaN, so a Counter can't match two identical NaN-bearing rows. Map every NaN
    # to one sentinel so the multiset comparison sees them as equal (the engine treats all
    # NaN as one group; this only fixes the *comparison*, not the data).
    if isinstance(x, float) and x != x:
        return "__nan__"
    return x


def _multiset(d: dict) -> collections.Counter:
    return collections.Counter(
        tuple(_canon(v) for v in row) for row in zip(*d.values(), strict=True)
    )


def _agg_dataset(src: pa.Table):
    return (
        bt.from_arrow(src)
        .group_by("k")
        .agg(
            s=col("v").sum(),
            n=col("v").count(),
            mn=col("v").min(),
            mx=col("v").max(),
            avg=col("v").mean(),
            nu=col("v").n_unique(),
            amx=col("v").arg_max(col("o")),
            amn=col("v").arg_min(col("o")),
        )
    )


@pytest.mark.parametrize("workers", [1, 2, 3, 7])
def test_int_key_distributed_equals_single_node(workers):
    ds = _agg_dataset(_INT_KEYS)
    single = _multiset(ds.collect().to_pydict())
    multi = _multiset(ds.collect(distributed=True, num_workers=workers).to_pydict())
    assert single == multi, f"distributed(workers={workers}) split or dropped a group"


@pytest.mark.parametrize("workers", [1, 2, 3, 7])
def test_float_edge_key_distributed_equals_single_node(workers):
    # No arg_* / o column here; float edge keys are the point.
    ds = (
        bt.from_arrow(_FLOAT_KEYS)
        .group_by("k")
        .agg(s=col("v").sum(), n=col("v").count(), mn=col("v").min(), mx=col("v").max())
    )
    single = ds.collect().to_pydict()
    multi = ds.collect(distributed=True, num_workers=workers).to_pydict()
    # Same number of groups: -0.0/0.0 -> one, all NaN -> one, all NULL -> one, plus the
    # ordinary keys. A shuffle that split any edge class would inflate this.
    assert len(single["k"]) == len(multi["k"]), (
        f"distributed(workers={workers}) produced {len(multi['k'])} groups "
        f"vs {len(single['k'])} single-node"
    )
    assert _multiset(single) == _multiset(multi)


def test_int_key_aggregate_matches_duckdb(duck):
    duck.register("t", _INT_KEYS)
    ds = _agg_dataset(_INT_KEYS)
    want = duck.sql(
        "SELECT k, sum(v) s, count(v) n, min(v) mn, max(v) mx, avg(v) avg, "
        "count(DISTINCT v) nu, arg_max(v, o) amx, arg_min(v, o) amn "
        "FROM t GROUP BY k"
    )
    assert_same(ds.collect(), want)
    assert_same(ds.collect(distributed=True, num_workers=3), want)


def test_float_edge_key_grouping_matches_duckdb(duck):
    # DuckDB groups -0.0 with 0.0, all NaN together, NULL on its own — the same classes
    # the engine's canonical float key produces. Sum/count/min/max are order-stable, so
    # this is a safe oracle (no float total-order subtlety).
    duck.register("t", _FLOAT_KEYS)
    ds = (
        bt.from_arrow(_FLOAT_KEYS)
        .group_by("k")
        .agg(s=col("v").sum(), n=col("v").count(), mn=col("v").min(), mx=col("v").max())
    )
    want = duck.sql("SELECT k, sum(v) s, count(v) n, min(v) mn, max(v) mx FROM t GROUP BY k")
    assert_same(ds.collect(), want)
    assert_same(ds.collect(distributed=True, num_workers=3), want)
