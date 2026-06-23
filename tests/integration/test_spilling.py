"""Out-of-core spilling aggregation == in-memory result (bounded-memory group-by)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count


def _streaming_dataset(n_batches=120, groups=2000):
    rng = np.random.default_rng(0)
    batches = [
        pa.record_batch(
            {
                "k": rng.integers(0, groups, 1000).astype("int64"),
                "v": rng.integers(0, 100, 1000).astype("int64"),
            }
        )
        for _ in range(n_batches)
    ]
    schema = batches[0].schema
    table = pa.Table.from_batches(batches)
    return (lambda: iter(batches)), schema, table


def _norm(t):
    return sorted(
        tuple(round(v, 6) if isinstance(v, float) else v for v in r.values()) for r in t.to_pylist()
    )


def test_spill_grouped_matches_in_memory():
    factory, schema, table = _streaming_dataset()
    agg = {"s": col("v").sum(), "n": count(), "a": col("v").mean(), "mx": col("v").max()}

    spilled = (
        bt.from_batches(factory, schema)
        .group_by("k")
        .agg(**agg)
        .collect(spill=True, num_partitions=16)
    )
    in_memory = bt.from_arrow(table).group_by("k").agg(**agg).collect()
    assert _norm(spilled) == _norm(in_memory)


@pytest.mark.parametrize("num_partitions", [1, 4, 64])
def test_spill_partition_count_invariant(num_partitions):
    # The result must not depend on the number of spill buckets.
    factory, schema, table = _streaming_dataset()
    spilled = (
        bt.from_batches(factory, schema)
        .group_by("k")
        .agg(s=col("v").sum())
        .collect(spill=True, num_partitions=num_partitions)
    )
    in_memory = bt.from_arrow(table).group_by("k").agg(s=col("v").sum()).collect()
    assert _norm(spilled) == _norm(in_memory)


def test_spill_global_aggregate():
    factory, schema, table = _streaming_dataset()
    spilled = (
        bt.from_batches(factory, schema)
        .group_by()
        .agg(s=col("v").sum(), n=count())
        .collect(spill=True)
    )
    in_memory = bt.from_arrow(table).group_by().agg(s=col("v").sum(), n=count()).collect()
    assert spilled.to_pylist() == in_memory.to_pylist()


def test_spill_with_stddev():
    # Mergeable 3-column state (var/stddev) survives partition-and-spill.
    factory, schema, table = _streaming_dataset()
    spilled = (
        bt.from_batches(factory, schema)
        .group_by("k")
        .agg(sd=col("v").std())
        .collect(spill=True, num_partitions=8)
    )
    in_memory = bt.from_arrow(table).group_by("k").agg(sd=col("v").std()).collect()
    assert _norm(spilled) == _norm(in_memory)


def test_spill_list_state_aggregates():
    # median + n_unique carry a per-group `ListArray` partial state; this verifies
    # that variable-length state survives the Arrow-IPC spill round-trip and merges
    # correctly per bucket.
    factory, schema, table = _streaming_dataset()
    spilled = (
        bt.from_batches(factory, schema)
        .group_by("k")
        .agg(m=col("v").median(), nd=col("v").n_unique())
        .collect(spill=True, num_partitions=8)
    )
    in_memory = (
        bt.from_arrow(table)
        .group_by("k")
        .agg(m=col("v").median(), nd=col("v").n_unique())
        .collect()
    )
    assert _norm(spilled) == _norm(in_memory)


def _join_datasets(n_batches=80, groups=2000):
    rng = np.random.default_rng(1)
    left = [
        pa.record_batch(
            {
                "k": rng.integers(0, groups, 1000).astype("int64"),
                "v": rng.integers(0, 100, 1000).astype("int64"),
            }
        )
        for _ in range(n_batches)
    ]
    right = [
        pa.record_batch(
            {
                "k": np.arange(groups, dtype="int64"),
                "name": [f"n{i}" for i in range(groups)],
            }
        )
    ]
    lt, rt = pa.Table.from_batches(left), pa.Table.from_batches(right)
    return (lambda: iter(left)), left[0].schema, (lambda: iter(right)), right[0].schema, lt, rt


@pytest.mark.parametrize("how", ["inner", "left", "right"])
def test_spill_join_matches_in_memory(how):
    lf, ls, rf, rs, lt, rt = _join_datasets()
    spilled = (
        bt.from_batches(lf, ls)
        .join(bt.from_batches(rf, rs), on="k", how=how)
        .collect(spill=True, num_partitions=16)
    )
    in_memory = bt.from_arrow(lt).join(bt.from_arrow(rt), on="k", how=how).collect()
    assert _norm(spilled) == _norm(in_memory)


@pytest.mark.parametrize("num_partitions", [1, 8, 64])
def test_spill_join_partition_invariant(num_partitions):
    lf, ls, rf, rs, lt, rt = _join_datasets()
    spilled = (
        bt.from_batches(lf, ls)
        .join(bt.from_batches(rf, rs), on="k")
        .collect(spill=True, num_partitions=num_partitions)
    )
    in_memory = bt.from_arrow(lt).join(bt.from_arrow(rt), on="k").collect()
    assert _norm(spilled) == _norm(in_memory)


def _sort_dataset(n_batches=100, key_range=100000):
    rng = np.random.default_rng(2)
    batches = [
        pa.record_batch(
            {
                "k": rng.integers(0, key_range, 1000).astype("int64"),
                "v": rng.integers(0, 100, 1000).astype("int64"),
            }
        )
        for _ in range(n_batches)
    ]
    return (lambda: iter(batches)), batches[0].schema, pa.Table.from_batches(batches)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("num_partitions", [1, 16, 64])
def test_spill_sort_key_order(descending, num_partitions):
    factory, schema, table = _sort_dataset()
    spilled = (
        bt.from_batches(factory, schema)
        .sort("k", descending=descending)
        .collect(spill=True, num_partitions=num_partitions)
    )
    in_memory = bt.from_arrow(table).sort("k", descending=descending).collect()
    # The sorted key sequence is deterministic; compare it exactly, plus the
    # full row multiset (tie order among equal keys may differ).
    assert spilled.column("k").to_pylist() == in_memory.column("k").to_pylist()
    assert _norm(spilled) == _norm(in_memory)


def test_spill_sort_top_n():
    factory, schema, table = _sort_dataset()
    spilled = (
        bt.from_batches(factory, schema)
        .sort("k", descending=True)
        .limit(25)
        .collect(spill=True, num_partitions=16)
    )
    in_memory = bt.from_arrow(table).sort("k", descending=True).limit(25).collect()
    assert spilled.column("k").to_pylist() == in_memory.column("k").to_pylist()


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("num_partitions", [1, 16, 64])
def test_spill_global_window_matches_in_memory(descending, num_partitions):
    # A *global* (no PARTITION BY) window over a single plain-column ORDER BY streams
    # via ordered-bucket offsetting (range-partition → per-bucket window → offset).
    # rank/dense_rank/running-sum/count/min/max are deterministic under ties, so the
    # spilled result equals the in-memory kernel as a multiset for any bucket count.
    factory, schema, table = _sort_dataset()

    def q(ds):
        return ds.window(
            order_by=[("k", descending)],
            functions={
                "rk": "rank",
                "dr": "dense_rank",
                "s": ("sum", "v"),
                "c": ("count", "v"),
                "mn": ("min", "v"),
                "mx": ("max", "v"),
            },
        )

    spilled = q(bt.from_batches(factory, schema)).collect(spill=True, num_partitions=num_partitions)
    in_memory = q(bt.from_arrow(table)).collect()
    assert _norm(spilled) == _norm(in_memory)


def test_spill_global_window_streams_via_iter_batches():
    # iter_batches() over a global window streams the same bounded-memory pipeline.
    factory, schema, table = _sort_dataset()

    def q(ds):
        return ds.window(order_by=[("k", False)], functions={"rk": "rank", "s": ("sum", "v")})

    streamed = pa.Table.from_batches(list(q(bt.from_batches(factory, schema)).iter_batches()))
    in_memory = q(bt.from_arrow(table)).collect()
    assert _norm(streamed) == _norm(in_memory)
