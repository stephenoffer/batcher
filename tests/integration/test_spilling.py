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


@pytest.mark.parametrize("group", ["k", None])
def test_streaming_partial_aggregate_matches_whole_partition(group):
    """The map-side streaming fold (per-chunk partial → combine) equals one partial over
    the whole partition — the mergeable invariant that lets the map side aggregate a
    partition without ever materializing it (the #1 distributed memory peak)."""
    import json

    import batcher._native as nat

    from batcher.config import active_config
    from batcher.dist.executor import _relabel_single_source
    from batcher.dist.executors.partition_io import streaming_partial_aggregate

    rng = np.random.default_rng(5)
    batches = [
        pa.record_batch(
            {
                "k": rng.integers(0, 50, 2000).astype("int64"),
                "v": rng.integers(0, 100, 2000).astype("int64"),
            }
        )
        for _ in range(6)
    ]
    ds = bt.from_arrow(pa.Table.from_batches(batches))
    grouped = ds.group_by(group) if group else ds.group_by()
    agg = grouped.agg(s=col("v").sum(), n=count(), mx=col("v").max())._plan
    gk = json.dumps([{"expr": k.expr.to_ir(), "alias": k.alias} for k in agg.group_keys])
    aj = json.dumps([s.agg.to_ir(s.alias) for s in agg.aggregates])
    map_plan, _ = _relabel_single_source(agg.input)
    map_ir = json.dumps(map_plan.to_ir())
    cfg = active_config().engine_config_json()

    # chunk_bytes=1 forces a fold per batch (six chunks); the finalized result must equal
    # the single-node aggregate over the whole table.
    partial = streaming_partial_aggregate(nat, map_ir, gk, aj, iter(batches), cfg, 1)
    streamed = pa.Table.from_batches([nat.combine_finalize(gk, aj, [partial])])
    single = grouped.agg(s=col("v").sum(), n=count(), mx=col("v").max()).collect()
    assert _norm(streamed) == _norm(single)


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


@pytest.mark.parametrize("codec", ["lz4", "zstd", None])
def test_spill_result_invariant_under_compression(codec):
    # Spill compression is perf-only: the result must be identical whether the
    # spilled IPC streams are uncompressed, LZ4, or ZSTD (the read path
    # auto-detects). Covers value-list state (median) + constant state (sum) so the
    # codec threads through every spill path.
    from batcher.config import Config, MemoryConfig, config_context

    factory, schema, table = _streaming_dataset()
    agg = {"s": col("v").sum(), "m": col("v").median()}
    cfg = Config().replace(memory=MemoryConfig(spill_compression=codec))
    with config_context(cfg):
        spilled = (
            bt.from_batches(factory, schema)
            .group_by("k")
            .agg(**agg)
            .collect(spill=True, num_partitions=8)
        )
    in_memory = bt.from_arrow(table).group_by("k").agg(**agg).collect()
    assert _norm(spilled) == _norm(in_memory)


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


def _find_join(plan):
    """The first `Join` node at or under `plan` (an outer join wraps it in a Project)."""
    from batcher.plan.logical import Join

    node = plan
    while node is not None and not isinstance(node, Join):
        node = getattr(node, "input", None)
    return node


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_reduce_join_paths_spilling_matches_direct(tmp_path, how):
    """The bounded-memory grace reducer (re-partition both buckets to disk, join one
    sub-bucket pair at a time) equals a single in-memory join over the whole bucket —
    under skew (one very hot key) and null keys, with each side split across mapper
    files. This is the per-reducer spill that keeps a skewed shuffle join off OOM."""
    import json

    import batcher._native as nat

    from batcher.config import active_config
    from batcher.dist.executors.join import _join_reducer_ir
    from batcher.dist.shuffle_io import write_ipc
    from batcher.dist.spill_breakers import reduce_join_paths_spilling

    rng = np.random.default_rng(7)
    left_files, all_left = [], []
    for m in range(3):  # three "mapper" contributions; key 3 is hot, file 1 has nulls
        ks = np.concatenate([np.full(400, 3), rng.integers(0, 50, 200)]).astype("int64")
        rng.shuffle(ks)
        karr = pa.array(
            [None if (m == 1 and i < 5) else int(k) for i, k in enumerate(ks)], type=pa.int64()
        )
        b = pa.record_batch({"k": karr, "v": rng.integers(0, 100, len(ks)).astype("int64")})
        all_left.append(b)
        left_files.append(write_ipc([b], str(tmp_path / f"l{m}.arrow")))
    right_files, all_right = [], []
    for m, (lo, hi) in enumerate([(0, 25), (25, 55)]):  # unique keys, two mapper files
        b = pa.record_batch(
            {
                "k": pa.array(range(lo, hi), type=pa.int64()),
                "name": [f"n{i}" for i in range(lo, hi)],
            }
        )
        all_right.append(b)
        right_files.append(write_ipc([b], str(tmp_path / f"r{m}.arrow")))

    joined = bt.from_arrow(pa.Table.from_batches(all_left)).join(
        bt.from_arrow(pa.Table.from_batches(all_right)), on="k", how=how
    )
    join = _find_join(joined._plan)
    join_ir = json.dumps(_join_reducer_ir(join))
    cfg = active_config().engine_config_json()

    direct = nat.execute_plan(join_ir, [all_left, all_right], cfg)
    graced = reduce_join_paths_spilling(
        join_ir,
        list(join.left_keys),
        list(join.right_keys),
        left_files,
        right_files,
        str(tmp_path),
        4,
        cfg,
    )
    # None-safe multiset comparison (outer/left joins null-extend, so rows hold None).
    schema = (direct or graced)[0].schema
    norm = lambda t: sorted(repr(r) for r in t.to_pylist())  # noqa: E731
    assert norm(pa.Table.from_batches(graced, schema)) == norm(
        pa.Table.from_batches(direct, schema)
    )


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


def test_spill_global_value_list_matches_in_memory():
    # A *global* (no GROUP BY) value-list aggregate must spill correctly: the mixed
    # path (median + a constant-state aggregate) and the lone paths (n_unique, mode)
    # all have an empty group-key set, which the bounded spill aligns without a group
    # sort. Regression for the empty-group-key spill crash exposed by auto-budgeting.
    factory, schema, table = _streaming_dataset()

    def q(ds):
        return ds.group_by().agg(
            m=col("v").median(),
            s=col("v").sum(),
            d=col("v").n_unique(),
            md=col("v").mode(),
        )

    spilled = q(bt.from_batches(factory, schema)).collect(spill=True, num_partitions=16)
    in_memory = q(bt.from_arrow(table)).collect()
    assert _norm(spilled) == _norm(in_memory)


def test_spill_empty_group_by_matches_in_memory():
    # An empty input (0 rows) has no working set, so it must take the in-memory path
    # and yield zero groups even when spilling is requested — the grace path's
    # concat-of-nothing crash is the regression this guards.
    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    empty = pa.table({"k": [], "v": []}, schema=schema)
    spilled = (
        bt.from_arrow(empty).group_by("k").agg(s=col("v").sum(), n=count()).collect(spill=True)
    )
    assert spilled.num_rows == 0
    assert spilled.column_names == ["k", "s", "n"]
