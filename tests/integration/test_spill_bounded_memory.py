"""Out-of-core spill stays correct and leaves no scratch behind — the reliability
contract for the bounded-memory input tap (`_iter_spill_morsels`).

These cover the pieces the existing spill suite did not: a source that emits
thousands of *tiny* batches (so the input tap's coalescing runs) and a single
*over-large* batch (so its splitting runs) — both must match the in-memory result;
that the per-query scratch dir is removed on both the success and the error path (no
disk leak under repeated/failing queries); and that concurrent spilling queries
sharing one scratch root do not corrupt each other's results.
"""

from __future__ import annotations

import concurrent.futures
import json
import os

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count
from batcher.config import Config, MemoryConfig, config_context

pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration


def _one_big_batch(n=2_000_000, groups=300_000):
    rng = np.random.default_rng(7)
    return pa.record_batch(
        {
            "k": rng.integers(0, groups, n).astype("int64"),
            "v": rng.integers(0, 100, n).astype("int64"),
        }
    )


def _norm(t: pa.Table) -> list:
    return sorted(tuple(r.values()) for r in t.to_pylist())


def _leftover_dirs(root: str) -> list[str]:
    return [d for d in os.listdir(root) if d.startswith(("batcher_spill_", "batcher_win_spill_"))]


def _tiny_batches(n=400_000, groups=60_000, batch_rows=256):
    rng = np.random.default_rng(11)
    ks = rng.integers(0, groups, n).astype("int64")
    vs = rng.integers(0, 100, n).astype("int64")
    return [
        pa.record_batch({"k": ks[i : i + batch_rows], "v": vs[i : i + batch_rows]})
        for i in range(0, n, batch_rows)
    ]


def test_tiny_batch_source_spill_matches_in_memory():
    # A source that emits thousands of tiny batches exercises the coalescing half of
    # the input tap. Aggregate/distinct spill over 256-row batches must equal the
    # in-memory result — coalescing reshapes the row stream, it must not change it.
    tiny = _tiny_batches()
    table = pa.Table.from_batches(tiny)

    agg_spilled = (
        bt.from_batches(lambda: iter(tiny), tiny[0].schema)
        .group_by("k")
        .agg(s=col("v").sum(), n=count())
        .collect(spill=True, num_partitions=16)
    )
    agg_mem = bt.from_arrow(table).group_by("k").agg(s=col("v").sum(), n=count()).collect()
    assert _norm(agg_spilled) == _norm(agg_mem)

    distinct_spilled = (
        bt.from_batches(lambda: iter(tiny), tiny[0].schema)
        .distinct()
        .collect(spill=True, num_partitions=16)
    )
    distinct_mem = bt.from_arrow(table).distinct().collect()
    assert _norm(distinct_spilled) == _norm(distinct_mem)


def test_tiny_batch_source_sort_matches_in_memory():
    tiny = _tiny_batches()
    table = pa.Table.from_batches(tiny)
    spilled = (
        bt.from_batches(lambda: iter(tiny), tiny[0].schema)
        .sort("v")
        .collect(spill=True, num_partitions=8)
    )
    in_memory = bt.from_arrow(table).sort("v").collect()
    assert spilled.column("v").to_pylist() == in_memory.column("v").to_pylist()


def test_empty_source_spill():
    # The input tap yields no morsels for an all-empty source; the spill executors
    # must still return the right shape (0 grouped rows; exactly one row for a global
    # aggregate). A dataset of any size includes the zero-row one.
    empty = [
        pa.record_batch({"k": pa.array([], type=pa.int64()), "v": pa.array([], type=pa.int64())})
        for _ in range(4)
    ]
    schema = empty[0].schema
    grouped = (
        bt.from_batches(lambda: iter(empty), schema)
        .group_by("k")
        .agg(s=col("v").sum())
        .collect(spill=True, num_partitions=8)
    )
    assert grouped.num_rows == 0
    global_agg = bt.from_batches(lambda: iter(empty), schema).agg(n=count()).collect(spill=True)
    assert global_agg.to_pydict() == {"n": [0]}


def test_single_row_source_spill():
    one = [pa.record_batch({"k": pa.array([7]), "v": pa.array([42])})]
    out = (
        bt.from_batches(lambda: iter(one), one[0].schema)
        .group_by("k")
        .agg(s=col("v").sum())
        .collect(spill=True, num_partitions=8)
    )
    assert out.to_pydict() == {"k": [7], "s": [42]}


def test_ultra_wide_rows_spill_matches_in_memory():
    # Rows far larger than the input-chunk ceiling (a ~1 MiB blob per row): the tap
    # must split them (never accumulate the whole run in memory) and still produce the
    # correct result. This is the many-small vs one-huge boundary at the row level.
    wide = [
        pa.record_batch(
            {"k": pa.array((np.arange(40) % 8).tolist()), "b": pa.array(["z" * 1_000_000] * 40)}
        )
        for _ in range(6)
    ]
    schema = wide[0].schema
    spilled = (
        bt.from_batches(lambda: iter(wide), schema)
        .group_by("k")
        .agg(n=count())
        .collect(spill=True, num_partitions=8)
    )
    in_memory = bt.from_arrow(pa.Table.from_batches(wide)).group_by("k").agg(n=count()).collect()
    assert _norm(spilled) == _norm(in_memory)


def test_single_oversized_batch_aggregate_matches_in_memory():
    # from_batches with ONE 2M-row batch: the source yields a batch far over the
    # input ceiling, so the spilling path must re-morselize and split it. Result must
    # still equal the in-memory aggregation.
    big = _one_big_batch()

    def agg(ds):
        return ds.group_by("k").agg(s=col("v").sum(), n=count(), mx=col("v").max())

    spilled = agg(bt.from_batches(lambda: iter([big]), big.schema)).collect(
        spill=True, num_partitions=64
    )
    in_memory = agg(bt.from_arrow(pa.Table.from_batches([big]))).collect()
    assert _norm(spilled) == _norm(in_memory)


def test_single_oversized_batch_sort_matches_in_memory():
    big = _one_big_batch()
    spilled = (
        bt.from_batches(lambda: iter([big]), big.schema)
        .sort("v")
        .collect(spill=True, num_partitions=32)
    )
    in_memory = bt.from_arrow(pa.Table.from_batches([big])).sort("v").collect()
    # Sort is order-sensitive: compare the sorted key column position-for-position.
    assert spilled.column("v").to_pylist() == in_memory.column("v").to_pylist()


def test_scratch_is_cleaned_after_a_spilling_query(tmp_path):
    root = str(tmp_path)
    big = _one_big_batch()
    cfg = Config().replace(memory=MemoryConfig(spill_dir=root, max_memory_bytes=1))
    with config_context(cfg):
        out = (
            bt.from_batches(lambda: iter([big]), big.schema)
            .group_by("k")
            .agg(s=col("v").sum())
            .collect(spill=True, num_partitions=32)
        )
    assert out.num_rows > 0
    assert _leftover_dirs(root) == []  # per-query scratch dir removed


def test_scratch_is_cleaned_on_the_error_path(tmp_path, monkeypatch):
    # A failure in the reduce phase must still trigger the `finally` cleanup — a
    # spilling query that raises mid-flight must not leak its on-disk scratch.
    import batcher._native as nat

    root = str(tmp_path)
    big = _one_big_batch()
    real_combine = nat.combine_finalize
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] >= 1:
            raise RuntimeError("injected reduce failure")
        return real_combine(*a, **k)

    monkeypatch.setattr(nat, "combine_finalize", boom)
    cfg = Config().replace(memory=MemoryConfig(spill_dir=root, max_memory_bytes=1))
    with config_context(cfg), pytest.raises(RuntimeError, match="injected reduce failure"):
        (
            bt.from_batches(lambda: iter([big]), big.schema)
            .group_by("k")
            .agg(s=col("v").sum())
            .collect(spill=True, num_partitions=32)
        )
    assert _leftover_dirs(root) == []  # scratch removed despite the mid-flight error


def _dist_json(group_cols, aggs):
    """The (group_keys_json, aggregates_json) the distributed aggregate ships to workers."""
    gk = json.dumps([{"expr": col(c).to_ir(), "alias": c} for c in group_cols])
    ag = json.dumps([spec.to_ir(alias) for alias, spec in aggs])
    return gk, ag


def test_distributed_reduce_spills_high_cardinality(tmp_path):
    # The distributed reducer folds its shuffle partials into one running group state; a
    # high-cardinality GROUP BY makes that state exceed a worker's RAM. `combine_finalize_spilling`
    # reads the shuffle files one at a time and grace-partitions them to disk, so the reducer
    # completes out of core. Drive the real FFI end-to-end (native partial/partition + on-disk
    # Arrow-IPC shuffle files) and assert the spilling reduce equals the in-memory reduce and the
    # single-node aggregate — the mergeable invariant holding out-of-core, no Ray needed.
    from batcher.dist.shuffle_io import read_ipc, write_ipc

    nat = pytest.importorskip("batcher._native", reason="native engine not built")

    # ~120k rows, ~mostly-distinct keys → a reducer's merged state is large relative to a tiny
    # budget, forcing real grace partitioning + its recursion.
    rng = np.random.default_rng(3)
    n = 120_000
    ks = rng.integers(0, 100_000, n).astype("int64")
    vs = rng.integers(0, 1000, n).astype("int64")
    table = pa.table({"k": ks, "v": vs})
    batches = table.to_batches(max_chunksize=8_192)

    gk_json, ag_json = _dist_json(["k"], [("s", col("v").sum()), ("n", count())])

    # MAP: partial-aggregate each chunk, then hash-shuffle its partials to `n_reducers`.
    n_reducers = 5
    reducer_paths: list[list[str]] = [[] for _ in range(n_reducers)]
    work = str(tmp_path)
    for mi, b in enumerate(batches):
        partial = nat.partial_aggregate(gk_json, ag_json, [b])
        buckets = nat.partition_batches([partial], [0], n_reducers)
        for r, bucket in enumerate(buckets):
            if bucket:
                p = os.path.join(work, f"m{mi}_r{r}.arrow")
                write_ipc(bucket, p)
                reducer_paths[r].append(p)

    def _reduce(spilling: bool) -> list:
        rows: list = []
        for paths in reducer_paths:
            if not paths:
                continue
            if spilling:
                # A 1 KiB budget forces the deepest out-of-core path for every non-trivial reducer.
                out = nat.combine_finalize_spilling(gk_json, ag_json, list(paths), 1024, work, None)
            else:
                partials = [b for p in paths for b in read_ipc(p)]
                out = nat.combine_finalize(gk_json, ag_json, partials)
            if out.num_rows:
                rows.extend(pa.Table.from_batches([out]).to_pylist())
        return sorted(tuple(r.values()) for r in rows)

    spilled = _reduce(spilling=True)
    in_memory = _reduce(spilling=False)
    single_node = _norm(
        bt.from_arrow(table).group_by("k").agg(s=col("v").sum(), n=count()).collect()
    )
    assert spilled == in_memory
    assert spilled == single_node


def test_distributed_reduce_spill_all_empty_reducer(tmp_path):
    # A reducer that receives only empty keyed shards must spill to a zero-row result, not
    # raise — the out-of-core path must tolerate the all-empty bucket the in-memory fold does.
    from batcher.dist.shuffle_io import write_ipc

    nat = pytest.importorskip("batcher._native", reason="native engine not built")
    gk_json, ag_json = _dist_json(["k"], [("s", col("v").sum())])
    empty = pa.record_batch(
        {"k": pa.array([], type=pa.int64()), "v": pa.array([], type=pa.int64())}
    )
    partial = nat.partial_aggregate(gk_json, ag_json, [empty])
    p = os.path.join(str(tmp_path), "empty.arrow")
    write_ipc([partial], p)
    out = nat.combine_finalize_spilling(gk_json, ag_json, [p], 1, str(tmp_path), None)
    assert out.num_rows == 0


def test_concurrent_spilling_queries_share_a_root_without_corruption(tmp_path):
    # Each query gets its own per-query subdir under the shared root; running several
    # at once must not cross-contaminate results or leak scratch.
    root = str(tmp_path)
    batches = {g: _one_big_batch(groups=g) for g in (100_000, 200_000, 300_000)}
    expected = {
        g: _norm(
            bt.from_arrow(pa.Table.from_batches([b])).group_by("k").agg(s=col("v").sum()).collect()
        )
        for g, b in batches.items()
    }
    cfg = Config().replace(memory=MemoryConfig(spill_dir=root))

    def run(g):
        with config_context(cfg):
            return g, _norm(
                bt.from_batches(lambda b=batches[g]: iter([b]), batches[g].schema)
                .group_by("k")
                .agg(s=col("v").sum())
                .collect(spill=True, num_partitions=32)
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results = dict(ex.map(run, batches))
    for g in batches:
        assert results[g] == expected[g]
    assert _leftover_dirs(root) == []
