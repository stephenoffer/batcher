"""Distributed execution equals single-node execution.

The distributed aggregation reuses the engine's mergeable primitives across Ray
workers (disk Arrow-IPC shuffle), so its result must be identical to single-node.
These tests are the cross-machine analogue of the partition-independence invariant.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    import ray

    ray.init(num_cpus=4, include_dashboard=False, logging_level="ERROR", ignore_reinit_error=True)
    yield
    ray.shutdown()


def _data():
    rng = np.random.default_rng(7)
    n = 200_000
    return pa.table({"k": rng.integers(0, 30, n), "v": rng.integers(0, 100, n).astype("int64")})


def _norm(table: pa.Table) -> set:
    return {
        tuple(round(v, 6) if isinstance(v, float) else v for v in row.values())
        for row in table.to_pylist()
    }


@pytest.mark.parametrize("transport", ["disk", "flight"])
def test_distributed_list_state_aggregates(transport):
    """median + n_unique carry a per-group ListArray partial state; verify it
    survives the disk AND Flight shuffle and merges to the single-node result."""
    t = _data()

    def q(ds, **kw):
        return ds.group_by("k").agg(m=col("v").median(), nd=col("v").n_unique()).collect(**kw)

    single = q(bt.from_arrow(t))
    distrib = q(bt.from_arrow(t), distributed=True, num_workers=4, transport=transport)
    assert _norm(single) == _norm(distrib)


def test_distributed_grouped_aggregate_matches_single_node():
    t = _data()

    def q(ds):
        return ds.group_by("k").agg(
            s=col("v").sum(), n=count(), a=col("v").mean(), hi=col("v").max()
        )

    single = q(bt.from_arrow(t)).collect()
    distrib = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4)
    assert _norm(single) == _norm(distrib)


def test_distributed_global_aggregate_matches_single_node():
    t = _data()

    def q(ds):
        return ds.group_by().agg(s=col("v").sum(), n=count(), a=col("v").mean())

    single = q(bt.from_arrow(t)).collect().to_pydict()
    distrib = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4).to_pydict()
    assert single == distrib


def test_distributed_with_post_aggregation_ops():
    t = _data()

    def q(ds):
        return ds.group_by("k").agg(s=col("v").sum()).sort("s", descending=True).limit(5)

    single = q(bt.from_arrow(t)).collect().to_pylist()
    distrib = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4).to_pylist()
    assert single == distrib  # ordered (sort+limit), must match exactly


def test_distributed_distinct_matches_single_node():
    # DISTINCT dedups across workers via the aggregate shuffle (group-by-all-cols).
    t = _data()

    def q(ds):
        return ds.select("k", "v").distinct()

    single = q(bt.from_arrow(t)).collect()
    distrib = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4)
    assert _norm(single) == _norm(distrib)


def test_distributed_distinct_with_filter_and_post_sort():
    # Filter below the DISTINCT (breaker-free input) and a sort above it (post-op).
    t = _data()

    def q(ds):
        return ds.filter(col("v") > 50).select("k").distinct().sort("k")

    single = q(bt.from_arrow(t)).collect().to_pylist()
    distrib = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4).to_pylist()
    assert single == distrib  # sorted → exact match


def test_distributed_window_partition_aggregate_matches_single_node():
    # Whole-partition window aggregate: rows shuffle by partition key `k`, each
    # partition is computed whole on one reducer, the union equals single-node.
    t = _data()

    def q(ds):
        return ds.window(partition_by=["k"], functions={"tot": ("sum", "v"), "hi": ("max", "v")})

    single = q(bt.from_arrow(t)).collect()
    distrib = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4)
    assert _norm(single) == _norm(distrib)


def test_distributed_window_running_aggregate_matches_single_node():
    # Running (ORDER BY) window aggregate — order within each partition must be
    # intact on the reducer, which it is because the whole partition lands there.
    t = _data()

    def q(ds):
        return ds.window(
            partition_by=["k"],
            order_by=[("v", False)],
            functions={"rn": "row_number", "rs": ("sum", "v")},
        )

    single = q(bt.from_arrow(t)).collect()
    distrib = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4)
    assert _norm(single) == _norm(distrib)


@pytest.mark.parametrize("transport", ["disk", "flight"])
def test_distributed_streaming_source_partitions_in_bounded_memory(transport):
    """A non-splittable streaming source distributes correctly without the driver
    materializing the whole source.

    Previously the driver did `pa.Table.from_batches(read_source(...))`, holding the
    entire source at once (driver OOM on a larger-than-RAM stream). The partitioner
    now streams `iter_source` one batch at a time; a live-batch counter in the
    factory proves the driver never holds more than a single batch concurrently, and
    the result still equals single-node.
    """
    t = _data()
    chunks = t.to_batches(max_chunksize=8_192)  # many small batches to stream
    live = 0
    peak = 0

    def factory():
        nonlocal live, peak

        def gen():
            nonlocal live, peak
            for b in chunks:
                live += 1
                peak = max(peak, live)
                yield b
                live -= 1

        return gen()

    def q(ds):
        return ds.group_by("k").agg(s=col("v").sum(), n=count())

    single = q(bt.from_arrow(t)).collect()
    distrib = q(bt.from_batches(factory, t.schema)).collect(
        distributed=True, num_workers=4, transport=transport
    )
    assert _norm(single) == _norm(distrib)
    # The disk path streams into IPC files one batch at a time; the Flight path
    # accumulates per-worker references but still pulls the source one batch at a
    # time (no whole-source Table concat). Either way the generator is never driven
    # to hold more than one batch live on the driver.
    assert peak == 1, f"driver held {peak} batches at once; expected streaming (1)"


def test_distributed_multikey_sort_matches_single_node():
    # Leading key `k` range-partitions; ties broken by `v` within each bucket.
    # Lots of ties on `k` (0..29 over 200k rows) stress the equal-value boundary.
    t = _data()

    def q(ds):
        return ds.sort("k", "v", descending=[False, True])

    single = q(bt.from_arrow(t)).collect().to_pylist()
    distrib = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4).to_pylist()
    assert single == distrib  # globally ordered → exact, position-by-position match


def test_distributed_union_all_matches_single_node():
    # UNION ALL of two aggregated branches: each branch distributes, then concat.
    t = _data()

    def q(ds_factory):
        a = ds_factory().filter(col("v") < 50).group_by("k").agg(s=col("v").sum())
        b = ds_factory().filter(col("v") >= 50).group_by("k").agg(s=col("v").sum())
        return a.union(b)

    single = q(lambda: bt.from_arrow(t)).collect()
    distrib = q(lambda: bt.from_arrow(t)).collect(distributed=True, num_workers=4)
    assert _norm(single) == _norm(distrib)


def test_distributed_union_distinct_matches_single_node():
    t = _data()

    def q(ds_factory):
        a = ds_factory().select("k")
        b = ds_factory().filter(col("v") > 50).select("k")
        return a.union(b, distinct=True)

    single = q(lambda: bt.from_arrow(t)).collect()
    distrib = q(lambda: bt.from_arrow(t)).collect(distributed=True, num_workers=4)
    assert _norm(single) == _norm(distrib)


def _join_data():
    rng = np.random.default_rng(11)
    n = 100_000
    left = pa.table({"k": rng.integers(0, 100, n), "lv": rng.integers(0, 50, n).astype("int64")})
    right = pa.table({"k": np.arange(100), "label": [f"g{i}" for i in range(100)]})
    return bt.from_arrow(left), bt.from_arrow(right)


def _rowset(table: pa.Table) -> set:
    cols = table.column_names
    return {tuple(r[c] for c in cols) for r in table.to_pylist()}


@pytest.mark.parametrize("how", ["inner", "left", "right"])
def test_distributed_join_matches_single_node(how):
    left, right = _join_data()
    single = left.join(right, on="k", how=how).collect()
    distrib = left.join(right, on="k", how=how).collect(distributed=True, num_workers=4)
    assert _rowset(single) == _rowset(distrib)


def test_distributed_broadcast_runtime_guard_falls_back_to_shuffle(monkeypatch):
    """The planner picks broadcast from an estimate, but the distributed executor's
    runtime guard sees the materialized build side exceed the (tiny) configured
    threshold and falls back to a shuffle join — same result, no driver OOM."""
    import dataclasses

    from batcher.config import active_config, config_context
    from batcher.kyber.rules import selection

    left, right = _join_data()
    single = left.join(right, on="k").collect()
    # Planner threshold huge → it marks the join broadcast; config threshold tiny →
    # the executor's runtime guard rejects the actual build side and shuffles instead.
    monkeypatch.setattr(selection, "_broadcast_max_bytes", lambda: 1 << 40)
    cfg = active_config()
    guarded_cfg = cfg.replace(optimizer=dataclasses.replace(cfg.optimizer, broadcast_max_bytes=1))
    with config_context(guarded_cfg):
        guarded = left.join(right, on="k").collect(distributed=True, num_workers=4)
    assert _rowset(guarded) == _rowset(single)


@pytest.mark.parametrize("how", ["inner", "left", "semi", "anti"])
def test_distributed_broadcast_equals_shuffle_and_single_node(how, monkeypatch):
    # Tiny right side → the planner marks the join broadcast, so the distributed
    # run takes the no-shuffle broadcast path. Forcing the byte threshold to -1
    # makes the same query take the co-partition shuffle path. All three (broadcast
    # distributed, shuffle distributed, single-node) must produce the same rows.
    from batcher.kyber.rules import selection

    left, right = _join_data()
    single = left.join(right, on="k", how=how).collect()
    bcast = left.join(right, on="k", how=how).collect(distributed=True, num_workers=4)

    monkeypatch.setattr(selection, "_broadcast_max_bytes", lambda: -1)
    shuffled = left.join(right, on="k", how=how).collect(distributed=True, num_workers=4)

    assert _rowset(bcast) == _rowset(single)
    assert _rowset(shuffled) == _rowset(single)


@pytest.mark.parametrize("transport", ["disk", "flight"])
def test_distributed_iter_batches_streams_result_off_driver(transport):
    # iter_batches(distributed=True) runs the breaker with materialize=False so the
    # result stays partitioned on the workers, then streams it back one reducer bucket
    # at a time — the driver never holds the whole result. Rows must equal collect, and
    # the result comes back in multiple buckets (not one collected table).
    t = _data()
    single = bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=count()).collect()
    batches = list(
        bt.from_arrow(t)
        .group_by("k")
        .agg(s=col("v").sum(), n=count())
        .iter_batches(distributed=True, num_workers=4, transport=transport)
    )
    got = pa.Table.from_batches(batches) if batches else single.slice(0, 0)
    assert _rowset(got) == _rowset(single)
    assert len(batches) >= 1  # streamed back per reducer bucket, not one driver table


def test_distributed_iter_batches_join_equals_collect():
    # The same streaming terminal for a distributed join (a multi-source breaker).
    left, right = _join_data()
    single = left.join(right, on="k").collect()
    batches = list(left.join(right, on="k").iter_batches(distributed=True, num_workers=4))
    got = pa.Table.from_batches(batches) if batches else single.slice(0, 0)
    assert _rowset(got) == _rowset(single)


def test_distributed_broadcast_join_correct_with_speculation_enabled():
    # The broadcast join's probe barrier now goes through `gather_with_backups`, so a
    # slow probe task can be backed up under speculation. The probe tasks are
    # deterministic (a left chunk joined against the full broadcast right), so the
    # result is identical to single-node whether or not a backup fires.
    from batcher.config import DistributedConfig

    left, right = _join_data()
    single = left.join(right, on="k").collect()
    scoped = bt.Config().replace(distributed=DistributedConfig(speculation_max_backups=2))
    with bt.config_context(scoped):
        distrib = left.join(right, on="k").collect(distributed=True, num_workers=4)
    assert _rowset(distrib) == _rowset(single)


@pytest.mark.parametrize("how", ["inner", "left", "semi", "anti"])
def test_distributed_skew_join_salting_equals_single_node(how, monkeypatch):
    # A skewed join: key 0 dominates the left (probe) side. With skew salting on, the
    # hot key's probe rows fan across reducers while its build rows are replicated to
    # all of them, so the hot key never overloads one reducer. The result must still
    # equal single-node (salting only moves work between reducers, never the relation).
    # Force the shuffle path (broadcast threshold = -1) so salting is actually exercised.
    from batcher.config import DistributedConfig
    from batcher.kyber.rules import selection

    monkeypatch.setattr(selection, "_broadcast_max_bytes", lambda: -1)

    rng = np.random.default_rng(7)
    # Left: key 0 is hot (1000 rows ≈ 33%); keys 1..20 are cold (100 each).
    lk = np.concatenate([np.zeros(1000, "int64"), np.repeat(np.arange(1, 21), 100)])
    left = bt.from_arrow(pa.table({"k": lk, "lv": rng.integers(0, 10, lk.size).astype("int64")}))
    # Right: key 0 has a handful of rows; keys 1..20 ~20 each.
    rk = np.concatenate([np.zeros(10, "int64"), np.repeat(np.arange(1, 21), 20)])
    right = bt.from_arrow(pa.table({"k": rk, "rv": rng.integers(0, 10, rk.size).astype("int64")}))

    single = left.join(right, on="k", how=how).collect()

    scoped = bt.Config().replace(
        distributed=DistributedConfig(skew_join_salt=4, skew_join_fraction=0.1)
    )
    with bt.config_context(scoped):
        salted = left.join(right, on="k", how=how).collect(distributed=True, num_workers=4)

    # Row COUNT, not just the row set: salting must not duplicate (or drop) rows. The
    # set comparison alone can't catch duplication when the data has repeated rows
    # (which it does here), so assert the multiset size too.
    assert single.num_rows == salted.num_rows
    assert _rowset(single) == _rowset(salted)


@pytest.mark.parametrize("keys", [["k"], ["k", "g"]])
def test_distributed_join_then_aggregate_fused(keys):
    # An aggregate grouped by (a superset of) the join key over an inner join is
    # distributed by reusing the join's co-partitioning: each reducer joins AND
    # aggregates its bucket, with no second shuffle and no full-join collection on the
    # driver. Every group shares one join-key value → it lies in one bucket → the
    # per-bucket aggregate is complete, so the union equals single-node — even for a
    # non-mergeable aggregate like median.
    fact = bt.from_arrow(
        pa.table(
            {
                "k": [1, 1, 2, 2, 3, 1, 2],
                "g": ["a", "a", "b", "b", "c", "a", "b"],
                "v": [10, 20, 30, 40, 50, 60, 70],
            }
        )
    )
    dim = bt.from_arrow(pa.table({"k": [1, 2, 3], "d": [100, 200, 300]}))

    def q():
        return (
            fact.join(dim, on="k")
            .group_by(*keys)
            .agg(s=col("v").sum(), hi=col("v").max(), med=col("v").median())
        )

    single = q().collect()
    distrib = q().collect(distributed=True, num_workers=4)
    assert _rowset(single) == _rowset(distrib)


@pytest.mark.parametrize("how", ["inner", "semi"])
def test_distributed_runtime_bloom_join_equals_single_node(how, monkeypatch):
    # A selective join: the probe (left) side ranges over 1000 keys but the build
    # (right) side has only 0..49, so a bloom over the build keys prunes ~95% of probe
    # rows before the shuffle. The result must still equal single-node — the bloom has
    # no false negatives, so pruning only drops provably-non-matching rows. Nulls in
    # the probe key (never matched by an equi-join) must also be handled.
    from batcher.config import DistributedConfig
    from batcher.kyber.rules import selection

    monkeypatch.setattr(selection, "_broadcast_max_bytes", lambda: -1)  # force the shuffle path

    rng = np.random.default_rng(11)
    lk = rng.integers(0, 1000, 5000).astype("int64")
    left_tbl = pa.table(
        {
            "k": pa.array([None if i % 500 == 0 else int(v) for i, v in enumerate(lk)], pa.int64()),
            "lv": pa.array(rng.integers(0, 10, lk.size).astype("int64")),
        }
    )
    keys = np.arange(50, dtype="int64")
    right_tbl = pa.table({"k": pa.array(keys), "rv": pa.array(keys)})
    left, right = bt.from_arrow(left_tbl), bt.from_arrow(right_tbl)

    single = left.join(right, on="k", how=how).collect()
    scoped = bt.Config().replace(distributed=DistributedConfig(runtime_bloom_join=True))
    with bt.config_context(scoped):
        bloomed = left.join(right, on="k", how=how).collect(distributed=True, num_workers=4)

    assert _rowset(single) == _rowset(bloomed)


@pytest.mark.parametrize("how", ["inner", "semi"])
def test_distributed_runtime_bloom_join_multikey_equals_single_node(how, monkeypatch):
    # The runtime bloom prunes on the row-encoded *composite* key (k1, k2), not just a
    # single column. The build side covers a small region of the (k1, k2) grid; the
    # bloom drops probe rows outside it before the shuffle. Result must equal
    # single-node — multi-key membership has no false negatives either.
    from batcher.config import DistributedConfig
    from batcher.kyber.rules import selection

    monkeypatch.setattr(selection, "_broadcast_max_bytes", lambda: -1)  # force the shuffle path

    rng = np.random.default_rng(7)
    n = 5000
    left_tbl = pa.table(
        {
            "k1": pa.array(rng.integers(0, 100, n).astype("int64")),
            "k2": pa.array(rng.integers(0, 100, n).astype("int64")),
            "lv": pa.array(rng.integers(0, 10, n).astype("int64")),
        }
    )
    # Build side: only k1,k2 both < 10 → a 10×10 corner of the 100×100 probe grid.
    bk = np.arange(10, dtype="int64")
    g1, g2 = np.meshgrid(bk, bk)
    right_tbl = pa.table(
        {
            "k1": pa.array(g1.ravel()),
            "k2": pa.array(g2.ravel()),
            "rv": pa.array(np.arange(g1.size, dtype="int64")),
        }
    )
    left, right = bt.from_arrow(left_tbl), bt.from_arrow(right_tbl)

    single = left.join(right, on=["k1", "k2"], how=how).collect()
    scoped = bt.Config().replace(distributed=DistributedConfig(runtime_bloom_join=True))
    with bt.config_context(scoped):
        joined = left.join(right, on=["k1", "k2"], how=how)
        bloomed = joined.collect(distributed=True, num_workers=4)

    assert _rowset(single) == _rowset(bloomed)


@pytest.mark.parametrize("direction", ["backward", "forward"])
def test_distributed_asof_by_keys_matches_single_node(direction):
    """ASOF join with `by` keys co-partitions both sides by those keys; each bucket is
    an independent nearest-`on` match, so the union equals single-node. Includes a
    left `by` group ("D") absent from the right — its rows must be emitted with null
    right columns (left-style), proving empty-right buckets are handled."""
    rng = np.random.default_rng(41)
    n = 40_000
    syms = np.array(["A", "B", "C", "D"])
    left = pa.table(
        {
            "sym": pa.array(syms[rng.integers(0, 4, n)]),
            "ts": pa.array(np.sort(rng.integers(0, 1_000_000, n)).astype("int64")),
            "price": pa.array(rng.integers(0, 100, n).astype("int64")),
        }
    )
    m = 8_000
    right = pa.table(
        {
            # No "D" on the right → "D" left rows match nothing.
            "sym": pa.array(np.array(["A", "B", "C"])[rng.integers(0, 3, m)]),
            "ts": pa.array(np.sort(rng.integers(0, 1_000_000, m)).astype("int64")),
            "bid": pa.array(rng.integers(0, 50, m).astype("int64")),
        }
    )

    def q(ds_factory):
        return ds_factory(left).join_asof(ds_factory(right), on="ts", by="sym", direction=direction)

    single = q(bt.from_arrow).collect()
    distrib = q(bt.from_arrow).collect(distributed=True, num_workers=4)
    assert _rowset(single) == _rowset(distrib)


def test_gather_with_backups_relaunches_and_wins_straggler():
    import time

    import ray

    from batcher.carbonite.resilience import SpeculationPolicy, gather_with_backups

    @ray.remote
    def _task(i: int, delay: float) -> int:
        time.sleep(delay)
        return i

    # Task 2 is a hard straggler (5 s); the rest are fast. A backup re-issues it
    # fast and the barrier takes whichever finishes first.
    refs = [_task.remote(i, 0.05 if i != 2 else 5.0) for i in range(4)]
    relaunched: list[int] = []

    def relaunch(i: int):
        relaunched.append(i)
        return _task.remote(i, 0.05)  # the backup is fast

    pol = SpeculationPolicy(max_backups=1, min_finished_frac=0.5, straggler_factor=1.5)
    out = gather_with_backups(refs, relaunch, pol, poll_seconds=0.1)
    assert out == [0, 1, 2, 3]  # correct results, in order
    assert relaunched == [2]  # only the straggler got a backup


def test_distributed_aggregate_correct_with_speculation_enabled():
    # With speculation enabled, the distributed aggregate still equals single-node
    # (backups are deterministic; the result is identical whether or not one fires).
    from batcher.config import DistributedConfig

    t = _data()
    single = bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=count()).collect()
    scoped = bt.Config().replace(distributed=DistributedConfig(speculation_max_backups=2))
    with bt.config_context(scoped):
        distrib = (
            bt.from_arrow(t)
            .group_by("k")
            .agg(s=col("v").sum(), n=count())
            .collect(distributed=True, num_workers=4)
        )
    assert _rowset(single) == _rowset(distrib)


def test_distributed_sort_correct_with_speculation_enabled():
    # The disk sort's sample/map/reduce barriers go through `gather_with_backups`,
    # so with speculation enabled a straggler can be backed up; the deterministic
    # tasks make the globally-sorted result identical to single-node regardless.
    from batcher.config import DistributedConfig

    t = _data()
    single = bt.from_arrow(t).sort("k", "v", descending=[False, True]).collect().to_pylist()
    scoped = bt.Config().replace(distributed=DistributedConfig(speculation_max_backups=2))
    with bt.config_context(scoped):
        distrib = (
            bt.from_arrow(t)
            .sort("k", "v", descending=[False, True])
            .collect(distributed=True, num_workers=4)
            .to_pylist()
        )
    assert single == distrib  # globally ordered → exact, position-by-position


def test_distributed_window_correct_with_speculation_enabled():
    # Same for the disk window shuffle: its map/reduce barriers are backed up under
    # speculation, and each whole partition still lands on one reducer → identical.
    from batcher.config import DistributedConfig

    t = _data()

    def q(ds):
        return ds.window(partition_by=["k"], functions={"tot": ("sum", "v"), "hi": ("max", "v")})

    single = q(bt.from_arrow(t)).collect()
    scoped = bt.Config().replace(distributed=DistributedConfig(speculation_max_backups=2))
    with bt.config_context(scoped):
        distrib = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4)
    assert _norm(single) == _norm(distrib)


def test_distributed_map_batches_matches_single_node():
    # Distributed batch-inference pipeline: read -> filter -> map_batches(model).
    import numpy as np

    def embed(batch: pa.RecordBatch) -> pa.RecordBatch:
        x = np.asarray(batch.column("x"))
        y = np.asarray(batch.column("y"))
        return batch.append_column("emb", pa.array((x * 0.5 + y).astype("float64")))

    n = 100_000
    t = pa.table({"x": np.arange(n) % 500, "y": (np.arange(n) % 7).astype("int64")})

    def pipe(ds):
        return ds.filter(col("x") >= 250).map_batches(
            embed, batch_size=20_000, output_columns=["x", "y", "emb"]
        )

    single = pipe(bt.from_arrow(t)).collect()
    distrib = pipe(bt.from_arrow(t)).collect(distributed=True, num_workers=4)

    def multiset(tb):
        c = tb.column_names
        return sorted(tuple(r[k] for k in c) for r in tb.to_pylist())

    assert single.num_rows == distrib.num_rows
    assert multiset(single) == multiset(distrib)


def test_distributed_falls_back_for_unsupported_shape():
    # A plain filter/project has no shuffle breaker → single-node fallback path.
    t = pa.table({"a": [1, 2, 3, 4]})
    out = bt.from_arrow(t).filter(col("a") > 1).select("a").collect(distributed=True, num_workers=4)
    assert out.to_pydict() == {"a": [2, 3, 4]}


# --- Arrow Flight transport (network shuffle, object store bypassed) ----------


def test_flight_grouped_aggregate_matches_single_node():
    t = _data()

    def q(ds):
        return ds.group_by("k").agg(s=col("v").sum(), n=count(), a=col("v").mean())

    single = q(bt.from_arrow(t)).collect()
    flight = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4, transport="flight")
    assert _norm(single) == _norm(flight)


def test_flight_global_aggregate_matches_single_node():
    t = _data()

    def q(ds):
        return ds.group_by().agg(s=col("v").sum(), n=count())

    single = q(bt.from_arrow(t)).collect().to_pydict()
    flight = (
        q(bt.from_arrow(t)).collect(distributed=True, num_workers=4, transport="flight").to_pydict()
    )
    assert single == flight


@pytest.mark.parametrize("how", ["inner", "left", "right"])
def test_flight_join_matches_single_node(how):
    left, right = _join_data()
    single = left.join(right, on="k", how=how).collect()
    flight = left.join(right, on="k", how=how).collect(
        distributed=True, num_workers=4, transport="flight"
    )
    assert _rowset(single) == _rowset(flight)


def test_flight_splittable_source_matches_single_node(tmp_path):
    """A splittable source (Parquet row-groups) over the Flight path is shared-nothing:
    each worker gets a split-manifest as a Ray arg and reads its row-groups directly —
    no driver-local work_dir. (Also guards the path where the old code read a manifest
    as if it were an IPC file.)"""
    import pyarrow.parquet as pq

    rng = np.random.default_rng(33)
    n = 100_000
    t = pa.table(
        {"k": rng.integers(0, 40, n).astype("int64"), "v": rng.integers(0, 100, n).astype("int64")}
    )
    path = str(tmp_path / "t.parquet")
    pq.write_table(t, path, row_group_size=10_000)  # 10 row-groups → 10 splits

    def q(ds):
        return ds.group_by("k").agg(s=col("v").sum(), n=count())

    single = q(bt.read.parquet(path)).collect(distributed=False)
    flight = q(bt.read.parquet(path)).collect(distributed=True, num_workers=4, transport="flight")
    assert _norm(single) == _norm(flight)


def _spy_distributed_map(monkeypatch) -> list:
    """Record each `_distributed_map` invocation (the parallel fan-out path)."""
    import batcher.dist.executors.map as map_mod

    calls: list = []
    real = map_mod._distributed_map

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(map_mod, "_distributed_map", spy)
    return calls


def test_distributed_breaker_free_scan_fans_out_over_splits(tmp_path, monkeypatch):
    # A breaker-free scan/filter/project over a SPLITTABLE source distributes: each
    # worker reads its own row-groups in parallel (the distributed-scan case) instead of
    # one node reading the whole source. Result must equal single-node, and the parallel
    # path must actually be taken (`_distributed_map` invoked).
    import pyarrow.parquet as pq

    rng = np.random.default_rng(11)
    n = 80_000
    t = pa.table(
        {
            "k": rng.integers(0, 1000, n).astype("int64"),
            "v": rng.integers(0, 100, n).astype("int64"),
        }
    )
    path = str(tmp_path / "scan.parquet")
    pq.write_table(t, path, row_group_size=10_000)  # 8 row-groups → splittable

    def q(ds):
        return ds.filter(col("v") >= 50).select("k", "v")

    single = q(bt.read.parquet(path)).collect(distributed=False)
    calls = _spy_distributed_map(monkeypatch)
    distrib = q(bt.read.parquet(path)).collect(distributed=True, num_workers=4)
    assert calls == [1]  # the breaker-free splittable scan fanned out, not single-node
    assert _rowset(distrib) == _rowset(single)


def test_distributed_breaker_free_in_memory_stays_single_node(monkeypatch):
    # An in-memory source is NOT shipped to workers for a breaker-free pipeline — that
    # would cost more than the parallel CPU saves — so it stays single-node. Result is
    # still correct; only the routing differs.
    t = _data()
    calls = _spy_distributed_map(monkeypatch)
    got = (
        bt.from_arrow(t)
        .filter(col("v") >= 50)
        .select("k", "v")
        .collect(distributed=True, num_workers=4)
    )
    assert calls == []  # in-memory stayed single-node (no fan-out)
    single = bt.from_arrow(t).filter(col("v") >= 50).select("k", "v").collect()
    assert _rowset(got) == _rowset(single)


def test_distributed_iter_batches_scan_streams_off_driver(tmp_path):
    # iter_batches(distributed=True) over a breaker-free splittable scan fans the read
    # out across workers AND streams each worker's output back one partition at a time —
    # the driver never holds the whole scan result. Rows must equal collect, and the
    # result comes back in multiple partition-sized batches (not one collected table).
    import pyarrow.parquet as pq

    rng = np.random.default_rng(5)
    n = 80_000
    t = pa.table(
        {
            "k": rng.integers(0, 1000, n).astype("int64"),
            "v": rng.integers(0, 100, n).astype("int64"),
        }
    )
    path = str(tmp_path / "scan.parquet")
    pq.write_table(t, path, row_group_size=10_000)  # 8 row-groups → splittable

    def q(ds):
        return ds.filter(col("v") >= 50).select("k", "v")

    single = q(bt.read.parquet(path)).collect(distributed=False)
    batches = list(q(bt.read.parquet(path)).iter_batches(distributed=True, num_workers=4))
    got = pa.Table.from_batches(batches) if batches else single.slice(0, 0)
    assert _rowset(got) == _rowset(single)
    assert len(batches) >= 2  # streamed back per partition, not one driver table


def test_locality_aware_scheduling_equals_single_node():
    # Locality-aware reducer placement hosts each reducer where its bucket concentrates,
    # so its fetches become same-node hits. It is RESULT-PRESERVING — which actor runs a
    # reducer never changes the output — so the aggregate must equal single-node whether
    # placement is on or off. (On a single-node test cluster the placement path is fully
    # exercised: every bucket's data is on the one node, so affinity fires for each.)
    from batcher.config import DistributedConfig

    t = _data()
    single = bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=count()).collect()
    scoped = bt.Config().replace(distributed=DistributedConfig(locality_aware_scheduling=True))
    with bt.config_context(scoped):
        distrib = (
            bt.from_arrow(t)
            .group_by("k")
            .agg(s=col("v").sum(), n=count())
            .collect(distributed=True, num_workers=4, transport="flight")
        )
    assert _norm(single) == _norm(distrib)


def test_flight_distinct_matches_single_node():
    """DISTINCT over the Flight (Carbonite) aggregate shuffle equals single-node."""
    t = _data()

    def q(ds):
        return ds.select("k").distinct()

    single = q(bt.from_arrow(t)).collect()
    flight = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4, transport="flight")
    assert _norm(single) == _norm(flight)


def test_flight_window_matches_single_node():
    """A window partitioned by a column, hash-shuffled over Flight, equals single-node."""
    t = _data()

    def q(ds):
        return ds.window(partition_by=["k"], functions={"s": ("sum", "v"), "c": ("count", "v")})

    single = q(bt.from_arrow(t)).collect()
    flight = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4, transport="flight")
    assert _norm(single) == _norm(flight)


# --- distributed sort (range-partitioned) ------------------------------------


@pytest.mark.parametrize("descending", [False, True])
def test_distributed_sort_matches_single_node(descending):
    rng = np.random.default_rng(13)
    t = pa.table(
        {
            "k": rng.integers(0, 100000, 80000).astype("int64"),
            "v": rng.integers(0, 100, 80000).astype("int64"),
        }
    )
    distrib = (
        bt.from_arrow(t).sort("k", descending=descending).collect(distributed=True, num_workers=4)
    )
    single = bt.from_arrow(t).sort("k", descending=descending).collect()
    assert distrib.column("k").to_pylist() == single.column("k").to_pylist()
    assert _norm(distrib) == _norm(single)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_first", [False, True])
def test_flight_sort_matches_single_node(descending, nulls_first):
    """The Flight range-sort (sketch-sampled boundaries, no driver materialization)
    equals single-node for every desc/nulls ordering — including real Arrow nulls,
    which must land at the correct end of the *post-`desc`* concatenation."""
    rng = np.random.default_rng(21)
    n = 80_000
    keys = rng.integers(0, 100_000, n).astype("int64")
    t = pa.table(
        {
            "k": pa.array(keys, mask=rng.random(n) < 0.05),
            "v": pa.array(rng.integers(0, 100, n).astype("int64")),
        }
    )
    single = bt.from_arrow(t).sort("k", descending=descending, nulls_first=nulls_first).collect()
    flight = (
        bt.from_arrow(t)
        .sort("k", descending=descending, nulls_first=nulls_first)
        .collect(distributed=True, num_workers=4, transport="flight")
    )
    assert single.column("k").to_pylist() == flight.column("k").to_pylist()


def test_flight_sort_skewed_keys_match_single_node():
    """A heavily skewed leading key (90% one value) still sorts correctly over Flight —
    boundary imbalance affects only balance, never the result."""
    rng = np.random.default_rng(22)
    n = 80_000
    keys = np.where(rng.random(n) < 0.9, 5, rng.integers(0, 100_000, n)).astype("int64")
    t = pa.table({"k": keys, "v": rng.integers(0, 100, n).astype("int64")})
    single = bt.from_arrow(t).sort("k").collect()
    flight = bt.from_arrow(t).sort("k").collect(distributed=True, num_workers=4, transport="flight")
    assert single.column("k").to_pylist() == flight.column("k").to_pylist()


def test_distributed_sort_top_n():
    rng = np.random.default_rng(14)
    t = pa.table(
        {
            "k": rng.integers(0, 100000, 50000).astype("int64"),
            "v": rng.integers(0, 100, 50000).astype("int64"),
        }
    )
    distrib = (
        bt.from_arrow(t)
        .sort("k", descending=True)
        .limit(15)
        .collect(distributed=True, num_workers=4)
    )
    single = bt.from_arrow(t).sort("k", descending=True).limit(15).collect()
    assert distrib.column("k").to_pylist() == single.column("k").to_pylist()


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_first", [False, True])
def test_distributed_disk_sort_nulls_match_single_node(descending, nulls_first):
    """The disk range-sort routes real Arrow nulls to the correct end of the
    *post-`desc`* concatenation, exactly like single-node — the shared `bucketize`
    null-bucket logic the Flight path uses now also backs the disk path."""
    rng = np.random.default_rng(23)
    n = 80_000
    keys = rng.integers(0, 100_000, n).astype("int64")
    t = pa.table(
        {
            "k": pa.array(keys, mask=rng.random(n) < 0.05),
            "v": pa.array(rng.integers(0, 100, n).astype("int64")),
        }
    )
    single = bt.from_arrow(t).sort("k", descending=descending, nulls_first=nulls_first).collect()
    distrib = (
        bt.from_arrow(t)
        .sort("k", descending=descending, nulls_first=nulls_first)
        .collect(distributed=True, num_workers=4)
    )
    assert single.column("k").to_pylist() == distrib.column("k").to_pylist()


def test_distributed_disk_sort_never_reads_full_source_on_driver(tmp_path, monkeypatch):
    """The disk sort samples boundaries from per-worker KLL sketches, so a splittable
    source's rows are read only inside the worker tasks — never materialized on the
    driver. Spy on the driver-side `read_source` to prove it is never called."""
    import pyarrow.parquet as pq

    from batcher.io import source as source_mod

    rng = np.random.default_rng(34)
    n = 100_000
    t = pa.table(
        {
            "k": rng.integers(0, 100_000, n).astype("int64"),
            "v": rng.integers(0, 100, n).astype("int64"),
        }
    )
    path = str(tmp_path / "t.parquet")
    pq.write_table(t, path, row_group_size=10_000)  # 10 row-groups → splittable

    # The driver-side eager read path goes through `read_source`; the splittable
    # sketch-sampling path never touches it. Spy on it (call-time import resolves to
    # this patched attribute) to prove the driver never materializes the source.
    calls: list = []
    real_read = source_mod.read_source
    monkeypatch.setattr(
        source_mod, "read_source", lambda *a, **k: calls.append(1) or real_read(*a, **k)
    )

    single = bt.read.parquet(path).sort("k").collect(distributed=False)
    distrib = bt.read.parquet(path).sort("k").collect(distributed=True, num_workers=4)
    assert single.column("k").to_pylist() == distrib.column("k").to_pylist()
    assert not calls, "driver read the full source instead of sketch-sampling per worker"


def test_flight_shuffle_correct_under_tight_credit_window():
    """The Flight shuffle result is identical no matter how tight the credit window.

    Forcing a window of 1 (strict lock-step backpressure) through Carbonite's config
    exercises the credit-governed reducer<-mapper channels at their tightest; the
    distributed aggregate and join must still equal the single-node result. This is
    the end-to-end proof that credit flow control bounds memory without changing
    semantics."""
    from batcher.config import Config, FlowControlConfig, config_context

    t = _data()
    with config_context(Config().replace(flow_control=FlowControlConfig(default_credits=1))):

        def agg(ds):
            return ds.group_by("k").agg(s=col("v").sum(), n=count())

        single = agg(bt.from_arrow(t)).collect()
        distrib = agg(bt.from_arrow(t)).collect(distributed=True, num_workers=4, transport="flight")
        assert _norm(single) == _norm(distrib)


def test_flight_adaptive_credits_match_single_node():
    """AIMD adaptive shuffle credits (window grows/shrinks with memory pressure)
    must not change the result — flow control bounds memory, never semantics."""
    from batcher.config import Config, DistributedConfig, config_context

    t = _data()

    def q(ds):
        return ds.group_by("k").agg(s=col("v").sum(), n=count())

    single = q(bt.from_arrow(t)).collect(distributed=False)
    with config_context(Config().replace(distributed=DistributedConfig(adaptive_credits=True))):
        adaptive = q(bt.from_arrow(t)).collect(distributed=True, num_workers=4, transport="flight")
    assert _norm(single) == _norm(adaptive)


def test_distributed_honors_engine_config_from_context():
    """A non-default `ExecutionConfig.morsel_rows` set via `config_context` flows
    through the driver into every Ray worker's `execute_plan` (it can't read the
    driver's context itself). A tiny morsel forces many morsels per worker — the
    aggregate, join, and sort paths must still equal the single-node result, proving
    the engine-config threading works end-to-end without changing semantics."""
    from batcher.config import Config, ExecutionConfig, config_context

    tiny = Config().replace(execution=ExecutionConfig(morsel_rows=512))
    with config_context(tiny):
        t = _data()
        agg_single = bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=count()).collect()
        agg_dist = (
            bt.from_arrow(t)
            .group_by("k")
            .agg(s=col("v").sum(), n=count())
            .collect(distributed=True, num_workers=4)
        )
        assert _norm(agg_single) == _norm(agg_dist)

        left, right = _join_data()
        join_single = left.join(right, on="k", how="inner").collect()
        join_dist = left.join(right, on="k", how="inner").collect(distributed=True, num_workers=4)
        assert _rowset(join_single) == _rowset(join_dist)

        sort_single = bt.from_arrow(t).sort("k", "v").collect().to_pylist()
        sort_dist = (
            bt.from_arrow(t).sort("k", "v").collect(distributed=True, num_workers=4).to_pylist()
        )
        assert sort_single == sort_dist
