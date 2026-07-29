"""Two operators that had no distributed path and raised on distributed data.

Both reached `_unsupported` and raised rather than silently running the whole query on one
node, so neither was a wrong answer — but each was a hard failure on exactly the data
distribution is for.

- **Fixed-count `sample(n=...)`** keeps the `n` smallest-hash rows of the WHOLE relation
  (the *fraction* form is a per-row predicate and already rode the map path). It is
  mergeable top-N: a row among the globally `n` smallest hashes is also among its own
  partition's `n` smallest, so the union of the per-partition results contains the global
  answer and re-applying the operator to that union selects exactly it.

- **`RangeJoin`** has no equality to co-partition on — hashing `a.x` and the `b.y` values
  it is less than sends them to different buckets — so the shuffle every other join uses is
  not merely slower, it is wrong. Broadcasting the build side is the shape that works: each
  probe task sees the whole right, so a left row's match set is computed inside its own
  partition.

Every test asserts the distributed result is *identical* to the single-node one, which is
the invariant that matters (`CLAUDE.md` #7).
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher._internal.errors import PlanError

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

WORKERS = 4


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(WORKERS)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def split_source(tmp_path_factory):
    """A genuinely splittable source: several Parquet files, several row-groups each."""
    d = tmp_path_factory.mktemp("split")
    rng = np.random.default_rng(11)
    for i in range(4):
        pq.write_table(
            pa.table(
                {
                    "x": np.arange(i * 500, (i + 1) * 500, dtype="int64"),
                    "g": rng.integers(0, 5, 500).astype("int64"),
                }
            ),
            d / f"p{i}.parquet",
            row_group_size=100,
        )
    return str(d)


def _sorted_rows(table: pa.Table) -> list[tuple]:
    return sorted(tuple(row.values()) for row in table.to_pylist())


# --- fixed-count sample --------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("n", [0, 1, 7, 100, 1_500, 5_000])
def test_distributed_fixed_count_sample_equals_single_node(split_source, n):
    """Every size, including 0, 1, and n larger than the relation (keep everything)."""
    ds = bt.read.parquet(split_source).sample(n=n, seed=42)
    single = ds.collect(distributed=False)
    dist = ds.collect(distributed=True, num_workers=WORKERS)
    assert dist.num_rows == min(n, 2_000)
    assert _sorted_rows(dist) == _sorted_rows(single)


@pytest.mark.integration
def test_distributed_fixed_count_sample_is_not_n_per_partition(split_source):
    """The bug this path exists to prevent: running the operator per partition and
    concatenating would return `n x workers` rows."""
    got = (
        bt.read.parquet(split_source)
        .sample(n=10, seed=1)
        .collect(distributed=True, num_workers=WORKERS)
    )
    assert got.num_rows == 10


@pytest.mark.integration
def test_distributed_fixed_count_sample_with_operators_above(split_source):
    ds = bt.read.parquet(split_source).sample(n=200, seed=7).filter(bt.col("x") > 100).select("x")
    assert _sorted_rows(ds.collect(distributed=True, num_workers=WORKERS)) == _sorted_rows(
        ds.collect(distributed=False)
    )


@pytest.mark.integration
def test_distributed_fraction_sample_still_matches(split_source):
    """The fraction form is partition-independent and rides the map path; it must not
    have been disturbed by the fixed-count branch landing above it."""
    ds = bt.read.parquet(split_source).sample(fraction=0.25, seed=3)
    assert _sorted_rows(ds.collect(distributed=True, num_workers=WORKERS)) == _sorted_rows(
        ds.collect(distributed=False)
    )


# --- range (inequality) join ---------------------------------------------------


@pytest.fixture(scope="module")
def bands(tmp_path_factory):
    """A small build side — the canonical range-join shape (events against a few bands)."""
    d = tmp_path_factory.mktemp("bands")
    pq.write_table(
        pa.table({"lo": [0, 500, 1_000, 1_500], "tier": ["a", "b", "c", "d"]}),
        d / "bands.parquet",
    )
    return str(d)


@pytest.mark.integration
def test_distributed_range_join_equals_single_node(split_source, bands):
    left = bt.read.parquet(split_source)
    right = bt.read.parquet(bands)
    ds = left.join(right, how="cross").filter(bt.col("x") < bt.col("lo"))
    single = ds.collect(distributed=False)
    dist = ds.collect(distributed=True, num_workers=WORKERS)
    assert dist.num_rows == single.num_rows
    assert _sorted_rows(dist) == _sorted_rows(single)


@pytest.mark.integration
@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
def test_distributed_range_join_every_inequality(split_source, bands, op):
    left = bt.read.parquet(split_source)
    right = bt.read.parquet(bands)
    pred = {
        "<": bt.col("x") < bt.col("lo"),
        "<=": bt.col("x") <= bt.col("lo"),
        ">": bt.col("x") > bt.col("lo"),
        ">=": bt.col("x") >= bt.col("lo"),
    }[op]
    ds = left.join(right, how="cross").filter(pred)
    assert _sorted_rows(ds.collect(distributed=True, num_workers=WORKERS)) == _sorted_rows(
        ds.collect(distributed=False)
    )


@pytest.mark.integration
def test_distributed_band_join_two_conditions(split_source, tmp_path_factory):
    """Two inequalities (an interval containment) — the IEJoin shape."""
    d = tmp_path_factory.mktemp("intervals")
    pq.write_table(
        pa.table({"lo": [0, 400, 900], "hi": [300, 800, 1_400], "tier": ["a", "b", "c"]}),
        d / "iv.parquet",
    )
    left = bt.read.parquet(split_source)
    right = bt.read.parquet(str(d))
    ds = left.join(right, how="cross").filter(
        (bt.col("x") >= bt.col("lo")) & (bt.col("x") <= bt.col("hi"))
    )
    assert _sorted_rows(ds.collect(distributed=True, num_workers=WORKERS)) == _sorted_rows(
        ds.collect(distributed=False)
    )


@pytest.mark.integration
def test_distributed_range_join_with_operators_above(split_source, bands):
    left = bt.read.parquet(split_source)
    right = bt.read.parquet(bands)
    ds = (
        left.join(right, how="cross")
        .filter(bt.col("x") < bt.col("lo"))
        .group_by("tier")
        .agg(n=bt.count())
    )
    assert _sorted_rows(ds.collect(distributed=True, num_workers=WORKERS)) == _sorted_rows(
        ds.collect(distributed=False)
    )


@pytest.mark.integration
def test_oversized_build_side_refuses_rather_than_replicating(split_source, monkeypatch):
    """There is no shuffle fallback for an inequality, so an unbroadcastable build side
    must raise with the fix — not silently run on one node, nor replicate into an OOM."""
    import batcher.dist.executors.join as dj

    monkeypatch.setattr(dj, "l3_cache_bytes", lambda: 1)
    cfg = bt.config()
    monkeypatch.setattr(type(cfg.optimizer), "resolved_broadcast_max_bytes", lambda self, _c: 1)

    left = bt.read.parquet(split_source)
    right = bt.read.parquet(split_source)
    ds = left.join(right, how="cross").filter(bt.col("x") < bt.col("g"))
    with pytest.raises(PlanError, match="broadcast"):
        ds.collect(distributed=True, num_workers=WORKERS)


# --- distributed UNION streams branch by branch --------------------------------


@pytest.mark.integration
def test_distributed_union_streams_without_collecting_every_branch(split_source, bands):
    """`_distributed_union` runs each branch to a driver table and concatenates, so routing
    a union there materializes the whole result. The streaming terminal now decomposes the
    union first, sending each branch through the distributed router on its own — the driver
    holds one branch's one bucket."""
    import batcher.api.terminal.core as tc

    a = bt.read.parquet(split_source).select("x")
    b = bt.read.parquet(split_source).filter(bt.col("x") > 1_000).select("x")
    ds = a.union(b)

    calls: list[int] = []
    original = tc._collect
    try:
        tc._collect = lambda *ar, **k: (calls.append(1), original(*ar, **k))[1]
        streamed = pa.Table.from_batches(
            list(ds.iter_batches(distributed=True, num_workers=WORKERS))
        )
    finally:
        tc._collect = original

    assert calls == []
    assert _sorted_rows(streamed) == _sorted_rows(ds.collect(distributed=False))


@pytest.mark.integration
def test_distributed_union_of_breaker_branches_matches_single_node(split_source):
    """Each branch is itself a breaker, so each takes the distributed bucket-at-a-time
    path underneath the union."""
    src = bt.read.parquet(split_source)
    ds = src.group_by("g").agg(n=bt.count()).union(src.group_by("g").agg(n=bt.count()))
    streamed = pa.Table.from_batches(list(ds.iter_batches(distributed=True, num_workers=WORKERS)))
    assert _sorted_rows(streamed) == _sorted_rows(ds.collect(distributed=False))


@pytest.mark.integration
def test_distributed_union_preserves_concatenation_order(split_source):
    """Branch 0's rows, then branch 1's — the contract an order-independent comparison
    cannot see."""
    a = bt.read.parquet(split_source).filter(bt.col("x") < 10).select("x")
    b = bt.read.parquet(split_source).filter(bt.col("x") >= 1_990).select("x")
    ds = a.union(b)
    streamed = pa.Table.from_batches(
        list(ds.iter_batches(distributed=True, num_workers=WORKERS))
    ).to_pydict()["x"]
    assert streamed == list(range(10)) + list(range(1_990, 2_000))
