"""A distributed breaker must not funnel its result through the driver.

An aggregate reduces, so collecting its result on the driver costs the size of a summary.
A **sort** and a **window** emit one row per input row, a join can emit more rows than
either input, and a high-cardinality `distinct` removes almost nothing — for those the
collect is the whole relation through one process, which is the term that decides whether
a query runs at all on a cluster.

Two things are asserted, and both matter:

* the *result* is unchanged — for the sort, compared **order-dependently**, since
  `assert_same`-style multiset comparison cannot see an ordering bug at all; and
* the driver-collecting path is not taken — proved by making it raise, rather than by
  timing or by peak memory, so the test fails loudly if a shape silently regresses back
  onto it.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

WORKERS = 4


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(WORKERS)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def facts(cluster_tmp_dir):
    """A splittable Parquet source: four files, ten row groups each.

    Built under `cluster_tmp_dir` rather than `tmp_path_factory`: the workers read these
    files themselves (that is the whole point of a splittable source), so a driver-local
    path resolves on the driver and nowhere else. On a multi-node cluster that reads back
    as `FileNotFoundError` from a Ray task, which looks like an engine defect and is not
    one. See `tests/conftest.py::cluster_tmp_dir`.
    """
    root = cluster_tmp_dir / "no_materialize_facts"
    root.mkdir(parents=True, exist_ok=True)
    n = 4000
    table = pa.table(
        {
            "a": [(i * 7919) % 1000 for i in range(n)],
            "g": [f"k{i % 37}" for i in range(n)],
            "v": [float(i % 101) for i in range(n)],
        }
    )
    for i in range(4):
        pq.write_table(
            table.slice(i * (n // 4), n // 4),
            str(root / f"part{i}.parquet"),
            row_group_size=100,
        )
    return str(root)


@pytest.fixture(scope="module")
def dim(cluster_tmp_dir):
    root = cluster_tmp_dir / "no_materialize_dim"
    root.mkdir(parents=True, exist_ok=True)
    path = str(root / "dim.parquet")
    pq.write_table(
        pa.table({"g": [f"k{i}" for i in range(37)], "lbl": [f"L{i}" for i in range(37)]}), path
    )
    return path


@pytest.fixture
def no_driver_collect(monkeypatch):
    """Make the collect-then-reshard write path fail, so taking it is a test failure.

    That path is the fallback for a shape which could not keep its result partitioned. It
    still exists and is still correct; what this fixture asserts is that the shapes below
    no longer *reach* it.
    """
    import batcher.dist.executors.write as write_mod

    def _refuse(*_args, **_kwargs):
        raise AssertionError("the distributed write collected its result on the driver")

    monkeypatch.setattr(write_mod, "_distributed_write", _refuse)


def _sorted_ids(plan_result) -> list:
    return plan_result.to_pydict()["a"]


# --------------------------------------------------------------------------------------
# The result is unchanged, and for the sort that means *in order*.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("descending", [False, True])
def test_distributed_sort_streams_back_in_order(facts, descending):
    """`iter_batches(distributed=True)` over an ORDER BY yields the exact single-node row
    order. The buckets are ranges, so listing them in range order is the sorted relation;
    an off-by-one in that ordering is invisible to any multiset comparison."""
    q = lambda: bt.read.parquet(facts).sort("a", descending=descending)  # noqa: E731
    want = _sorted_ids(q().collect())

    batches = list(q().iter_batches(distributed=True, num_workers=WORKERS))
    got = pa.Table.from_batches(batches).to_pydict()["a"]
    assert got == want

    assert _sorted_ids(q().collect(distributed=True, num_workers=WORKERS)) == want


def test_distributed_sort_keeps_its_result_partitioned(facts):
    """The executor hands back a partitioned source rather than a table under
    `materialize=False` — the property the streaming terminal above depends on."""
    from batcher.dist import execute_distributed

    ds = bt.read.parquet(facts).sort("a")
    result = execute_distributed(
        ds._plan, ds._sources, num_workers=WORKERS, transport="disk", materialize=False
    )
    try:
        assert not isinstance(result, pa.Table), "the sort collected on the driver"
        assert result.row_count() == 4000
        # And the buckets, read in the order the source advertises them, ARE the sort.
        scanned = pa.Table.from_batches(result.read(), schema=result.schema())
        assert scanned.to_pydict()["a"] == ds.collect().to_pydict()["a"]
    finally:
        result.cleanup()


def test_distributed_window_streams_back(facts):
    q = lambda: bt.read.parquet(facts).with_columns(  # noqa: E731
        s=bt.col("v").sum().over(partition_by="g")
    )
    want = sorted(zip(*q().collect().to_pydict().values(), strict=True))
    batches = list(q().iter_batches(distributed=True, num_workers=WORKERS))
    got = sorted(zip(*pa.Table.from_batches(batches).to_pydict().values(), strict=True))
    assert got == want


# --------------------------------------------------------------------------------------
# ... and the driver never holds it.
# --------------------------------------------------------------------------------------


def test_distributed_sort_write_never_collects(facts, cluster_tmp_path, no_driver_collect):
    out = str(cluster_tmp_path / "sorted")
    want = bt.read.parquet(facts).sort("a").collect().to_pydict()["a"]
    manifest = (
        bt.read.parquet(facts).sort("a").write.parquet(out, distributed=True, num_workers=WORKERS)
    )
    assert manifest.total_rows == len(want)
    assert sorted(bt.read.parquet(out).collect().to_pydict()["a"]) == sorted(want)


def test_distributed_aggregate_write_never_collects(facts, cluster_tmp_path, no_driver_collect):
    out = str(cluster_tmp_path / "agg")
    q = lambda: bt.read.parquet(facts).group_by("g").agg(s=bt.col("v").sum())  # noqa: E731
    want = sorted(zip(*q().collect().to_pydict().values(), strict=True))
    q().write.parquet(out, distributed=True, num_workers=WORKERS)
    got = sorted(zip(*bt.read.parquet(out).collect().to_pydict().values(), strict=True))
    assert got == want


def test_distributed_distinct_write_never_collects(facts, cluster_tmp_path, no_driver_collect):
    out = str(cluster_tmp_path / "distinct")
    want = sorted(bt.read.parquet(facts).select("g").distinct().collect().to_pydict()["g"])
    bt.read.parquet(facts).select("g").distinct().write.parquet(
        out, distributed=True, num_workers=WORKERS
    )
    assert sorted(bt.read.parquet(out).collect().to_pydict()["g"]) == want


def test_distributed_join_write_never_collects(facts, dim, cluster_tmp_path, no_driver_collect):
    out = str(cluster_tmp_path / "joined")
    q = lambda: bt.read.parquet(facts).join(bt.read.parquet(dim), on="g")  # noqa: E731
    want = q().count()
    q().write.parquet(out, distributed=True, num_workers=WORKERS)
    assert bt.read.parquet(out).count() == want


def test_distributed_window_write_never_collects(facts, cluster_tmp_path, no_driver_collect):
    out = str(cluster_tmp_path / "windowed")
    q = lambda: bt.read.parquet(facts).with_columns(  # noqa: E731
        s=bt.col("v").sum().over(partition_by="g")
    )
    want = sorted(zip(*q().collect().to_pydict().values(), strict=True))
    q().write.parquet(out, distributed=True, num_workers=WORKERS)
    got = sorted(zip(*bt.read.parquet(out).collect().to_pydict().values(), strict=True))
    assert got == want


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param("window", id="window"),
        pytest.param("distinct_on", id="distinct_on"),
    ],
)
def test_empty_keyed_shuffle_keeps_the_column_types(facts, shape):
    """A shuffle whose every bucket came back empty must still carry the real column
    types.

    The empty result used to be built from a list of column *names*
    (`pa.table({name: []})`), which types every column `null`. That is a
    `distributed != single-node` divergence in column types, on the one case where such a
    divergence hides longest: it appears only when the filter happens to match nothing,
    and then breaks a downstream concat or a typed write rather than the query itself.
    """
    empty = bt.read.parquet(facts).filter(bt.col("a") > 10_000)
    q = (
        empty.with_columns(s=bt.col("v").sum().over(partition_by="g"))
        if shape == "window"
        else empty.distinct(["g"])
    )
    want = q.collect().schema
    got = q.collect(distributed=True, num_workers=WORKERS).schema
    assert got.names == want.names
    assert list(got.types) == list(want.types)


def test_empty_breaker_write_leaves_a_readable_relation(facts, cluster_tmp_path):
    """Every bucket empty still writes one empty file, so the distributed result is a
    readable empty table rather than an absent path — the divergence from single-node the
    breaker-free streaming write already guards against."""
    out = str(cluster_tmp_path / "empty")
    bt.read.parquet(facts).filter(bt.col("a") > 10_000).sort("a").write.parquet(
        out, distributed=True, num_workers=WORKERS
    )
    assert bt.read.parquet(out).count() == 0
