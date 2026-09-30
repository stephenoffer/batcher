"""Key-range-aligned joins, run on a real cluster, before and after a warm shuffle fleet exists.

`dist.executors.aligned` runs each key range as a Ray task, or, when an earlier query left a
warm shuffle fleet, on that fleet's actors. Both must give the single-node answer, and the
second must leave the fleet in place for the next shuffling query instead of tearing it down
to find cores (see `aligned.run._run_units`).
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import ray_session_fixture
from batcher import col

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

_WORKERS = 2

_ray_session = ray_session_fixture(4)


@pytest.fixture(scope="module")
def star(cluster_scratch) -> dict[str, str]:
    """`orders` and `lineitem` stored in `orderkey` order, several files each."""
    orders_dir, items_dir = cluster_scratch("aligned_orders"), cluster_scratch("aligned_items")
    for part in range(4):
        keys = list(range(part * 500, (part + 1) * 500))
        pq.write_table(
            pa.table({"o_ok": keys, "o_ck": [k % 97 for k in keys]}),
            orders_dir / f"p{part}.parquet",
        )
        rows = [k for k in keys for _ in range(3)]
        pq.write_table(
            pa.table({"l_ok": rows, "l_price": [float(i % 13) for i in range(len(rows))]}),
            items_dir / f"p{part}.parquet",
        )
    cust_dir = cluster_scratch("aligned_customer")
    pq.write_table(
        pa.table({"c_ck": list(range(97)), "c_name": [f"c{k % 11}" for k in range(97)]}),
        cust_dir / "p0.parquet",
    )
    return {"orders": str(orders_dir), "lineitem": str(items_dir), "customer": str(cust_dir)}


def _query(star):
    orders, items = bt.read.parquet(star["orders"]), bt.read.parquet(star["lineitem"])
    return (
        items.join(orders, left_on="l_ok", right_on="o_ok")
        .group_by("o_ck")
        .agg(rev=col("l_price").sum(), n=bt.count())
    )


def _rows(table: pa.Table) -> list[tuple]:
    return sorted(tuple(r.values()) for r in table.to_pylist())


def test_aligned_join_matches_single_node_as_tasks_and_on_a_warm_fleet(star):
    from batcher.dist.fleet import _fleet

    ds = _query(star)
    expected = _rows(ds.collect(distributed=False))
    assert _rows(ds.collect(distributed=True, num_workers=_WORKERS)) == expected
    # Warm a session fleet with a query that shuffles, then run the aligned join again: it
    # must run on the fleet's actors, give the same answer, and leave the fleet standing.
    shuffle = bt.read.parquet(star["lineitem"]).group_by("l_ok").agg(n=bt.count())
    shuffle.collect(distributed=True, num_workers=_WORKERS, transport="flight")
    fleet = _fleet._SESSION
    if fleet is None:
        pytest.skip("this cluster ran the shuffle without a session fleet")
    assert _rows(ds.collect(distributed=True, num_workers=_WORKERS)) == expected
    assert _fleet._SESSION is fleet, "the aligned run tore down the warm fleet"


def test_a_broadcast_read_on_each_node_matches_single_node(star, monkeypatch):
    """An unfiltered dimension travels as a recipe the workers evaluate, not as rows."""
    from batcher.dist.executors.aligned import local
    from batcher.dist.executors.aligned import run as aligned_run

    monkeypatch.setattr(local, "LOCAL_BROADCAST_BYTES", 0)
    built = []
    real = aligned_run.local_broadcast
    monkeypatch.setattr(
        aligned_run, "local_broadcast", lambda *a, **k: built.append(1) or real(*a, **k)
    )
    orders, items = bt.read.parquet(star["orders"]), bt.read.parquet(star["lineitem"])
    cust = bt.read.parquet(star["customer"])
    ds = (
        items.join(orders, left_on="l_ok", right_on="o_ok")
        .join(cust, left_on="o_ck", right_on="c_ck")
        .group_by("l_ok")
        .agg(rev=col("l_price").sum(), names=col("c_name").count_distinct())
    )
    expected = _rows(ds.collect(distributed=False))
    assert _rows(ds.collect(distributed=True, num_workers=_WORKERS)) == expected
    # Positive control: the dimension went to the workers as a recipe.
    assert built, "no broadcast was read on the workers"
