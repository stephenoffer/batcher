"""`group_by(maintain_order=True)` emits the same group order on one node and on many.

The order is a computed property (a row number, its per-group minimum, a sort), so unlike the
hash order of a plain `group_by` it is a guarantee rather than an observation, and it must
survive a real shuffle. The input is four Parquet files so the distributed run genuinely
splits it, and a group first appears only in the last file, which a per-partition numbering
would misplace. Compared order-sensitively against the single-node result and against the
first-appearance order computed in Python.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_tables_equal
from _ray_cluster import init_test_ray, shutdown_test_ray

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="ray not installed")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(2)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("maintain_order")
    keys = ["m", "c", "x", "a", "q"]
    for part in range(4):
        n = 20_000
        g = [keys[(i * 7 + part) % len(keys)] for i in range(n)]
        if part == 3:
            g[-1] = "late"
        pq.write_table(
            pa.table({"g": g, "v": list(range(part * n, part * n + n))}),
            root / f"part-{part}.parquet",
        )
    return str(root / "*.parquet")


def test_the_distributed_order_equals_the_single_node_order(source):
    ds = (
        bt.read.parquet(source)
        .group_by("g", maintain_order=True)
        .agg(n=bt.count(), total=bt.col("v").sum())
    )
    one = ds.collect()
    many = ds.collect(distributed=True)
    assert_tables_equal(many, one, ordered=True)

    rows = bt.read.parquet(source).collect()
    first_seen = list(dict.fromkeys(rows.column("g").to_pylist()))
    assert one.column("g").to_pylist() == first_seen
    assert first_seen[-1] == "late"
