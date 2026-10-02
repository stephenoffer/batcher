"""Row-local work above a distributed sort must stay on the workers, and stay sorted.

A distributed sort leaves its range buckets where they were sorted when the caller asks for a
partitioned result (`materialize=False`, which `iter_batches(distributed=True)` and the staged
executor both do), and hands back handles in range order. It declined whenever anything sat
above the sort, collecting the whole relation onto the driver instead (audit finding F095) --
and one thing always sits above a computed sort key: the projection that drops the hidden key
the range partitioner cut on. So `sort(col("a") * -1)` streamed nothing.

A `Filter` or a `Project` over each range, read in range order, is that operator over the
sorted relation, so both transports now fold them into every reducer's plan. The comparison
is **ordered** throughout: the rows must come back in the sort's order, and an
order-independent comparison is exactly what would let a bucket emitted out of range order
pass.

The source is a four-file Parquet directory, so the distributed route is taken rather than
the in-memory short cut; the spies are the positive control that the partitioned result was
actually produced rather than a collected table re-chunked.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_tables_equal
from _ray_cluster import ray_session_fixture

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="ray not installed")

_N = 80_000
_WORKERS = 2

_ray_session = ray_session_fixture(4)


@pytest.fixture(scope="module")
def splittable(cluster_scratch) -> str:
    directory = cluster_scratch("sort_row_local_above")
    for part in range(4):
        keys = range(part, _N, 4)
        pq.write_table(
            pa.table(
                {
                    "t": pa.array([(k * 7919) % _N for k in keys], pa.int64()),
                    "x": pa.array([None if k % 13 == 0 else k % 11 for k in keys]),
                }
            ),
            directory / f"p{part}.parquet",
        )
    return str(directory)


_SHAPES = {
    "computed_key": lambda ds: ds.sort(bt.col("t") * -1),
    "projection_above": lambda ds: ds.sort("t").select("t", y=bt.col("x") * 2),
    "descending_projection": lambda ds: ds.sort("t", descending=True).select(
        z=bt.col("t") + 1, x="x"
    ),
}


def _spy_published(monkeypatch, transport: str) -> list[int]:
    """Count partitioned results as they are built, per transport."""
    seen: list[int] = []
    if transport == "disk":
        import batcher.dist.executors.partition_io as pio

        original = pio.materialize_reduce_output

        def spy(*args, **kwargs):
            seen.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(pio, "materialize_reduce_output", spy)
    else:
        import batcher.dist.fleet as fleet

        base = fleet.FlightMaterializedSource

        class Spy(base):
            def __init__(self, *args, **kwargs):
                seen.append(1)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(fleet, "FlightMaterializedSource", Spy)
    return seen


@pytest.mark.parametrize("transport", ["disk", "flight"])
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_row_local_work_above_a_sort_streams_partitioned(splittable, monkeypatch, shape, transport):
    ds = _SHAPES[shape](bt.read.parquet(splittable))
    want = ds.collect(distributed=False)
    seen = _spy_published(monkeypatch, transport)
    batches = list(ds.iter_batches(distributed=True, num_workers=_WORKERS, transport=transport))
    got = pa.Table.from_batches(batches)
    assert seen, "the sort collected instead of leaving its buckets partitioned"
    assert got.column_names == want.column_names
    assert_tables_equal(got, want, ordered=True)


@pytest.mark.parametrize("transport", ["disk", "flight"])
def test_a_limited_sort_still_slices_after_the_order(splittable, transport):
    """The control on the other side: a `limit` keeps the assembly, and a projection above it
    must see the limited rows, so the projection is not folded into any reducer."""
    ds = bt.read.parquet(splittable).sort("t").limit(_N - 7).select("t")
    got = ds.collect(distributed=True, num_workers=_WORKERS, transport=transport)
    assert_tables_equal(got, ds.collect(distributed=False), ordered=True)
