"""`Dataset.to_ray_dataset` — the return leg of `bt.from_ray_dataset`.

A Batcher result feeding a Ray Train / Tune / Serve stage has to arrive as a real
`ray.data.Dataset`, with the schema the query declared and the rows it produced. These
tests pin the round trip, the empty-result schema (the case a naive implementation loses),
and the block sizing that keeps a large result from becoming tens of thousands of tiny Ray
blocks.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _ray():
    ray = pytest.importorskip("ray", reason="ray not installed")
    pytest.importorskip("ray.data", reason="ray.data not installed")
    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        configure_logging=False,
        log_to_driver=False,
    )


def test_round_trips_through_ray_data():
    """Rows and column types survive both directions.

    `from_ray_dataset` -> query -> `to_ray_dataset` is the shape a user porting off Ray Data
    incrementally actually runs, so the two halves have to agree on more than row count.
    """
    import ray.data

    source = ray.data.from_arrow(pa.table({"k": [1, 2, 3, 4], "v": [10, 20, 30, 40]}))
    out = bt.from_ray_dataset(source).filter(bt.col("v") > 15).to_ray_dataset()
    rows = sorted(r["v"] for r in out.take_all())
    assert rows == [20, 30, 40]
    assert set(out.schema().names) == {"k", "v"}


def test_preserves_the_schema_of_an_empty_result():
    """A query that filters everything away still has a schema, and Ray must see it.

    A Ray Dataset built from zero blocks reports `schema() is None`, so every downstream
    operation fails on a column the user can see in `ds.schema`. One empty block carries it
    across, and this is the assertion that keeps it there.
    """
    out = bt.from_pydict({"x": [1, 2, 3], "s": ["a", "b", "c"]}).filter(bt.col("x") > 99)
    rds = out.to_ray_dataset()
    assert rds.count() == 0
    assert set(rds.schema().names) == {"x", "s"}


def test_coalesces_engine_morsels_into_ray_sized_blocks():
    """Many engine morsels become few Ray blocks.

    Engine morsels are 16,384 rows; a Ray Dataset of one block per morsel schedules badly
    because per-block task overhead dominates. Asserted by *count*, not by timing: with a
    tiny target every morsel is its own block, and with the default target they collapse to
    one — the two directions together prove the coalescing is real and is driven by the
    target rather than by luck.
    """
    ds = bt.from_pydict({"x": list(range(100_000))})
    assert ds.to_ray_dataset().num_blocks() == 1
    fine = ds.to_ray_dataset(block_size_bytes=1)
    assert fine.num_blocks() > 1
    assert fine.count() == 100_000


def test_batch_size_controls_the_engine_batches_it_is_built_from():
    """`batch_size` rebatches before coalescing, so a caller can pin the block granularity.

    With a one-byte block target every engine batch becomes its own block, which makes the
    rebatching observable — otherwise coalescing hides it.
    """
    ds = bt.from_pydict({"x": list(range(1000))})
    blocks = ds.to_ray_dataset(batch_size=100, block_size_bytes=1).num_blocks()
    assert blocks == 10


def test_reads_rays_own_target_block_size():
    """Block sizing follows Ray Data's `DataContext`, not a Batcher constant.

    A cluster that tuned `target_max_block_size` expects every dataset in its pipeline to
    block the same way, including the ones Batcher hands it.
    """
    from ray.data import DataContext

    from batcher.api.dataset._export import _ray_target_block_bytes

    assert _ray_target_block_bytes() == DataContext.get_current().target_max_block_size
