"""A distributed batch-inference WRITE must never route its result through the driver.

`write_parquet(distributed=True)` on a `map_batches`/inference plan used to be excluded from
the streaming distributed write, so it fell through to `collect(distributed=True)` and landed
the whole post-inference result on the driver — an unconditional OOM for any job whose output
exceeds one node, which is every large embedding or scoring job.

These tests pin the two things that make the fix real: each worker writes its own shard (the
rows never come back), and the written result is identical to the single-node one.
"""

from __future__ import annotations

import sys

import pyarrow as pa
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytest.importorskip("batcher._native", reason="native engine not built")
ray = pytest.importorskip("ray", reason="ray not installed")

pytestmark = pytest.mark.integration

ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


class _Embedder:
    """Stands in for a model: widens each row the way a real embedding stage does."""

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        x = batch.column("x").to_pylist()
        return pa.RecordBatch.from_pydict({"x": x, "vec": [[float(v), float(v) + 1.0] for v in x]})


def _source(tmp_path, rows: int = 64):
    path = str(tmp_path / "in.parquet")
    bt.from_pydict({"x": list(range(rows))}).write.parquet(path)
    return bt.read(path)


def test_distributed_inference_write_matches_the_single_node_write(tmp_path):
    out_dist = str(tmp_path / "dist")
    out_local = str(tmp_path / "local")

    _source(tmp_path).map_batches(_Embedder).write.parquet(out_dist, distributed=True)
    _source(tmp_path).map_batches(_Embedder).write.parquet(out_local)

    got = bt.read(out_dist, format="parquet").collect().sort_by("x")
    want = bt.read(out_local, format="parquet").collect().sort_by("x")
    assert got.to_pydict() == want.to_pydict()
    assert got.num_rows == 64


def test_the_inference_write_takes_the_streaming_path_not_a_driver_collect(tmp_path):
    """The proof that rows do not transit the driver: `_distributed_write_plan` is what runs,
    and it hands back a manifest of locators rather than a table."""
    from batcher.dist.executors import write as write_mod

    seen: dict = {}
    real = write_mod._distributed_write_plan

    def _spy(*args, **kwargs):
        manifest = real(*args, **kwargs)
        seen["manifest"] = manifest
        return manifest

    write_mod._distributed_write_plan = _spy
    try:
        _source(tmp_path).map_batches(_Embedder).write.parquet(
            str(tmp_path / "spied"), distributed=True
        )
    finally:
        write_mod._distributed_write_plan = real

    assert "manifest" in seen, "the UDF write did not take the streaming distributed path"
    assert seen["manifest"].files, "no data files were written by the workers"


def test_an_inference_write_that_filters_everything_still_commits_a_readable_output(tmp_path):
    out = str(tmp_path / "empty")
    _source(tmp_path).filter(bt.col("x") < 0).map_batches(_Embedder).write.parquet(
        out, distributed=True
    )
    assert bt.read(out, format="parquet").collect().num_rows == 0
