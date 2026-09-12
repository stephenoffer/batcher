"""A distributed `map_batches(Model).write.parquet(...)` builds the model only on workers.

Scoring a corpus and writing the scores is the canonical batch-inference job, and it built the
model on the **driver** before writing a row -- not as the write, as a *schema probe*.
`WriteManifest.schema` is documented as attached by "the driver, which knows the plan's output
type", and for a `map_batches` stage the driver does not: an arbitrary Python callable's output
type is only knowable by running it. So `_write` called `_schema`, whose last resort executes
the plan under a zero-row limit, which builds the UDF.

On a machine chosen for orchestration rather than inference that is a model load with nothing
to load onto. Measured on a GPU-less head node against a real 8xT4 fleet: the query died inside
`cupy` with `cudaErrorInsufficientDriver` on a 1.9 GiB input, and on a 29 GiB one the driver was
OOM-killed at 16.7 GB -- both before a single row was written, and neither with a message naming
the driver.

The assertion needs no device and no GPU cluster, because the question is *which process*
builds the model. A worker builds it in another process, where it cannot touch this module's
counter; a driver-side build lands in the counter directly. So `_BUILDS == []` after a
successful distributed write is exactly the property, and it reads the same on any hardware.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="the distributed write path needs Ray")
pytest.importorskip("batcher._native", reason="native engine not built")

import sys  # noqa: E402

import ray  # noqa: E402

ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])

_N = 4000

#: Every construction of `Model` *in this process*. A worker's build happens elsewhere.
_BUILDS: list[int] = []


class Model:
    """A load-once factory, the shape `map_batches` documents for a model."""

    def __init__(self) -> None:
        _BUILDS.append(1)

    def __call__(self, batch):
        v = batch.column("v").to_pylist()
        return pa.table({"pred": pa.array([x * 2.0 for x in v])})


@pytest.fixture(scope="module")
def parquet_path(cluster_tmp_dir):
    """A multi-row-group Parquet file, so the source really splits and really distributes."""
    import pyarrow.parquet as pq

    table = pa.table({"v": pa.array([float(i % 97) * 0.5 for i in range(_N)], pa.float64())})
    path = cluster_tmp_dir / "inference_write.parquet"
    pq.write_table(table, path, row_group_size=100)
    return str(path)


def test_the_driver_never_builds_the_model(parquet_path, cluster_tmp_dir):
    out = str(cluster_tmp_dir / "scores_off_driver")
    _BUILDS.clear()
    manifest = (
        bt.read.parquet(parquet_path)
        .map_batches(Model, output_columns=["pred"], batch_format="pyarrow", concurrency=2)
        .write.parquet(out, distributed=True)
    )
    assert bt.read.parquet(out).count() == _N, "the write must produce every row"
    assert _BUILDS == [], (
        f"the model was built {len(_BUILDS)} time(s) on the driver; a schema probe that "
        "executes the plan is what does that, and on a real inference job the driver is the "
        "one machine with no device and no room for the corpus"
    )
    # The schema still reaches the commit -- it is the only reason it was ever asked for.
    assert manifest.schema is not None and manifest.schema.names == ["pred"]


def test_the_written_scores_are_the_single_node_answer(parquet_path, cluster_tmp_dir):
    """Distributed == single-node, so the fix is not buying speed with a different answer."""
    out = str(cluster_tmp_dir / "scores_dist_equiv")
    scored = bt.read.parquet(parquet_path).map_batches(
        Model, output_columns=["pred"], batch_format="pyarrow", concurrency=2
    )
    scored.write.parquet(out, distributed=True)
    expected = sorted(scored.collect(distributed=False).to_pydict()["pred"])
    assert sorted(bt.read.parquet(out).collect().to_pydict()["pred"]) == expected
