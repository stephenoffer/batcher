"""A `map_batches(Model).write.parquet(...)` must not build the model on the driver.

Scoring a corpus and writing the scores is the canonical batch-inference job, and it ran the
UDF on the **driver** before writing a row. Not as the write: as a *schema probe*.
`WriteManifest.schema` is documented as attached by "the driver, which knows the plan's output
type", and for a `map_batches` stage the driver does not -- an arbitrary Python callable's
output type is only knowable by running it. So `_write` called `_schema`, whose last resort
executes the plan under a zero-row limit, which builds the UDF.

For a model that is a model load on a machine chosen for orchestration rather than for
inference. Measured on a GPU-less head node against a real 8xT4 fleet: the query died inside
`cupy` with `cudaErrorInsufficientDriver` on a 1.9 GiB input, and on a 29 GiB one the driver
was OOM-killed at 16.7 GB. Both before a single row was written, and neither with a message
naming the driver.

These run with no cluster and no device: the point is *where* the model is built, and a
counter in `__init__` sees that on any machine.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit

_BUILDS: list[int] = []


class Model:
    """A load-once factory that records every construction, wherever it happens."""

    def __init__(self) -> None:
        _BUILDS.append(1)

    def __call__(self, batch):
        return {"pred": np.asarray(batch["x"], dtype="float64") * 2.0}


@pytest.fixture(autouse=True)
def _fresh():
    _BUILDS.clear()
    yield
    _BUILDS.clear()


def _scored():
    table = pa.table({"x": pa.array(np.arange(64, dtype="float64"))})
    return bt.from_arrow(table).map_batches(Model, output_columns=["pred"], batch_format="numpy")


def test_the_declared_schema_never_executes_the_plan():
    # The static half of `_schema`, split out so the write can ask the cheap question alone.
    # A `map_batches` output type is genuinely unknowable statically, so it must answer `None`
    # rather than reach for the engine.
    from batcher.api.terminal.core import _declared_schema

    ds = _scored()
    assert _declared_schema(ds._plan, ds._sources) is None
    assert _BUILDS == [], "the static analysis must not build the UDF"
    # A bare scan is the arm it can answer, and it answers without touching the engine.
    plain = bt.from_arrow(pa.table({"x": pa.array([1.0, 2.0])}))
    assert _declared_schema(plain._plan, plain._sources).names == ["x"]


def test_a_worker_reports_the_schema_it_wrote(tmp_path):
    # The mechanism the fix rests on: the shard writer returns `(locators, schema)`, so the
    # driver gets the output type from the process that produced the rows instead of deriving
    # it by running the UDF on itself.
    from batcher.dist.executors.map import _write_udf_output

    batch = pa.record_batch({"pred": pa.array([1.0, 2.0, 3.0])})
    spec = {"fmt": "parquet", "sink_kwargs": None, "path": str(tmp_path / "shard"), "shards": 1}
    files, schema = _write_udf_output([batch], spec, 0)
    assert files and schema is not None
    assert schema.names == ["pred"]
