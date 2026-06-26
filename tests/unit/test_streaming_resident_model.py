"""A streaming inference query loads its model once, not once per micro-batch.

`api/streaming.py::_build_run_batch` prebuilds the `map_batches` class UDFs once for the
whole stream (`core.prebuild_factories`), so a load-once model is reused across every
micro-batch trigger instead of reloaded each time (the resident-inference contract).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.streaming import _build_run_batch

pytestmark = pytest.mark.unit

_BUILDS: list[int] = []


class _CountingModel:
    """A load-once model UDF: counts how many times it is instantiated."""

    def __init__(self) -> None:
        _BUILDS.append(1)

    def __call__(self, batch: pa.RecordBatch) -> dict:
        d = batch.to_pydict()
        d["y"] = [v + 1 for v in d["x"]]
        return d


def test_model_built_once_across_micro_batches():
    _BUILDS.clear()
    ds = bt.from_pydict({"x": [1, 2, 3]}).ml.map_batches(_CountingModel)
    run_batch = _build_run_batch(ds._plan, ds._sources)
    assert _BUILDS == [1]  # built once when the stream's runner is constructed

    micro = pa.RecordBatch.from_pydict({"x": [10, 11]})
    out = None
    for _ in range(4):  # four micro-batch triggers
        out = run_batch(micro)
    assert _BUILDS == [1]  # still once — reused, never reloaded per trigger
    assert out[0].to_pydict()["y"] == [11, 12]
