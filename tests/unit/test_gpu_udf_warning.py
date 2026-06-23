"""GPU map_batches with a plain function warns (the model-reload-per-batch foot-gun).

Ray Data's most common inference mistake is a plain-function UDF on a GPU stage:
the model reloads on every batch. Batcher warns at plan-construction time and
points at the class/factory spelling that loads once per worker. CPU stages and
class UDFs do not warn. Pure plan construction — no engine needed.
"""

from __future__ import annotations

import warnings

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PerformanceWarning


class _Model:
    def __call__(self, batch):  # loaded once per worker
        return batch


def test_gpu_plain_function_warns():
    ds = bt.from_arrow(pa.table({"x": [1, 2, 3]}))
    with pytest.warns(PerformanceWarning, match="once per worker"):
        ds.ml.map_batches(lambda b: b, num_gpus=1, output_columns=["x"])


def test_gpu_class_does_not_warn():
    ds = bt.from_arrow(pa.table({"x": [1, 2, 3]}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        ds.ml.map_batches(_Model, num_gpus=1, output_columns=["x"])


def test_cpu_plain_function_does_not_warn():
    ds = bt.from_arrow(pa.table({"x": [1, 2, 3]}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds.ml.map_batches(lambda b: b, output_columns=["x"])  # num_gpus=0 default
