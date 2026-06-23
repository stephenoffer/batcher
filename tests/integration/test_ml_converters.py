"""Framework converters — Arrow batches → NumPy / PyTorch."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from batcher.ml import to_numpy_batches, to_torch_iterable


def _batches() -> list[pa.RecordBatch]:
    return [
        pa.RecordBatch.from_arrays(
            [pa.array([1, 2], type=pa.int64()), pa.array([1.5, 2.5], type=pa.float64())],
            names=["a", "b"],
        ),
        pa.RecordBatch.from_arrays(
            [pa.array([3], type=pa.int64()), pa.array([3.5], type=pa.float64())],
            names=["a", "b"],
        ),
    ]


def test_to_numpy_batches_all_columns():
    out = list(to_numpy_batches(_batches()))
    assert len(out) == 2
    assert out[0].keys() == {"a", "b"}
    np.testing.assert_array_equal(out[0]["a"], np.array([1, 2]))
    np.testing.assert_array_equal(out[1]["b"], np.array([3.5]))


def test_to_numpy_batches_column_subset():
    out = list(to_numpy_batches(_batches(), columns=["a"]))
    assert all(d.keys() == {"a"} for d in out)


def test_to_torch_iterable():
    torch = pytest.importorskip("torch")
    ds = to_torch_iterable(_batches())
    items = list(iter(ds))
    assert len(items) == 2
    assert torch.equal(items[0]["a"], torch.tensor([1, 2]))
    assert torch.equal(items[1]["b"], torch.tensor([3.5]))


def test_to_torch_iterable_skips_non_numeric():
    pytest.importorskip("torch")
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1], type=pa.int64()), pa.array(["hi"], type=pa.string())],
        names=["n", "s"],
    )
    ds = to_torch_iterable([batch])
    item = next(iter(ds))
    assert "n" in item and "s" not in item  # string column dropped from tensors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
