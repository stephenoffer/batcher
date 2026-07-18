"""Framework converters — Arrow batches → NumPy / PyTorch."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from batcher.io.formats.ml.tensor import to_tensor_column
from batcher.ml import to_numpy_batches, to_tf_dataset, to_torch_iterable


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


def test_to_tf_dataset_feature_column_keeps_inner_shape():
    """A fixed-size-list feature column must survive `to_tf_dataset` as `(None, W)`.

    Regression: the output signature pinned every column to `shape=(None,)`, so a 2-D
    feature column made `from_generator` raise `InvalidArgumentError` ("element of shape
    (n, W) where an element of shape (None,) was expected") the moment it was iterated —
    a hard crash on exactly the multi-dimensional columns tf.data is used for.
    """
    pytest.importorskip("tensorflow")
    feat = pa.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type=pa.list_(pa.float64(), 3))
    label = pa.array([0, 1], type=pa.int64())
    batch = pa.RecordBatch.from_arrays([feat, label], names=["features", "label"])

    elements = list(to_tf_dataset([batch]))
    assert len(elements) == 1
    element = elements[0]
    assert element["features"].shape.as_list() == [2, 3]
    assert element["label"].shape.as_list() == [2]
    np.testing.assert_array_equal(
        element["features"].numpy(), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    # Row alignment: feature row i lines up with label row i.
    np.testing.assert_array_equal(element["label"].numpy(), np.array([0, 1]))


def test_to_tf_dataset_image_tensor_column_keeps_shape():
    """A fixed-shape-tensor (image) column must survive `to_tf_dataset` as `(None, H, W, C)`."""
    pytest.importorskip("tensorflow")
    nd = np.arange(2 * 2 * 2 * 3).reshape(2, 2, 2, 3).astype(np.uint8)
    batch = pa.RecordBatch.from_arrays([to_tensor_column(nd)], names=["image"])

    element = next(iter(to_tf_dataset([batch])))
    assert element["image"].shape.as_list() == [2, 2, 2, 3]
    np.testing.assert_array_equal(element["image"].numpy(), nd)


def test_to_tf_dataset_plain_numeric_unchanged():
    """A plain 1-D numeric column still comes through as `(None,)`."""
    pytest.importorskip("tensorflow")
    element = next(iter(to_tf_dataset(_batches())))
    assert element["a"].shape.as_list() == [2]
    np.testing.assert_array_equal(element["a"].numpy(), np.array([1, 2]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
