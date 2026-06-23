"""Fixed-shape-tensor columns — construction, ingestion, and tensor conversion.

Locks the Phase-0 tensor path: `from_numpy` / the NumPy reader build canonical
``arrow.fixed_shape_tensor`` columns for rank >= 2 rows, the helpers in
`batcher.io.tensor_type` classify them, and `column_to_tensor` takes the fast
``to_numpy_ndarray`` branch back to a correctly-shaped torch tensor.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.io.formats.ml.tensor import (
    as_tensor_column,
    is_tensor_column,
    tensor_type,
    to_tensor_column,
)


def test_tensor_type_factory():
    t = tensor_type(pa.uint8(), (2, 3))
    assert isinstance(t, pa.FixedShapeTensorType)
    assert t.shape == [2, 3]


def test_tensor_type_rejects_empty_shape():
    with pytest.raises(ValueError, match="non-empty shape"):
        tensor_type(pa.float32(), ())


def test_to_tensor_column_and_classify():
    arr = np.arange(4 * 2 * 3, dtype=np.uint8).reshape(4, 2, 3)
    col = to_tensor_column(arr)
    assert is_tensor_column(col)
    assert col.to_numpy_ndarray().shape == (4, 2, 3)


def test_to_tensor_column_needs_rank_two():
    with pytest.raises(ValueError, match="ndim >= 2"):
        to_tensor_column(np.arange(5, dtype=np.uint8))


def test_is_tensor_column_false_for_plain():
    assert not is_tensor_column(pa.array([1, 2, 3]))
    assert not is_tensor_column(pa.int64())


def test_as_tensor_column_reinterprets_fixed_size_list():
    # A non-nullable item field (as the decode kernel emits) must still wrap cleanly.
    typ = pa.list_(pa.field("item", pa.uint8(), nullable=False), 12)
    storage = pa.FixedSizeListArray.from_arrays(pa.array(range(2 * 12), type=pa.uint8()), type=typ)
    ext = as_tensor_column(storage, (2, 2, 3))
    assert is_tensor_column(ext)
    assert ext.type.shape == [2, 2, 3]
    assert ext.to_numpy_ndarray().shape == (2, 2, 2, 3)


def test_as_tensor_column_rejects_non_list():
    with pytest.raises(ValueError, match="FixedSizeList"):
        as_tensor_column(pa.array([1, 2, 3]), (3,))


def test_from_numpy_4d_is_tensor_column():
    imgs = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    tbl = bt.from_numpy(imgs, column="img").collect()
    field = tbl.schema.field("img")
    assert is_tensor_column(field.type)
    assert field.type.shape == [2, 2, 3]
    assert np.array_equal(tbl.column("img").combine_chunks().to_numpy_ndarray(), imgs)


def test_from_numpy_2d_stays_fixed_size_list():
    emb = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    tbl = bt.from_numpy(emb, column="e").collect()
    assert pa.types.is_fixed_size_list(tbl.schema.field("e").type)


def test_numpy_reader_preserves_shape(tmp_path):
    imgs = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    path = tmp_path / "imgs.npy"
    np.save(path, imgs)
    tbl = bt.read.numpy(str(path)).collect()
    field = tbl.schema.field("data")
    assert is_tensor_column(field.type)
    assert field.type.shape == [4, 4]
    assert np.array_equal(tbl.column("data").combine_chunks().to_numpy_ndarray(), imgs)


def test_column_to_tensor_fast_path_shape():
    torch = pytest.importorskip("torch")
    imgs = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)
    tbl = bt.from_numpy(imgs, column="img").collect()
    from batcher.ml.loader import column_to_tensor

    t = column_to_tensor(tbl.column("img"))
    assert isinstance(t, torch.Tensor)
    assert tuple(t.shape) == (4, 2, 2, 3)
    # Writable copy decoupled from the immutable Arrow buffer.
    t[0, 0, 0, 0] = 99
    assert np.array_equal(tbl.column("img").combine_chunks().to_numpy_ndarray(), imgs)
