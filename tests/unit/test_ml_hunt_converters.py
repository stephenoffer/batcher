"""Bug-hunt regressions for the Arrow -> NumPy / torch conversion path (ml/converters, ml/loader).

The defect: a fixed-size-list column with a NULL row was flattened with
``pa.Array.flatten()``, which *drops* the null row's slot entirely. The reshape
``(-1, width)`` then produced fewer rows than the batch, silently misaligning that
column against its sibling columns (a feature row sliding out from under its label) —
data corruption for ML training. The offset-aware child slice keeps every row and
surfaces a null row as NaN.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from batcher.ml.converters import _column_to_numpy, to_numpy_batches


def _fsl(values: list, width: int, dtype: pa.DataType) -> pa.Array:
    return pa.array(values, type=pa.list_(dtype, width))


def test_fixed_size_list_null_row_keeps_row_count() -> None:
    # A null row in the middle must not shrink the column.
    arr = _fsl([[1.0, 2.0, 3.0], None, [7.0, 8.0, 9.0]], 3, pa.float32())
    out = _column_to_numpy(arr)
    assert out.shape == (3, 3)
    assert np.isnan(out[1]).all()
    np.testing.assert_array_equal(out[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(out[2], np.array([7.0, 8.0, 9.0], dtype=np.float32))


def test_fixed_size_list_null_row_stays_aligned_with_siblings() -> None:
    # This is the corruption: feature column and label column must have the same length,
    # and each feature row must still correspond to its own label.
    feat = _fsl([[1, 2], None, [5, 6]], 2, pa.int64())
    lab = pa.array([10, 20, 30], type=pa.int64())
    batch = pa.record_batch({"feat": feat, "label": lab})
    d = next(to_numpy_batches([batch]))
    assert d["feat"].shape[0] == d["label"].shape[0] == 3
    # The label at the null-feature position is preserved (not shifted onto row 2's feature).
    np.testing.assert_array_equal(d["label"], np.array([10, 20, 30]))
    np.testing.assert_array_equal(d["feat"][0], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(d["feat"][2], np.array([5.0, 6.0]))


def test_fixed_size_list_sliced_batch_respects_offset() -> None:
    # A sliced (offset) fixed-size-list column must read its own rows, not the buffer head.
    arr = _fsl([[1, 2], [3, 4], [5, 6], [7, 8]], 2, pa.int64())
    out = _column_to_numpy(arr.slice(2, 2))
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, np.array([[5, 6], [7, 8]]))


def test_fixed_size_list_no_nulls_preserves_integer_dtype() -> None:
    arr = _fsl([[1, 2], [3, 4]], 2, pa.int64())
    out = _column_to_numpy(arr)
    assert out.dtype.kind == "i"
    np.testing.assert_array_equal(out, np.array([[1, 2], [3, 4]]))
