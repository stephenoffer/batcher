"""Regression: `serving_udf` must shape a numeric fixed-size-list feature column as ``(N, W)``.

`ml.serving.base._column_to_numpy` promises tensor columns "keep their ``(N, *shape)``
form" (the same contract the training/loader path honors). It previously handled only the
`FixedShapeTensor` extension type and fell through to ``to_numpy`` for a numeric
``FixedSizeList<T, W>`` feature/embedding column — which yields an opaque dtype=object
array of per-row arrays, so a vectorized serving model silently received the wrong shape.
The two other conversion sites (`ml.converters`, `ml.loader.tensors`) reshape it to
``(N, W)``; this test pins serving to the same behavior.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from batcher.ml.serving.base import _column_to_numpy, serving_udf


def test_fixed_size_list_feature_column_is_2d_float() -> None:
    """A numeric ``FixedSizeList<T, W>`` column converts to a real ``(N, W)`` float array."""
    vec = pa.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type=pa.list_(pa.float64(), 3))
    arr = _column_to_numpy(vec)
    assert arr.dtype.kind == "f"  # a real float matrix, not an object array
    assert arr.ndim == 2
    assert arr.shape == (2, 3)
    np.testing.assert_array_equal(arr, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_serving_udf_feeds_model_a_real_tensor_and_stays_aligned() -> None:
    """The model sees a 2-D float feature matrix, and its output stays row-aligned."""
    vec = pa.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type=pa.list_(pa.float64(), 3))
    batch = pa.record_batch({"f": vec, "id": [10, 20]})

    seen: dict[str, object] = {}

    class _Client:
        def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            f = inputs["f"]
            seen["kind"] = f.dtype.kind
            seen["ndim"] = f.ndim
            # A vectorized model does real array math — impossible on an object array.
            return {"y": f.astype(np.float64) @ np.ones(3)}

    udf = serving_udf(lambda: _Client(), input_columns=["f"])()
    out = udf(batch)

    assert seen["kind"] == "f"  # a real float matrix, not an object array
    assert seen["ndim"] == 2
    assert out.column("y").to_pylist() == [6.0, 15.0]
    assert out.column("id").to_pylist() == [10, 20]  # sibling column stays aligned


def test_fixed_shape_tensor_column_still_keeps_shape() -> None:
    """The prior `FixedShapeTensor` handling is preserved (no regression)."""
    from batcher.io.formats.ml.tensor import to_tensor_column

    tc = to_tensor_column(np.arange(6, dtype=np.float32).reshape(2, 3))
    arr = _column_to_numpy(tc)
    assert arr.shape == (2, 3)
    assert arr.dtype == np.dtype(np.float32)
