"""Fixed-shape tensor columns — multi-dimensional arrays as one Arrow column.

A column where every row is an ``N``-dimensional tensor of the *same* shape (an
image ``(H, W, C)``, a patch grid, a feature map). Batcher uses Arrow's **canonical**
``arrow.fixed_shape_tensor`` extension type so the shape travels with the data: it is
stored as a ``FixedSizeList`` of the value type and carries its shape in the Arrow
field metadata, so a column built here round-trips through the zero-copy FFI boundary
into the Rust engine and back with its shape intact — no IR tag, no two-sided
contract. Choosing the canonical type (over a bespoke one) is what makes that free.

This is the storage half of the ML tensor path: `from_numpy` and the NumPy/image
readers build these columns, and `batcher.ml.loader.column_to_tensor` turns one back
into a correctly-shaped, zero-copy-friendly torch tensor for a training loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    import numpy as np

__all__ = ["as_tensor_column", "is_tensor_column", "tensor_type", "to_tensor_column"]


def as_tensor_column(storage: pa.Array, shape: tuple[int, ...]) -> pa.Array:
    """Reinterpret a `FixedSizeList` array as a fixed-shape-tensor column (zero-copy).

    The decode kernels produce a flat ``FixedSizeList<value, prod(shape)>``; wrapping
    it in the canonical tensor type attaches the per-row `shape` with no data copy, so
    it converts to a correctly-shaped ``(n, *shape)`` tensor downstream.

    Args:
        storage: a ``FixedSizeListArray`` whose list size equals ``prod(shape)``.
        shape: the per-row tensor shape (e.g. ``(224, 224, 3)``).

    Raises:
        ValueError: if `storage` is not a fixed-size-list array.
    """
    if not pa.types.is_fixed_size_list(storage.type):
        raise ValueError(f"as_tensor_column expects a FixedSizeList, got {storage.type}")
    ext = tensor_type(storage.type.value_type, shape)
    # Align the storage to the extension's exact storage type (e.g. the engine emits a
    # non-nullable item field; the canonical tensor type wants a nullable one).
    if storage.type != ext.storage_type:
        storage = storage.cast(ext.storage_type)
    return pa.ExtensionArray.from_storage(ext, storage)


def tensor_type(value_type: pa.DataType, shape: tuple[int, ...]) -> pa.DataType:
    """The Arrow fixed-shape-tensor type for rows of `value_type` with `shape`.

    Args:
        value_type: the per-element Arrow type (e.g. ``pa.uint8()``, ``pa.float32()``).
        shape: the per-row tensor shape (e.g. ``(224, 224, 3)``); must be non-empty.

    Returns:
        A ``pyarrow.FixedShapeTensorType`` (canonical ``arrow.fixed_shape_tensor``),
        backed by a ``FixedSizeList`` of size ``prod(shape)``.

    Raises:
        ValueError: if `shape` is empty.
    """
    if not shape:
        raise ValueError("tensor_type requires a non-empty shape")
    return pa.fixed_shape_tensor(value_type, list(shape))


def to_tensor_column(ndarray: np.ndarray) -> pa.Array:
    """Build a fixed-shape-tensor column from a NumPy array.

    The array's **leading axis is the row axis**: an ``(n, *shape)`` array becomes a
    length-``n`` column whose rows are ``shape``-shaped tensors. The array is made
    C-contiguous first (Arrow stores tensors row-major).

    Args:
        ndarray: an array with ``ndim >= 2`` (``shape[0]`` rows of ``shape[1:]``).

    Returns:
        A ``pyarrow.FixedShapeTensorArray``.

    Raises:
        ValueError: if `ndarray` has fewer than 2 dimensions.
    """
    import numpy as np

    arr = np.ascontiguousarray(ndarray)
    if arr.ndim < 2:
        raise ValueError(f"to_tensor_column needs ndim >= 2 (rows, *shape), got {arr.ndim}-D")
    return pa.FixedShapeTensorArray.from_numpy_ndarray(arr)


def is_tensor_column(array: pa.Array | pa.ChunkedArray | pa.DataType) -> bool:
    """Whether `array` (or its type) is a fixed-shape-tensor column."""
    dtype = array if isinstance(array, pa.DataType) else array.type
    return isinstance(dtype, pa.FixedShapeTensorType)
