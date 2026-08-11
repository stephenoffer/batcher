"""Variable-shape tensor columns — arrays of differing shape in one Arrow column.

The sibling of `tensor`, for the case it cannot express: a column where each row is an
``N``-dimensional array but the shapes differ from row to row. Mixed-resolution images are
the ordinary case in multimodal preprocessing — decode a folder of photos and no two are
``(H, W, 3)`` for the same ``H`` and ``W`` — and Arrow's canonical fixed-shape tensor type
requires one shape for the whole column, so such a batch could not be typed at all. The error
it produced named the problem accurately and left the user to solve it by resizing or by
keeping the bytes encoded.

The layout is a plain struct, ``{data: binary, shape: list<int32>, dtype: string}``: the row's
buffer row-major, the row's shape, and the element type that reads the buffer back. That
choice is the point. It is ordinary Arrow, so it crosses the zero-copy FFI boundary, writes to
Parquet, survives a shuffle, and passes through every relational operator **with no engine
change, no IR tag, and no wire-contract change** — a struct of a binary and two small fields
is nothing the engine has to learn. A bespoke extension type would have needed the Rust side
to know it.

**`data` is a binary buffer and not a list of elements**, which is not a detail. The FFI
boundary normalizes narrow numerics on the way in (Int8/16/32 → Int64, Float16/32 → Float64),
so a ``list<uint8>`` image column arrives as ``list<int64>`` — eight bytes per pixel, for the
one workload this type exists to carry. A binary column is passed through untouched, so a
uint8 image stays one byte per pixel and a float32 embedding stays four.

What it deliberately does not do is make a ragged column a *tensor* to the frameworks
downstream. There is no single torch tensor for rows of differing shape, so the bridges decode
it to per-row arrays rather than silently padding.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "is_ragged_tensor_column",
    "ragged_from_values",
    "ragged_tensor_type",
    "ragged_to_numpy",
    "to_ragged_tensor_column",
]

_DATA = "data"
_SHAPE = "shape"
_DTYPE = "dtype"

#: The struct fields, in order. `dtype` is a per-row string rather than column metadata
#: because Arrow field metadata is not guaranteed across the FFI boundary or a Parquet round
#: trip, and a column that cannot say how to read its own bytes is a column that decodes
#: wrongly rather than loudly. It is one short, highly repetitive string per row, which every
#: encoder this data meets compresses to nothing.
_FIELDS = (
    pa.field(_DATA, pa.binary()),
    pa.field(_SHAPE, pa.list_(pa.int32())),
    pa.field(_DTYPE, pa.string()),
)


def ragged_tensor_type() -> pa.DataType:
    """The Arrow type of a variable-shape tensor column.

    Returns:
        ``struct<data: binary, shape: list<int32>, dtype: string>``.
    """
    return pa.struct(_FIELDS)


def is_ragged_tensor_column(array: pa.Array | pa.ChunkedArray | pa.DataType) -> bool:
    """Whether `array` (or type) is a variable-shape tensor column built by this module.

    Structural, not nominal: a struct whose fields are exactly ``data`` (binary), ``shape``
    (a list of integers) and ``dtype`` (a string). Nothing else in the engine produces that
    shape, and matching on it rather than on a name means a column that survived a Parquet
    round trip is still recognized.

    Every check accepts the *large* variant of its type, and the field order is compared as a
    set. Both matter in practice rather than in theory: a round trip through polars widens
    `binary` to `large_binary` and `list` to `large_list`, and one through pandas reorders the
    struct's fields. A stricter match would have quietly stopped recognizing the column after
    a `batch_format` conversion, which is where it is most likely to be handled.

    Args:
        array: An Arrow array, chunked array, or data type.

    Returns:
        True when the value is a variable-shape tensor column.
    """
    dtype = array if isinstance(array, pa.DataType) else array.type
    if not pa.types.is_struct(dtype) or dtype.num_fields != len(_FIELDS):
        return False
    names = {dtype.field(i).name for i in range(dtype.num_fields)}
    if names != {_DATA, _SHAPE, _DTYPE}:
        return False
    data, shape, kind = dtype.field(_DATA), dtype.field(_SHAPE), dtype.field(_DTYPE)
    if not (pa.types.is_binary(data.type) or pa.types.is_large_binary(data.type)):
        return False
    listish = pa.types.is_list(shape.type) or pa.types.is_large_list(shape.type)
    if not (listish and pa.types.is_integer(shape.type.value_type)):
        return False
    return pa.types.is_string(kind.type) or pa.types.is_large_string(kind.type)


def to_ragged_tensor_column(arrays: Sequence[Any]) -> pa.Array:
    """Build a variable-shape tensor column from a sequence of NumPy arrays.

    Every array is stored as its own row-major buffer beside its own shape and dtype, so the
    rows need not agree on shape, rank, or element type. A `None` row is null in every field,
    which is what keeps a ragged column nullable like any other.

    Args:
        arrays: One NumPy array (or `None`) per row.

    Returns:
        A `StructArray` of the type `ragged_tensor_type` describes.
    """
    import numpy as np

    data: list[bytes | None] = []
    shapes: list[list[int] | None] = []
    kinds: list[str | None] = []
    for item in arrays:
        if item is None:
            data.append(None)
            shapes.append(None)
            kinds.append(None)
            continue
        arr = np.ascontiguousarray(item)
        data.append(arr.tobytes())
        shapes.append(list(arr.shape))
        kinds.append(arr.dtype.str)
    dtype = ragged_tensor_type()
    return pa.StructArray.from_arrays(
        [
            pa.array(data, type=dtype.field(_DATA).type),
            pa.array(shapes, type=dtype.field(_SHAPE).type),
            pa.array(kinds, type=dtype.field(_DTYPE).type),
        ],
        fields=list(_FIELDS),
    )


def ragged_to_numpy(array: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Decode a variable-shape tensor column to a NumPy object array of per-row arrays.

    An object array rather than one stacked array, because rows of differing shape have no
    stacked form. Each element is a real `ndarray` with the row's own shape and dtype; a null
    row decodes to `None`.

    Args:
        array: A column `is_ragged_tensor_column` accepts.

    Returns:
        A 1-D object array of length ``len(array)``.
    """
    import numpy as np

    combined = array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array
    rows = combined.to_pylist()
    out = np.empty(len(rows), dtype=object)
    for index, row in enumerate(rows):
        if row is None or row.get(_DATA) is None:
            out[index] = None
            continue
        flat = np.frombuffer(row[_DATA], dtype=np.dtype(row[_DTYPE]))
        out[index] = flat.reshape(tuple(row[_SHAPE]))
    return out


def ragged_from_values(values: Sequence[Any]) -> pa.Array | None:
    """A ragged column for `values`, or `None` when they are not ragged NumPy arrays.

    The predicate the conversion paths call before giving up on a column: it answers "are
    these several NumPy arrays that only differ in shape?" and builds the column if so. Rows
    that disagree on *rank* are included — a column of 2-D and 3-D arrays is still ragged, and
    excluding it would leave the harder half of the problem unsolved.

    **A column of 1-D arrays declines**, for the same compatibility reason
    `tensor.tensor_from_values` declines it: ``[vec, other_vec]`` already converts to a
    ``list<T>`` column, ragged lengths and all, so claiming it here would change the schema of
    every existing embedding column. Only the shapes Arrow genuinely cannot type are claimed.

    Args:
        values: The column's values, as handed to Arrow.

    Returns:
        The built column, or None if this is not a ragged-array column.
    """
    import numpy as np

    if not isinstance(values, Sequence) or isinstance(values, str | bytes) or not values:
        return None
    # Cheap gate first. This runs on every column of every `from_pydict`, so a column of
    # numbers or strings must cost one `isinstance` rather than a full scan.
    head = next((v for v in values if v is not None), None)
    if not isinstance(head, np.ndarray):
        return None
    present = [v for v in values if v is not None]
    if not all(isinstance(v, np.ndarray) and v.ndim >= 2 for v in present):
        return None
    if len({v.shape for v in present}) < 2:
        return None  # one shape — the fixed-shape tensor column is the right answer
    return to_ragged_tensor_column(list(values))
