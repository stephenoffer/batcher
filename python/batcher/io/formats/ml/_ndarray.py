"""NumPy-slice → Arrow conversion shared by the HDF5 and Zarr array readers.

Both formats expose a dense N-D array whose leading axis is the row axis, and both
map it to Arrow the same way: a 1-D array becomes one ``value`` column, a 2-D array
one column per trailing index (``c0``, ``c1``, …). Centralized here so the two
readers cannot drift, and so a schema can be derived from the array's *metadata*
(dtype + shape) without reading a single chunk.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

__all__ = ["schema_from_array_meta", "slice_to_batch"]


def slice_to_batch(array: Any, projection: list[str] | None) -> pa.RecordBatch:
    """Turn an in-memory numpy slice into an Arrow record batch."""
    if array.ndim == 1:
        data = {"value": array}
    else:
        data = {f"c{i}": array[:, i] for i in range(array.shape[1])}
    batch = pa.RecordBatch.from_pydict({k: pa.array(v) for k, v in data.items()})
    return batch.select(projection) if projection is not None else batch


def schema_from_array_meta(array: Any) -> pa.Schema:
    """The Arrow schema of `array` from its dtype and shape alone — reading no data.

    Mirrors `slice_to_batch` field-for-field for the 1-D and 2-D layouts it supports,
    so a reader answers `schema()` from the array's stored metadata instead of pulling
    a chunk (Zarr) or a hyperslab (HDF5) just to learn a schema the format already
    states. Falls back to an *empty*-slice read for anything it cannot type from
    metadata alone (a >2-D array, or a dtype ``pyarrow`` cannot map) — which still
    reads no rows and reproduces the exact behavior the old path had there.
    """
    try:
        value_type = pa.from_numpy_dtype(array.dtype)
    except (pa.ArrowNotImplementedError, NotImplementedError, TypeError):
        return slice_to_batch(array[0:0], None).schema
    if array.ndim == 1:
        return pa.schema([("value", value_type)])
    if array.ndim == 2:
        return pa.schema([(f"c{i}", value_type) for i in range(array.shape[1])])
    return slice_to_batch(array[0:0], None).schema
