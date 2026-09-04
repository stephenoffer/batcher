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
    columns = _columns(array)
    batch = pa.RecordBatch.from_pydict(columns)
    return batch.select(projection) if projection is not None else batch


def _columns(array: Any) -> dict[str, pa.Array]:
    """The Arrow columns one slice becomes, by this reader's layout rules.

    A **compound** dataset is a table and becomes one column per field. HDF5 stores a C
    struct that way, which is how most instrument, simulation and genomics files record
    their rows, and it is 1-D — so it took the ``value`` branch below, handed a `void`
    array to `pa.array`, and failed the read with ``ArrowNotImplementedError: Unsupported
    numpy type 20``: a message naming neither the dataset's fields nor the fact that they
    were the columns being asked for.

    The field split is `io.interop.structured_to_columns`, the same one `bt.from_numpy` and
    the `.npy` reader use, so a compound dataset reads as the table `bt.from_numpy` would
    make of the identical array — and a sub-array field keeps the embedding convention
    rather than being flattened.
    """
    from batcher.io.interop import structured_to_columns

    fields = structured_to_columns(array)
    if fields is not None:
        return fields
    if array.ndim == 1:
        data = {"value": array}
    else:
        data = {f"c{i}": array[:, i] for i in range(array.shape[1])}
    return {k: pa.array(v) for k, v in data.items()}


def schema_from_array_meta(array: Any) -> pa.Schema:
    """The Arrow schema of `array` from its dtype and shape alone — reading no data.

    Mirrors `slice_to_batch` field-for-field for the 1-D and 2-D layouts it supports,
    so a reader answers `schema()` from the array's stored metadata instead of pulling
    a chunk (Zarr) or a hyperslab (HDF5) just to learn a schema the format already
    states. Falls back to an *empty*-slice read for anything it cannot type from
    metadata alone (a >2-D array, or a dtype ``pyarrow`` cannot map) — which still
    reads no rows and reproduces the exact behavior the old path had there.
    """
    if array.dtype.names is not None:
        # A compound dtype has no single `value_type`; its schema is its fields, which the
        # empty-slice read below derives exactly and without touching a chunk.
        return slice_to_batch(array[0:0], None).schema
    try:
        value_type = pa.from_numpy_dtype(array.dtype)
    except (pa.ArrowNotImplementedError, NotImplementedError, TypeError):
        return slice_to_batch(array[0:0], None).schema
    if array.ndim == 1:
        return pa.schema([("value", value_type)])
    if array.ndim == 2:
        return pa.schema([(f"c{i}", value_type) for i in range(array.shape[1])])
    return slice_to_batch(array[0:0], None).schema
