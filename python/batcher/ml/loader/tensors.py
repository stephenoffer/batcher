"""Arrow → torch conversion, and moving the result to a device.

The leaf of the loader package: everything here turns one Arrow column or batch into tensors,
and nothing here knows about ranks, epochs, or shards. Split out of `ml/loader.py` when it
outgrew the module limit; the seam is "convert a batch" vs "decide which batch."
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["column_to_tensor"]


def column_to_tensor(array: pa.Array) -> Any | None:
    """Convert one Arrow column to a torch tensor, or ``None`` if not tensorizable.

    A `FixedShapeTensor` column becomes a shaped tensor; numeric columns convert; strings and
    other types return ``None`` so the caller drops them. The tensor owns **writable** memory
    decoupled from the (immutable) Arrow buffer, because a training loop mutates batches in
    place — sharing the buffer is "undefined behavior" per torch, and can corrupt the source.
    """
    import numpy as np
    import pyarrow as pa
    import torch

    # Collapse a ChunkedArray so an extension array exposes its typed `to_numpy_ndarray`.
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()

    if hasattr(array, "to_numpy_ndarray"):  # FixedShapeTensor extension array
        nd = array.to_numpy_ndarray()
    elif pa.types.is_fixed_size_list(array.type) and pa.types.is_primitive(array.type.value_type):
        # A feature/embedding vector stored as FixedSizeList<T, W> → an (n, W) tensor.
        # `to_numpy` would give an object array of per-row arrays; reshape the flat child
        # buffer instead, so the column is a real 2-D tensor rather than silently dropped.
        w = array.type.list_size
        n = len(array)
        # Slice the child by (offset, length) instead of `flatten()`, which drops null
        # rows and would return fewer than `n` rows — silently misaligning this column
        # against its siblings (a feature row sliding out from under its label).
        child = array.values.slice(array.offset * w, n * w).to_numpy(zero_copy_only=False)
        if child.dtype.kind not in "biuf":
            return None
        nd = child.reshape(n, w)
    else:
        try:
            nd = array.to_numpy(zero_copy_only=False)
        except (ValueError, TypeError):
            return None
        if nd.dtype.kind not in "biuf":  # bool/int/uint/float only
            return None
    # `copy=True` guarantees an owned, contiguous, writable buffer (Arrow's is read-only).
    return torch.from_numpy(np.array(nd, copy=True))


def tensorize(batch: Any, keep: list[str]) -> dict[str, Any]:
    """One batch as a `{column: tensor}` dict, dropping the non-tensorizable columns."""
    out = {}
    for c in keep:
        t = column_to_tensor(batch.column(c))
        if t is not None:
            out[c] = t
    return out


def to_torch_out(
    arrays: dict, arrays_to_torch: Any, collate_fn: Any, device: Any, pin_memory: bool = False
) -> Any:
    """One `{col: ndarray}` batch as the yielded output, optionally moved to `device`."""
    out = collate_fn(arrays) if collate_fn is not None else arrays_to_torch(arrays)
    if device is None:
        return out
    # Apple MPS has no 64-bit tensors; downcast so `device="auto"` works on Apple silicon.
    is_mps = str(device).startswith("mps")

    def _move(t: Any) -> Any:
        if not hasattr(t, "to"):
            return t
        if is_mps:
            t = _mps_safe_dtype(t)
        if pin_memory and hasattr(t, "pin_memory"):
            t = t.pin_memory()
        return t.to(device, non_blocking=pin_memory)

    if isinstance(out, dict):
        return {k: _move(v) for k, v in out.items()}
    return _move(out)


def _mps_safe_dtype(tensor: Any) -> Any:
    """Downcast a 64-bit tensor to 32-bit (MPS supports no 64-bit dtypes)."""
    import torch

    if tensor.dtype == torch.float64:
        return tensor.to(torch.float32)
    if tensor.dtype == torch.int64:
        return tensor.to(torch.int32)
    return tensor
