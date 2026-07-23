"""Arrow → torch conversion, and moving the result to a device.

The leaf of the loader package: everything here turns one Arrow column or batch into tensors,
and nothing here knows about ranks, epochs, or shards. Split out of `ml/loader.py` when it
outgrew the module limit; the seam is "convert a batch" vs "decide which batch."
"""

from __future__ import annotations

import warnings
from collections import deque
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


def tensorize(batch: Any, keep: list[str], collate_fn: Any = None) -> Any:
    """One batch as a `{column: tensor}` dict, dropping the non-tensorizable columns.

    With `collate_fn` the Arrow batch is handed over untouched instead — the escape hatch for
    the columns this cannot represent (string labels/ids, ragged sequences needing a padding
    collate and an attention mask). Without it, a non-tensorizable column is dropped, which
    `warn_dropped_columns` announces once per loader rather than letting a label vanish
    silently.
    """
    if collate_fn is not None:
        return collate_fn(batch)
    out = {}
    for c in keep:
        t = column_to_tensor(batch.column(c))
        if t is not None:
            out[c] = t
    return out


def warn_dropped_columns(probe: Any, keep: list[str], collate_fn: Any = None) -> list[str]:
    """Warn about the `keep` columns `tensorize` cannot convert, and return their names.

    Called once, at loader construction, on a one-row probe of the corpus. A dropped column is
    otherwise invisible: a string ``label``/``id`` simply does not appear in the yielded dict,
    and the training loop reads a `KeyError` — or worse, trains happily on what is left.
    """
    if collate_fn is not None:
        return []  # the caller collates the raw batch; nothing is dropped
    dropped = [c for c in keep if column_to_tensor(probe.column(c)) is None]
    if dropped:
        warnings.warn(
            f"loader is dropping non-tensorizable column(s) {dropped}: they cannot become "
            "torch tensors and will be missing from every yielded batch. Pass "
            "`columns=` to select the columns you want, or `collate_fn=` to receive the "
            "raw Arrow batch and collate them yourself.",
            UserWarning,
            stacklevel=3,
        )
    return dropped


class DeviceMover:
    """Move a yielded batch to `device`, keeping pinned staging buffers alive until it lands.

    ``t.pin_memory().to(device, non_blocking=True)`` is the standard fast host-to-device
    recipe and, written that way, a use-after-free: the pinned staging tensor is the *source*
    of an asynchronous DMA, and dropping the last reference to it the moment `.to()` returns
    frees that page-locked buffer while the copy is still in flight. The destination then holds
    whatever the allocator put there next. It is silent, load-dependent, and the background
    prefetch thread widens the window rather than closing it.

    So the staging tensors are retained here for `depth` batches, and on CUDA the copy is
    issued on a dedicated stream whose completion event the compute stream waits on — which is
    what actually overlaps the transfer with compute instead of merely claiming to.
    """

    __slots__ = ("_depth", "_device", "_is_mps", "_live", "_pin", "_stream")

    def __init__(self, device: Any, *, pin_memory: bool = False, depth: int = 2) -> None:
        """Bind to a target device; `depth` staging batches stay referenced in flight."""
        self._device = device
        self._pin = pin_memory
        # Apple MPS has no 64-bit tensors; downcast so `device="auto"` works on Apple silicon.
        self._is_mps = str(device).startswith("mps")
        self._depth = max(2, depth)
        self._live: deque = deque()
        self._stream = _copy_stream(device) if pin_memory else None

    def __call__(self, out: Any) -> Any:
        """Move one batch (a tensor or a `{name: tensor}` dict) to the device."""
        staging: list = []
        event = None
        if self._stream is not None:
            import torch

            with torch.cuda.stream(self._stream):
                moved = self._move_all(out, staging)
            event = torch.cuda.Event()
            event.record(self._stream)
            # Order the consumer's compute after the copy without blocking this thread.
            torch.cuda.current_stream().wait_event(event)
        else:
            moved = self._move_all(out, staging)
        if staging:
            self._retain(staging, event)
        return moved

    def _move_all(self, out: Any, staging: list) -> Any:
        if isinstance(out, dict):
            return {k: self._move(v, staging) for k, v in out.items()}
        return self._move(out, staging)

    def _move(self, t: Any, staging: list) -> Any:
        if not hasattr(t, "to"):
            return t
        if self._is_mps:
            t = _mps_safe_dtype(t)
        if self._pin and hasattr(t, "pin_memory"):
            t = t.pin_memory()
            staging.append(t)  # the DMA source — must outlive the copy
        return t.to(self._device, non_blocking=self._pin)

    def _retain(self, staging: list, event: Any) -> None:
        """Hold this batch's staging buffers, retiring the ones whose copy has completed."""
        self._live.append((event, staging))
        while len(self._live) > self._depth:
            old_event, _ = self._live.popleft()
            if old_event is not None:
                old_event.synchronize()  # the copy is done; the buffers may now be freed


def _copy_stream(device: Any) -> Any:
    """A dedicated CUDA stream for host-to-device copies, or ``None`` off CUDA."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is checked by the caller
        return None
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    return torch.cuda.Stream()


def _mps_safe_dtype(tensor: Any) -> Any:
    """Downcast a 64-bit tensor to 32-bit (MPS supports no 64-bit dtypes)."""
    import torch

    if tensor.dtype == torch.float64:
        return tensor.to(torch.float32)
    if tensor.dtype == torch.int64:
        return tensor.to(torch.int32)
    return tensor
