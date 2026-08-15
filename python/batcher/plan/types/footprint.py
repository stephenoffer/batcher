"""How much memory live Arrow data actually keeps resident.

`widths.py` answers what a *schema* costs per row, before any data exists. This module
answers what a `RecordBatch` or `Table` in hand costs the process right now, which is a
different question with a different failure mode.

The natural answer, `nbytes`, is wrong for it. `nbytes` is the size of the rows an object
*addresses*; it is not what that object keeps alive. Every zero-copy derivation in Arrow —
`slice`, `head`, `limit`, a partition cut by offset — addresses a window of its parent's
buffers and pins the whole parent. The two figures are then unrelated: a 10-row window of a
2M-row column reports 80 bytes and holds 16 MB.

That matters wherever a byte figure is used as a *bound* rather than as a statistic, and it
fails in the one direction a bound must not fail. A working set chunked to 64 MiB by
`nbytes` can hold gigabytes; a broadcast guard comparing `nbytes` against its threshold
passes a build side far over it and replicates it to every worker. Both are the OOM those
checks exist to prevent, arrived at through the check itself.

`retained_bytes` is that figure, and it over-counts where two columns share one buffer (a
dictionary encoded twice). That is deliberate. A bound that over-counts spills or chunks
sooner and costs some throughput; one that under-counts costs the process.

Neutral layer: imports only `pyarrow`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "logical_bytes",
    "retained_bytes",
    "total_logical_bytes",
    "total_retained_bytes",
]


def logical_bytes(data: Any) -> int:
    """Bytes the rows `data` addresses, on every Arrow layout — the safe `nbytes`.

    `nbytes` is not total on `pyarrow`: it walks the buffer layout and raises
    `ArrowTypeError("Extracting byte ranges not supported for type ...")` for the *view*
    layouts (`string_view`, `binary_view`, `list_view`). Those are not exotic. They are
    what a Parquet reader with view types enabled, DuckDB, Polars, and any Velox-backed
    producer hand over, so a bare `.nbytes` read is a crash waiting on the shape of
    someone else's data rather than on anything the caller did.

    A raising property also defeats the obvious guard: `getattr(batch, "nbytes", 0)`
    propagates the error rather than returning the default, because the default applies
    only to a *missing* attribute. Read the figure through here instead.

    Args:
        data: A `pyarrow.Table`, `RecordBatch`, `ChunkedArray`, or `Array`.

    Returns:
        The logical byte count, falling back to the total buffer size on a layout
        `nbytes` cannot walk, and to `0` for an object that reports neither.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.plan.types import logical_bytes
            >>> logical_bytes(pa.record_batch({"v": pa.array([1, 2, 3])}))
            24
            >>> logical_bytes(pa.array(["a", "b"], type=pa.string_view())) > 0
            True
    """
    try:
        return int(data.nbytes)
    except pa.ArrowTypeError:
        total = getattr(data, "get_total_buffer_size", None)
        try:
            return int(total()) if total is not None else 0
        except (TypeError, ValueError):  # pragma: no cover - an unusable API
            return 0
    except (AttributeError, TypeError, ValueError):
        return 0


def total_logical_bytes(items: Iterable[Any]) -> int:
    """Logical bytes across a sequence of batches or tables.

    The counterpart of `total_retained_bytes` for the sites that *report* a size or size a
    payload by what its rows address — a transfer's wire cost, a profile's `result_bytes`,
    a shuffle counter. It exists for the same reason: `sum(b.nbytes for b in batches)` is
    the shape those call sites reach for, and it raises on the Arrow view layouts, so the
    named helper is what keeps a bare `nbytes` from being written back in.

    Args:
        items: The Arrow objects to measure.

    Returns:
        The total logical byte count.
    """
    return sum(logical_bytes(item) for item in items)


def retained_bytes(data: Any) -> int:
    """Bytes the process cannot reclaim while `data` is referenced.

    Use this, not `nbytes`, wherever the number decides something: whether to spill, how
    much to hold in a chunk, whether a build side fits a broadcast. Use `logical_bytes` where
    the number is only reported, since it is the figure a user recognizes as their data's
    size — and never a bare `nbytes`, which raises on the view layouts.

    Args:
        data: A `pyarrow.Table`, `RecordBatch`, `ChunkedArray`, or `Array`.

    Returns:
        The retained byte count. Falls back to `nbytes`, then to `0`, for an object that
        reports neither — a footprint that cannot be measured is not a reason to raise from
        inside a memory guard.
    """
    total = getattr(data, "get_total_buffer_size", None)
    logical = logical_bytes(data)
    if total is None:
        return logical
    try:
        return max(int(total()), logical)
    except (TypeError, ValueError):  # pragma: no cover - an object with an unusable API
        return logical


def total_retained_bytes(items: Iterable[Any]) -> int:
    """Retained bytes across a sequence of batches or tables.

    The shape every caller here actually needs, because the thing being bounded is almost
    always a list of batches rather than one. Named so a call site cannot accidentally
    write `sum(b.nbytes ...)` back in while looking like it measured something.

    Args:
        items: The Arrow objects to measure.

    Returns:
        The total retained byte count.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.plan.types import total_retained_bytes
            >>> parent = pa.record_batch({"v": pa.array(range(100_000))})
            >>> window = parent.slice(0, 4)
            >>> window.nbytes
            32
            >>> total_retained_bytes([window]) > 100_000
            True
    """
    return sum(retained_bytes(item) for item in items)
