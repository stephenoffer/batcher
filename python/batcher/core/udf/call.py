"""The per-batch `map_batches` call boundary (Core, layer 3).

One Arrow batch in, Arrow batches out: this module owns everything that happens
*around* a single user `fn` call — reframing the batch to the requested
`batch_format`, isolating a failing batch by bisection (CUDA-OOM halving and
dirty-row tolerance), and normalizing whatever the `fn` returns back to
`RecordBatch`es. The callers own the *scheduling* of those calls: `execute`
walks the plan tree, `stream` overlaps the stages, `strategy` picks threads vs
processes — all three share this one boundary, so a UDF behaves identically on
every path.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

__all__: list[str] = []


def _resilient_call(
    call, sub: pa.RecordBatch, budget: list[int], is_gpu: bool
) -> list[pa.RecordBatch]:
    """Run a per-batch `call`, isolating failures by bisection — the unified OOM-halving +
    dirty-data-tolerance path.

    On a CUDA OOM (GPU stage) the batch is halved and retried (a too-large batch often fits at
    N/2; the per-row-independent outputs concatenate to the whole result); a single row that
    still OOMs is a genuine over-allocation and re-raises. On any OTHER error the batch is
    bisected to isolate the offending row(s): a failing single row is DROPPED (charged against
    `budget`, the ``max_errored_rows`` allowance) so a corrupt image / malformed record doesn't
    kill a long job — until the budget is exhausted, when it re-raises. With ``budget == 0``
    and a CPU stage this reduces to strict behavior (any error propagates), so a real bug on
    clean data still fails fast."""
    from batcher.ml.inference import _empty_cuda_cache, _is_cuda_oom

    try:
        return _coerce_udf_result(call(sub))
    except Exception as exc:
        oom = is_gpu and _is_cuda_oom(exc)
        if oom:
            _empty_cuda_cache()
        if sub.num_rows <= 1:
            if oom or budget[0] <= 0:
                raise  # genuine single-row over-allocation, or the error budget is spent
            budget[0] -= 1
            return []  # drop the one corrupt row and carry on
        mid = sub.num_rows // 2
        left = _resilient_call(call, sub.slice(0, mid), budget, is_gpu)
        return left + _resilient_call(call, sub.slice(mid), budget, is_gpu)


def _formatted(fn: Any, fmt: str) -> Any:
    """Wrap `fn` so it receives/returns `fmt` batches while the caller stays Arrow."""
    from batcher.ml.batch_format import result_to_arrowable, to_format

    def _call(batch: pa.RecordBatch) -> object:
        return result_to_arrowable(fn(to_format(batch, fmt)), fmt)

    return _call


def _coerce_udf_result(result: object) -> list[pa.RecordBatch]:
    """Normalize a `map_batches` return (RecordBatch / Table / column dict) to batches."""
    if isinstance(result, pa.RecordBatch):
        return [result]
    if isinstance(result, pa.Table):
        # A 0-row Table yields *no* batches, which would drop the stage's output schema (the
        # parent falls back to the input schema and a downstream ref to a UDF-added column
        # fails). Keep one empty batch so the schema survives, like a 0-row RecordBatch does.
        batches = result.to_batches()
        if batches:
            return batches
        cols = [pa.array([], type=f.type) for f in result.schema]
        return [pa.RecordBatch.from_arrays(cols, schema=result.schema)]
    if isinstance(result, dict):
        return [pa.RecordBatch.from_pydict(_tensorize_columns(result))]
    raise TypeError(
        "map_batches function must return a pyarrow RecordBatch, Table, or dict; "
        f"got {type(result).__name__}"
    )


def _tensorize_columns(result: dict) -> dict:
    """Turn any multi-dimensional NumPy value into a fixed-shape-tensor column.

    A `map_batches` `fn` (image decode, embedding, feature-map) commonly returns a
    ``(B, *shape)`` NumPy array per column — the Ray Data tensor-block shape.
    ``from_pydict`` can't build a column from a >1-D array, so multi-dim values are
    converted to the canonical ``arrow.fixed_shape_tensor`` column (`to_tensor_column`),
    which round-trips zero-copy through the FFI with its shape intact. 1-D arrays, lists,
    and Arrow arrays pass through untouched, so scalar/label columns are unchanged. This
    keeps the tensor path identical single-node and distributed, for every modality.
    """
    import numpy as np

    from batcher.io.formats.ml.tensor import to_tensor_column

    converted: dict = {}
    for name, value in result.items():
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            converted[name] = to_tensor_column(value)
        else:
            converted[name] = value
    return converted
