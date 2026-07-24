"""Execution of pipelines containing `map_batches` (opaque Python/ML operators).

Façade for the `map_batches` family, grouped by responsibility:

- `execute` — the tree walk that composes native relational operators with Python UDFs.
- `apply` — how one stage's `fn` runs over its batches: rebatching, the threads/processes/
  async/GPU-autobatch dispatch, retry/timeout, the error budget, and the load-once build.
- `lifecycle` — build a class UDF's model once and tear it down (its optional `close()`).
- `resilience` — retry-with-backoff + timeout policy for a flaky/external `fn`.
- `async_udf` — run an `async def` `fn`'s batches concurrently on one event loop.
- `call` — the per-batch call boundary: batch-format reframing, failure isolation,
  and result normalization back to Arrow.
- `strategy` — the measured policy: threads vs processes, and the per-batch row count.
- `stream` — the stage-overlapped streaming path for a linear map chain (CPU→GPU pipelines).
- `processes` — the warm, shared process pool that runs GIL-bound UDFs off the GIL.

The public import path `batcher.core.udf` is preserved; internals live in the submodules.
"""

from __future__ import annotations

from batcher.core.udf.execute import (
    build_udf_callable,
    execute_with_udfs,
    has_map_batches,
    prebuild_factories,
    stream_with_udfs,
)

__all__ = [
    "build_udf_callable",
    "execute_with_udfs",
    "has_map_batches",
    "prebuild_factories",
    "stream_with_udfs",
]
