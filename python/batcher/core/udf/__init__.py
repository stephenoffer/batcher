"""Execution of pipelines containing `map_batches` (opaque Python/ML operators).

Façade for the `map_batches` family, grouped by responsibility:

- `execute` — the tree walk that composes native relational operators with Python UDFs.
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
)

__all__ = ["build_udf_callable", "execute_with_udfs", "has_map_batches", "prebuild_factories"]
