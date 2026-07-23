"""Actor-pool batch inference — the ML data plane's orchestration layer.

The engine's `map_batches` runs a Python callback per batch, but reloads nothing:
for model inference you want the expensive resource (the model, the tokenizer, the
GPU context) loaded **once per worker** and reused across batches. `InferencePool`
does that — a fixed pool of workers, each built once from a factory, fed
dynamically-sized batches and run concurrently while preserving input order.

This is the control-plane orchestration twin of the native-pipeline primitives in
the Rust `bc-udf` crate (`OpaqueOperator`/`Rebatcher`/`BatchSizeController`): the
same dynamic-batching idea, applied here over whole Arrow batches for the
actor-pool path. Workers receive whole `pyarrow.RecordBatch`es — never per-row
Python — so the control plane never touches a tuple in the hot path.

Two responsibilities, two modules: `pool` is the pool and its batching/OOM/dispatch
control, `pipelines` is the HuggingFace ``transformers`` placement and class UDF.
The import path ``batcher.ml.inference`` is unchanged.
"""

from __future__ import annotations

from batcher.ml.inference.pipelines import (
    _configure_cpu_inference_threads,
    _cpu_inference_thread_target,
    _maybe_compile_pipeline,
    _pipeline_accel_kwargs,
    _primary_output,
    _should_shard_across_devices,
    transformers_pipeline_encoder,
)
from batcher.ml.inference.pool import (
    InferencePool,
    Worker,
    WorkerFactory,
    _DynamicBatcher,
    _empty_cuda_cache,
    _is_cuda_oom,
    _LatencyController,
    _run_with_oom_retry,
)

__all__ = ["InferencePool", "Worker", "WorkerFactory", "transformers_pipeline_encoder"]

# The underscored names are re-exported deliberately, not by accident: `core/udf/call.py`
# imports `_is_cuda_oom` / `_empty_cuda_cache`, and the unit suites reach for
# `_DynamicBatcher`, `_LatencyController`, `_run_with_oom_retry`,
# `_pipeline_accel_kwargs`, and `_maybe_compile_pipeline` through this module path. They
# stay private (absent from `__all__`) while keeping `batcher.ml.inference.<name>`
# resolvable exactly as it was before this module became a package.
_INTERNAL = (
    _DynamicBatcher,
    _LatencyController,
    _configure_cpu_inference_threads,
    _cpu_inference_thread_target,
    _empty_cuda_cache,
    _is_cuda_oom,
    _maybe_compile_pipeline,
    _pipeline_accel_kwargs,
    _primary_output,
    _run_with_oom_retry,
    _should_shard_across_devices,
)
