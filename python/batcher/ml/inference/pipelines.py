"""HuggingFace ``transformers.pipeline`` placement, precision, and the load-once class UDF.

The model-identifier path of `ds.ml.infer`. Everything here is about getting one
`transformers` pipeline onto the right device at the right precision — accelerator
placement (including sharding a model that exceeds one GPU), the half-precision dtype,
the CPU thread cap, and the vision-model `torch.compile` — and then wrapping it as a
class UDF whose identity is stable enough for the distributed warm pool to reuse.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from batcher._internal.logging import note_suppressed
from batcher.config import active_config

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["transformers_pipeline_encoder"]


def _pipeline_accel_kwargs(model: str = "") -> dict[str, Any]:
    """Zero-config accelerator placement + precision for a ``transformers.pipeline``.

    On a GPU worker: put the model on the detected device and, when the GPU has fast
    half-precision tensor cores, run it in BF16/FP16 (`recommend_inference_dtype`) —
    ~2x throughput on a compute-bound model at negligible inference quality loss. A
    no-op (CPU, FP32) where half gives no speedup or torch is absent, so correctness is
    never traded for the fast path. This is the per-GPU default Ray Data users set by hand.
    """
    kwargs: dict[str, Any] = {}
    try:
        import torch

        from batcher.ml.gpu import detect_backend, recommend_inference_dtype, torch_device

        # Vendor-agnostic device placement: `detect_backend` names only an available backend,
        # so NVIDIA/AMD (cuda), Intel (xpu), Apple (mps), TPU (xla) all get placed on the
        # accelerator. transformers wants an int for the primary CUDA device, a string else.
        backend = detect_backend()
        if backend != "cpu":
            dev = torch_device(backend)
            if _should_shard_across_devices(dev, model):
                # `device=0` pins the whole model to ONE device, so a model larger than a
                # single GPU raises OOM at load even on a node with room across its cards.
                # `device_map="auto"` lets accelerate shard the layers over every visible
                # device (and offload the overflow), which is the only way such a model
                # runs at all. Mutually exclusive with `device`, so it replaces it.
                kwargs["device_map"] = "auto"
            else:
                kwargs["device"] = 0 if dev == "cuda" else dev
        else:
            # Cap torch's BLAS thread pool to this actor's cores, so co-located CPU actors
            # don't each grab every host core.
            _configure_cpu_inference_threads()
        dtype = recommend_inference_dtype(backend)
        if dtype is not None:
            kwargs["torch_dtype"] = getattr(torch, dtype)
    except ImportError:
        return {}  # no torch → the CPU path is correct and silent
    except Exception as exc:
        # An accelerator present but *misconfigured* (broken driver, stale CUDA): silently
        # returning {} runs CPU FP32 — a 10-50x slowdown that looks like success. Warn.
        import warnings

        from batcher._internal.errors import PerformanceWarning

        warnings.warn(
            f"accelerator configuration failed ({type(exc).__name__}: {exc}); using CPU FP32 — "
            "inference may be far slower than expected. Check the GPU driver / CUDA runtime.",
            PerformanceWarning,
            stacklevel=2,
        )
        return {}
    return kwargs


def _should_shard_across_devices(device: str, model: str = "") -> bool:
    """Whether a ``transformers.pipeline`` should shard with ``device_map="auto"``.

    True only when this process sees more than one CUDA device, ``accelerate`` is installed
    to do the sharding, and the model does not fit one of those devices. A single-GPU actor
    keeps the explicit `device` pin, which is cheaper and avoids accelerate's dispatch hooks;
    without `accelerate`, `device_map` would raise where the pin at least loads. Ray pins a
    one-GPU actor to one visible device, so a packed inference actor keeps the pin.

    The footprint check is what keeps a *local* multi-GPU run honest. `device_map="auto"` is
    accelerate's **balanced** map, not a fill-first one: it spreads the layers evenly over
    every visible device whether or not they need spreading. A model that fits one card then
    runs as a naive pipeline with no micro-batching, so one device computes while the rest
    wait and a transfer crosses the bus at every stage boundary — slower than the single-device
    pin it replaced, on more hardware. An unreadable footprint keeps the old behaviour, because
    sharding a model that turns out not to fit is recoverable and pinning it is not."""
    if device != "cuda":
        return False  # xpu/mps/xla: accelerate's auto map is not a supported placement here
    try:
        import importlib.util

        import torch

        if importlib.util.find_spec("accelerate") is None:
            return False
        if torch.cuda.device_count() <= 1:
            return False
    except Exception:
        return False
    return not _fits_one_device(model)


def _fits_one_device(model: str) -> bool:
    """Whether `model`'s weights fit one visible device, `False` when that is unknown.

    Sized against the smallest visible device, since a heterogeneous node is bounded by its
    smallest card, and with the configured VRAM headroom — activations, the CUDA context, and
    allocator fragmentation are not in a weight footprint.

    The headroom is read from the configuration rather than from Carbonite's constant, and
    that is a layering constraint rather than a preference: `core.udf` reaches up into
    `ml.inference` (the debt recorded in the architecture rule), so anything this module
    imports becomes reachable from `core` — and a `core -> carbonite` edge breaks the
    subsystem-independence contract. `config` is neutral, and the operator's number is the
    better one to use anyway.
    """
    from batcher.config import active_config
    from batcher.ml.llm.engines.footprint import device_total_bytes, model_weight_bytes

    weights = model_weight_bytes(model) if model else None
    device = device_total_bytes()
    if not weights or not device:
        return False
    return weights <= device * (1.0 - active_config().accelerator.vram_headroom)


def _cpu_inference_thread_target() -> int:
    """The intra-op thread count a CPU-inference process should use.

    An explicit `OMP_NUM_THREADS` (Ray sets it to the actor's `num_cpus`, and a user may pin it)
    wins, so the per-actor allocation is honored; otherwise the container's usable core count
    (cgroup/affinity-aware, via `available_cpu_count`), never the host count.
    """
    import os

    from batcher._internal.hardware import available_cpu_count

    omp = os.environ.get("OMP_NUM_THREADS", "").strip()
    if omp.isdigit() and int(omp) > 0:
        return int(omp)
    return available_cpu_count()


def _configure_cpu_inference_threads() -> int | None:
    """Cap torch's CPU math-library thread pool to the cores this process may actually use.

    torch — and the Intel MKL / oneDNN, AMD BLIS/AOCL, or OpenBLAS backend it dispatches to —
    sizes its intra-op pool to the *host* core count, so a CPU-inference actor over-subscribes
    under a cgroup quota and thrashes *catastrophically* when co-located actors each grab every
    core. Sets the pool to `_cpu_inference_thread_target()`, only ever *lowering* it (never raising
    above torch's or a caller's explicit choice). Returns the applied count, or `None` when torch
    is absent. Best-effort and idempotent; result-invariant (a thread count, not a result)."""
    try:
        import torch
    except Exception:
        return None
    target = _cpu_inference_thread_target()
    try:
        if torch.get_num_threads() > target:
            torch.set_num_threads(target)
    except Exception:
        return None
    return target


def _maybe_compile_pipeline(pipe: Any) -> None:
    """`channels_last` + `torch.compile` a **vision (CNN)** model in place for ~2x GPU inference.

    Applied ONLY to convolutional models (a `Conv2d` present) — fixed input shapes and
    compute-heavy, where `torch.compile`'s kernel fusion + graph capture is a measured ~1.9x at
    inference-identical results (unchanged predicted labels, logits within fp16 tolerance). It is
    deliberately NOT applied to text transformers: their dynamic sequence lengths trigger
    per-shape recompiles and the work is tokenization-bound, so `torch.compile` there measured
    ~0.9x (a regression). Compiled once per worker; the warm pool amortizes the one-time compile
    over every batch. A no-op on CPU, when `distributed.torch_compile` is off, or on any failure
    — eager is the safe fallback, so a perf optimization never breaks or slows inference.

    The applicability gate is the detected accelerator backend, not `torch.cuda.is_available`.
    Inductor has backends for Intel XPU and Apple MPS as well as CUDA/ROCm, and every other
    accelerator decision in `ml/gpu.py` is vendor-agnostic; gating on CUDA alone silently
    denied the ~2x to every non-NVIDIA device. Any backend where compile does not apply still
    lands in the `except` below and keeps eager."""
    if not active_config().distributed.torch_compile:
        return
    try:
        import torch

        from batcher.ml.gpu import detect_backend

        model = getattr(pipe, "model", None)
        if detect_backend() == "cpu" or model is None:
            return
        if not any(isinstance(m, torch.nn.Conv2d) for m in model.modules()):
            return  # not a CNN — compile regresses dynamic-shape text models, so skip
        pipe.model = torch.compile(model.to(memory_format=torch.channels_last))
    except Exception as exc:
        note_suppressed(
            "ml", "compile the inference model", exc
        )  # eager fallback — never break inference for a perf optimization


@functools.cache
def transformers_pipeline_encoder(
    model: str,
    column: str,
    *,
    output_column: str = "prediction",
    task: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    model_kwargs: tuple[tuple[str, Any], ...] = (),
) -> type:
    """A load-once class UDF that runs a HuggingFace ``transformers`` pipeline.

    The model-identifier path of `ds.ml.infer`: builds ``transformers.pipeline(task,
    model=model)`` once per worker (the load-once GPU-inference pattern) and runs it
    over each batch's `column`, appending the pipeline's primary output per row as
    `output_column`. For a classification pipeline that is the predicted ``label``;
    for text generation the ``generated_text``; for speech recognition the ``text``;
    otherwise the raw scalar output. On a GPU worker the model is placed on the device
    and auto-cast to BF16/FP16 where the hardware benefits (see
    `_pipeline_accel_kwargs`). Needs ``transformers``
    (``batcher-engine[transformers]``).

    `column` is not restricted to text. Its Arrow type decides what the pipeline is fed
    (`_pipeline_inputs`): a string or binary column passes through as objects — text,
    paths, URLs, or encoded bytes the pipeline decodes itself — and a numeric list or
    tensor column becomes one NumPy array per row, which is what an audio, image, or
    video model wants. That is what makes a decoded waveform column feed an ASR model
    directly, with the decode and the resample staying in the data plane.

    An explicit `device` (``"cuda"``, ``"cpu"``, ``"mps"``) or `dtype` (``"float16"``,
    ``"bfloat16"``) overrides the zero-config accelerator placement; leaving both unset keeps
    the auto behavior. `model_kwargs` (passed as a hashable tuple of items so the memoization
    key stays stable) is forwarded to ``transformers.pipeline`` for model-load options such
    as ``trust_remote_code``.

    The generated class is **memoized** per ``(model, column, output_column, task, device,
    dtype, model_kwargs)`` so repeated ``ds.ml.infer(<same model>)`` calls return the *same*
    class object. The distributed warm-pool key is the UDF's identity
    (`dist…map._pipeline_signature`), so a stable class is what lets a session-warm inference
    pool be reused across `collect()`s instead of reloading the model every call — the whole
    point of the warm pool on the convenience path.
    """

    class _PipelineModel:
        def __init__(self) -> None:
            from batcher._internal.optional import require

            pipeline = require(
                "transformers",
                "pipeline",
                feature="ds.ml.infer(<model id>)",
                provides="transformers",
                extra="transformers",
            )
            kwargs = _pipeline_accel_kwargs(model)
            if device is not None:
                from batcher.ml.devices import resolve_device

                kwargs["device"] = resolve_device(device)
            if dtype is not None:
                from batcher.ml.devices import resolve_dtype

                kwargs["torch_dtype"] = resolve_dtype(dtype)
            kwargs.update(dict(model_kwargs))
            self._pipe = pipeline(task, model=model, **kwargs)
            _maybe_compile_pipeline(self._pipe)

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            import pyarrow as pa

            # A HF pipeline defaults to batch_size=1 — it runs the model once PER ROW,
            # starving the GPU (the classic pipeline footgun). Feed the whole Arrow batch
            # as one GPU batch (`batch_size=num_rows`) so the forward pass is actually
            # batched; the map_batches `batch_size` already bounds `num_rows`, and the
            # actor-pool OOM-halving is the safety net if a batch is too large for VRAM.
            inputs = _pipeline_inputs(batch.column(column))
            results = self._pipe(inputs, batch_size=max(1, len(inputs)))
            out_col = pa.array([_primary_output(r) for r in results])
            if output_column in batch.schema.names:
                idx = batch.schema.get_field_index(output_column)
                return batch.set_column(idx, output_column, out_col)
            return batch.append_column(output_column, out_col)

    return _PipelineModel


def _pipeline_inputs(column: pa.Array) -> list:
    """One batch column as the inputs a ``transformers`` pipeline accepts.

    A pipeline is polymorphic in its input and the column type is what says which shape this
    is, so the conversion is chosen from the schema rather than from the task name:

    * **string** and **binary** columns pass through as Python objects. That covers text, and
      it covers the file paths, URLs and raw encoded bytes that the image and audio pipelines
      accept and decode themselves.
    * a **numeric list / fixed-size-list / tensor** column becomes one NumPy array per row.
      That is a decoded waveform or a decoded image tensor, and it is the case that used to be
      broken: ``to_pylist()`` on a ``list<float>`` waveform hands the feature extractor a
      Python list of floats per row, which it rejects — so an ASR or audio-classification
      model over the decode path this project documents could not run through `ds.ml.infer`
      at all. Delegating to the one converter (`interop.arrays`) also means a
      `FixedShapeTensor` image column restores to its full ``(H, W, C)`` shape rather than
      arriving flat.
    """
    import pyarrow as pa

    dtype = column.type
    if (
        pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_binary(dtype)
        or pa.types.is_large_binary(dtype)
        or pa.types.is_fixed_size_binary(dtype)
    ):
        return column.to_pylist()
    from batcher.ml.converters import _column_to_numpy

    matrix = _column_to_numpy(column)
    if getattr(matrix, "dtype", None) is not None and matrix.dtype == object:
        return column.to_pylist()  # a nested/ragged column with no array form; let HF judge
    return list(matrix)


def _primary_output(result: Any) -> Any:
    """The single salient value of one pipeline result row (label / text / scalar).

    A one-element list is the common wrapper text-generation and summarization pipelines
    return, so it is unwrapped. A **multi**-element list is a genuinely multi-valued result
    — every entity a token-classification/NER pipeline found, or every candidate label —
    and each element's salient value is kept as a list, rather than silently discarding all
    but the first (the old ``result[0]``, which dropped every NER entity after the first).

    ``text`` is in the key list because it is what every **speech** pipeline calls its
    output (``automatic-speech-recognition`` returns ``{"text": ..., "chunks": [...]}``).
    Without it the whole dict fell through as the "scalar", so a transcription landed as a
    struct column carrying the timing chunks beside the words — a column nothing downstream
    reads as a transcript.
    """
    if isinstance(result, list):
        if not result:
            return None
        if len(result) > 1:
            return [_primary_output(item) for item in result]
        result = result[0]
    if isinstance(result, dict):
        for key in (
            "label",
            "generated_text",
            "summary_text",
            "translation_text",
            "answer",
            "text",
        ):
            if key in result:
                return result[key]
    return result
