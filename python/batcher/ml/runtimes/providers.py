"""Execution-provider and device selection for the local model runtimes.

A local runtime — ONNX Runtime, OpenVINO, TensorRT — runs the *same graph* on wildly
different hardware, and the only thing that changes is which backend it dispatches to.
Getting that choice wrong is the classic silent slowdown: a CPU-only session on a GPU
actor still returns correct answers, 10-50x slower, and nothing in the result says so.
So the choice is made here, once, from the accelerator the worker actually sees, and it
is shared by every runtime rather than pasted into each one.

Two rules the whole module follows:

* **Never name a provider the installed build does not have.** ONNX Runtime raises on an
  unknown provider name, so a fixed list would turn "you installed the CPU wheel" into a
  crash at model load. Every resolved list is intersected with what the runtime reports.
* **Never silently land on CPU when an accelerator was asked for.** An explicit
  ``providers=["cuda"]`` that cannot be honored warns, because that is exactly the case a
  user cannot see in the results.
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = [
    "PROVIDER_ALIASES",
    "RenamedPorts",
    "onnx_providers",
    "port_mapping",
    "resolve_device_id",
    "runtime_thread_target",
]

#: Friendly names accepted for an execution provider, mapped to ONNX Runtime's own spelling.
#: A caller may pass either; the full name passes through untouched so a provider added by a
#: newer runtime than this table knows about is never rejected.
PROVIDER_ALIASES: dict[str, str] = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "trt": "TensorrtExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "xnnpack": "XnnpackExecutionProvider",
    "webgpu": "WebGpuExecutionProvider",
    "cann": "CANNExecutionProvider",
    "nnapi": "NnapiExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
}

#: The order a GPU worker prefers when nothing is requested. TensorRT is deliberately *not*
#: first: it compiles the graph on the first batch, which can take minutes, and it falls back
#: per-subgraph to CUDA anyway — so it is a deliberate choice, never a default. CUDA and ROCm
#: are both listed because a build only ever has one of them, and the intersection below
#: drops whichever is absent.
_ACCELERATED_ORDER = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "CANNExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
)


def _canonical(name: str) -> str:
    """One provider name in ONNX Runtime's spelling, accepting a friendly alias."""
    return PROVIDER_ALIASES.get(name.strip().lower(), name)


def onnx_providers(
    requested: object,
    available: list[str],
    *,
    device_id: int | None = None,
    provider_options: dict[str, dict[str, Any]] | None = None,
) -> list[Any]:
    """The execution-provider list to hand ``onnxruntime.InferenceSession``.

    Resolution has three steps, in this order: expand the friendly aliases, drop anything
    this build does not offer, and append ``CPUExecutionProvider`` as the terminal fallback
    so a graph with one unsupported node still runs instead of raising.

    ``requested=None`` auto-selects: every accelerated provider this build offers, in
    `_ACCELERATED_ORDER`, but **only when an accelerator is actually visible**. A GPU-build
    wheel on a CPU-only node reports ``CUDAExecutionProvider`` as "available" and then fails
    at session creation, which is why availability alone is not the test.

    Args:
        requested: `None` to auto-select, or a provider name / sequence of names. Names may
            be friendly (``"cuda"``, ``"tensorrt"``) or ONNX Runtime's own spelling.
        available: what ``onnxruntime.get_available_providers()`` reported.
        device_id: the accelerator ordinal to pin GPU providers to; `None` resolves it from
            the worker's visible devices.
        provider_options: extra per-provider option dicts, keyed by either spelling. Merged
            over the defaults this function supplies (the device pin).

    Returns:
        A list for the ``providers=`` argument: each entry is either a provider name or a
        ``(name, options)`` pair when that provider has options to carry.
    """
    offered = list(available)
    names = _requested_names(requested, offered)
    if "CPUExecutionProvider" not in names and "CPUExecutionProvider" in offered:
        # The terminal fallback. ONNX Runtime assigns nodes to the first provider that
        # claims them and CPU claims everything, so appending it costs nothing on a graph
        # the accelerator fully covers and is the difference between running and raising
        # on a graph with one unsupported operator.
        names.append("CPUExecutionProvider")
    ordinal = resolve_device_id(device_id)
    options = {_canonical(k): dict(v) for k, v in (provider_options or {}).items()}
    return [_with_options(name, ordinal, options.get(name)) for name in names]


def _requested_names(requested: object, offered: list[str]) -> list[str]:
    """The caller's providers, canonicalized, de-duplicated, and filtered to `offered`."""
    if requested is None:
        return [p for p in _ACCELERATED_ORDER if p in offered] if _accelerator_visible() else []
    raw = [requested] if isinstance(requested, str) else list(requested)  # type: ignore[arg-type]
    names: list[str] = []
    for entry in raw:
        name = _canonical(entry if isinstance(entry, str) else str(entry))
        if name in names:
            continue
        if name not in offered:
            _warn_unavailable(name, offered)
            continue
        names.append(name)
    return names


def _warn_unavailable(name: str, offered: list[str]) -> None:
    """Warn that an explicitly requested provider is not in this build.

    Dropping it silently is what turns "I asked for the GPU" into a correct answer at CPU
    speed, which is invisible in the output and expensive in the bill.
    """
    if name == "CPUExecutionProvider":
        return  # appended unconditionally below; never worth a warning
    from batcher._internal.errors import PerformanceWarning

    warnings.warn(
        f"execution provider {name!r} is not available in this onnxruntime build "
        f"(it offers {offered}); falling back to the remaining providers. Install the "
        "matching build (e.g. onnxruntime-gpu) to use it.",
        PerformanceWarning,
        stacklevel=3,
    )


def _accelerator_visible() -> bool:
    """Whether this worker can actually see an accelerator, not merely link a GPU build."""
    from batcher.ml.gpu import detect_backend

    try:
        return detect_backend() != "cpu"
    except Exception:
        return False


def _with_options(name: str, ordinal: int, extra: dict[str, Any] | None) -> Any:
    """One provider entry, pinned to `ordinal` and merged with the caller's options."""
    defaults: dict[str, Any] = {}
    if name in ("CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider"):
        defaults["device_id"] = ordinal
    elif name == "TensorrtExecutionProvider":
        defaults["device_id"] = ordinal
        # Without a cache the engine is rebuilt from scratch in every worker on every run,
        # which for a transformer is minutes of GPU time before the first row is scored.
        defaults["trt_engine_cache_enable"] = True
    merged = {**defaults, **(extra or {})}
    return (name, merged) if merged else name


def resolve_device_id(device_id: int | None) -> int:
    """The accelerator ordinal a local runtime should bind to.

    An explicit `device_id` wins. Otherwise it is 0 — which is correct under every scheduler
    that scopes a worker to its own device (Ray sets ``CUDA_VISIBLE_DEVICES`` per actor, so
    the actor's card *is* ordinal 0), and is the only defensible guess when nothing does.
    """
    if device_id is not None:
        return max(0, int(device_id))
    return 0


def runtime_thread_target(requested: int | None) -> int:
    """Intra-op threads a local runtime should use, honoring the cgroup and the scheduler.

    A runtime sizes its own pool to the *host* core count, so two co-located inference
    actors each grab every core and thrash. `available_cpu_count` is affinity- and
    quota-aware, and an explicit ``OMP_NUM_THREADS`` (which Ray sets to the actor's
    ``num_cpus``) wins over it, matching what the transformers path already does.
    """
    if requested is not None and requested > 0:
        return int(requested)
    from batcher.ml.inference import _cpu_inference_thread_target

    return _cpu_inference_thread_target()


class RenamedPorts:
    """Maps dataset column names onto a runtime's own port names, around a `ServingClient`.

    A column is named for what it holds (``"tokens"``, ``"score"``); a model port is named
    for what the exporter or the forward signature called it (``"input_ids"``, ``"logits"``).
    Without this the two have to match, which forces a rename in the plan for a purely
    cosmetic reason — and a rename in the plan is a real projection the optimizer then has to
    carry through every stage above it.

    Both directions are optional and independent: pass only `inputs` to feed differently
    named columns, only `outputs` to land the results under different column names, or both.
    Delegates `batch_window` and `close` so the wrapped client keeps the whole `serving_udf`
    contract.
    """

    def __init__(
        self,
        client: Any,
        inputs: dict[str, str] | None = None,
        outputs: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._inputs = inputs or {}
        self._outputs = outputs or {}

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Feed the client under its own port names and relabel what it returns."""
        result = self._client.predict(
            {self._inputs.get(name, name): array for name, array in inputs.items()}
        )
        if not self._outputs:
            return result
        return {self._outputs.get(name, name): array for name, array in result.items()}

    def batch_window(self) -> int | None:
        """The wrapped client's declared batch window, when it declares one."""
        ask = getattr(self._client, "batch_window", None)
        return ask() if callable(ask) else None

    def close(self) -> None:
        """Release the wrapped client."""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def port_mapping(model_names: list[str], requested: object, count: int | None = None) -> dict:
    """A ``{model port: caller name}`` map, positional and length-safe.

    `requested` is what the caller asked the ports to be called; `model_names` is what the
    loaded model calls them. Pairing them positionally is the only correspondence available —
    a model's port order is its declaration order, and that is the order every exporter,
    every framework, and every user writes the two lists in.

    Returns an empty map when `requested` is `None` (keep the model's own names) or when the
    two lists already agree, so the common case costs no rename at all.
    """
    if requested is None:
        return {}
    names = list(requested) if not isinstance(requested, str) else [requested]
    ports = model_names[: count if count is not None else len(names)]
    return {port: name for port, name in zip(ports, names, strict=False) if port != name}
