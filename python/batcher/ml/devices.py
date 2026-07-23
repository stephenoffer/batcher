"""Zero-config device, dtype, and batch-size resolution for the ML surface.

Every ML entry point that touches a GPU has to answer the same three questions: which
device, which dtype, and how many rows per call. The ecosystem spells the answers as
``device="auto"``, ``dtype="float16"``, and a ``batch_size`` a user should not have to
guess. This module is those answers, once, so ``ds.ml.infer`` / ``embed`` / the loaders
share one detection path instead of each re-implementing it — and so a bad ``device`` or
``dtype`` fails with a "did you mean ...?" rather than a cryptic backend error.

It sits over `batcher.ml.gpu` (the hardware detection) and adds only the naming and
validation a Python user expects, keeping `gpu` focused on the accelerator facts.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError, did_you_mean

__all__ = [
    "available_devices",
    "default_batch_size",
    "default_dtype",
    "describe_accelerators",
    "get_device",
    "gpu_available",
    "resolve_device",
    "resolve_dtype",
    "validate_batch_size",
    "validate_num_gpus",
]

#: The device names a user may pass, mapped to `batcher.ml.gpu.torch_device` backends.
#: ``"auto"`` detects; the rest force a specific device (degrading to CPU if absent).
_DEVICE_BACKENDS = {
    "cpu": "cpu",
    "cuda": "cuda",
    "gpu": "cuda",
    "rocm": "rocm",
    "mps": "mps",
    "xpu": "xpu",
    "tpu": "tpu",
    "xla": "tpu",
}

#: Canonical torch dtype names, plus the abbreviations the ecosystem uses
#: interchangeably (``fp16``/``half`` for ``float16``, ``bf16`` for ``bfloat16``).
_DTYPE_ALIASES = {
    "float32": "float32",
    "float": "float32",
    "fp32": "float32",
    "float64": "float64",
    "double": "float64",
    "fp64": "float64",
    "float16": "float16",
    "half": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "long": "int64",
    "uint8": "uint8",
    "bool": "bool",
}


def resolve_device(device: str | None = "auto") -> str:
    """Resolve a device request to a concrete torch device string.

    ``"auto"`` (the default) and ``None`` detect the best available accelerator —
    CUDA/ROCm on NVIDIA/AMD, ``mps`` on Apple, ``xpu`` on Intel — and fall back to
    ``"cpu"`` when none is present, so the same code runs on a laptop and a GPU box
    unchanged. A concrete name (``"cuda"``, ``"cuda:1"``, ``"mps"``, ``"cpu"``) is
    honored, and an indexed device (``"cuda:1"``) passes through as written.

    Args:
        device: ``"auto"``/``None`` to detect, or a torch device string.

    Returns:
        A torch device string ready for ``tensor.to(...)``.

    Raises:
        PlanError: If `device` is an unrecognized name, with a "did you mean ...?" hint.

    Examples:
        .. doctest::

            >>> from batcher.ml.devices import resolve_device
            >>> resolve_device("cpu")
            'cpu'
    """
    from batcher.ml.gpu import torch_device

    if device is None or device == "auto":
        return torch_device()
    if not isinstance(device, str):
        raise PlanError(f"device must be a string or None, got {type(device).__name__}")
    base = device.split(":", 1)[0].lower()
    if base not in _DEVICE_BACKENDS:
        hint = did_you_mean(base, _DEVICE_BACKENDS)
        suffix = f" Did you mean {hint[0]!r}?" if hint else ""
        raise PlanError(
            f"device {device!r} is not a known device. Use 'auto', 'cpu', 'cuda', "
            f"'cuda:N', 'mps', or 'xpu'.{suffix}"
        )
    if ":" in device:
        # An index only means something for CUDA/ROCm; keep the user's spelling.
        return device.lower()
    return torch_device(_DEVICE_BACKENDS[base])


def get_device(device: str | None = "auto") -> str:
    """Resolve `device` to a concrete torch device string (Ray Data's ``get_device`` name).

    A thin alias of `resolve_device` under the name PyTorch/Ray users reach for first.

    Args:
        device: ``"auto"``/``None`` to detect, or a torch device string.

    Returns:
        A torch device string ready for ``tensor.to(...)``.

    Examples:
        .. doctest::

            >>> from batcher.ml.devices import get_device
            >>> get_device("cpu")
            'cpu'
    """
    return resolve_device(device)


def available_devices() -> list[str]:
    """The device strings usable on this machine, ``"cpu"`` always included.

    Returns:
        A list such as ``["cpu"]`` on a CPU-only box, or ``["cuda", "cpu"]`` where an
        accelerator is present.

    Examples:
        .. doctest::

            >>> from batcher.ml.devices import available_devices
            >>> "cpu" in available_devices()
            True
    """
    from batcher.ml.gpu import detect_backend, torch_device

    backend = detect_backend()
    if backend == "cpu":
        return ["cpu"]
    return [torch_device(backend), "cpu"]


def gpu_available() -> bool:
    """Whether an accelerator (CUDA/ROCm/MPS/XPU/TPU) is present, so GPU paths will run.

    Returns:
        ``True`` if any non-CPU accelerator is detected.

    Examples:
        .. doctest::

            >>> from batcher.ml.devices import gpu_available
            >>> isinstance(gpu_available(), bool)
            True
    """
    from batcher.ml.gpu import detect_backend

    return detect_backend() != "cpu"


def resolve_dtype(dtype: str | None = "auto", *, device: str | None = None) -> str:
    """Resolve a dtype request to a canonical torch dtype name.

    ``"auto"``/``None`` pick a sensible default for `device` (`default_dtype`): half
    precision on a GPU, full precision on CPU. Abbreviations are accepted the way the
    ecosystem writes them, so ``"fp16"``, ``"half"``, and ``"float16"`` all resolve to
    ``"float16"``, and ``"bf16"`` to ``"bfloat16"``.

    Args:
        dtype: ``"auto"``/``None`` to default, or a dtype name/abbreviation.
        device: The device the default is chosen for, when `dtype` is ``"auto"``.

    Returns:
        A canonical torch dtype name such as ``"float16"`` or ``"float32"``.

    Raises:
        PlanError: If `dtype` is unrecognized, with a "did you mean ...?" hint.

    Examples:
        .. doctest::

            >>> from batcher.ml.devices import resolve_dtype
            >>> resolve_dtype("fp16")
            'float16'
            >>> resolve_dtype("float32")
            'float32'
    """
    if dtype is None or dtype == "auto":
        return default_dtype(device)
    if not isinstance(dtype, str):
        raise PlanError(f"dtype must be a string or None, got {type(dtype).__name__}")
    key = dtype.lower().removeprefix("torch.")
    if key not in _DTYPE_ALIASES:
        hint = did_you_mean(key, _DTYPE_ALIASES)
        suffix = f" Did you mean {hint[0]!r}?" if hint else ""
        raise PlanError(
            f"dtype {dtype!r} is not a known dtype. Use 'auto', 'float32', 'float16', "
            f"'bfloat16', 'float64', or an int/uint/bool type.{suffix}"
        )
    return _DTYPE_ALIASES[key]


def default_dtype(device: str | None = None) -> str:
    """The default dtype for `device` — half precision on a GPU, ``float32`` on CPU.

    Half precision roughly doubles inference throughput and halves memory on a GPU with
    negligible accuracy loss for most models, while CPU kernels are typically slower in
    ``float16`` than ``float32``, so the default follows the device.

    Args:
        device: The target device (detected when ``None``).

    Returns:
        ``"float16"`` on an accelerator, ``"float32"`` on CPU.

    Examples:
        .. doctest::

            >>> from batcher.ml.devices import default_dtype
            >>> default_dtype("cpu")
            'float32'
    """
    resolved = resolve_device(device if device is not None else "auto")
    return "float32" if resolved == "cpu" else "float16"


def default_batch_size(*, device: str | None = None) -> int:
    """A sensible starting batch size for `device`, when none is given.

    Larger on an accelerator (more rows keep the device busy and amortize the
    host-to-device copy) and smaller on CPU (where a huge batch only grows latency and
    memory). It is a starting point, not a tuned value: profile with the real model and
    set `batch_size` explicitly for production throughput.

    Args:
        device: The target device (detected when ``None``).

    Returns:
        A positive row count to batch by.

    Examples:
        .. doctest::

            >>> from batcher.ml.devices import default_batch_size
            >>> default_batch_size(device="cpu")
            256
    """
    resolved = resolve_device(device if device is not None else "auto")
    return 256 if resolved == "cpu" else 1024


def describe_accelerators() -> str:
    """A one-line human summary of the detected accelerator, for logs and ``repr``.

    Returns:
        A string such as ``"cuda (1 device)"`` or ``"cpu (no accelerator detected)"``.

    Examples:
        .. doctest::

            >>> from batcher.ml.devices import describe_accelerators
            >>> isinstance(describe_accelerators(), str)
            True
    """
    from batcher.ml.gpu import detect_backend

    backend = detect_backend()
    if backend == "cpu":
        return "cpu (no accelerator detected)"
    return f"{backend} accelerator detected (device string {resolve_device('auto')!r})"


def validate_batch_size(batch_size: int | None, *, param: str = "batch_size") -> None:
    """Raise a clear error if `batch_size` is present but not a positive integer.

    ``None`` is always allowed (it means "let the engine choose"). A zero or negative
    value is caught here with a message naming the parameter, rather than surfacing deep
    in the engine as a bare ``range()`` or slice error.

    Args:
        batch_size: The batch size to check; ``None`` passes.
        param: The parameter name to name in the error (``"batch_size"`` by default).

    Raises:
        PlanError: If `batch_size` is not ``None`` and not a positive integer.

    Examples:
        .. doctest::

            >>> from batcher.ml.devices import validate_batch_size
            >>> validate_batch_size(None)
            >>> validate_batch_size(128)
    """
    if batch_size is None:
        return
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise PlanError(f"{param} must be a positive integer or None, got {batch_size!r}")


def validate_num_gpus(num_gpus: float) -> None:
    """Raise a clear error if `num_gpus` is negative.

    Fractional values are valid (they pack several small models onto one GPU), so only a
    negative request is rejected, with a message naming ``num_gpus``.

    Args:
        num_gpus: GPUs to reserve per worker; must be ``>= 0``.

    Raises:
        PlanError: If `num_gpus` is negative.

    Examples:
        .. doctest::

            >>> from batcher.ml.devices import validate_num_gpus
            >>> validate_num_gpus(0.5)
    """
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, (int, float)) or num_gpus < 0:
        raise PlanError(f"num_gpus must be a non-negative number, got {num_gpus!r}")
