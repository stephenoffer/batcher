"""PyTorch's caching allocator as a machine fact: how to read it, and how to configure it.

The *decision* of what to configure it with is Carbonite's (`accel.device.torch_alloc`, which
sizes a plan from the VRAM headroom and the device's tenancy). This is the mechanism under
that decision — the environment variable, the driver call, and the two statistics — and it
lives in the neutral layer because the ML inference path needs the same readings and cannot
import a subsystem.

The reading that matters is **fragmentation**. A device out-of-memory is two different
failures wearing one message: the device is genuinely full, or the memory is there in blocks
too small for the request. They need opposite responses, and only the allocator's own
reserved-versus-allocated split tells them apart.

Everything here is **vendor-agnostic through one table** (`TORCH_NAMESPACE`), because torch
gives every accelerator the same allocator API under a different module name:
``torch.cuda`` (NVIDIA and, through HIP, AMD), ``torch.xpu`` (Intel), ``torch.mps`` (Apple),
``torch.hpu`` (Intel Gaudi), ``torch.npu`` (Huawei Ascend, via ``torch_npu``). Each reading
used to be written out against ``torch.cuda`` alone, so a Gaudi or Ascend worker had no
fragmentation signal, no per-process cap, and no memory reading at all — which does not fail,
it just silently leaves the OOM ladder and the packing cap inert on exactly the hardware whose
memory is hardest to reason about.
"""

from __future__ import annotations

import os

__all__ = [
    "ALLOC_CONF_ENV",
    "FRAGMENTATION_THRESHOLD",
    "TORCH_NAMESPACE",
    "accelerator_namespace",
    "allocator_initialized",
    "device_memory_used_fraction",
    "device_total_memory_bytes",
    "device_utilization",
    "fragmentation_ratio",
    "set_alloc_conf",
    "set_memory_fraction",
]

#: The `torch` submodule each accelerator backend's allocator and device API lives under.
#:
#: ROCm maps to ``cuda`` because HIP shims the CUDA API — a ROCm build of torch *is*
#: ``torch.cuda``. TPU and Trainium are absent on purpose rather than by omission: both run
#: through XLA, which has no caching allocator to read, no per-process cap to set, and
#: releases memory by stepping its execution graph instead (`oom._xla_mark_step`).
TORCH_NAMESPACE = {
    "cuda": "cuda",
    "rocm": "cuda",
    "xpu": "xpu",
    "mps": "mps",
    "hpu": "hpu",
    "npu": "npu",
}

#: The variable PyTorch reads its allocator settings from. It is parsed **once**, when the
#: caching allocator first initializes, so writing it after the first tensor is allocated has
#: no effect and no error — which is why the caller runs from a worker's setup path rather
#: than from the stage that needs it.
ALLOC_CONF_ENV = "PYTORCH_CUDA_ALLOC_CONF"

#: Cached-but-unused share of the allocator's reservation above which the process is
#: fragmented rather than full. Below it, a failed allocation means the device is genuinely out
#: of memory and shrinking the batch is the answer; above it there is real memory the allocator
#: cannot hand out at the requested size, and releasing cached blocks is.
FRAGMENTATION_THRESHOLD = 0.25


def _torch():
    """The **already-imported** torch module, or `None`.

    A `sys.modules` lookup rather than an import, throughout. Importing torch to answer a
    question about torch's allocator would cost seconds and hundreds of megabytes on a worker
    that never used it — and a process that has not imported torch has no allocator to ask
    about, so the answer is `None` either way.
    """
    import sys

    return sys.modules.get("torch")


def accelerator_namespace(backend: str | None = None) -> object | None:
    """The `torch.<backend>` module for this worker's accelerator, or `None`.

    One resolution for every reading below, so a backend is taught to the whole memory layer
    by adding a row to `TORCH_NAMESPACE` rather than by remembering five call sites.

    The namespace is returned only when it reports a device available: a torch build can
    carry ``torch.xpu`` on a machine with no Intel GPU, and answering from a namespace with no
    device is how a reading becomes a plausible wrong number rather than an absent one.

    With `backend` omitted the detected backend is preferred and, if it yields nothing, the
    table is **scanned** for a namespace that does. The scan is not belt-and-braces: the
    detector answers "what host is this" from device nodes behind a cheap NVIDIA-shaped
    negative, and when that negative fires on a non-NVIDIA accelerator the memory reading
    disappears silently — the cap simply stops applying. `is_available()` is torch's own
    authority on whether a namespace can be read, so a memory question is settled by it rather
    than by a host-identification question. An explicit `backend` is answered exactly, with no
    scan, so a caller asking about one accelerator never gets another's numbers.

    Args:
        backend: The accelerator name (``cuda``/``rocm``/``xpu``/``mps``/``hpu``/``npu``).
            Detected from the host when omitted.

    Returns:
        The torch submodule, or `None` when torch is absent, the backend has no namespace, or
        the namespace reports no device.
    """
    torch = _torch()
    if torch is None:
        return None
    if backend is not None:
        # An explicit ask is answered exactly: `device_utilization("hpu")` must never answer
        # from `torch.xpu` because that is the namespace that happened to be available.
        name = TORCH_NAMESPACE.get(backend)
        return _available_namespace(torch, name) if name else None
    from batcher._internal.accelerators import accelerator_backend

    detected = TORCH_NAMESPACE.get(accelerator_backend())
    if detected is not None:
        namespace = _available_namespace(torch, detected)
        if namespace is not None:
            return namespace
    # The detector declined, so scan. It answers "what is this host" from device nodes and a
    # cheap NVIDIA-shaped negative first, and a reading that vanishes because that negative
    # fired is a reading nobody notices going missing — the memory cap simply stops applying.
    # torch's own `is_available()` is the authority on whether a namespace can be read, so ask
    # it directly rather than letting a host-identification question decide a memory question.
    for name in dict.fromkeys(TORCH_NAMESPACE.values()):
        namespace = _available_namespace(torch, name)
        if namespace is not None:
            return namespace
    return None


def _available_namespace(torch: object, name: str) -> object | None:
    """`torch.<name>` when it reports a device available, else `None`.

    MPS is the exception the API forces — it exposes ``torch.backends.mps.is_available``
    rather than ``torch.mps.is_available``. A namespace with no `is_available` at all is taken
    at face value, since a build that ships one without the probe has nothing else to ask.
    """
    namespace = getattr(torch, name, None)
    if namespace is None:
        return None
    try:
        if name == "mps":
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            return namespace if mps is not None and mps.is_available() else None
        available = getattr(namespace, "is_available", None)
        return namespace if available is None or available() else None
    except Exception:
        return None


def _reading(namespace: object, *names: str, ordinal: int | None = None) -> object | None:
    """The first of `names` this namespace defines and answers, or `None`.

    Several of these readings are spelled differently per vendor, and a vendor that does not
    implement one raises rather than returning zero. Trying in order and treating a failure as
    "not reported" is what keeps a backend that supports three of four readings from losing
    all four.

    The device argument is offered and then withdrawn, in that order, because the two shapes
    both exist: ``memory_reserved(device)`` takes one and ``recommended_max_memory()`` does
    not. Offering it first is what makes the reading describe the device this process is
    *computing on* rather than device 0 — which on a worker granted two boards is a different
    card, and so a fraction of the wrong capacity.
    """
    for name in names:
        fn = getattr(namespace, name, None)
        if fn is None:
            continue
        attempts = ((ordinal,), ()) if ordinal is not None else ((),)
        for args in attempts:
            try:
                value = fn(*args)
            except TypeError:
                continue  # this spelling does not take a device; try it without
            except Exception:
                break  # it took the call and refused; a different argument will not help
            if value is not None:
                return value
    return None


def device_total_memory_bytes(
    backend: str | None = None, ordinal: int = 0, *, _namespace: object | None = None
) -> int | None:
    """Total memory of the accelerator this process is computing on, or `None`.

    Read from torch rather than from a vendor SMI, so it works for every backend torch
    supports without a per-vendor library. `ordinal` is the *bound* device — a worker granted
    two boards computes on one at a time, and reading properties off device 0 is right only by
    coincidence on a mixed node.

    Args:
        backend: The accelerator name; detected when omitted.
        ordinal: The device index within the visible set.
        _namespace: An already-resolved namespace, to skip re-resolving it. Internal.

    Returns:
        The device's total memory in bytes, or `None` when it cannot be read.
    """
    namespace = _namespace or accelerator_namespace(backend)
    if namespace is None:
        return None
    # Apple shares system memory with the CPU, so "total" is the budget torch will use before
    # it starts paging, not a board's capacity.
    recommended = _reading(namespace, "recommended_max_memory")  # unified memory: no ordinal
    if recommended:
        return int(recommended)
    try:
        properties = namespace.get_device_properties(ordinal)  # type: ignore[attr-defined]
    except Exception:
        return None
    total = getattr(properties, "total_memory", None)
    return int(total) if total else None


def device_memory_used_fraction(
    backend: str | None = None, ordinal: int = 0, *, _namespace: object | None = None
) -> float | None:
    """Share of the accelerator's memory this process's allocator holds, in [0, 1].

    The **reserved** bytes, not the allocated ones: reserved is what the device cannot give to
    anyone else, so it is what a predictive out-of-memory guard has to steer by. Allocated
    would under-report by exactly the cached blocks that make the next growth step fail.

    Args:
        backend: The accelerator name; detected when omitted.
        ordinal: The device index within the visible set.
        _namespace: An already-resolved namespace, to skip re-resolving it. Internal.

    Returns:
        The fraction in [0, 1], or `None` when it cannot be read.
    """
    namespace = _namespace or accelerator_namespace(backend)
    if namespace is None:
        return None
    # The namespace is threaded through rather than resolved again. Resolving costs a
    # `sys.path` walk behind the backend detector, and this reading is taken **once per batch**
    # by the guard that keeps the batch-size climb out of an out-of-memory — so a second and
    # third resolution inside one sample is pure per-batch overhead. Measured at 154us a call
    # before the threading and 4us after.
    total = device_total_memory_bytes(backend, ordinal, _namespace=namespace)
    if not total:
        return None
    used = _reading(
        namespace,
        "memory_reserved",
        "current_allocated_memory",
        "memory_allocated",
        ordinal=ordinal,
    )
    if used is None:
        return None
    return max(0.0, min(1.0, float(used) / total))


def device_utilization(backend: str | None = None) -> float | None:
    """Accelerator busy fraction in [0, 1] as *torch* reports it, or `None`.

    Torch exposes a `utilization()` on the namespaces whose vendors provide one — Intel XPU,
    Gaudi and Ascend among them — reported as a percentage. This is the path for the
    accelerators with no SMI library the control plane can link; NVIDIA and AMD have richer
    per-device readings and go through their own probes.

    Args:
        backend: The accelerator name; detected when omitted.

    Returns:
        The busy fraction in [0, 1], or `None` where the vendor reports none.
    """
    namespace = accelerator_namespace(backend)
    if namespace is None:
        return None
    value = _reading(namespace, "utilization")
    if value is None:
        return None
    try:
        percent = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, percent / 100.0))


def fragmentation_ratio() -> float | None:
    """Share of this process's allocator reservation that is cached but not in use.

    ``(reserved - allocated) / reserved``. The number that distinguishes the two device OOMs:

    * **Low ratio** — the device is genuinely full. Shrink the batch.
    * **High ratio** — the memory exists, split across blocks too small for the request.
      Shrinking may not help at all, and releasing cached blocks will.

    Returns:
        The ratio in [0, 1], or `None` when torch is absent, unused, or has reserved nothing.
    """
    namespace = accelerator_namespace()
    if namespace is None:
        return None
    try:
        reserved = int(namespace.memory_reserved())  # type: ignore[attr-defined]
        if reserved <= 0:
            return None
        allocated = int(namespace.memory_allocated())  # type: ignore[attr-defined]
        return max(0.0, min(1.0, (reserved - allocated) / reserved))
    except Exception:
        return None


def allocator_initialized() -> bool:
    """Whether torch's CUDA caching allocator has already reserved anything.

    The check that turns "these settings did nothing" into something a caller can report: past
    this point `ALLOC_CONF_ENV` is parsed and a later write is silently ignored.
    """
    namespace = accelerator_namespace()
    if namespace is None:
        return False
    try:
        initialized = getattr(namespace, "is_initialized", None)
        if initialized is not None and not initialized():
            return False
        return int(namespace.memory_reserved()) > 0  # type: ignore[attr-defined]
    except Exception:
        return False


def set_alloc_conf(conf: str) -> bool:
    """Set `PYTORCH_CUDA_ALLOC_CONF`, unless an operator already set it themselves.

    Their value outranks any default: it is an explicit tuning decision, and the settings
    interact (a hand-set `max_split_size_mb` is incompatible with expandable segments), so
    merging would be worse than either.

    Args:
        conf: The settings string to install.

    Returns:
        True when this call installed it; False when it was already set.
    """
    if os.environ.get(ALLOC_CONF_ENV):
        return False
    os.environ[ALLOC_CONF_ENV] = conf
    return True


def set_memory_fraction(fraction: float) -> bool:
    """Cap this process's share of its device through torch's own allocator.

    The cap that makes packing several actors onto one board safe rather than merely dense: a
    stage that misjudges its footprint fails its own allocation instead of exhausting the
    device and taking every co-tenant down with it.

    Args:
        fraction: Share of the device this process may allocate, in (0, 1].

    Returns:
        True when the cap was applied, False when torch is absent or has no device.
    """
    namespace = accelerator_namespace()
    if namespace is None:
        return False
    cap = getattr(namespace, "set_per_process_memory_fraction", None)
    if cap is None:
        return False  # a backend whose allocator has no cap; packing there stays uncapped
    try:
        cap(float(fraction))
    except Exception:
        return False
    return True
