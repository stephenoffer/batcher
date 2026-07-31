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
"""

from __future__ import annotations

import os

__all__ = [
    "ALLOC_CONF_ENV",
    "FRAGMENTATION_THRESHOLD",
    "allocator_initialized",
    "fragmentation_ratio",
    "set_alloc_conf",
    "set_memory_fraction",
]

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


def fragmentation_ratio() -> float | None:
    """Share of this process's allocator reservation that is cached but not in use.

    ``(reserved - allocated) / reserved``. The number that distinguishes the two device OOMs:

    * **Low ratio** — the device is genuinely full. Shrink the batch.
    * **High ratio** — the memory exists, split across blocks too small for the request.
      Shrinking may not help at all, and releasing cached blocks will.

    Returns:
        The ratio in [0, 1], or `None` when torch is absent, unused, or has reserved nothing.
    """
    torch = _torch()
    if torch is None:
        return None
    try:
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return None
        reserved = int(cuda.memory_reserved())
        if reserved <= 0:
            return None
        allocated = int(cuda.memory_allocated())
        return max(0.0, min(1.0, (reserved - allocated) / reserved))
    except Exception:
        return None


def allocator_initialized() -> bool:
    """Whether torch's CUDA caching allocator has already reserved anything.

    The check that turns "these settings did nothing" into something a caller can report: past
    this point `ALLOC_CONF_ENV` is parsed and a later write is silently ignored.
    """
    torch = _torch()
    if torch is None:
        return False
    try:
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_initialized():
            return False
        return int(cuda.memory_reserved()) > 0
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
    torch = _torch()
    if torch is None:
        return False
    try:
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return False
        cuda.set_per_process_memory_fraction(float(fraction))
    except Exception:
        return False
    return True
