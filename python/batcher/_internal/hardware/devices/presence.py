"""Is there an accelerator in *this* process? — a three-valued answer, shared by two layers.

Separate from the richer device modules beside it because the question is different and so
is the tolerance for being wrong. Those report what a device *is* (capacity, utilization,
health) for code that has already decided to use one. This answers whether to warn a user
that the device they asked for is not here, and a false "no" would fire that warning on
every GPU query run on an actual GPU box.

So the answer is deliberately three-valued: `True`, `False`, or `None` for "cannot tell".
A host whose devices cannot be read is not a host without devices.
"""

from __future__ import annotations

__all__ = ["local_accelerator_present"]


def local_accelerator_present() -> bool | None:
    """Whether this process can see an accelerator: `True`, `False`, or `None` if unknown.

    Two independent readings, because either alone has a blind spot. The hardware layer's
    device scope answers without importing torch, but reads NVML and so sees nothing on a
    non-NVIDIA accelerator or a host without the bindings — its silence is not evidence.
    Torch answers for every backend it supports and is consulted second, because it is the
    heavier import; callers gate this behind a plan that already intends to load a model, so
    that import costs nothing the query was not about to pay anyway.

    Returns:
        `True` if a device is visible, `False` if the absence is positively established, and
        `None` when neither reading could answer.

    Examples:
        .. doctest::

            >>> from batcher._internal.hardware.devices.presence import local_accelerator_present
            >>> local_accelerator_present() in (True, False, None)
            True
    """
    try:
        from batcher._internal.hardware.devices import device_scope

        if device_scope().count > 0:
            return True
    except Exception:  # a probe must never be the thing that fails a query
        pass
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            return True
        for backend in ("xpu", "mps"):
            module = getattr(torch, backend, None)
            if module is not None and module.is_available():
                return True
        return False
    except Exception:
        return None
