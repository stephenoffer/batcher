"""Telling a GPU backend that declined from one that is broken.

The backend is written to fall back, and that tolerance is right: an unsupported shape, a
device out of memory, a lost worker, a cluster with no GPU. It is also how two whole-path
outages shipped unnoticed here — a moved autoscale helper that made every multi-device fan-out
raise `ImportError`, and a fan-out called without one of its keyword arguments — because a
handler written for "the device declined" cannot tell those apart from one.

The fallback is unchanged either way: the CPU engine answers the query and returns the same
rows. What changes is whether a backend that has stopped working says so.

Kept in its own module with no intra-package imports, so `route`, `translate` and `fanout` can
all reach it without the cycle that putting it in any of them would create.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed

#: Exception types that mean the GPU backend itself is broken rather than that this plan cannot
#: run on a device. A missing symbol, a renamed attribute, a call whose signature no longer
#: matches: none of those are properties of the query, and every one of them is silent under a
#: handler written for "the device declined". Both of the whole-path outages this file has had
#: were of exactly this shape — an `ImportError` from a moved autoscale helper that disabled
#: every multi-device fan-out, and a `TypeError` from a fan-out called without one of its
#: keyword arguments — and both looked, from the outside, like a query that was simply slow.
_BACKEND_DEFECTS = (ImportError, AttributeError, NameError, TypeError)

#: `verify.DeviceDivergence` is a defect too, and is matched by name rather than by import: it
#: lives in `verify`, which imports this module, so naming the class here would close a cycle.
#: A divergence is the one failure on this path that is never a decline — the tier's contract
#: is that the device changes where a plan runs, never what it computes.
_DEFECT_NAMES = frozenset({"DeviceDivergence"})


def note_gpu_failure(step: str, exc: BaseException) -> None:
    """Record a GPU-path failure, loudly when it reads as a defect rather than a decline.

    The fallback is the same either way — the CPU engine runs the query and returns the same
    rows — so this changes nothing about the answer. What it changes is whether a backend that
    has stopped working says so. A debug note is right for "this device is out of memory" and
    wrong for "this function does not exist", and the two were previously indistinguishable.

    Args:
        step: A short, stable name for what was attempted.
        exc: The exception being suppressed.
    """
    if not isinstance(exc, _BACKEND_DEFECTS) and type(exc).__name__ not in _DEFECT_NAMES:
        note_suppressed("api", step, exc)
        return
    import logging

    from batcher._internal.logging import get_logger, log_kv

    log_kv(
        get_logger("api"),
        logging.WARNING,
        "the GPU backend is not usable and the CPU engine ran this query instead",
        step=step,
        error=type(exc).__name__,
        detail=str(exc),
    )
