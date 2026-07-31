"""Surviving a device out-of-memory: telling the three kinds apart, answering the right one.

The engine already halves a batch and retries when a GPU stage OOMs. That recovers the common
case and mishandles two others, because "out of memory" on a device is three different
failures wearing one message:

* **Too large.** The batch genuinely does not fit. Halving is exactly right.
* **Fragmented.** There is enough free memory and no single block big enough for the request.
  Halving *may* work by accident, and often does not — the next request is still larger than
  the largest hole. Releasing the allocator's cached blocks is what actually helps, and doing
  it costs nothing the retry was not already going to pay.
* **Occupied.** A co-tenant on the device holds the memory. Halving this process's batch to
  one row will not recover it, and retrying at all just burns the attempt budget — the
  scheduler needs to hear that the device is full, not that this batch is big.

Getting the classification wrong is not merely inefficient. A fragmented process that halves
its way down to a single row has thrown away all of its throughput and still fails; an
occupied device that is retried sixteen times turns one placement mistake into minutes of
wasted GPU time across the fleet.

One more thing the halving retry did not do, and it decides how much room the retry gets:
**collect before releasing.** PyTorch cannot return a cached block whose tensor is still
referenced, and a dead tensor is only unreferenced once Python has collected it. On the OOM
path the batch that failed is usually still held by a traceback frame, so emptying the cache
without collecting first frees a fraction of what it could.

Classification and release only. What to do with the verdict — shrink, and *remember* the size
that failed so the batch-size controller stops climbing back into it — belongs to the caller
(`ml.autobatch`), because it is a decision about a workload rather than a fact about a device.

Pure inspection plus best-effort framework calls; nothing here allocates. Every entry point is
a no-op where torch is absent, so a CPU stage pays nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from batcher._internal.hardware.devices.torch_memory import (
    FRAGMENTATION_THRESHOLD,
    fragmentation_ratio,
)

__all__ = [
    "OomKind",
    "OomVerdict",
    "classify_oom",
    "is_device_oom",
    "release_device_cache",
]


class OomKind(Enum):
    """Why a device allocation failed, which decides what to do about it.

    Attributes:
        TOO_LARGE: The request genuinely exceeds what is free. Shrink and retry.
        FRAGMENTED: Enough is free, in blocks too small for the request. Release cached blocks
            and retry at the same size before shrinking.
        OCCUPIED: Another process holds the device's memory. Shrinking cannot recover it, so
            surface it as a placement failure instead of retrying.
    """

    TOO_LARGE = "too_large"
    FRAGMENTED = "fragmented"
    OCCUPIED = "occupied"


@dataclass(frozen=True, slots=True)
class OomVerdict:
    """What kind of device OOM this was, and what the caller should do next.

    Attributes:
        kind: The classification.
        fragmentation: This process's cached-but-unused share when it was measurable.
        own_fraction: Share of the device's used memory this process is responsible for, when
            the driver attributed it. Low means a co-tenant owns the problem.
        detail: A short human-readable reason, for the error a caller eventually raises.
    """

    kind: OomKind
    fragmentation: float | None = None
    own_fraction: float | None = None
    detail: str = ""

    @property
    def should_retry_same_size(self) -> bool:
        """Whether releasing cached blocks alone is likely to make the same request fit.

        True only for fragmentation. This is the retry that costs nothing and recovers the
        batch intact, and it is worth exactly one attempt: if the request still fails after
        the cache is gone, the memory genuinely is not there and the caller should shrink.
        """
        return self.kind is OomKind.FRAGMENTED

    @property
    def should_shrink(self) -> bool:
        """Whether halving the batch is a response that can work.

        False for an occupied device, where this process's batch is not what filled it.
        """
        return self.kind is not OomKind.OCCUPIED


def is_device_oom(exc: BaseException) -> bool:
    """Whether `exc` is an accelerator out-of-memory error, checked without importing torch.

    Structural rather than `isinstance`, deliberately: this runs on a failure path in a
    process that may not have torch loaded, and importing a multi-second dependency to decide
    how to report an error is a poor trade. The type name covers
    `torch.cuda.OutOfMemoryError`; the message covers the older
    `RuntimeError: CUDA out of memory`, XLA's `RESOURCE_EXHAUSTED`, and the HIP phrasing.

    Args:
        exc: The exception a device stage raised.

    Returns:
        True when this is a device memory exhaustion rather than a model or data error.

    Examples:
        .. doctest::

            >>> from batcher._internal.hardware.devices import is_device_oom
            >>> is_device_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
            True
            >>> is_device_oom(ValueError("bad column"))
            False
    """
    if type(exc).__name__ in ("OutOfMemoryError", "ResourceExhaustedError"):
        return True
    message = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "out of memory" in message
        or "resource_exhausted" in message
        # HIP reports exhaustion under its own name, so a ROCm host matched none of the
        # phrasings above and lost the halving retry entirely.
        or "hip_error_out_of_memory" in message
    )


def classify_oom(exc: BaseException, *, device: int | None = None) -> OomVerdict:
    """Decide which of the three device-OOM failures `exc` is.

    The classification is evidence-based and degrades toward the response that is always safe.
    Where nothing can be measured — no torch, no driver attribution, a backend with no
    allocator statistics — the verdict is `TOO_LARGE`, because shrinking is the answer that
    cannot make things worse. Only positive evidence moves it: a measured fragmentation ratio
    past the threshold, or a device whose memory is measurably somebody else's.

    Args:
        exc: The exception that was raised; classified only for a genuine device OOM.
        device: Physical device index the failure happened on, or `None` for the current one.

    Returns:
        The verdict, always `TOO_LARGE` for a non-OOM exception so a caller that classifies
        unconditionally still gets a safe answer.
    """
    if not is_device_oom(exc):
        return OomVerdict(OomKind.TOO_LARGE, detail="not a device out-of-memory error")
    fragmentation = fragmentation_ratio()
    own = _own_share_of_device(device)
    # Occupancy is checked first and wins: it is the one verdict that says *retrying is
    # futile*, and a co-tenant filling the device can leave this process's own small
    # reservation looking fragmented at the same time.
    if own is not None and own < _CO_TENANT_SHARE:
        return OomVerdict(
            OomKind.OCCUPIED,
            fragmentation=fragmentation,
            own_fraction=own,
            detail=(
                f"another process holds {(1.0 - own):.0%} of this device's used memory; "
                "shrinking this stage's batch cannot recover it"
            ),
        )
    if fragmentation is not None and fragmentation >= FRAGMENTATION_THRESHOLD:
        return OomVerdict(
            OomKind.FRAGMENTED,
            fragmentation=fragmentation,
            own_fraction=own,
            detail=(
                f"{fragmentation:.0%} of this process's device reservation is cached but "
                "unused, so the memory exists in blocks too small for this request"
            ),
        )
    return OomVerdict(
        OomKind.TOO_LARGE,
        fragmentation=fragmentation,
        own_fraction=own,
        detail="the device is full at this batch size",
    )


#: Share of a device's *used* memory below which this process is a bystander rather than the
#: cause. Deliberately low: a stage that owns even a third of what is resident is still a
#: plausible cause of its own OOM, and misreading that as someone else's problem would skip
#: the shrink that would have fixed it. Only a process holding almost none of a full device
#: has positive evidence that it is not the one that filled it.
_CO_TENANT_SHARE = 0.25


def _own_share_of_device(device: int | None) -> float | None:
    """This process's share of a device's resident memory, or `None` when unattributable.

    `None` — not zero — whenever the driver will not attribute memory per process, which is
    the normal case inside a container that cannot see other PID namespaces and on a MIG
    instance. Reporting zero there would classify every OOM on every containerized fleet as
    somebody else's, which is precisely backwards.
    """
    from batcher._internal.hardware.devices.scope import current_physical_index, device_scope
    from batcher._internal.hardware.nvml import own_device_memory

    index = current_physical_index() if device is None else device
    if index is None:
        return None
    scope = device_scope()
    used = (scope.used or {}).get(index)
    if not used:
        return None
    mine = own_device_memory(index)
    if mine is None:
        return None
    return max(0.0, min(1.0, mine / used))


def release_device_cache() -> bool:
    """Return cached device blocks to the driver, collecting first so there is more to return.

    The ordering is the point, and it is the whole reason this wraps the release rather than
    calling it directly. PyTorch's allocator can only release a cached block once no tensor
    references it, and on an out-of-memory path the tensors of the batch that just failed are
    still alive in the exception's traceback frames. Emptying the cache first — which is what
    every hand-rolled retry does — releases whatever happens to be unreferenced already and
    leaves the rest, so the retry runs with much of the memory that just overflowed. One
    `gc.collect()` beforehand drops those frames, which is the difference between a retry that
    has room and one that fails identically.

    Only frameworks **already imported** are touched. Importing torch to free memory would
    cost seconds and several hundred megabytes on a worker that never used it, which is the
    opposite of the goal, and a process that never imported torch has no torch cache to free.
    RMM is deliberately absent: a cuDF pool returns memory to the driver only on an explicit
    reinitialize, which would tear down every outstanding device buffer in the process — a far
    larger hammer than an OOM retry is allowed to swing.

    Vendor-agnostic: NVIDIA and AMD share ``torch.cuda`` (ROCm shims the CUDA API), Intel is
    ``torch.xpu``, Apple ``torch.mps``, and XLA has no cache to empty at all — a TPU or
    Trainium releases by stepping its execution graph.

    Returns:
        True when at least one backend released something, False where none applies — which is
        not a failure, just a worker with no framework loaded.
    """
    import gc
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return _xla_mark_step()
    gc.collect()
    released = False
    for name in ("cuda", "xpu", "mps"):
        backend = getattr(torch, name, None)
        empty = getattr(backend, "empty_cache", None)
        if empty is None:
            continue
        try:
            available = getattr(backend, "is_available", None)
            if name == "mps" or available is None or available():
                empty()
                released = True
        except Exception:
            # A backend compiled in but with no device raises here. That is the normal case on
            # a CPU-only worker and must never propagate out of a recovery path, whose entire
            # job is to be the thing that does not fail.
            continue
    return _xla_mark_step() or released


def _xla_mark_step() -> bool:
    """Step XLA's execution graph, which is how a TPU or Trainium releases device memory."""
    import sys

    if "torch_xla" not in sys.modules:
        return False
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]

        xm.mark_step()
    except Exception:
        return False
    return True
