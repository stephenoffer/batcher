"""What kind of failure this was, and therefore what to do with it.

Every retry decision in the engine reduces to one question — *will doing this again work?* —
and the honest answer has three parts, not two:

* **Retry here.** A CUDA out-of-memory when several actors peaked together, a model endpoint
  returning 429. The next attempt on the same worker usually succeeds.
* **Retry somewhere else.** A device that has faulted, a filesystem that went read-only, a
  node being reclaimed. Retrying *here* fails identically every time, and because a scheduler
  with a free slot on that node will keep offering it, the retries walk the entire queue onto
  the one broken machine. This is the failure mode that turns one bad node into a dead job,
  and it is invisible to a classifier that only knows "transient" and "not".
* **Do not retry.** A `TypeError` in a UDF fails the same way on every worker in the fleet.
  Retrying it burns the recovery budget, delays the real error by minutes, and finally
  surfaces as a resource error with the original traceback gone.

There is a fourth thing to know that is not about retrying at all. **Some failures mean the
results already produced are wrong**, not merely missing — an uncontained ECC fault on a device
returned a number, and the number is bad. Retrying past one of those completes the job
successfully with corrupt output, which is worse than the crash it avoided. `results_untrusted`
carries that, and nothing else in the taxonomy implies it.

Classification is on the exception's *type name and message text*, walking the cause chain,
because the real error is raised by torch, an HTTP client, a vendor SDK, or Ray and arrives here
already wrapped two or three deep. That is unavoidably heuristic, so the taxonomy is
deliberately conservative in one direction: an unrecognized failure is `"application"` — do not
retry, do not move — because wrongly retrying a deterministic bug across a fleet is the more
expensive mistake, and it is the one that hides its own cause.

Carbonite owns this: it is a protection concern, and both the single-node executor and the
distributed scheduler need the same answer. It imports nothing from Ray or torch, so it
classifies a failure on a driver that has neither installed.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CATEGORIES",
    "FailureClass",
    "classify_failure",
    "failure_class",
    "is_retryable",
    "must_move",
    "results_untrusted",
]


@dataclass(frozen=True, slots=True)
class FailureClass:
    """What one category of failure implies for the scheduler.

    Attributes:
        name: The category name.
        retryable: Whether another attempt can succeed at all. False means the failure is
            deterministic and every retry re-fails identically.
        must_move: Whether the retry has to land on a *different* device or node. The field
            that stops a retry storm: without it a scheduler with a free slot on the broken
            machine offers it again, and again, until the queue is exhausted there.
        results_untrusted: Whether work already completed under this failure may have returned
            wrong data. True only where a device kept running and answered incorrectly, which
            is a correctness condition and not a scheduling one.
        summary: One line an operator reads in an incident log.
    """

    name: str
    retryable: bool
    must_move: bool
    results_untrusted: bool
    summary: str


#: The taxonomy, keyed by category name.
#:
#: `must_move` is the field to read carefully, because the two obvious groupings are both
#: wrong. A device OOM does *not* move: the device is fine and the next attempt at a smaller
#: size fits, whereas moving it just relocates a memory-pressure problem onto a peer and loses
#: the warm model weights. A host OOM *does* move: the kernel already decided this node cannot
#: hold the working set, and it will decide the same thing again in thirty seconds.
CATEGORIES: dict[str, FailureClass] = {
    "preemption": FailureClass(
        "preemption",
        retryable=True,
        must_move=True,
        results_untrusted=False,
        summary="the node was reclaimed by its provider or scheduler",
    ),
    "worker_lost": FailureClass(
        "worker_lost",
        retryable=True,
        must_move=False,
        results_untrusted=False,
        summary="the worker process died without reporting a cause",
    ),
    "device_oom": FailureClass(
        "device_oom",
        retryable=True,
        must_move=False,
        results_untrusted=False,
        summary="the accelerator ran out of memory",
    ),
    "host_oom": FailureClass(
        "host_oom",
        retryable=True,
        must_move=True,
        results_untrusted=False,
        summary="the host ran out of memory and the kernel killed the process",
    ),
    "device_fault": FailureClass(
        "device_fault",
        retryable=True,
        must_move=True,
        results_untrusted=False,
        summary="the accelerator reported a hardware fault and needs a reset",
    ),
    "device_corruption": FailureClass(
        "device_corruption",
        retryable=True,
        must_move=True,
        results_untrusted=True,
        summary="the accelerator returned data that is not what was written",
    ),
    "network": FailureClass(
        "network",
        retryable=True,
        must_move=False,
        results_untrusted=False,
        summary="a peer or remote service was unreachable",
    ),
    "throttled": FailureClass(
        "throttled",
        retryable=True,
        must_move=False,
        results_untrusted=False,
        summary="a remote service asked for a slower rate",
    ),
    "timeout": FailureClass(
        "timeout",
        retryable=True,
        must_move=False,
        results_untrusted=False,
        summary="an operation exceeded its deadline",
    ),
    "storage": FailureClass(
        "storage",
        retryable=True,
        must_move=True,
        results_untrusted=False,
        summary="local storage was full, read-only, or failing",
    ),
    "application": FailureClass(
        "application",
        retryable=False,
        must_move=False,
        results_untrusted=False,
        summary="the failure is deterministic and will recur on every worker",
    ),
}

#: Category to the message fragments that identify it, tried in this order.
#:
#: Order is load-bearing where two categories share vocabulary. "out of memory" appears in both
#: a CUDA OOM and a kernel OOM kill, and the two have opposite `must_move` answers, so the
#: device-specific spellings are matched first. A generic "out of memory" with no device marker
#: falls through to the host category, which is the safer of the two to be wrong about: moving
#: a device OOM costs a warm cache, while not moving a host OOM costs the node.
_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "device_corruption",
        (
            "uncorrectable ecc",
            "uncontained ecc",
            "double-bit ecc",
            "ecc error",
            "xid 48",
            "xid 95",
        ),
    ),
    (
        "device_fault",
        (
            "cuda error: unspecified launch failure",
            "cuda error: an illegal memory access",
            "device-side assert",
            "cuda error: unknown error",
            "cudaerrorecc",
            "gpu has fallen off the bus",
            "nvml_error",
            "hip error",
            "no cuda-capable device is detected",
            "cuda error: invalid device ordinal",
        ),
    ),
    (
        "device_oom",
        (
            "cuda out of memory",
            "hip out of memory",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
            "out of memory on device",
            "xpu out of memory",
            "mps backend out of memory",
            # The spellings a GPU task's allocation failure actually arrives in when the
            # frame is cuDF rather than torch. RMM raises `rmm::out_of_memory` and
            # `rmm::bad_alloc`, the C++ allocator underneath raises `std::bad_alloc`, and the
            # CUDA driver reports `cudaErrorMemoryAllocation` — none of which contains the
            # phrase "cuda out of memory". Without them a cuDF worker's OOM classified as a
            # deterministic bug and was never retried at a smaller size, which is the one
            # response that would have worked.
            "out_of_memory",
            "bad_alloc",
            "cudaerrormemoryallocation",
            "insufficient memory",
            # gRPC's RESOURCE_EXHAUSTED, which is how an allocation failure inside a remote
            # inference server reaches the caller.
            "resource_exhausted",
        ),
    ),
    (
        "preemption",
        (
            "preempted",
            "spot interruption",
            "instance-action",
            "node is draining",
            "scheduled for termination",
            "sigterm",
        ),
    ),
    (
        "host_oom",
        (
            "out of memory",
            "oom-kill",
            "killed by the kernel",
            "cannot allocate memory",
        ),
    ),
    (
        "storage",
        (
            "no space left on device",
            "read-only file system",
            "disk quota exceeded",
            "input/output error",
            "structure needs cleaning",
        ),
    ),
    (
        "throttled",
        (
            "too many requests",
            "429",
            "503",
            "service unavailable",
            "temporarily unavailable",
            "slow_down",
            "rate limit",
            "throttl",
            # What a busy *model* endpoint says instead of a status code. An inference
            # stage is the main consumer of this taxonomy, and "the server is overloaded"
            # (OpenAI) / "model is currently loading" (HuggingFace) are the two phrasings
            # it meets — both of which the next attempt usually serves.
            "overloaded",
            "is currently loading",
            "model_not_ready",
            "not ready yet",
        ),
    ),
    (
        "network",
        (
            "connection reset",
            "connection aborted",
            "connection refused",
            "broken pipe",
            "nccl",
            "socket closed",
            "unreachable",
            "name or service not known",
            "bad gateway",
            "502",
        ),
    ),
    (
        "timeout",
        (
            "timed out",
            "timeout",
            "deadline exceeded",
        ),
    ),
    (
        "worker_lost",
        (
            "worker died",
            "actor died",
            "raylet",
            "node failure",
            "the actor is dead",
            "owner has died",
            "segmentation fault",
            "bus error",
        ),
    ),
)

#: Exception *type* names that settle the category without reading a message. Checked first,
#: because a type is a much stronger signal than a substring — `MemoryError` means what it says,
#: where the word "timeout" in someone's error text may be part of a parameter name.
_BY_TYPE: dict[str, str] = {
    "MemoryError": "host_oom",
    "TimeoutError": "timeout",
    # Ray's `GetTimeoutError` is a *caller-imposed* deadline, not a failure of the work: the
    # task is very likely still running. Retrying it duplicates work that was never lost and
    # then times out again on the same deadline. Classified as the do-not-retry category
    # deliberately, in agreement with the distributed scheduler's own fatal-error list.
    "GetTimeoutError": "application",
    "ConnectionResetError": "network",
    "ConnectionRefusedError": "network",
    "ConnectionAbortedError": "network",
    "BrokenPipeError": "network",
    "OutOfMemoryError": "host_oom",  # Ray's memory monitor killing a task under node pressure
    "NodePreemptedError": "preemption",
    "ActorDiedError": "worker_lost",
    "ActorUnavailableError": "worker_lost",
    "WorkerCrashedError": "worker_lost",
    "NodeDiedError": "worker_lost",
    "LocalRayletDiedError": "worker_lost",
    "RetryableShuffleError": "network",
    "FatalShuffleError": "application",
    "TaskCancelledError": "application",
    "RuntimeEnvSetupError": "application",
}

#: `errno` values that name a storage condition outright. An `OSError` carries the kernel's own
#: verdict in a field, which beats matching the message text it was formatted into — the text
#: is localized on some systems and the number never is.
_STORAGE_ERRNOS = frozenset({28, 30, 122, 5, 117})  # ENOSPC, EROFS, EDQUOT, EIO, EUCLEAN


def _chain(exc: BaseException):
    """The exception and everything it was raised from, without cycling."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def failure_class(exc: BaseException) -> str:
    """The category name for a failure.

    Args:
        exc: The exception, however deeply wrapped. Ray fuses a remote exception's original
            type into the class it raises locally, and the cause chain is walked, so a torch
            OOM inside a `RayTaskError` classifies as a device OOM rather than as an unknown.

    Returns:
        A key of `CATEGORIES`. `"application"` for anything unrecognized — deliberately the
        do-not-retry answer, because retrying a deterministic bug across a fleet costs more
        than failing fast on a transient one that was misread.
    """
    for cur in _chain(exc):
        name = type(cur).__name__
        if name in _BY_TYPE:
            return _BY_TYPE[name]
        if getattr(cur, "errno", None) in _STORAGE_ERRNOS:
            return "storage"
        # The type name *and* the message, because a remote failure arrives as Ray's own
        # wrapper class with the worker-side traceback — including the original exception's
        # name — formatted into the message. Matching the class name alone would see only
        # `RayTaskError` and classify every remote failure as unknown.
        text = f"{name} {cur}".lower()
        for category, markers in _MARKERS:
            if any(marker in text for marker in markers):
                return category
    return "application"


def classify_failure(exc: BaseException) -> FailureClass:
    """The full class record for a failure.

    Args:
        exc: The exception, however deeply wrapped.

    Returns:
        The `FailureClass` naming whether to retry, whether the retry must move, and whether
        results already produced are suspect.
    """
    return CATEGORIES[failure_class(exc)]


def is_retryable(exc: BaseException) -> bool:
    """Whether another attempt at this work can succeed.

    Args:
        exc: The exception, however deeply wrapped.

    Returns:
        False for a deterministic failure, which every retry would reproduce.
    """
    return classify_failure(exc).retryable


def must_move(exc: BaseException) -> bool:
    """Whether a retry has to land on a different device or node.

    The answer that stops a retry storm. A scheduler that keeps a free slot on the broken
    machine will keep offering it, so a failure that is local to the host must be told to move
    or it walks the whole queue onto one node.

    Args:
        exc: The exception, however deeply wrapped.

    Returns:
        True when the same placement would fail again for the same reason.
    """
    return classify_failure(exc).must_move


def results_untrusted(exc: BaseException) -> bool:
    """Whether work already completed alongside this failure may be wrong.

    Not a scheduling question. A job that retries past a device that returned corrupted data
    finishes successfully and writes out the corruption, which is strictly worse than failing.

    Args:
        exc: The exception, however deeply wrapped.

    Returns:
        True only for failures documented as returning bad data rather than no data.
    """
    return classify_failure(exc).results_untrusted
