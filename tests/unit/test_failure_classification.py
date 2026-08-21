"""What kind of failure this was, and therefore whether — and where — to retry it.

The taxonomy exists because "transient or not" is not enough to keep a job alive on an unstable
fleet. There are three answers, not two, and the third is the one that matters most: a failure
that will recur *on this machine specifically* must be retried somewhere else, or the scheduler
keeps offering the same free slot and the retries walk the whole queue onto one broken node.

The tests below fix the three answers, and the fourth thing that is not about retrying at all:
whether the results already produced can be trusted.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.resilience import classify as c

pytestmark = pytest.mark.unit


def _wrapped(inner: BaseException, outer_text: str = "RayTaskError") -> BaseException:
    """`inner` as it arrives from a worker: re-raised inside another exception."""
    try:
        try:
            raise inner
        except BaseException as exc:
            raise RuntimeError(outer_text) from exc
    except BaseException as exc:
        return exc


def test_a_device_oom_retries_in_place_because_the_device_is_fine():
    verdict = c.classify_failure(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert verdict.name == "device_oom"
    assert verdict.retryable is True
    # Moving it would relocate a sizing problem onto a peer and throw away the warm weights.
    assert verdict.must_move is False


def test_a_host_oom_moves_because_the_kernel_already_decided_this_node_cannot_hold_it():
    assert c.must_move(MemoryError()) is True
    assert c.failure_class(RuntimeError("oom-kill:constraint=CONSTRAINT_MEMCG")) == "host_oom"


def test_a_device_fault_moves_and_a_throttled_endpoint_does_not():
    assert c.must_move(RuntimeError("CUDA error: unspecified launch failure")) is True
    assert c.must_move(RuntimeError("HTTP 429 Too Many Requests")) is False


def test_a_deterministic_bug_is_not_retried_at_all():
    # The expensive mistake in the other direction: retrying this across a fleet burns the
    # budget, delays the real error, and finally reports it as a resource problem.
    assert c.is_retryable(TypeError("unsupported operand type(s)")) is False
    assert c.failure_class(TypeError("nope")) == "application"


def test_an_unrecognized_failure_is_treated_as_a_bug_not_as_a_transient():
    class SomethingNew(Exception):
        pass

    assert c.failure_class(SomethingNew("who knows")) == "application"
    assert c.is_retryable(SomethingNew("who knows")) is False


def test_the_cause_chain_is_walked_because_a_remote_failure_arrives_wrapped():
    wrapped = _wrapped(RuntimeError("CUDA out of memory"))
    assert c.failure_class(wrapped) == "device_oom"


def test_a_storage_errno_beats_the_message_text():
    # The kernel's verdict is in a field. The text it was formatted into is localized on some
    # systems; the number never is.
    assert c.failure_class(OSError(28, "No space left on device")) == "storage"
    assert c.failure_class(OSError(30, "Read-only file system")) == "storage"
    assert c.must_move(OSError(28, "")) is True


def test_only_a_corrupting_device_fault_marks_results_untrusted():
    # A device that fell off the bus returned nothing, so retrying past it is safe. One that
    # took an uncontained ECC error returned a wrong number and kept going, so a job that
    # retries past it finishes successfully and writes out the corruption.
    assert c.results_untrusted(RuntimeError("Xid 95: uncontained ECC error")) is True
    assert c.results_untrusted(RuntimeError("GPU has fallen off the bus")) is False
    assert c.results_untrusted(TypeError("nope")) is False


def test_preemption_is_not_a_sign_of_an_unhealthy_node():
    verdict = c.classify_failure(RuntimeError("instance-action: terminate"))
    assert verdict.name == "preemption"
    assert verdict.retryable is True
    assert verdict.must_move is True


@pytest.mark.parametrize(
    "message",
    [
        "std::bad_alloc",
        "rmm::out_of_memory: CUDA error at: ../include/rmm/mr/device",
        "RuntimeError: cudaErrorMemoryAllocation out of memory",
        "insufficient memory to allocate the output column",
        "StatusCode.RESOURCE_EXHAUSTED",
    ],
)
def test_a_cudf_workers_allocation_failure_is_a_device_oom(message):
    # These are the spellings a GPU task's OOM actually arrives in when the frame is cuDF
    # rather than torch, and none of them contains the phrase "cuda out of memory". Read as
    # an application bug, they were never retried at a smaller size — the one response that
    # would have worked.
    assert c.failure_class(RuntimeError(message)) == "device_oom"
    assert c.is_retryable(RuntimeError(message)) is True


def test_a_device_marker_wins_over_the_generic_out_of_memory_text():
    # Both spellings contain "out of memory" and the two have opposite `must_move` answers,
    # so the ordering of the marker table is load-bearing rather than incidental.
    assert c.failure_class(RuntimeError("CUDA out of memory")) == "device_oom"
    assert c.failure_class(RuntimeError("Out of memory: Killed process 4")) == "host_oom"


def test_every_category_in_the_table_is_reachable_and_complete():
    for name, record in c.CATEGORIES.items():
        assert record.name == name
        assert record.summary
        # A category that is neither retryable nor a move instruction must be the terminal one:
        # anything else would be a silent "give up" with no explanation attached.
        if not record.retryable:
            assert name == "application"


@pytest.mark.parametrize(
    "exc",
    [
        KeyError("timeout"),
        AssertionError("expected status 429, got 200"),
        TypeError("cannot concat 'connection reset by peer' to int"),
        AttributeError("'NoneType' object has no attribute 'cuda out of memory'"),
        IndexError("list index out of range"),
        ZeroDivisionError("division by zero"),
    ],
)
def test_a_deterministic_error_is_not_retried_for_words_in_its_message(exc):
    # The message of these types is *data*, not diagnosis: a `KeyError`'s message is the key
    # that was missing, so a UDF doing `config["timeout"]` against a dict that lacks it
    # raises exactly `KeyError: 'timeout'`. The marker scan read that as the `timeout`
    # category and declared it retryable — the deterministic bug retried across the whole
    # fleet that this module's docstring calls the more expensive mistake. Each of these
    # fails identically on every worker no matter what string it carries.
    assert c.failure_class(exc) == "application"
    assert c.is_retryable(exc) is False


def test_suppressing_the_text_scan_still_lets_a_wrapped_cause_win():
    # Only the *text* scan is suppressed, never the chain walk: a real transient wrapped in
    # a deterministic type must still classify by its cause, or the fix above would trade
    # one misclassification for its mirror image.
    try:
        try:
            raise ConnectionResetError(104, "Connection reset by peer")
        except ConnectionResetError as inner:
            raise KeyError("timeout") from inner
    except KeyError as exc:
        assert c.failure_class(exc) == "network"
        assert c.is_retryable(exc) is True


def test_rays_synthesized_wrapper_is_still_scanned_by_text():
    # Ray fuses the remote type into its wrapper's *name* (`RayTaskError(TypeError)`), which
    # is not an exact match for a bare `TypeError`, so a remote failure's formatted
    # worker-side traceback is still read exactly as it was before.
    wrapper = type("RayTaskError(TypeError)", (RuntimeError,), {})
    exc = wrapper("ray::task() ... RuntimeError: CUDA out of memory")
    assert c.failure_class(exc) == "device_oom"
    assert c.is_retryable(exc) is True


# --- The code spellings every cloud SDK actually uses -------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # S3, and the commonest transient failure a large write meets: a bucket written
        # faster than a prefix will take. Boto and s3fs surface the code verbatim with no
        # status code in the message, so a spaced marker matched nothing at all.
        ("An error occurred (SlowDown) when calling the PutObject operation", "throttled"),
        ("Please reduce your request rate.", "throttled"),
        ("An error occurred (RateLimitExceeded) when calling the objects.insert", "throttled"),
        ("ServerBusy: Egress is over the account limit.", "throttled"),
        ("TooManyRequests: retry after 3s", "throttled"),
        ("ProvisionedThroughputExceededException: rate exceeded for shard", "throttled"),
        ("OperationTimedOut", "timeout"),
        ("RemoteDisconnected('Remote end closed connection without response')", "network"),
        ("IncompleteRead(1024 bytes read, 4096 more expected)", "network"),
    ],
)
def test_a_concatenated_error_code_is_classified_like_its_spaced_phrase(message, expected):
    # The markers are substring matches on the message text, so `too many requests` does not
    # match `TooManyRequests` and `slow_down` does not match `SlowDown`. Every one of these
    # fell through to `application` — *not retryable* — and failed the whole query on a
    # condition the next attempt would have served.
    assert c.failure_class(RuntimeError(message)) == expected
    assert c.is_retryable(RuntimeError(message)) is True


def test_a_throttle_is_retried_here_rather_than_moved():
    # A remote service asking for a slower rate says nothing about this node, so moving the
    # work buys nothing and gives up a warm worker.
    exc = RuntimeError("An error occurred (SlowDown) when calling the PutObject operation")
    assert c.must_move(exc) is False
    assert c.results_untrusted(exc) is False


def test_the_storage_vocabulary_agrees_with_the_io_retry_loop():
    """Every storage condition `io` calls transient must be retryable here too.

    There are two classifiers, deliberately: `io.base._transient` answers "retry this read or
    write", and this one answers "retry, move, or trust the results" for the scheduler. They
    describe the same failures from different sides, and two lists of the same conditions is
    exactly the shape that drifts — this one had already fallen behind on the *code* spellings
    (`SlowDown`, `ServerBusy`, `TooManyRequests`) that `io` carries, so a throttle the reader
    retried happily failed the task outright once the scheduler saw it.

    The database half of `io`'s list is named below rather than asserted on: a deadlock or a
    serialization failure is retried by the write path itself, one layer down, and the
    scheduler seeing one means that loop already gave up.
    """
    from batcher.io.base._transient import _TRANSIENT_MARKERS

    # Conditions the IO retry loop owns end to end. A deadlocked or serializing transaction is
    # re-run by the writer, and "try again" is too generic to carry a category on its own.
    io_owned = {
        "deadlock",
        "could not serialize access",
        "serialization failure",
        "lock wait timeout",
        "database is locked",
        "database table is locked",
        "concurrent update",
        "restart transaction",
        "retry transaction",
        "too many connections",
        "server closed the connection",
        "ssl connection has been closed",
        "the database system is starting up",
        "request has expired",
        "operation aborted",
        "try again",
        "backenderror",
        "internal error",
        "internal server error",
        "internalerror",
    }
    missed = [
        marker
        for marker in _TRANSIENT_MARKERS
        if marker not in io_owned and not c.is_retryable(RuntimeError(f"backend said: {marker}"))
    ]
    assert not missed, (
        "storage conditions the IO layer retries but the scheduler would not: "
        f"{missed}. Add them to `classify._MARKERS`, or to `io_owned` above with the reason."
    )
