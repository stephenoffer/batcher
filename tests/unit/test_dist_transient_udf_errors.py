"""A transient inference failure must not kill a multi-hour distributed job.

Both gather loops treated every `RayTaskError` as deterministic ("resubmitting cannot help").
That is true of a bug in the UDF and false of the failure modes that actually dominate GPU
batch inference: a CUDA OOM when several actors peak together, a throttled or briefly
unavailable model endpoint, a socket timeout. Each of those clears on a retry, and failing the
whole query on one discards every partition already inferred.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime.policies import _is_transient_udf_error

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"),
        OSError("Connection reset by peer"),
        TimeoutError("request timed out after 30s"),
        RuntimeError("HTTP 503 Service Unavailable"),
        RuntimeError("HTTP 429 Too Many Requests"),
        RuntimeError("AWS Error SLOW_DOWN during PutObject"),
        RuntimeError("NCCL timeout on rank 3"),
    ],
)
def test_resource_and_remote_service_failures_are_retryable(exc):
    assert _is_transient_udf_error(exc)


@pytest.mark.parametrize(
    "exc",
    [
        TypeError("unsupported operand type(s) for +: 'int' and 'str'"),
        KeyError("embedding"),
        ValueError("expected 384 dimensions, got 768"),
        AttributeError("'NoneType' object has no attribute 'encode'"),
    ],
)
def test_a_deterministic_udf_bug_is_not_retryable(exc):
    """Retrying these only burns the budget and delays an error the user must see."""
    assert not _is_transient_udf_error(exc)


def test_a_transient_cause_is_found_through_the_exception_chain():
    """The real cause is raised by torch / an HTTP client and arrives already wrapped."""
    try:
        try:
            raise RuntimeError("CUDA out of memory")
        except RuntimeError as inner:
            raise ValueError("udf failed on batch 12") from inner
    except ValueError as outer:
        assert _is_transient_udf_error(outer)


def test_the_chain_walk_terminates_on_a_reference_cycle():
    a = RuntimeError("outer")
    b = RuntimeError("inner")
    a.__cause__ = b
    b.__cause__ = a
    assert _is_transient_udf_error(a) is False
