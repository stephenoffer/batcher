"""The retry classifier against the error strings the object stores actually raise.

The markers were spaced English phrases, but a store reports a machine-readable *error
code*, and those codes are CamelCase with no spaces -- so `InternalError`,
`ServiceUnavailable`, `ServerBusy`, `rateLimitExceeded` and `backendError` all fell
through. Falling through does not merely skip a retry: `ErrorPolicy` then records the blip
as a corrupt file, so under ``on_error="skip"`` a healthy object's rows were dropped with
nothing but a warning. Each string below is the shape its SDK produces.
"""

from __future__ import annotations

import pytest

from batcher.io.base._transient import is_transient

pytestmark = pytest.mark.unit

TRANSIENT = [
    # AWS / botocore
    "An error occurred (SlowDown) when calling the GetObject operation",
    "An error occurred (InternalError) when calling the GetObject operation",
    "An error occurred (ServiceUnavailable) when calling the GetObject operation",
    "An error occurred (RequestTimeout) when calling the PutObject operation",
    "An error occurred (503) when calling GetObject: Service Unavailable",
    'Could not connect to the endpoint URL: "https://s3.amazonaws.com/bucket"',
    # Google Cloud Storage
    "429 GET https://storage.googleapis.com/x: rateLimitExceeded",
    "503 GET https://storage.googleapis.com/x: backendError",
    "500 GET https://storage.googleapis.com/x: internalError",
    # Azure Blob / ADLS
    "ErrorCode:ServerBusy The server is busy.",
    "ErrorCode:OperationTimedOut Operation could not be completed within the specified time.",
    "ErrorCode:InternalError The server encountered an internal error.",
    # HDFS and generic sockets
    "Failed on local exception: java.io.IOException: Please try again later",
    "Read timed out",
    "Connection reset by peer",
]

PERMANENT = [
    "An error occurred (NoSuchKey) when calling GetObject: Not Found",
    "An error occurred (NoSuchBucket) when calling ListObjects",
    "An error occurred (AccessDenied) when calling GetObject",
    "An error occurred (InvalidAccessKeyId) when calling GetObject",
    "HTTP 403 Forbidden",
    "Permission denied: /data/x.parquet",
    "The request signature we calculated does not match the signature you provided",
    "Unauthorized",
    # An unrecognized failure is a bug, not a blip: never retried.
    "ValueError: bad schema",
    "KeyError: column missing",
]


@pytest.mark.parametrize("message", TRANSIENT)
def test_a_retryable_store_failure_is_retried(message):
    assert is_transient(Exception(message)) is True


@pytest.mark.parametrize("message", PERMANENT)
def test_a_fact_that_will_not_change_is_not_retried(message):
    assert is_transient(Exception(message)) is False


def test_a_socket_level_failure_is_transient_whatever_it_says():
    assert is_transient(TimeoutError("")) is True
    assert is_transient(ConnectionError("")) is True


def test_a_permanent_marker_vetoes_a_transient_one():
    # "Not Found" alongside "internal error" must stay permanent: retrying a 404 only
    # spends the budget before failing anyway.
    assert is_transient(Exception("404 Not Found (internal error page)")) is False
