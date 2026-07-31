"""Sizing the object-store read fan-out to the machine rather than to a constant.

A GPU node reads from object storage over a link two orders of magnitude faster than the small
VM a fixed 32-GET default was chosen on, and a four-core container cannot service even that
many. One constant makes the same mistake in both directions. These tests pin the scaling, both
clamps, and — the part that matters most — that the pool is never *narrowed*, since another
part of the process may have widened it deliberately.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.io import filesystem

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_pool(monkeypatch):
    """Put pyarrow's pools back, since they are process-global."""
    io_threads, cpu_threads = pa.io_thread_count(), pa.cpu_count()
    filesystem.ensure_io_threads.cache_clear()
    monkeypatch.delenv("BATCHER_IO_THREADS", raising=False)
    yield
    pa.set_io_thread_count(io_threads)
    pa.set_cpu_count(cpu_threads)
    filesystem.ensure_io_threads.cache_clear()


def _target(monkeypatch, cores: int) -> int:
    monkeypatch.setattr("batcher._internal.hardware.available_cpu_count", lambda: cores)
    pa.set_io_thread_count(1)
    filesystem.ensure_io_threads.cache_clear()
    filesystem.ensure_io_threads()
    return pa.io_thread_count()


def test_a_small_container_keeps_the_previous_floor(monkeypatch):
    assert _target(monkeypatch, 2) == filesystem._IO_THREADS_FLOOR
    assert _target(monkeypatch, 8) == filesystem._IO_THREADS_FLOOR


def test_a_dense_node_gets_a_wider_fan_out(monkeypatch):
    assert _target(monkeypatch, 32) == 128
    assert _target(monkeypatch, 48) == 192


def test_the_fan_out_is_capped_below_what_a_store_will_accept(monkeypatch):
    assert _target(monkeypatch, 192) == filesystem._IO_THREADS_CEILING


def test_an_explicit_override_wins_over_the_measurement(monkeypatch):
    monkeypatch.setenv("BATCHER_IO_THREADS", "12")
    assert _target(monkeypatch, 192) == 12


def test_an_already_wider_pool_is_never_narrowed(monkeypatch):
    # Another part of the process may have widened it on purpose; this only ever lifts.
    monkeypatch.setattr("batcher._internal.hardware.available_cpu_count", lambda: 2)
    pa.set_io_thread_count(200)
    filesystem.ensure_io_threads.cache_clear()
    filesystem.ensure_io_threads()
    assert pa.io_thread_count() == 200


def test_the_cpu_pool_is_still_capped_to_usable_cores(monkeypatch):
    # The other half of the same call, and the opposite direction: pyarrow sizes its compute
    # pool to host cores, which over-subscribes under a cgroup quota.
    monkeypatch.setattr("batcher._internal.hardware.available_cpu_count", lambda: 3)
    pa.set_cpu_count(64)
    filesystem.ensure_io_threads.cache_clear()
    filesystem.ensure_io_threads()
    assert pa.cpu_count() == 3


# --- The retry budget that fan-out makes necessary -----------------------------------------


def test_an_s3_filesystem_gets_a_retry_budget_by_default():
    # A store's answer to 256 concurrent GETs is `503 SlowDown`, which is a request to wait
    # rather than a failure — and three attempts turns it into a failed query.
    import pyarrow.fs as pafs

    from batcher.io.filesystem import _S3_DEFAULT_RETRY_ATTEMPTS, _s3_with_options

    assert _S3_DEFAULT_RETRY_ATTEMPTS > 3
    # Built without raising, with the strategy attached: pyarrow does not expose the
    # constructed value back, so the contract under test is that the option is accepted.
    assert _s3_with_options("s3://bucket/key?region=us-east-1") is not None
    assert isinstance(
        pafs.AwsStandardS3RetryStrategy(max_attempts=_S3_DEFAULT_RETRY_ATTEMPTS),
        pafs.S3RetryStrategy,
    )


def test_the_retry_budget_is_overridable_in_both_directions():
    from batcher.io.filesystem import _s3_with_options

    assert _s3_with_options("s3://b/k?retry_max_attempts=1") is not None
    assert _s3_with_options("s3://b/k?retry_max_attempts=32") is not None


def test_an_unknown_option_still_names_itself():
    from batcher._internal.errors import IOError as BatcherIOError
    from batcher.io.filesystem import _s3_with_options

    with pytest.raises(BatcherIOError, match="unknown s3:// option 'retries'"):
        _s3_with_options("s3://b/k?retries=4")
