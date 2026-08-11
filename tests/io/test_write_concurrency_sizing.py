"""How wide a partitioned write publishes, sized for where the files are going.

A PUT to an object store is tens of milliseconds of latency, so throughput tracks
requests in flight rather than cores available to encode them -- the same asymmetry the
read path already sizes for. Bounded by the core count, a small worker published four
files of a 200-partition write at a time and spent the rest of its life on sockets.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware import available_cpu_count
from batcher.io.base.sink import _REMOTE_WRITE_CONCURRENCY, _write_concurrency

pytestmark = pytest.mark.unit


def test_a_local_write_is_sized_by_cores():
    assert _write_concurrency(1000, "/data/out") == available_cpu_count()
    assert _write_concurrency(1000, "file:///data/out") == available_cpu_count()


def test_an_object_store_write_is_sized_by_requests_in_flight():
    wide = max(available_cpu_count(), _REMOTE_WRITE_CONCURRENCY)
    for uri in ("s3://bucket/out", "gs://bucket/out", "abfs://c@a/out"):
        assert _write_concurrency(1000, uri) == wide
    assert wide >= _REMOTE_WRITE_CONCURRENCY


def test_it_never_exceeds_the_number_of_files():
    assert _write_concurrency(3, "s3://bucket/out") == 3
    assert _write_concurrency(1, "/data/out") == 1


def test_it_never_returns_zero():
    assert _write_concurrency(0, "s3://bucket/out") == 1
