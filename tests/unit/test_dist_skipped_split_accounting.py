"""Under `on_read_error="skip"`, a job's data loss must be answerable per query.

`skipped_splits()` is a cumulative worker-process counter that is never reset. On a
persistent fleet worker serving many queries it answers "how much has this process ever
skipped", which cannot tell you whether *your* petabyte scan quietly dropped a corrupt
shard. `drain_skipped_splits()` is the per-query reading the driver can sum.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors import scan_read

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_counter():
    scan_read.drain_skipped_splits()
    yield
    scan_read.drain_skipped_splits()


class _Split:
    path = "s3://bucket/corrupt.parquet"


def test_a_skip_is_counted():
    scan_read._record_skipped(_Split(), OSError("bad magic bytes"))
    assert scan_read.skipped_splits() == 1


def test_draining_returns_the_count_and_resets_it():
    for _ in range(3):
        scan_read._record_skipped(_Split(), OSError("truncated footer"))
    assert scan_read.drain_skipped_splits() == 3
    assert scan_read.skipped_splits() == 0, "a persistent worker must start the next query at 0"


def test_a_second_query_on_the_same_worker_does_not_inherit_the_first_ones_losses():
    scan_read._record_skipped(_Split(), OSError("query 1 loss"))
    assert scan_read.drain_skipped_splits() == 1
    scan_read._record_skipped(_Split(), OSError("query 2 loss"))
    assert scan_read.drain_skipped_splits() == 1, "cumulative counting would report 2 here"


def test_draining_a_clean_worker_reports_no_loss():
    assert scan_read.drain_skipped_splits() == 0
