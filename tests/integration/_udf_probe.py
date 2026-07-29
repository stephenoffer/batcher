"""UDFs for `test_udf_isolation_e2e`, in an importable module rather than a closure.

A `forkserver` child re-imports the callable by qualified name, so a UDF defined inside a
test function cannot be run on the process path at all — the pool fails with
`BrokenProcessPool` and the test silently measures the *thread* fallback instead. Which is
to say: putting these here is not tidiness, it is the difference between the test
exercising the path it claims to and quietly exercising the one that has no isolation.
"""

from __future__ import annotations

import os

import pyarrow as pa

__all__ = ["report_environment"]


def report_environment(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Report what credentials this UDF can see, and which process it is in."""
    rows = batch.num_rows
    return pa.record_batch(
        {
            "secret": pa.array([os.environ.get("AWS_SECRET_ACCESS_KEY") or "<gone>"] * rows),
            "helper": pa.array([os.environ.get("BATCHER_SECRET_COMMAND") or "<gone>"] * rows),
            "pid": pa.array([os.getpid()] * rows, type=pa.int64()),
        }
    )
