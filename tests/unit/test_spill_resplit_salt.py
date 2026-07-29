"""Re-splitting a spilled bucket must actually spread it.

Every grace path sizes its bucket count from an *average* — total bytes over the memory
envelope — so under key skew one bucket holds far more than its share. The answer is to
re-partition that bucket into sub-buckets and reduce them one at a time, and the aggregate's
out-of-core reduce has done exactly that for a long time.

It did nothing. Bucket assignment reads the **low hash bits** at a power-of-two bucket count,
and both the parent count (16 by default) and the sub-bucket count (8) are powers of two — so
every row in parent bucket `b` re-partitions to `b & 7`. One sub-bucket, always, at every
level of the recursion. The reduce wrote and re-read the whole over-large bucket three times,
changed nothing, and then combined it anyway.

Nothing could catch this. The result was always correct — a re-partition that moves no rows
is still a valid partition — and the only symptom was memory, on the skewed inputs the guard
exists for. These tests assert the property the recursion actually depends on: that a
re-split *separates* rows it did not separate before.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher._internal.native import engine


def _keys(n: int) -> pa.RecordBatch:
    return pa.record_batch({"k": pa.array(range(n), type=pa.int64())})


def _nonempty(buckets) -> int:
    return sum(1 for b in buckets if any(rb.num_rows for rb in b))


def _rows_of(batch: pa.RecordBatch, bucket_batches) -> pa.RecordBatch:
    """The rows of one bucket, as a single batch (or a 0-row batch)."""
    rows = [rb for rb in bucket_batches if rb.num_rows]
    if not rows:
        return batch.slice(0, 0)
    return pa.Table.from_batches(rows).combine_chunks().to_batches()[0]


def test_an_unsalted_resplit_of_a_power_of_two_bucket_moves_no_rows():
    """The defect, pinned so it cannot come back as "an optimization"."""
    nat = engine()
    batch = _keys(20_000)
    parent = nat.partition_batches([batch], [0], 16)
    bucket0 = _rows_of(batch, parent[0])
    assert bucket0.num_rows > 100, "need a populated parent bucket"

    unsalted = nat.partition_batches([bucket0], [0], 8)
    assert _nonempty(unsalted) == 1, (
        "an unsalted 16-way -> 8-way re-split unexpectedly spread; if bucket assignment "
        "changed, the salted recursion's justification needs re-checking"
    )


def test_a_salted_resplit_spreads_the_same_bucket():
    nat = engine()
    batch = _keys(20_000)
    parent = nat.partition_batches([batch], [0], 16)
    bucket0 = _rows_of(batch, parent[0])

    salted = nat.partition_batches_salted([bucket0], [0], 8, 0x9E3779B97F4A7C15 | 1)
    assert _nonempty(salted) == 8, (
        f"a salted re-split reached only {_nonempty(salted)} of 8 sub-buckets — the grace "
        f"recursion still cannot separate a bucket it already failed to fit"
    )
    # No row is lost or duplicated.
    assert sum(rb.num_rows for b in salted for rb in b) == bucket0.num_rows


def test_salt_zero_is_the_unsalted_assignment():
    """A salt of 0 must be the identity: the unsalted bucket of a key is the cluster-wide
    contract every reducer and both sides of a distributed join agree on. Salting is only
    ever a local decision about how to re-split."""
    nat = engine()
    batch = _keys(5_000)
    for n in (2, 8, 16, 100):
        plain = nat.partition_batches([batch], [0], n)
        salted = nat.partition_batches_salted([batch], [0], n, 0)
        assert [[rb.to_pydict() for rb in b] for b in plain] == [
            [rb.to_pydict() for rb in b] for b in salted
        ], f"salt 0 changed the bucket assignment at {n} partitions"


def test_a_salt_keeps_equal_keys_together():
    """The salt depends on the recursion depth, never on the row — which is what keeps each
    sub-bucket an exact partial reduce. Duplicated keys must land in one sub-bucket."""
    nat = engine()
    # Every key appears three times, in a scattered order.
    vals = [i % 40 for i in range(600)]
    batch = pa.record_batch({"k": pa.array(vals, type=pa.int64())})
    salted = nat.partition_batches_salted([batch], [0], 7, 12345)
    where: dict[int, int] = {}
    for i, bucket in enumerate(salted):
        for rb in bucket:
            for k in rb.column("k").to_pylist():
                assert where.setdefault(k, i) == i, f"key {k} was split across sub-buckets"


@pytest.mark.parametrize("bad_salt", [0, 1, 2**63])
def test_salted_partition_validates_like_the_unsalted_one(bad_salt):
    """The salted entry point must reject the same bad arguments, so it cannot become a way
    around the validation."""
    nat = engine()
    # `ValueError`, not a bare `Exception`: the engine reports a bad argument as
    # `PyValueError`, and a panic escaping the FFI would surface as something else.
    with pytest.raises(ValueError):
        nat.partition_batches_salted([_keys(4)], [99], 4, bad_salt)
    with pytest.raises(ValueError):
        nat.partition_batches_salted([_keys(4)], [0], 0, bad_salt)
