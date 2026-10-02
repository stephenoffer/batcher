"""The aligned route declines an aggregate cut whose partials pass 32-bit string offsets.

The driver merges every unit's partial groups as one relation, and TPC-H q16 at SF1000
returned 2.4 GB of `Utf8` partials (distinct (group, supplier) pairs carrying `p_type`), which
cannot be one 32-bit-offset array. `_past_string_offsets` is the check that sends such a cut
to the shuffle route; the limit is lowered here so the test needs kilobytes, not gigabytes.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.dist.executors.aligned import run

pytestmark = pytest.mark.unit


def _batches(n: int, width: int) -> list[pa.RecordBatch]:
    return [
        pa.record_batch({"k": pa.array(range(n)), "s": pa.array(["x" * width] * n)})
        for _ in range(3)
    ]


def test_strings_past_the_limit_across_batches_decline(monkeypatch):
    monkeypatch.setattr(run, "_STRING_OFFSET_LIMIT", 1_000)
    # Each batch's strings fit; together they do not, which is the case the merge meets.
    batches = _batches(10, 50)
    assert sum(b.column(1).nbytes for b in batches[:1]) < 1_000
    assert run._past_string_offsets(batches) is True


def test_strings_inside_the_limit_and_non_string_columns_do_not(monkeypatch):
    monkeypatch.setattr(run, "_STRING_OFFSET_LIMIT", 1_000)
    assert run._past_string_offsets(_batches(2, 10)) is False
    wide_ints = [pa.record_batch({"k": pa.array(range(10_000))})]
    assert run._past_string_offsets(wide_ints) is False
    assert run._past_string_offsets([]) is False
