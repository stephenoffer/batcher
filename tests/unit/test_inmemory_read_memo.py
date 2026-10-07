"""`InMemorySource.read` memoizes its projected batches per projection.

The memo is only an optimization, so what is pinned is that it cannot change an answer: a
repeated projection returns the same batches (the same batch objects, not re-projected
copies), a different projection is projected afresh, a caller mutating the returned list
does not reach the memo, and a narrow column is still widened on the memoized path.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.io.source import InMemorySource


def _source() -> InMemorySource:
    batches = [
        pa.record_batch({"a": pa.array([1, 2], pa.int32()), "b": ["x", "y"]}),
        pa.record_batch({"a": pa.array([3], pa.int32()), "b": ["z"]}),
    ]
    return InMemorySource(batches)


def test_a_repeated_projection_returns_the_same_batches():
    src = _source()
    first = src.read(["a"])
    second = src.read(["a"])
    assert [b.num_rows for b in first] == [2, 1]
    assert all(x is y for x, y in zip(first, second, strict=True)), "re-projected, not memoized"
    assert first[0].schema.field("a").type == pa.int64(), "the narrow column is widened"


def test_projections_are_kept_apart_and_the_list_is_the_callers():
    src = _source()
    narrow = src.read(["b"])
    narrow.clear()
    assert [b.schema.names for b in src.read(["b"])] == [["b"], ["b"]]
    assert [b.schema.names for b in src.read()] == [["a", "b"], ["a", "b"]]
    assert [b.schema.names for b in src.read(["b", "a"])] == [["b", "a"], ["b", "a"]]
