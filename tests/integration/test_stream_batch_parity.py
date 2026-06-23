"""Batch/stream API parity: the same code runs over bounded and unbounded sources.

`iter_batches()` chooses the execution mode automatically (no `streaming=` flag);
a bounded source can additionally `collect()`, while an unbounded one fails fast on
any materializing terminal instead of hanging.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

_SCHEMA = pa.schema([("x", pa.int64())])


def _batches():
    for i in range(3):
        yield pa.RecordBatch.from_pydict({"x": [i * 2, i * 2 + 1]}, schema=_SCHEMA)


@pytest.mark.integration
def test_is_streaming_flag():
    assert bt.from_pydict({"x": [1, 2, 3]}).is_streaming is False
    assert bt.from_batches(_batches, _SCHEMA, bounded=False).is_streaming is True
    # A finite factory defaults to bounded.
    assert bt.from_batches(_batches, _SCHEMA).is_streaming is False


@pytest.mark.integration
def test_same_pipeline_bounded_and_unbounded():
    # Identical transform code over a bounded and an "unbounded" source.
    def pipe(ds):
        return ds.filter(bt.col("x") > 1)

    bounded = pipe(bt.from_pydict({"x": [0, 1, 2, 3, 4, 5]}))
    unbounded = pipe(bt.from_batches(_batches, _SCHEMA, bounded=False))

    assert sorted(v for b in bounded.iter_batches() for v in b.column("x").to_pylist()) == [
        2,
        3,
        4,
        5,
    ]
    assert sorted(v for b in unbounded.iter_batches() for v in b.column("x").to_pylist()) == [
        2,
        3,
        4,
        5,
    ]


@pytest.mark.integration
def test_collect_on_unbounded_raises():
    ds = bt.from_batches(_batches, _SCHEMA, bounded=False).filter(bt.col("x") > 1)
    with pytest.raises(PlanError, match="unbounded"):
        ds.collect()
    with pytest.raises(PlanError, match="unbounded"):
        ds.count()
    with pytest.raises(PlanError, match="unbounded"):
        ds.to_pydict()


@pytest.mark.integration
def test_unbounded_through_breaker_raises_not_hangs():
    # A sort over an unbounded source cannot stream → fail fast, do not hang.
    ds = bt.from_batches(_batches, _SCHEMA, bounded=False).sort("x")
    with pytest.raises(PlanError, match="unbounded"):
        list(ds.iter_batches())
