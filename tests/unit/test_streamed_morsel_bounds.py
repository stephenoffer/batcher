"""A streamed stage's morsel has a floor AND a ceiling; it used to have only a floor.

`_target_rows` is the batch the consumer stage declared, and both the producer and the relay
regrouped their output to *at least* that many rows. For narrow rows that is harmless. For the
rows this pipeline exists to carry it is fatal: a scan morsel is sized in **rows** (16,384 by
default), so one input chunk of a decode stage produces 16,384 decoded images -- about 2.4 GB --
and the relay published it as a single morsel while the producer held it whole.

Measured on 400,000 384x384 JPEGs over 8 T4s and 8 CPU nodes, with every other change of that
session reverted: `OutOfMemoryError: 3 worker(s) were killed due to the node running low on
memory ... 28.53GB / 30.00GB`. With the three bounds below the same run completes.

The bounds are unit-testable without a cluster because each is a pure regrouping: what is under
test is that no morsel exceeds the declared batch, and that regrouping still preserves every
row in order.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from batcher.dist.streaming.producers import coalesce

pytestmark = pytest.mark.unit


def _rb(n: int, start: int = 0) -> pa.RecordBatch:
    return pa.record_batch({"v": pa.array(np.arange(start, start + n, dtype="int64"))})


def _values(batches) -> list[int]:
    return [v for b in batches for v in b.column("v").to_pylist()]


@pytest.mark.parametrize(
    "sizes",
    [
        [1000],  # one oversized batch — the shape that OOMed
        [10, 10, 10],  # several short ones, merged
        [5],  # a single short batch
        [300, 7, 900],  # mixed, needing both merge and split
        [256, 256],  # already exactly on target
    ],
)
def test_coalesce_never_exceeds_the_target(sizes):
    target = 256
    start, batches = 0, []
    for n in sizes:
        batches.append(_rb(n, start))
        start += n
    out = coalesce(batches, target)
    # Every morsel but the last is exactly the declared batch; the last may be short because
    # there are no more rows, which is correct rather than a violation.
    assert all(b.num_rows == target for b in out[:-1]), [b.num_rows for b in out]
    assert all(b.num_rows <= target for b in out), [b.num_rows for b in out]
    # And regrouping is still lossless and order-preserving.
    assert _values(out) == _values(batches)


def test_coalesce_leaves_the_batches_alone_when_disabled():
    # `0` means the consumer declared no batch size, and then the engine's own morsel
    # granularity is the right answer — including for an oversized batch.
    batches = [_rb(1000)]
    out = coalesce(batches, 0)
    assert [b.num_rows for b in out] == [1000]
    assert _values(out) == _values(batches)


def test_coalesce_handles_an_empty_input():
    assert coalesce([], 256) == []


def test_the_producer_caps_both_the_input_it_pulls_and_the_morsel_it_publishes():
    """The producer's two halves of the same bound, exercised without Ray.

    `_next_input` is what keeps resident memory down -- capping the *published* morsel alone
    does not, because the slices are zero-copy views on the same oversized parent, which stays
    resident until the last of them drains.
    """
    from batcher.dist.streaming.producers import ProducerActor

    if ProducerActor is None:  # pragma: no cover - ray optional
        pytest.skip("ray is not installed")
    # Drive the two methods directly on a bare instance: what is under test is the regrouping,
    # not the Flight session an `__init__` would build.
    actor = ProducerActor.__ray_metadata__.modified_class.__new__(
        ProducerActor.__ray_metadata__.modified_class
    )
    from collections import deque

    actor._target_rows = 256
    actor._pending = deque([_rb(1000)])
    actor._inp_rest = None
    actor._it = iter([])

    first = actor._take_batch()
    assert first.num_rows == 256, "an oversized pending batch must be split, not published whole"
    assert actor._pending and actor._pending[0].num_rows == 744, "the remainder carries forward"

    # `_next_input` caps what is fed to the sub-plan in the first place.
    actor._pending = deque()
    actor._it = iter([_rb(1000)])
    actor._inp_rest = None
    chunk = actor._next_input()
    assert chunk.num_rows == 256, "the input chunk must be capped, not just the output morsel"
    assert actor._inp_rest is not None and actor._inp_rest.num_rows == 744
    # Draining it yields every row, in order, and nothing larger than the cap.
    seen = [chunk]
    while (nxt := actor._next_input()) is not None:
        assert nxt.num_rows <= 256
        seen.append(nxt)
    assert _values(seen) == list(range(1000))
