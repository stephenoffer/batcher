"""`iter_batches(batch_size=N)` must yield exactly-N-row batches (contract test).

`batch_size` is documented as "rebatch the output to this many rows". The per-path
chunkers used to slice each engine batch/chunk independently, so an unevenly-batched
result (a sorted output, a filtered scan spanning morsels) leaked a short batch at
*every* boundary — 1000, 1000, 651, 1000, … — instead of at the end only. The engine's
own `map_batches` rebatch coalesces correctly, so this was a divergence in the driver
path; these tests pin the exact-size contract across the streaming and materializing
paths, and that the rebatched rows still equal a plain `collect`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.integration


def _big_dataset(n: int = 40_000) -> bt.Dataset:
    return bt.from_pydict(
        {"g": [i % 20 for i in range(n)], "v": [(i * 7) % 1000 for i in range(n)]}
    )


def _sizes(plan: bt.Dataset, batch_size: int) -> list[int]:
    batches = list(plan.iter_batches(batch_size))
    assert all(isinstance(b, pa.RecordBatch) for b in batches)
    return [b.num_rows for b in batches]


def _multiset(table: pa.Table) -> list[tuple]:
    data = table.to_pydict()
    cols = sorted(data)
    return sorted(tuple(data[c][i] for c in cols) for i in range(table.num_rows))


@pytest.mark.parametrize(
    "make_plan",
    [
        pytest.param(lambda ds: ds.filter(bt.col("v") > 500), id="streaming-filter"),
        pytest.param(lambda ds: ds.sort("v").select("v"), id="materialized-sort"),
        pytest.param(lambda ds: ds, id="plain-scan"),
        pytest.param(lambda ds: ds.top_k(2503, "v"), id="top-n"),
    ],
)
def test_batches_are_exactly_batch_size_except_last(make_plan) -> None:
    plan = make_plan(_big_dataset())
    batch_size = 1000
    sizes = _sizes(plan, batch_size)
    assert sizes, "expected at least one batch for a non-empty result"
    # Every batch but the last holds exactly `batch_size` rows; the last is the remainder.
    assert all(s == batch_size for s in sizes[:-1]), sizes
    assert 0 < sizes[-1] <= batch_size, sizes


@pytest.mark.parametrize(
    "make_plan",
    [
        pytest.param(lambda ds: ds.filter(bt.col("v") > 500), id="streaming-filter"),
        pytest.param(lambda ds: ds.sort("v").select("v"), id="materialized-sort"),
        pytest.param(lambda ds: ds.group_by("g").agg(s=bt.col("v").sum()), id="aggregate"),
    ],
)
def test_rebatched_rows_equal_collect(make_plan) -> None:
    plan = make_plan(_big_dataset())
    batches = list(plan.iter_batches(700))
    rebatched = pa.Table.from_batches(batches, schema=batches[0].schema)
    assert _multiset(rebatched) == _multiset(plan.to_arrow())


def test_empty_result_yields_no_batches() -> None:
    ds = _big_dataset(100)
    assert list(ds.filter(bt.col("v") > 10_000).iter_batches(4)) == []


def test_non_positive_batch_size_raises() -> None:
    ds = _big_dataset(100)
    with pytest.raises(PlanError):
        list(ds.iter_batches(0))
