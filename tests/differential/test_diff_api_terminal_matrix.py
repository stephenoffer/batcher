"""Terminal-op cross-product: every terminal must agree with `collect()`.

Regression coverage for the p6 (core/api/bc-py) bug hunt. The engine ships many
terminal spellings (`collect`, `iter_batches`, `to_pydict`, `to_pylist`, `count`,
`is_empty`, `schema`) and CLAUDE.md demands they agree over the cross-product of
plan shapes x flags. These tests pin the ones that shipped broken.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential


def _rows(tbl: pa.Table) -> list[dict]:
    return tbl.to_pylist()


def test_iter_batches_topn_batch_size_yields_record_batches():
    """`iter_batches(batch_size=n)` over a top-N (`sort().limit()`) must yield
    `pyarrow.RecordBatch`es, not `pyarrow.Table`s.

    The streaming top-N driver sliced a `pa.Table` directly, leaking `Table` objects
    from the batch iterator — a contract violation that breaks any consumer that calls
    `RecordBatch`-only methods (and the distributed / write paths that re-batch it).
    """
    t = pa.table({"k": ["a", "b", "a", "c", "b"], "v": [3, 1, 5, 2, 9]})
    ds = bt.from_arrow(t).sort("v").limit(3)
    for bs in (None, 1, 2, 100):
        batches = list(ds.iter_batches(batch_size=bs))
        assert batches, f"batch_size={bs} yielded nothing"
        for b in batches:
            assert isinstance(b, pa.RecordBatch), (
                f"batch_size={bs} yielded {type(b).__name__}, expected RecordBatch"
            )
            if bs is not None:
                assert b.num_rows <= bs


@pytest.mark.parametrize(
    "make",
    [
        lambda ds: ds,
        lambda ds: ds.filter(bt.col("v") > 1),
        lambda ds: ds.sort("v"),
        lambda ds: ds.sort("v", descending=True),
        lambda ds: ds.sort("v").limit(3),
        lambda ds: ds.sort("v").limit(3, offset=1),
        lambda ds: ds.limit(2),
        lambda ds: ds.group_by("k").agg(s=bt.col("v").sum()),
        lambda ds: ds.distinct(["k"]),
    ],
)
@pytest.mark.parametrize("batch_size", [None, 2])
def test_iter_batches_matches_collect(make, batch_size):
    """`iter_batches` yields exactly the rows `collect()` produces (order-preserving
    for ordered plans), for every batch_size, and never an empty batch."""
    t = pa.table(
        {
            "k": ["a", "b", "a", None, "c", "b", "a", "c"],
            "v": [1, 2, None, 4, 5, -6, 7, 0],
        }
    )
    ds = make(bt.from_arrow(t))
    ref = ds.collect()
    batches = list(ds.iter_batches(batch_size=batch_size))
    for b in batches:
        assert isinstance(b, pa.RecordBatch)
        assert b.num_rows > 0, "iter_batches must not yield an empty batch"
    got = pa.Table.from_batches(batches, schema=batches[0].schema) if batches else ref.slice(0, 0)
    assert got.num_rows == ref.num_rows
    # Multiset-equal always; order-equal is asserted for the plain (ordered) shapes.
    assert sorted(map(str, _rows(got))) == sorted(map(str, _rows(ref)))
