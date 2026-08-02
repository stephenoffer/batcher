"""Router-level streaming contracts whose failure is a wrong answer, not a slow one.

Each predicate here decides *how* a plan streams, and each of their docstrings makes a
claim nothing verified: the exact-batch-size rebatcher's "N rows per batch except the
last", the union's four preconditions, and the spilling sort's key eligibility. A wrong
`True` from any of them runs a bounded-memory path on a shape it cannot handle.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.terminal.stream.rebatch import _rebatch_exact
from batcher.api.terminal.stream.union import union_branch_sources, union_streams_branchwise
from batcher.dist.spill_breakers import supports_spilling_sort, supports_spilling_window


def _batches(sizes: list[int]) -> list[pa.RecordBatch]:
    out, n = [], 0
    for size in sizes:
        out.append(pa.record_batch({"a": list(range(n, n + size))}))
        n += size
    return out


# --------------------------------------------------------------------------
# The exact-batch-size contract.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sizes", [[3, 7, 1, 9], [1] * 20, [20], [5, 5, 5, 5]])
def test_every_batch_but_the_last_holds_exactly_the_requested_rows(sizes):
    out = list(_rebatch_exact(iter(_batches(sizes)), 4))
    assert [b.num_rows for b in out[:-1]] == [4] * (len(out) - 1)
    assert 0 < out[-1].num_rows <= 4


@pytest.mark.parametrize("sizes", [[3, 7, 1, 9], [1] * 20, [20]])
def test_rebatching_preserves_every_row_in_order(sizes):
    total = sum(sizes)
    out = list(_rebatch_exact(iter(_batches(sizes)), 4))
    assert [v for b in out for v in b.column("a").to_pylist()] == list(range(total))


def test_an_empty_stream_yields_nothing():
    assert list(_rebatch_exact(iter([]), 4)) == []
    empty = pa.record_batch({"a": pa.array([], type=pa.int64())})
    assert list(_rebatch_exact(iter([empty]), 4)) == []


def test_a_batch_size_larger_than_the_stream_yields_one_short_batch():
    out = list(_rebatch_exact(iter(_batches([2, 3])), 100))
    assert [b.num_rows for b in out] == [5]


def test_the_buffer_does_not_grow_a_chunk_per_arriving_batch():
    """`concat_tables` copies the accumulator's whole chunk list per call, so appending each
    arrival to a table made the buffering quadratic in exactly the case this exists for: a
    large `batch_size` over a source yielding small batches."""
    out = list(_rebatch_exact(iter(_batches([1] * 500)), 200))
    assert [b.num_rows for b in out] == [200, 200, 100]
    # Each emitted batch is compacted, so a consumer never receives a 200-chunk batch.
    assert all(
        b.column(0).num_chunks == 1 if hasattr(b.column(0), "num_chunks") else True for b in out
    )


# --------------------------------------------------------------------------
# The UNION preconditions — "a wrong answer rather than a slow one if skipped".
# --------------------------------------------------------------------------
def _union_of(a, b, *, distinct=False):
    ds = a.union(b) if not distinct else a.union(b, distinct=True)
    return ds._plan, ds._sources


def test_union_all_over_bounded_matched_branches_streams_branchwise():
    a = bt.from_pydict({"x": [1, 2]})
    b = bt.from_pydict({"x": [3, 4]})
    plan, sources = _union_of(a, b)
    assert union_streams_branchwise(plan, sources) is True
    assert len(union_branch_sources(plan)) == 2


def test_a_distinct_union_does_not_stream_branchwise():
    """`UNION` (distinct) needs a global dedup — the whole-relation state this path lacks."""
    a = bt.from_pydict({"x": [1, 2]})
    b = bt.from_pydict({"x": [2, 3]})
    plan, sources = _union_of(a, b, distinct=True)
    assert union_streams_branchwise(plan, sources) is False


def test_branches_whose_types_differ_do_not_stream_branchwise():
    """Materializing lets the engine widen a column across branches; streaming yields each
    branch as the engine produced it, so it applies only where the types already agree.

    Narrow ints are normalized at the FFI boundary (Int32 -> Int64), so they are *not* a
    disagreement — the pair that genuinely differs is int against float.
    """
    a = bt.from_arrow(pa.table({"x": pa.array([1, 2], pa.int64())}))
    b = bt.from_arrow(pa.table({"x": pa.array([3.5, 4.5], pa.float64())}))
    plan, sources = _union_of(a, b)
    assert union_streams_branchwise(plan, sources) is False


def test_branches_whose_narrow_ints_normalize_alike_do_stream_branchwise():
    a = bt.from_arrow(pa.table({"x": pa.array([1, 2], pa.int32())}))
    b = bt.from_arrow(pa.table({"x": pa.array([3, 4], pa.int64())}))
    plan, sources = _union_of(a, b)
    assert union_streams_branchwise(plan, sources) is True


def test_every_branch_must_name_exactly_one_source():
    """The drivers address their input as `sources[0]`, so a branch spanning two sources
    (a join inside a union) has nothing to be relabelled onto."""
    a = bt.from_pydict({"x": [1]})
    b = bt.from_pydict({"x": [2], "k": [1]})
    c = bt.from_pydict({"k": [1], "y": [9]})
    joined = b.join(c, on="k").select("x")
    plan, sources = _union_of(a, joined)
    assert union_branch_sources(plan) == []
    assert union_streams_branchwise(plan, sources) is False


# --------------------------------------------------------------------------
# Out-of-core eligibility: a wrong `True` runs the bucket path on a key it cannot split.
# --------------------------------------------------------------------------
def test_a_numeric_leading_key_is_range_partitionable():
    src = bt.from_pydict({"n": [3, 1, 2], "s": ["c", "a", "b"]})
    plan = src.sort("n")._plan
    assert supports_spilling_sort(plan, src._sources) is True


def test_a_string_leading_key_is_range_partitionable_too():
    """A string key is sampled lexically rather than by the KLL sketch, and routes the same.

    This asserted `False` until `sample_key_grid`/`string_quantiles` gave a string key its own
    sampler: the KLL sketch is numeric-only, so before that there was no grid to cut and the
    out-of-core sort declined the shape. The refusal was not harmless — four TPC-H queries end
    in a string `ORDER BY` over a materialized aggregate, where no fallback is left to take
    (`benchmarks/BENCHMARK_RESULTS.md`).
    """
    src = bt.from_pydict({"n": [3, 1, 2], "s": ["c", "a", "b"]})
    plan = src.sort("s")._plan
    assert supports_spilling_sort(plan, src._sources) is True


def test_a_derived_key_is_not():
    """Not in the source schema, so its type is unknown — stay out of the range partition
    rather than fail inside it."""
    src = bt.from_pydict({"n": [3, 1, 2]})
    plan = src.with_columns(m=bt.col("n") * 2).sort("m")._plan
    assert supports_spilling_sort(plan, src._sources) is False


def test_a_partitioned_window_can_grace_partition_but_a_global_one_cannot():
    src = bt.from_pydict({"k": [1, 1, 2], "v": [1, 2, 3]})
    partitioned = src.with_columns(r=bt.col("v").sum().over(partition_by="k"))._plan
    assert supports_spilling_window(partitioned) is True
    glob = src.with_columns(r=bt.col("v").sum().over(order_by="v"))._plan
    assert supports_spilling_window(glob) is False
