"""`total_retained_bytes` charges each Arrow buffer once across a whole sequence.

A slice pins its parent, so one slice must be charged the parent. But N sibling slices of
one parent pin that parent *once*: charging every slice the whole parent made a 1,000-way
split of an 8 MB table measure ~8 GB, which declined broadcasts the planner admitted and
shrank UDF batches to the morsel floor (BT-025).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.plan.types import retained_bytes, total_retained_bytes

pytestmark = pytest.mark.unit

_ROWS = 1_000_000


def _parent() -> pa.Table:
    return pa.table({"v": pa.array(range(_ROWS), type=pa.int64())})


def test_many_slices_of_one_parent_are_charged_the_parent_once() -> None:
    table = _parent()
    slices = table.to_batches(max_chunksize=1_000)
    assert len(slices) == 1_000
    parent = retained_bytes(table)
    assert total_retained_bytes(slices) == pytest.approx(parent, rel=0.01)


def test_one_small_slice_still_charges_the_whole_parent() -> None:
    table = _parent()
    window = table.slice(0, 10)
    assert window.nbytes == 80
    assert total_retained_bytes([window]) >= _ROWS * 8


def test_distinct_parents_still_add() -> None:
    a, b = _parent(), _parent()
    assert total_retained_bytes([a, b]) == retained_bytes(a) + retained_bytes(b)


def test_dictionary_and_nested_buffers_are_counted() -> None:
    words = pa.array(["alpha", "beta", "alpha"] * 1_000).dictionary_encode()
    nested = pa.array([[1, 2], [3], []] * 1_000)
    batch = pa.record_batch({"w": words, "n": nested})
    assert total_retained_bytes([batch]) >= retained_bytes(batch)
    # Two halves of one batch share every buffer: still one set of buffers.
    halves = [batch.slice(0, 1_500), batch.slice(1_500)]
    whole = total_retained_bytes([batch])
    assert total_retained_bytes(halves) == pytest.approx(whole, rel=0.01)


def test_view_layout_and_non_arrow_items_do_not_raise() -> None:
    views = pa.array(["x" * 40, "y"], type=pa.string_view())
    chunked = pa.chunked_array([views])
    assert total_retained_bytes([chunked, views]) >= views.get_total_buffer_size()
    assert total_retained_bytes([object()]) == 0
