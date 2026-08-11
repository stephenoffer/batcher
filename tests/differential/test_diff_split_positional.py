"""Differential tests for the positional split family against DuckDB.

`split_at_indices` and `split_proportionately` cut a dataset into consecutive row ranges,
which is Ray Data's spelling for the operation and the one capability the competitor sweep
found Batcher missing. Neither adds an operator: both lower to `with_row_index` plus a
range filter, so what needs pinning is not a kernel but the *contract* — that the parts are
disjoint, that together they are the input, and that each one holds the rows it claims.

DuckDB is a usable oracle for that despite having no such method, because a positional
range over an ordered relation is exactly ``ORDER BY k LIMIT (hi - lo) OFFSET lo``. Every
case here sorts first so the position of a row is defined rather than incidental, and then
holds each part against that query.

`assert_same_ordered` throughout, deliberately. A split's contract is positional, so the
order-independent `assert_same` would pass while a part returned the right *rows* in the
wrong ranges — which is the defect class a positional operation actually has.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered, assert_tables_equal

pytestmark = pytest.mark.differential

#: Wide enough to span several morsels, so a part boundary can fall inside one rather than
#: only ever landing on the 16,384-row edge where a bug would be invisible.
_N = 50_000


def _table(n: int = _N) -> pa.Table:
    """Sorted keys with a payload, so a row's position is well defined and carried."""
    return pa.table(
        {
            "k": pa.array(list(range(n)), pa.int64()),
            "p": pa.array([f"row-{i}" for i in range(n)], pa.string()),
        }
    )


def _duck_range(duck, lo: int, hi: int | None, n: int):
    """The rows DuckDB puts at positions ``[lo, hi)`` of the sorted relation."""
    stop = n if hi is None else min(hi, n)
    count = max(stop - lo, 0)
    return duck.sql(f"SELECT k, p FROM t ORDER BY k LIMIT {count} OFFSET {lo}")


@pytest.mark.parametrize(
    "indices",
    [
        [1],
        [_N // 2],
        [2, 5],
        [100, 20_000, 33_333],
        [16_384],  # exactly a morsel boundary
        [16_383, 16_385],  # straddling one
    ],
)
def test_each_part_holds_its_own_row_range(duck, indices):
    """Every part matches the DuckDB rows at the positions it claims."""
    t = _table()
    duck.register("t", t)
    parts = bt.from_arrow(t).sort("k").split_at_indices(indices)
    assert len(parts) == len(indices) + 1
    for lo, hi, part in zip([0, *indices], [*indices, None], parts, strict=True):
        assert_same_ordered(part.collect(), _duck_range(duck, lo, hi, _N))


def test_the_parts_reassemble_into_the_input():
    """Disjoint and covering: concatenating the parts in order gives the original rows back."""
    t = _table()
    ds = bt.from_arrow(t).sort("k")
    parts = ds.split_at_indices([7, 1_000, 40_000])
    rejoined = [v for part in parts for v in part.collect().column("k").to_pylist()]
    assert rejoined == list(range(_N))


@pytest.mark.parametrize(
    ("indices", "sizes"),
    [
        ([0], [0, _N]),  # a leading zero gives an empty first part
        ([_N], [_N, 0]),  # a cut at the end gives an empty last part
        ([_N + 1_000], [_N, 0]),  # past the end is empty, not an error
        ([5, 5], [5, 0, _N - 5]),  # a repeated index gives an empty middle part
    ],
)
def test_degenerate_cuts_give_empty_parts_rather_than_errors(indices, sizes):
    """Empty parts are a legitimate answer — this matches numpy.split and Ray Data."""
    parts = bt.from_arrow(_table()).sort("k").split_at_indices(indices)
    assert [p.count() for p in parts] == sizes


def test_proportions_match_the_indices_they_imply(duck):
    """`split_proportionately` is `split_at_indices` with the cuts computed from the count."""
    t = _table(1_000)
    duck.register("t", t)
    parts = bt.from_arrow(t).sort("k").split_proportionately([0.2, 0.5])
    assert [p.count() for p in parts] == [200, 500, 300]
    for lo, hi, part in zip([0, 200, 700], [200, 700, None], parts, strict=True):
        assert_same_ordered(part.collect(), _duck_range(duck, lo, hi, 1_000))


def test_proportions_agree_with_ray_datas_own_example():
    """The example in Ray Data's docstring, so a ported script keeps its answer."""
    parts = bt.range(0, 10).split_proportionately([0.2, 0.5])
    assert [p.to_pydict()["value"] for p in parts] == [
        [0, 1],
        [2, 3, 4, 5, 6],
        [7, 8, 9],
    ]


def test_indices_example_agrees_with_ray_datas_own():
    """Likewise for `split_at_indices`, whose example is the one migrants copy."""
    parts = bt.range(0, 10).split_at_indices([2, 5])
    assert [p.to_pydict()["value"] for p in parts] == [
        [0, 1],
        [2, 3, 4],
        [5, 6, 7, 8, 9],
    ]


def test_an_empty_input_splits_into_empty_parts(duck):
    """No rows to place, so every part is empty and nothing raises."""
    t = _table(0)
    duck.register("t", t)
    parts = bt.from_arrow(t).sort("k").split_at_indices([3, 9])
    assert [p.count() for p in parts] == [0, 0, 0]
    for part in parts:
        assert_same_ordered(part.collect(), duck.sql("SELECT k, p FROM t WHERE false"))


def test_nulls_in_the_payload_ride_along(duck):
    """A split is positional and never inspects a value, so nulls must survive untouched."""
    t = pa.table(
        {
            "k": pa.array(list(range(10)), pa.int64()),
            "p": pa.array([None if i % 3 == 0 else f"v{i}" for i in range(10)], pa.string()),
        }
    )
    duck.register("t", t)
    parts = bt.from_arrow(t).sort("k").split_at_indices([4])
    assert_same_ordered(parts[0].collect(), _duck_range(duck, 0, 4, 10))
    assert_same_ordered(parts[1].collect(), _duck_range(duck, 4, None, 10))


def test_every_terminal_agrees_on_a_part():
    """`collect`, `iter_batches` and `collect(spill=True)` return one part identically.

    A split is a filter over a row index, so a path that numbered rows differently would
    return the right *count* from the wrong range — a count check cannot see that.
    """
    ds = bt.from_arrow(_table()).sort("k")
    part = ds.split_at_indices([1_000, 20_000])[1]
    collected = part.collect()
    streamed = pa.Table.from_batches(list(part.iter_batches()), schema=collected.schema)
    assert_tables_equal(streamed, collected, ordered=True)
    assert_tables_equal(part.collect(spill=True), collected, ordered=True)


def test_a_part_stays_lazy_until_it_is_collected():
    """Nothing materializes: the parts are plans, which is what Ray Data's version is not."""
    parts = bt.from_arrow(_table()).sort("k").split_at_indices([10])
    assert all(isinstance(p, bt.Dataset) for p in parts)
    # Only the part that is asked for does any work.
    assert parts[0].count() == 10


@pytest.mark.parametrize("existing", ["__bc_split_idx", "__bc_split_idx_"])
def test_a_column_named_like_the_internal_index_is_not_disturbed(existing):
    """The row index is escaped, so a user column of that name survives with its own values.

    `tail` and `gather_every` raise on this collision. It is escaped here because this method
    returns several datasets, so the failure would surface far from the call that caused it.
    """
    t = pa.table({existing: pa.array([9, 8, 7], pa.int64()), "x": pa.array([1, 2, 3], pa.int64())})
    parts = bt.from_arrow(t).split_at_indices([1])
    assert [p.columns for p in parts] == [[existing, "x"], [existing, "x"]]
    assert parts[0].collect().column(existing).to_pylist() == [9]
    assert parts[1].collect().column(existing).to_pylist() == [8, 7]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda ds: ds.split_at_indices([]), "must not be empty"),
        (lambda ds: ds.split_at_indices([-1]), "must be >= 0"),
        (lambda ds: ds.split_at_indices([5, 2]), "non-decreasing"),
        (lambda ds: ds.split_proportionately([]), "must not be empty"),
        (lambda ds: ds.split_proportionately([0.0]), "must be > 0"),
        (lambda ds: ds.split_proportionately([-0.5]), "must be > 0"),
        (lambda ds: ds.split_proportionately([0.6, 0.5]), "sum to less than 1"),
        (lambda ds: ds.split_proportionately([1.0]), "sum to less than 1"),
    ],
)
def test_invalid_arguments_raise_before_any_work(call, message):
    """Validation is at the API edge, so a bad split fails at build time, not mid-scan."""
    with pytest.raises(bt.PlanError, match=message):
        call(bt.from_arrow(_table(10)))


def test_too_many_parts_for_the_rows_is_refused():
    """Ray Data guarantees non-empty parts here, so an impossible request must raise."""
    with pytest.raises(bt.PlanError, match="non-empty parts"):
        bt.from_arrow(_table(2)).split_proportionately([0.3, 0.3, 0.3])
