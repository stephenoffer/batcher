"""An integer-keyed ASOF join at a size that takes its parallel path, held to DuckDB.

`join_asof` with an integer or temporal `on` key and at most one integer `by` key runs a
dedicated kernel (`bc_runtime::join::asof`): each side's `(by, on, row)` tuples are put in key
order by a parallel counting sort and the two are merged. The small fixtures elsewhere fit in
one row range of that sort, so they never exercise what makes it parallel -- several ranges
reserving disjoint slices of each group, and the merge starting each chunk of probes partway
into the right side. These fixtures are tens of thousands of rows, past the sort's range
threshold, and are built to stress the places such a kernel goes wrong:

* a left side **not** in `on` order, so groups must be sorted after the counting pass;
* duplicate `(by, on)` pairs on the right, where the tie must go to the same row DuckDB picks;
* nulls in both `by` and `on` on both sides, which must match nothing;
* `by` groups present on one side only;
* a dense `by` range (counting sort) and a sparse one (comparison sort).

Every left row carries a unique `rid`. The matched right row is identified by its own unique
`rid`, so a wrong match cannot hide behind an equal payload.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

_LEFT_ROWS = 60_000
_RIGHT_ROWS = 45_000


def _sides(seed: int, by_scale: int, ordered_left: bool) -> tuple[pa.Table, pa.Table]:
    rng = random.Random(seed)

    def maybe(v, p=0.03):
        return None if rng.random() < p else v

    # 700 groups on the left, 650 on the right with an offset: some groups exist on one side.
    l_by = [maybe(rng.randrange(700) * by_scale) for _ in range(_LEFT_ROWS)]
    l_t = [maybe(rng.randrange(5_000)) for _ in range(_LEFT_ROWS)]
    if ordered_left:
        l_t = sorted(t for t in l_t if t is not None) + [None] * sum(t is None for t in l_t)
    # Right times come from a narrow range so (by, t) pairs repeat: ties are the point.
    r_by = [maybe((rng.randrange(650) + 30) * by_scale) for _ in range(_RIGHT_ROWS)]
    r_t = [maybe(rng.randrange(1_500) * 3) for _ in range(_RIGHT_ROWS)]
    left = pa.table(
        {
            "rid": pa.array(range(_LEFT_ROWS), pa.int64()),
            "g": pa.array(l_by, pa.int64()),
            "t": pa.array(l_t, pa.int64()),
        }
    )
    right = pa.table(
        {
            "g": pa.array(r_by, pa.int64()),
            "t": pa.array(r_t, pa.int64()),
            "rrid": pa.array(range(_RIGHT_ROWS), pa.int64()),
        }
    )
    return left, right


_SHAPES = {
    "dense_ordered": (11, 1, True),
    "dense_unordered": (12, 1, False),
    "sparse_unordered": (13, 1_000_003_007, False),
}


@pytest.fixture(scope="module", params=sorted(_SHAPES))
def sides(request) -> tuple[pa.Table, pa.Table]:
    return _sides(*_SHAPES[request.param])


def _duck_asof(duck, left, right, op: str):
    duck.register("l", left)
    duck.register("r", right)
    # DuckDB breaks a tie between equal right keys arbitrarily, so the oracle names the row
    # Batcher must pick -- the first of the tied rows in input order for a forward match, the
    # last for a backward one (the general kernel's stable order) -- by joining back on the
    # matched key and taking the extreme `rrid`.
    agg = "min" if op == "<=" else "max"
    return duck.sql(
        f"""
        WITH m AS (
            SELECT l.rid, l.g, l.t, r.t AS rt FROM l ASOF LEFT JOIN r
            ON l.g = r.g AND l.t {op} r.t
        )
        SELECT m.rid, m.g, m.t, (SELECT {agg}(r.rrid) FROM r WHERE r.g = m.g AND r.t = m.rt)
            AS rrid
        FROM m
        """
    )


@pytest.mark.parametrize("direction, op", [("backward", ">="), ("forward", "<=")])
def test_a_large_integer_asof_matches_duckdb(duck, sides, direction, op):
    left, right = sides
    out = (
        bt.from_arrow(left)
        .join_asof(bt.from_arrow(right), on="t", by="g", direction=direction)
        .select("rid", "g", "t", "rrid")
        .collect()
    )
    assert out.num_rows == _LEFT_ROWS
    assert_same(out, _duck_asof(duck, left, right, op))


def test_strict_matching_at_scale_matches_duckdb(duck, sides):
    """`allow_exact_matches=False` is DuckDB's strict `>`."""
    left, right = sides
    out = (
        bt.from_arrow(left)
        .join_asof(bt.from_arrow(right), on="t", by="g", allow_exact_matches=False)
        .select("rid", "g", "t", "rrid")
        .collect()
    )
    assert_same(out, _duck_asof(duck, left, right, ">"))


def test_some_rows_match_and_some_do_not(sides):
    """A control: the fixtures must exercise both outcomes, or the comparisons prove little."""
    left, right = sides
    out = bt.from_arrow(left).join_asof(bt.from_arrow(right), on="t", by="g").collect()
    matched = out.num_rows - out.column("rrid").null_count
    assert 0 < matched < out.num_rows


def _by_rid(table: pa.Table) -> pa.Table:
    return table.sort_by([("rid", "ascending")])


@pytest.mark.parametrize("mode", ["iter_batches", "spill"])
def test_streamed_and_spilled_paths_agree_at_scale(sides, mode):
    left, right = sides
    ds = bt.from_arrow(left).join_asof(bt.from_arrow(right), on="t", by="g")
    if mode == "spill":
        got = ds.collect(spill=True, num_partitions=4)
    else:
        got = pa.Table.from_batches(list(ds.iter_batches()))
    assert_tables_equal(_by_rid(got), _by_rid(ds.collect()), ordered=True)


@pytest.mark.parametrize("n_left, n_right", [(0, 10), (10, 0), (1, 1), (0, 0)])
def test_tiny_and_empty_sides(duck, n_left, n_right):
    left, right = _sides(5, 1, True)
    left, right = left.slice(0, n_left), right.slice(0, n_right)
    out = (
        bt.from_arrow(left)
        .join_asof(bt.from_arrow(right), on="t", by="g")
        .select("rid", "g", "t", "rrid")
        .collect()
    )
    assert out.num_rows == n_left
    assert_same(out, _duck_asof(duck, left, right, ">="))
