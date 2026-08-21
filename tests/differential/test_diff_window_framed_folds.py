"""Explicit `ROWS`/`GROUPS` frames over the fold aggregates — vs DuckDB.

The framed window path had a hand-written sliding kernel per function, so only the
original five aggregates honoured a frame; `product`, `bool_and`, `bool_or`, `bit_and`,
`bit_or` and `bit_xor` rejected one at plan time. They all have one now, from a *single*
generalization: `FifoSum`, the two-stack "queue from two stacks" the float sum already
used, was never specific to addition.

Why that structure rather than the obvious slide is the thing these tests are really
pinning. The naive O(1) update applies the entering value and **un-applies** the leaving
one, which needs an inverse — and these folds do not have one. `product` cannot divide a
zero back out; `bit_and` and `bool_and` cannot un-AND at all. So the fixture deliberately
contains a **zero** and a **false**: with a subtract-based slide, a window that has passed
over either can never recover, and every later row is wrong.

Frames are exercised trailing, centred (both edges moving), and unbounded-preceding,
because the two-stack only reloads when its pop side empties — a bug in that reload shows
up only once the window has slid far enough to trigger it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered
from batcher import col

FOLDS = [
    ("product", lambda: col("x").product(), "product(x)"),
    ("bool_and", lambda: col("b").bool_and(), "bool_and(b)"),
    ("bool_or", lambda: col("b").bool_or(), "bool_or(b)"),
    ("bit_and", lambda: col("i").bit_and(), "bit_and(i)"),
    ("bit_or", lambda: col("i").bit_or(), "bit_or(i)"),
    ("bit_xor", lambda: col("i").bit_xor(), "bit_xor(i)"),
]

FRAMES = [
    ((-1, 0), "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW"),
    ((-2, 1), "ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING"),
    ((None, 0), "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"),
    ((0, 2), "ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING"),
]


@pytest.fixture
def table(duck):
    """A zero, a false, and a null in each column.

    The zero and the false are the load-bearing rows: they are what an inverse-based
    slide cannot undo, so a kernel that subtracts instead of re-folding gets every row
    after them wrong.
    """
    t = pa.table(
        {
            "g": ["a"] * 5 + ["b"] * 4,
            "o": [1, 2, 3, 4, 5, 1, 2, 3, 4],
            "x": [1.0, 2.0, 0.0, 4.0, 5.0, 2.0, None, 4.0, 8.0],
            "i": [6, 4, 0, 12, 8, 3, 7, 5, 1],
            "b": [True, False, True, True, True, True, True, False, None],
        }
    )
    duck.register("t", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize(("name", "build", "duck_agg"), FOLDS)
@pytest.mark.parametrize(("frame", "duck_frame"), FRAMES)
def test_framed_fold_matches_duckdb(duck, table, name, build, duck_agg, frame, duck_frame):
    out = (
        bt.from_arrow(table)
        .select(g=col("g"), o=col("o"), r=build().over("g", order_by="o", frame=frame))
        .sort("g", "o")
        .collect()
    )
    expected = duck.sql(
        f"SELECT g, o, {duck_agg} OVER (PARTITION BY g ORDER BY o {duck_frame}) r "
        "FROM t ORDER BY g, o"
    )
    assert_same_ordered(out, expected)


@pytest.mark.differential
def test_a_zero_does_not_poison_the_rest_of_the_product(duck, table):
    """The case that decides the whole design, asserted on its own.

    A slide that divided the leaving value out would hold `0` forever after row 3 of
    group `a`, or produce infinity. Re-folding the window's actual contents cannot.
    """
    got = (
        bt.from_arrow(table)
        .select(g=col("g"), o=col("o"), r=col("x").product().over("g", order_by="o", frame=(-1, 0)))
        .sort("g", "o")
        .to_pydict()["r"]
    )
    # rows 3 and 4 of group `a` bracket the zero; row 5 has slid past it entirely.
    assert got[2] == 0.0 and got[3] == 0.0
    assert got[4] == 20.0, "the window slid past the zero and must recover"


@pytest.mark.differential
def test_a_false_does_not_poison_the_rest_of_bool_and(duck, table):
    """The same argument for a boolean AND, which has no inverse at all."""
    got = (
        bt.from_arrow(table)
        .select(
            g=col("g"), o=col("o"), r=col("b").bool_and().over("g", order_by="o", frame=(-1, 0))
        )
        .sort("g", "o")
        .to_pydict()["r"]
    )
    assert got[1] is False and got[2] is False
    assert got[3] is True, "the window slid past the false and must recover"


@pytest.mark.differential
def test_an_all_null_frame_is_null_not_the_fold_identity(duck, table):
    """An empty (or all-null) frame is NULL, not `1` / `true` / `0`.

    Null rows are never pushed, so the fold's identity element is never needed — which is
    what keeps `product` over an all-null window from answering `1`.
    """
    nulls = pa.table({"o": [1, 2, 3], "x": [None, None, None]})
    duck.register("nulls", nulls)
    query = (
        "SELECT o, product(x) OVER (ORDER BY o ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) r "
        "FROM nulls ORDER BY o"
    )
    out = (
        bt.from_arrow(nulls)
        .select(o=col("o"), r=col("x").product().over(order_by="o", frame=(-1, 0)))
        .sort("o")
        .collect()
    )
    assert_same_ordered(out, duck.sql(query))
    assert out.column("r").to_pylist() == [None, None, None]


@pytest.mark.differential
def test_the_aggregates_with_no_sliding_form_still_refuse_a_frame():
    """`count_distinct` needs a multiset and `median` needs an order statistic — neither is
    expressible as this slide's combine, so both must keep raising rather than being handed
    a frame the kernel cannot honour.

    `var`/`stddev` were on this list and are no longer: the slide never required an
    *operator*, only an associative combine, and Welford has one in Chan's parallel formula.
    Merging two sorted halves is associative too, which is exactly why `median` needs saying
    out loud — it *could* be carried here, at `O(k)` a step, and that is the cost the slide
    exists to avoid.
    """
    ds = bt.from_pydict({"g": ["a", "a"], "o": [1, 2], "x": [1.0, 2.0]})
    for build in (lambda: col("x").n_unique(), lambda: col("x").median()):
        with pytest.raises(Exception, match="frame"):
            ds.select(r=build().over("g", order_by="o", frame=(-1, 0))).collect()
