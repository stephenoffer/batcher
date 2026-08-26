"""`with_columns` must return its new columns in the order they were written.

Found by an experiment that was then reverted, which is why it needs a home of its own:
`assert_same` was briefly tightened to compare column names positionally instead of as a set.
That tightening was wrong in general — it failed 49 join tests where Batcher follows SQL:2016
§7.7 and DuckDB does not, so the check would have enforced the comparator's deviation (see
`test_diff_join_column_order.py`). But among those failures were three that were **not** an
oracle disagreement, and reverting the tightening put them back out of sight.

This is those. `SELECT *, dr, gsum` has a defined column order in SQL, Python keyword
arguments preserve theirs, and the caller wrote `dr` before `gsum`. Batcher answers
`g, v, gsum, dr`.

Measured shape of the defect, which is narrower than "with_columns reorders":

    with_columns(a=v + 1,  b=v + 2)        -> g, v, a, b     correct
    with_columns(a=v + 1,  b=sum().over()) -> g, v, a, b     correct
    with_columns(a=rank(), b=sum().over()) -> g, v, b, a     WRONG
    three window expressions               -> g, v, b, a, c  WRONG

So it takes **two or more window expressions**, and it is a consequence of how they nest:
each becomes its own `Window` node, and `Window.available_columns()` is
`input.available_columns() + [alias]`, so whichever node ends up outermost contributes its
alias last.

**The nesting was not what put them in the wrong order, though — the optimizer was.** The
plan `with_columns` builds already reads `g, v, a, b`; Kyber's `transpose_adjacent_windows`
(Spark's `TransposeWindow`) then swaps the two nodes into a canonical spec order so that
`collapse_adjacent_windows` can merge equal specs, and the swap carries the output columns
with it. Its own docstring recorded the gap without seeing it: the aliases must be disjoint
"so the column **set** above the pair is unchanged either way" — and a set is not an order.
A rule required to be semantics-preserving was changing the result.

**Fixed** in `kyber/rules/relational/windows.py`: the swapped pair is wrapped in a `Project`
restoring the original column order. That costs the rule nothing, because
`collapse_adjacent_windows` matches a `Window` over a `Window` and the pair is still directly
stacked beneath the projection.

The values are right in every case; only the order differs. That is precisely why nothing
caught it: `assert_same` compares names as a set, `to_pydict()` comparisons are keyed by
name, and a caller reading columns by name never notices. It surfaces on
`collect().column_names`, on a positional write to a format with no header, and on anything
that consumes the Arrow schema in order.

These began as `xfail(strict=True)` — asserting the behaviour that *should* hold, so that
fixing the ordering turned them into failures and forced their retirement. They are plain
assertions now. The two controls stay: without them these could be passing because some
general reordering happens to agree, rather than because the specific defect is gone.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

pytest.importorskip("duckdb")

_T = pa.table({"g": ["a", "a", "b"], "v": [1, 2, 3]})


def _rank():
    return bt.dense_rank().over(partition_by="g", order_by="v")


def _group_sum():
    return bt.col("v").sum().over(partition_by="g")


def _row_number():
    return bt.row_number().over(partition_by="g", order_by="v")


def test_plain_expressions_keep_their_written_order():
    """The control. Without this, the assertions below could be passing because of some
    general reordering that happens to agree, rather than because the specific defect is
    gone — and the diagnosis in the module docstring would be unsupported.
    """
    got = bt.from_arrow(_T).with_columns(a=bt.col("v") + 1, b=bt.col("v") + 2).collect()
    assert got.column_names == ["g", "v", "a", "b"]


def test_one_window_beside_a_plain_expression_keeps_its_order():
    """Also correct today, and it localizes the defect: one `Window` node nests nothing."""
    got = bt.from_arrow(_T).with_columns(a=bt.col("v") + 1, b=_group_sum()).collect()
    assert got.column_names == ["g", "v", "a", "b"]


def test_two_window_expressions_keep_their_written_order():
    got = bt.from_arrow(_T).with_columns(a=_rank(), b=_group_sum()).collect()
    assert got.column_names == ["g", "v", "a", "b"], (
        "the caller wrote `a` before `b`; SQL's `SELECT *, a, b` and Python's keyword order "
        "both say `a` comes first"
    )


def test_three_window_expressions_keep_their_written_order():
    got = bt.from_arrow(_T).with_columns(a=_rank(), b=_group_sum(), c=_row_number()).collect()
    assert got.column_names == ["g", "v", "a", "b", "c"]


def test_two_window_expressions_match_duckdbs_column_order(duck):
    """The differential form, so the claim does not rest on reading the SQL standard.

    `SELECT *, dr, gsum` is unambiguous and DuckDB answers `g, v, dr, gsum`. Unlike the
    `USING`-join divergence this file's docstring refers to, there is no reading of the
    specification under which Batcher's order is the correct one.
    """
    duck.register("t", _T)
    got = bt.from_arrow(_T).with_columns(dr=_rank(), gsum=_group_sum()).collect()
    expected = duck.sql(
        "SELECT *, dense_rank() OVER (PARTITION BY g ORDER BY v) AS dr, "
        "sum(v) OVER (PARTITION BY g) AS gsum FROM t"
    ).to_arrow_table()
    assert got.column_names == expected.column_names


def test_the_values_are_correct_whatever_the_order(duck):
    """Bounding the defect: this is a presentation bug, not a wrong answer.

    Worth asserting, because "the columns come out in the wrong order" would be a far more
    serious finding if it meant the *data* had been transposed with the names.
    """
    duck.register("t", _T)
    got = bt.from_arrow(_T).with_columns(dr=_rank(), gsum=_group_sum()).collect().to_pydict()
    expected = (
        duck.sql(
            "SELECT *, dense_rank() OVER (PARTITION BY g ORDER BY v) AS dr, "
            "sum(v) OVER (PARTITION BY g) AS gsum FROM t"
        )
        .to_arrow_table()
        .to_pydict()
    )
    assert set(got) == set(expected)
    for name in got:
        assert sorted(got[name], key=repr) == sorted(expected[name], key=repr), name


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(lambda ds: ds.collect(), id="collect"),
        pytest.param(lambda ds: ds.collect(spill=True, num_partitions=4), id="spill"),
        pytest.param(lambda ds: pa.Table.from_batches(list(ds.iter_batches())), id="iter_batches"),
    ],
)
def test_the_order_holds_on_every_execution_path(run):
    """The correction is a `Project` the optimizer inserts, so every path has to carry it.

    The out-of-core and streaming paths peel row-wise operators off the top of a plan and
    re-apply them to the breaker's result — and the restoring projection *is* one of those.
    A path that peeled it and forgot to re-apply would return the transposed order again,
    on exactly the entry points a `collect()`-only test never reaches.
    """
    ds = bt.from_arrow(_T).with_columns(a=_rank(), b=_group_sum())
    assert run(ds).column_names == ["g", "v", "a", "b"]
