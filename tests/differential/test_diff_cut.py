"""`Expr.cut` — binning a measurement into labelled intervals, checked against DuckDB.

`cut` lowers to a `CASE` chain, so DuckDB's own `CASE` is an exact oracle: if the two
disagree, the desugaring is wrong. The cases that matter are the ones people get wrong
by hand — which side each interval is closed on, what happens exactly *on* a break, and
where a null goes (it must not silently land in the top bin, which is what a naive
`CASE WHEN x <= b THEN … ELSE top END` does).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

_VALUES = [-100, 0, 11, 12, 13, 18, 19, 20, 64, 65, 66, 1000, None]


def _table() -> pa.Table:
    """The values plus an explicit index, so both engines can be ordered identically."""
    return pa.table(
        {
            "i": pa.array(range(len(_VALUES)), type=pa.int64()),
            "x": pa.array(_VALUES, type=pa.int64()),
        }
    )


def _duck_case(duck, breaks: list[int], labels: list[str], left_closed: bool) -> list[str | None]:
    """The hand-written CASE chain `cut` desugars to. `labels` has `len(breaks) + 1` entries:
    one per WHEN branch, and the last as the ELSE."""
    op = "<" if left_closed else "<="
    branches = " ".join(
        f"WHEN x {op} {b} THEN '{name}'" for b, name in zip(breaks, labels[:-1], strict=True)
    )
    duck.register("t", _table())
    sql = (
        f"SELECT CASE WHEN x IS NULL THEN NULL {branches} ELSE '{labels[-1]}' END AS c "
        "FROM t ORDER BY i"
    )
    return [r[0] for r in duck.sql(sql).fetchall()]


def _batcher_cut(breaks: list[int], labels: list[str], left_closed: bool) -> list[str | None]:
    ds = bt.from_arrow(_table())
    binned = bt.col("x").cut(breaks, labels, left_closed=left_closed)
    return ds.select("i", c=binned).sort("i").to_pydict()["c"]


@pytest.mark.parametrize("left_closed", [False, True], ids=["right_closed", "left_closed"])
def test_it_matches_duckdbs_equivalent_case_chain(duck, left_closed):
    breaks = [12, 19, 65]
    labels = ["child", "teen", "adult", "senior"]
    got = _batcher_cut(breaks, labels, left_closed)
    assert got == _duck_case(duck, breaks, labels, left_closed)


def test_the_break_value_itself_lands_in_the_lower_bin_when_right_closed():
    """`(-inf, 12]` contains 12; `(12, 19]` does not."""
    got = bt.from_pydict({"x": [11, 12, 13]}).select(c=bt.col("x").cut([12, 19])).to_pydict()["c"]
    assert got == ["(-inf, 12]", "(-inf, 12]", "(12, 19]"]


def test_the_break_value_lands_in_the_upper_bin_when_left_closed():
    """`[-inf, 12)` excludes 12; `[12, 19)` contains it."""
    got = (
        bt.from_pydict({"x": [11, 12, 13]})
        .select(c=bt.col("x").cut([12, 19], left_closed=True))
        .to_pydict()["c"]
    )
    assert got == ["[-inf, 12)", "[12, 19)", "[12, 19)"]


def test_a_null_becomes_a_null_bin_not_the_top_bin():
    """The trap the guard exists for: null comparisons are null, so a naive CASE chain
    falls through every WHEN into the ELSE and mislabels missing data as the top bin."""
    got = bt.from_pydict({"x": [None, 100]}).select(c=bt.col("x").cut([50])).to_pydict()["c"]
    assert got == [None, "(50, inf]"]


def test_the_default_labels_are_interval_notation():
    got = bt.from_pydict({"x": [0]}).select(c=bt.col("x").cut([1, 5, 10])).to_pydict()["c"]
    assert got == ["(-inf, 1]"]
    labels = bt.from_pydict({"x": [0, 3, 7, 20]}).select(c=bt.col("x").cut([1, 5, 10]))
    assert labels.to_pydict()["c"] == ["(-inf, 1]", "(1, 5]", "(5, 10]", "(10, inf]"]


def test_float_breaks_render_without_a_trailing_zero_when_whole():
    got = bt.from_pydict({"x": [0.0]}).select(c=bt.col("x").cut([1.0, 2.5])).to_pydict()["c"]
    assert got == ["(-inf, 1]"]
    got = bt.from_pydict({"x": [2.0]}).select(c=bt.col("x").cut([1.0, 2.5])).to_pydict()["c"]
    assert got == ["(1, 2.5]"]


def test_a_single_break_makes_two_bins():
    got = bt.from_pydict({"x": [0, 10]}).select(c=bt.col("x").cut([5], ["lo", "hi"])).to_pydict()
    assert got["c"] == ["lo", "hi"]


def test_it_works_on_a_float_column_with_negative_and_extreme_values():
    ds = bt.from_pydict({"x": [-1e18, -0.5, 0.0, 0.5, 1e18]})
    got = ds.select(c=bt.col("x").cut([0.0], ["neg", "pos"])).to_pydict()["c"]
    assert got == ["neg", "neg", "neg", "pos", "pos"]  # 0.0 is right-closed into "neg"


def test_it_composes_into_a_group_by(duck):
    """The whole point: bin, then count per bin. It must survive being a group key."""
    table = pa.table({"x": pa.array(list(range(100)), type=pa.int64())})
    duck.register("t", table)
    got = (
        bt.from_arrow(table)
        .with_columns(bin=bt.col("x").cut([25, 50, 75], ["q1", "q2", "q3", "q4"]))
        .group_by("bin")
        .agg(n=bt.count())
        .sort("bin")
        .to_pydict()
    )
    want = duck.sql(
        "SELECT CASE WHEN x <= 25 THEN 'q1' WHEN x <= 50 THEN 'q2' "
        "WHEN x <= 75 THEN 'q3' ELSE 'q4' END AS bin, count(*) AS n "
        "FROM t GROUP BY bin ORDER BY bin"
    ).to_arrow_table()
    assert got["bin"] == want.column("bin").to_pylist()
    assert got["n"] == want.column("n").to_pylist()


def test_it_adds_no_plan_node():
    """`cut` is sugar over CASE; it must not introduce IR the engine has to learn."""
    expr = bt.col("x").cut([1, 2])
    ir = expr.to_ir()
    assert ir["e"] == "case", ir["e"]


@pytest.mark.parametrize(
    ("breaks", "labels", "match"),
    [
        ([], None, "must not be empty"),
        ([2, 1], None, "strictly increasing"),
        ([1, 1], None, "strictly increasing"),
        ([1], ["only-one"], "2 bins"),
        ([1, 2], ["a", "b", "c", "d"], "3 bins"),
    ],
)
def test_bad_arguments_are_rejected_at_the_api_edge(breaks, labels, match):
    with pytest.raises(PlanError, match=match):
        bt.col("x").cut(breaks, labels)
