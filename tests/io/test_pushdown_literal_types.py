"""Pushing a filter down must never turn a working query into an Arrow kernel error.

Predicate pushdown is an optimization: the engine's own `Filter` re-checks every row, so a
term the scanner cannot evaluate has to be *declined* rather than pushed. Arrow does not
decline — it raises `ArrowNotImplementedError` / `ArrowInvalid` from inside whatever task
built the scan, naming two Arrow types and no column.

`_comparable` is the guard that decides. This test is the cross-product it has to cover:
every column type against every literal type a caller can write, compared against the same
query run without pushdown. Two things must hold, and the second matters more than the
first — a raise is loud, a silently wrong answer is not.
"""

from __future__ import annotations

import datetime as dt
import decimal
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

pytestmark = pytest.mark.io

_COLUMNS = {
    "i64": pa.array([1, 2, 3], pa.int64()),
    "f64": pa.array([1.0, 2.0, 3.0], pa.float64()),
    "s": pa.array(["a", "b", "c"]),
    "d": pa.array([dt.date(2024, 1, i) for i in (1, 2, 3)], pa.date32()),
    "ts": pa.array([dt.datetime(2024, 1, i) for i in (1, 2, 3)], pa.timestamp("us")),
    "bo": pa.array([True, False, True]),
    "dec": pa.array(
        [decimal.Decimal("1.50"), decimal.Decimal("2.00"), decimal.Decimal("3.50")],
        pa.decimal128(5, 2),
    ),
}

#: One literal per type a caller can write, each chosen to select the middle row where the
#: comparison is meaningful, so a dropped term shows up as a different answer.
_LITERALS = {
    "int": 2,
    "float": 2.0,
    "str": "b",
    "date": dt.date(2024, 1, 2),
    "ts": dt.datetime(2024, 1, 2),
    "bool": True,
    "decimal": decimal.Decimal("2.00"),
}


@pytest.fixture(scope="module")
def table():
    return pa.table({**_COLUMNS, "n": pa.array([1, 2, 3], pa.int64())})


@pytest.fixture(scope="module")
def pushed(tmp_path_factory, table):
    """A Hive tree, which routes to the partition-aware reader and so pushes the filter."""
    root = str(tmp_path_factory.mktemp("pd") / "t")
    os.makedirs(f"{root}/g=x")
    pq.write_table(table, f"{root}/g=x/p.parquet")
    return bt.read.parquet(root)


#: Sentinel for "the engine refused this comparison", so a decline is a *value* the test
#: can assert on rather than control flow that removes the test from the run.
_DECLINED = object()


def _answer(dataset, build, column):
    """The rows `dataset` returns for this predicate, or `_DECLINED` if it refuses.

    Only `RuntimeError` is treated as a decline: that is what the FFI raises for every one
    of the 48 unsupported pairs here, while a `ColumnNotFoundError`, a `PlanError` or an
    `AssertionError` means something else broke and must reach the report as a failure.
    """
    try:
        return sorted(dataset.filter(build(bt.col(column))).to_pydict()["n"])
    except RuntimeError:
        return _DECLINED


@pytest.mark.parametrize("column", sorted(_COLUMNS))
@pytest.mark.parametrize("literal", sorted(_LITERALS))
@pytest.mark.parametrize("op", ["eq", "gt"])
def test_pushdown_answers_exactly_what_the_engine_answers(pushed, table, column, literal, op):
    value = _LITERALS[literal]
    build = (lambda c: c == value) if op == "eq" else (lambda c: c > value)

    # 48 of the 98 pairs are comparisons the engine declines outright (`bo > 'b'`). This
    # used to `pytest.skip` on them, which is the wrong shape twice over: it threw away the
    # only interesting half of the contract — that pushdown must decline *exactly where the
    # engine does*, never silently answer a comparison the engine refuses — and it did so
    # behind a bare `except Exception`, so an engine regression on a pair that works today
    # would have quietly removed it from the run instead of failing. `lint-skips` reads
    # module-level guards by design and cannot see a skip taken mid-body, so nothing would
    # have reported the loss.
    #
    # Both paths are therefore run and their *verdicts* compared. Measured on this tree:
    # all 48 decline on both sides and the other 50 answer on both sides.
    wanted = _answer(bt.from_arrow(table), build, column)
    got = _answer(pushed, build, column)
    if wanted is _DECLINED:
        assert got is _DECLINED, (
            f"the engine declines {column} vs {literal!r} but the pushdown path answered "
            f"{got!r} — a scan that evaluates a term the engine itself refuses is the "
            f"silently-wrong answer this file exists to catch"
        )
        return
    # No `pytest.raises` anywhere on the answering half: an Arrow kernel error there IS the
    # failure being guarded against, and letting it propagate names the pair in the test id.
    assert got == wanted


@pytest.mark.parametrize(
    ("column", "value"),
    [
        # `WHERE price = 2` against DECIMAL(5,2): Arrow rescales the literal into the
        # column's precision and raised `Precision is not great enough`.
        ("dec", 2),
        # Arrow promotes between numeric widths but has no `equal(int64, bool)`.
        ("i64", True),
        ("f64", True),
        # The one typing date partition keys created: `equal(date32, string)`.
        ("d", "2024-01-02"),
    ],
)
def test_the_specific_pairs_that_used_to_raise(pushed, table, column, value):
    wanted = sorted(bt.from_arrow(table).filter(bt.col(column) == value).to_pydict()["n"])
    assert sorted(pushed.filter(bt.col(column) == value).to_pydict()["n"]) == wanted
