"""A date/time function over a text column parses it, and does so for the whole family.

This file previously pinned the opposite: a build-time *refusal* of a string input, on the
premise that the engine answered such a column with a column of nulls. That premise was
false. `eval/temporal/date.rs` hoists a text-to-timestamp cast ahead of the whole `.dt`
surface, deliberately, and says why in the engine source: it accepts *more* than the DuckDB
oracle (which rejects the string form at bind time) rather than answering differently from
it, which is the only direction a compatibility convenience may go. Spark and pandas ports
depend on it, and `tests/unit/test_sql_spark_dialect_names.py` pins the SQL half.

So the refusal rejected queries the engine answers correctly. What remains true, and is what
these tests hold, is that the parse is *uniform*: every node kind in the family takes text,
and an unparseable value yields a null for that row alone -- `TRY_CAST` semantics, per value,
which no plan-time rule reading only the column's type could reproduce.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _table() -> pa.Table:
    return pa.table(
        {
            "s": pa.array(["2020-01-01", "2020-02-01"], pa.string()),
            "d": pa.array([datetime.date(2020, 1, 1), datetime.date(2020, 3, 1)], pa.date32()),
            "n": pa.array([1, 2], pa.int64()),
        }
    )


# One per temporal node kind, so a node that stops being covered fails here. Each maps a
# column to the answer the family must give for `_table()`'s text column and its date column
# alike -- the point being that the two agree.
_TEMPORAL = {
    "DateFunc": (lambda c: c.dt.year(), [2020, 2020]),
    "DateTrunc": (
        lambda c: c.dt.truncate("day"),
        [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 2, 1)],
    ),
    "DateOffset": (
        lambda c: c.dt.offset_by("1d"),
        [datetime.datetime(2020, 1, 2), datetime.datetime(2020, 2, 2)],
    ),
    "Strftime": (lambda c: c.dt.strftime("%Y"), ["2020", "2020"]),
}


@pytest.mark.parametrize("node", sorted(_TEMPORAL), ids=sorted(_TEMPORAL))
def test_a_string_input_is_parsed_not_refused(node: str) -> None:
    """The whole family reads text, rather than a subset of it by accident of kernel."""
    build, expected = _TEMPORAL[node]
    out = bt.from_arrow(_table()).select(x=build(bt.col("s"))).to_pydict()
    assert out["x"] == expected


@pytest.mark.parametrize("node", sorted(_TEMPORAL), ids=sorted(_TEMPORAL))
def test_text_lands_on_the_timestamp_spelling(node: str) -> None:
    """Text is not a second semantics: it parses to a timestamp and follows that path.

    Held against the *timestamp* spelling rather than the date one, because the hoist casts
    to `Timestamp(us)` and `offset_by` is type-preserving — so the date spelling of that node
    returns a `date` and the text spelling the same instant as a `datetime`. Comparing the two
    directly would assert the cast away.
    """
    build, _ = _TEMPORAL[node]
    text = bt.from_arrow(_table()).select(x=build(bt.col("s"))).to_pydict()["x"]
    stamps = (
        bt.from_pydict({"t": [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 2, 1)]})
        .select(x=build(bt.col("t")))
        .to_pydict()["x"]
    )
    assert text == stamps


def test_an_unparseable_value_nulls_only_its_own_row() -> None:
    """Per-value `TRY_CAST` semantics -- which is why no plan-time type rule can decide this.

    A rule reading the column's *type* sees `string` for this column and for a wholly valid
    one alike, so refusing on the type would have to refuse both.
    """
    ds = bt.from_pydict({"s": ["not-a-date", "2020-01-02", None]})
    assert ds.select(x=bt.col("s").dt.year()).to_pydict()["x"] == [None, 2020, None]


def test_a_numeric_input_raises_rather_than_being_read_as_an_epoch() -> None:
    """Recorded because the removed rule's docstring claimed the opposite.

    It stated that an integer input "works and is read as an epoch value", alongside the
    (also false) claim about strings. It does not: only text and null are hoisted, and an
    Int64 reaches Arrow's kernel and raises. The engine reporting this one for itself is
    exactly why it needs no plan-time rule.
    """
    with pytest.raises(RuntimeError, match="Year does not support"):
        bt.from_arrow(_table()).select(x=bt.col("n").dt.year()).collect()
