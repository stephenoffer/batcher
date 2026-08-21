"""`TO_TIMESTAMP(n, scale)` reads the scale it was given.

sqlglot records the scale as the decimal exponent of the unit, so 0 is seconds, 3 is
milliseconds, 6 microseconds and 9 nanoseconds. The lookup table carried 3, 6 and 9 and
defaulted everything else to milliseconds, which made scale 0 -- the seconds spelling a
Snowflake port reaches for first -- return an instant 1000x off. Scale 0 and scale 3
produced the *same* timestamp, from a query that raised nothing and returned a plausible
date in 1970.

DuckDB has no scale argument on `to_timestamp`, so it cannot be the oracle for the scaled
forms directly. Each scale is pinned instead to one instant, computed from the epoch count
rather than typed out, and that instant is separately checked against DuckDB by reading its
epoch back out of a timestamp literal. The unscaled `to_timestamp` is deliberately not the
anchor: DuckDB renders its result as TIMESTAMPTZ in the session time zone where Batcher
returns naive UTC, a rendering difference the translator documents and accepts.

Every case below passes a *literal*. A column argument takes a different branch, decided
by the column's declared type rather than by the argument's syntax; the last two tests
pin both readings.
"""

from __future__ import annotations

import datetime as _dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

#: 2023-11-14T22:13:20Z, as a second count.
_EPOCH_SECONDS = 1_700_000_000

#: The same instant, computed rather than written out, as a naive UTC timestamp.
_INSTANT = _dt.datetime.fromtimestamp(_EPOCH_SECONDS, tz=_dt.UTC).replace(tzinfo=None)

_ONE_ROW = pa.table({"n": pa.array([_EPOCH_SECONDS], type=pa.int64())})


@pytest.mark.parametrize(
    ("scale", "multiplier"),
    [(0, 1), (3, 1_000), (6, 1_000_000), (9, 1_000_000_000)],
)
def test_each_scale_names_the_unit_it_says(scale, multiplier):
    """Scale 0 and scale 3 must not agree, which is exactly what the default made them do."""
    scaled = _EPOCH_SECONDS * multiplier
    got = bt.sql(f"SELECT TO_TIMESTAMP({scaled}, {scale}) AS r FROM t", t=_ONE_ROW)
    assert got.to_pydict()["r"] == [_INSTANT], f"scale {scale} landed wrong"


def test_the_anchor_instant_matches_duckdb_read_as_an_epoch(duck):
    """The constant the scaled cases are pinned to is itself checked against DuckDB.

    Not via `to_timestamp`, whose result DuckDB renders as TIMESTAMPTZ in the session zone
    while Batcher returns a naive UTC timestamp (the same instant, a different rendering --
    a divergence `_sql/parser/expressions/temporal.py` documents deliberately). The epoch
    count of a fixed timestamp literal has no such ambiguity, so that is what is compared.
    """
    sql = "SELECT epoch(TIMESTAMP '2023-11-14 22:13:20') AS r FROM t"
    duck.register("t", _ONE_ROW)
    assert_same(bt.sql(sql, t=_ONE_ROW).collect(), duck.sql(sql))
    assert duck.sql(sql).fetchall()[0][0] == _EPOCH_SECONDS


def test_seconds_and_milliseconds_disagree_by_exactly_a_thousand():
    """The bug stated directly: the two scales used to return one instant."""
    secs = bt.sql(f"SELECT TO_TIMESTAMP({_EPOCH_SECONDS}, 0) AS r FROM t", t=_ONE_ROW)
    millis = bt.sql(f"SELECT TO_TIMESTAMP({_EPOCH_SECONDS}, 3) AS r FROM t", t=_ONE_ROW)
    first, second = secs.to_pydict()["r"][0], millis.to_pydict()["r"][0]
    assert first != second
    assert (first.year, second.year) == (2023, 1970)


def test_an_unrecognized_scale_refuses_rather_than_guessing():
    """There is no safe default: guessing one is what produced the 1000x error."""
    with pytest.raises(Exception, match=r"(?i)scale"):
        bt.sql(f"SELECT TO_TIMESTAMP({_EPOCH_SECONDS}, 1) AS r FROM t", t=_ONE_ROW).collect()


def test_the_scaled_form_survives_a_partitioned_collect():
    """The unit is fixed at translation, so every partition must resolve it identically."""
    table = pa.table({"k": pa.array(list(range(16)), type=pa.int64())})
    ds = bt.sql(f"SELECT k, TO_TIMESTAMP({_EPOCH_SECONDS}, 0) AS r FROM t", t=table)
    assert ds.collect().to_pydict() == ds.repartition(4).collect().to_pydict()


def test_an_epoch_column_builds_a_timestamp_like_duckdb(duck):
    """A stored epoch *column* builds a timestamp, exactly as the literal form does.

    `epoch_ms(n)` is two functions under one name, and which one a call means depends on
    the argument's *type*. Reading the syntax alone ("is it an integer literal?") sent a
    column down the extract branch, which returned an epoch count -- 0 for every row of a
    small column -- where DuckDB builds a timestamp. The translator now binds the
    relation's column types before the expression is built (`_Translator.bind_scope`), so
    the type decides.
    """
    table = pa.table({"n": pa.array([_EPOCH_SECONDS], type=pa.int64())})
    sql = "SELECT epoch_ms(n) AS r FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_a_timestamp_column_still_reads_its_epoch_out(duck):
    """The other reading must not regress: a timestamp argument extracts the count."""
    table = pa.table({"ts": pa.array([_INSTANT], type=pa.timestamp("us"))})
    sql = "SELECT epoch_ms(ts) AS r FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))
