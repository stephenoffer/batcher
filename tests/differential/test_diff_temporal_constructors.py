"""Temporal constructors against DuckDB: `make_date`, `make_timestamp`, epoch reads.

The direction opposite the `.dt` field extractions. DuckDB has a direct counterpart for
each (`make_date`, `make_timestamp`, `to_timestamp`, `epoch_ms`), so each is compared
against it rather than against a hand-written expectation.

The one deliberate divergence is on an impossible date. DuckDB raises on `make_date(2023,
2, 29)`; Batcher answers null, so a scan of dirty upstream integers cannot be aborted by
one bad row. That case is kept out of the oracle comparisons and asserted on its own, the
same split the JSON tests make for malformed documents.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher._internal.errors import PlanError

# Dates that exist, so both engines agree: the epoch, a leap day, a year end, and a null.
VALID = pa.table(
    {
        "y": [1970, 2024, 1999, 2000, None],
        "m": [1, 2, 12, 1, 1],
        "d": [1, 29, 31, 1, 1],
        "h": [0, 13, 23, 0, 0],
        "mi": [0, 45, 59, 0, 0],
        "s": [0, 30, 59, 0, 0],
    }
)


def test_make_date_matches_duckdb(duck):
    duck.register("t", VALID)
    out = bt.from_arrow(VALID).select(r=bt.make_date(col("y"), col("m"), col("d"))).collect()
    assert_same(out, duck.sql("SELECT make_date(y, m, d) r FROM t"))


def test_make_timestamp_matches_duckdb(duck):
    duck.register("t", VALID)
    out = (
        bt.from_arrow(VALID)
        .select(r=bt.make_timestamp(col("y"), col("m"), col("d"), col("h"), col("mi"), col("s")))
        .collect()
    )
    assert_same(out, duck.sql("SELECT make_timestamp(y, m, d, h, mi, s) r FROM t"))


def test_make_timestamp_clock_parts_default_to_midnight():
    out = (
        bt.from_arrow(VALID)
        .select(r=bt.make_timestamp(col("y"), col("m"), col("d")))
        .to_pydict()["r"]
    )
    assert out[0] == dt.datetime(1970, 1, 1, 0, 0, 0)
    assert out[1] == dt.datetime(2024, 2, 29, 0, 0, 0)
    assert out[4] is None


EPOCHS = pa.table({"t": [0, 1_700_000_000, -1, 946_684_800, None]})


def test_from_epoch_seconds_matches_duckdb(duck):
    duck.register("e", EPOCHS)
    out = bt.from_arrow(EPOCHS).select(r=bt.from_epoch(col("t"))).collect()
    # DuckDB's `to_timestamp` returns TIMESTAMP WITH TIME ZONE, which renders in the
    # session's local zone; Batcher's timestamps are tz-naive UTC instants, so the
    # oracle is read back at UTC. The underlying instants are the same either way.
    assert_same(out, duck.sql("SELECT (to_timestamp(t) AT TIME ZONE 'UTC') r FROM e"))


def test_from_epoch_millis_matches_duckdb(duck):
    duck.register("e", EPOCHS)
    out = bt.from_arrow(EPOCHS).select(r=bt.from_epoch(col("t"), "ms")).collect()
    assert_same(out, duck.sql("SELECT epoch_ms(t) r FROM e"))


def test_every_unit_names_the_same_instant():
    # The whole point of naming the unit: the same instant expressed in four units must
    # produce one timestamp. A `cast("timestamp")` gets three of these four wrong.
    want = dt.datetime(2023, 11, 14, 22, 13, 20)
    for unit, value in [
        ("s", 1_700_000_000),
        ("ms", 1_700_000_000_000),
        ("us", 1_700_000_000_000_000),
        ("ns", 1_700_000_000_000_000_000),
    ]:
        t = pa.table({"t": [value]})
        got = bt.from_arrow(t).select(r=bt.from_epoch(col("t"), unit)).to_pydict()["r"]
        assert got == [want], unit


def test_a_bare_cast_would_have_read_epoch_seconds_as_microseconds():
    # The failure `from_epoch` exists to prevent, pinned so it cannot be "simplified"
    # back into a cast: Arrow assumes microseconds, so epoch seconds land in 1970.
    t = pa.table({"t": [1_700_000_000]})
    ds = bt.from_arrow(t)
    assert ds.select(r=col("t").cast("timestamp")).to_pydict()["r"] == [
        dt.datetime(1970, 1, 1, 0, 28, 20)
    ]
    assert ds.select(r=bt.from_epoch(col("t"), "s")).to_pydict()["r"] == [
        dt.datetime(2023, 11, 14, 22, 13, 20)
    ]


def test_from_unix_date_matches_duckdb(duck):
    t = pa.table({"d": [0, 19782, -1, None]})
    duck.register("u", t)
    out = bt.from_arrow(t).select(r=bt.from_unix_date(col("d"))).collect()
    # DuckDB spells it as a day offset from the epoch date, which widens to TIMESTAMP;
    # narrow it back so the comparison is date-to-date.
    assert_same(out, duck.sql("SELECT (DATE '1970-01-01' + INTERVAL (d) DAY)::DATE r FROM u"))


def test_round_trip_through_the_field_extractions():
    # `make_date` and the `.dt` fields are inverses, which is the strongest statement
    # available without an oracle: any date reconstructed from its own parts is itself.
    t = pa.table({"d": [dt.date(1970, 1, 1), dt.date(2024, 2, 29), dt.date(1999, 12, 31)]})
    out = (
        bt.from_arrow(t)
        .select(r=bt.make_date(col("d").dt.year(), col("d").dt.month(), col("d").dt.day()))
        .to_pydict()["r"]
    )
    assert out == list(t.column("d").to_pylist())


def test_epoch_round_trips_through_dt_epoch():
    t = pa.table({"ts": [dt.datetime(2023, 11, 14, 22, 13, 20), dt.datetime(1969, 6, 1)]})
    out = bt.from_arrow(t).select(r=bt.from_epoch(col("ts").dt.epoch(), "s")).to_pydict()["r"]
    assert out == list(t.column("ts").to_pylist())


@pytest.mark.parametrize(
    ("y", "m", "d"),
    [(2023, 2, 29), (2024, 13, 1), (2024, 1, 0), (2024, 0, 1), (2024, 4, 31)],
)
def test_an_impossible_date_is_null_rather_than_an_error(y, m, d):
    # DuckDB raises on each of these. Batcher answers null so one bad row in a scan of
    # dirty upstream integers cannot abort the query.
    t = pa.table({"y": [y], "m": [m], "d": [d]})
    got = bt.from_arrow(t).select(r=bt.make_date(col("y"), col("m"), col("d"))).to_pydict()
    assert got["r"] == [None]


def test_out_of_range_clock_fields_are_null():
    t = pa.table({"h": [24, 0, 0], "mi": [0, 60, 0], "s": [0, 0, 60]})
    got = (
        bt.from_arrow(t)
        .select(r=bt.make_timestamp(2024, 1, 1, col("h"), col("mi"), col("s")))
        .to_pydict()["r"]
    )
    assert got == [None, None, None]


def test_an_overflowing_epoch_value_is_null_not_a_wrapped_instant():
    t = pa.table({"t": [2**62, 0]})
    got = bt.from_arrow(t).select(r=bt.from_epoch(col("t"))).to_pydict()["r"]
    assert got == [None, dt.datetime(1970, 1, 1)]


def test_unknown_epoch_unit_fails_at_plan_build():
    with pytest.raises(PlanError, match="unit must be one of"):
        bt.from_epoch(col("t"), "seconds")
