"""SQL names reachable after the interval/JSON/temporal-construction wave, vs DuckDB.

Three families that a migrating DuckDB or Spark query types constantly and that used to
raise: sub-day `INTERVAL` arithmetic, the JSON inspection functions, and the timestamp
*constructors* (`strptime`, `epoch_ms`, `make_timestamp`, `time_bucket`). Each case is
asserted against DuckDB's own answer rather than a remembered constant.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

# `ts + INTERVAL n <unit>` for every unit the engine can express. The sub-day units
# were refused outright before this wave, and QUARTER/DECADE/CENTURY/MILLENNIUM were
# refused even though the months component could express them.
INTERVAL_QUERIES = [
    "SELECT TIMESTAMP '2024-03-05 06:07:08' + INTERVAL 1 HOUR AS r",
    "SELECT TIMESTAMP '2024-03-05 06:07:08' - INTERVAL 90 MINUTE AS r",
    "SELECT TIMESTAMP '2024-03-05 06:07:08' + INTERVAL 30 SECOND AS r",
    "SELECT TIMESTAMP '2024-03-05 06:07:08' + INTERVAL 2 HOURS AS r",
    "SELECT TIMESTAMP '2024-03-05 06:07:08' + INTERVAL 500 MILLISECOND AS r",
    "SELECT TIMESTAMP '2024-03-05 06:07:08' + INTERVAL 250 MICROSECOND AS r",
    "SELECT TIMESTAMP '2024-03-05 23:59:59' + INTERVAL 1 SECOND AS r",
    "SELECT date_add(TIMESTAMP '2024-03-05 06:07:08', INTERVAL 3 HOUR) AS r",
    "SELECT TIMESTAMP '2024-03-05 06:07:08' - INTERVAL 8 HOUR AS r",
]

# The calendar units, whose answer is a DATE in the engine and a TIMESTAMP in DuckDB —
# same calendar value, so they are compared as text.
CALENDAR_QUERIES = [
    ("SELECT DATE '2024-03-05' + INTERVAL 2 QUARTER AS r", "2024-09-05"),
    ("SELECT DATE '2024-03-05' + INTERVAL 1 DECADE AS r", "2034-03-05"),
    ("SELECT DATE '2024-03-05' + INTERVAL 1 CENTURY AS r", "2124-03-05"),
    ("SELECT DATE '2024-03-05' + INTERVAL 1 MILLENNIUM AS r", "3024-03-05"),
    ("SELECT DATE '2024-01-31' + INTERVAL 1 MONTH AS r", "2024-02-29"),
]


@pytest.mark.parametrize("q", INTERVAL_QUERIES)
def test_sub_day_interval_units_match_duckdb(duck, q):
    assert_same(bt.sql(q).collect(), duck.sql(q))


@pytest.mark.parametrize(("q", "want"), CALENDAR_QUERIES)
def test_calendar_interval_units_match_duckdb(duck, q, want):
    got = bt.sql(q).to_pydict()["r"][0]
    assert str(got) == want
    assert str(duck.sql(q).fetchall()[0][0]).startswith(want)


def test_a_date_plus_a_sub_day_interval_promotes_to_a_timestamp(duck):
    # DuckDB promotes the operand to TIMESTAMP because a Date32 cannot carry an hour;
    # the engine casts for the same reason, so both answer with the time of day.
    q = "SELECT DATE '2024-03-05' + INTERVAL 1 HOUR AS r"
    assert_same(bt.sql(q).collect(), duck.sql(q))


def test_an_unknown_interval_unit_is_still_refused():
    with pytest.raises(NotImplementedError, match="INTERVAL unit"):
        bt.sql("SELECT DATE '2024-03-05' + INTERVAL 1 FORTNIGHT AS r").collect()


@pytest.fixture
def docs(duck):
    t = pa.table(
        {
            "j": [
                '{"a": 1, "b": "x", "c": [1, 2], "d": {"e": 1}}',
                "[1, 2, 3]",
                '{"a": null}',
            ]
        }
    )
    duck.register("docs", t)
    return t


JSON_QUERIES = [
    "SELECT json_valid(j) AS r FROM docs",
    "SELECT json_exists(j, '$.a') AS r FROM docs",
    "SELECT json_array_length(j, '$.c') AS r FROM docs",
    "SELECT json_keys(j, '$.d') AS r FROM docs",
]


@pytest.mark.parametrize("q", JSON_QUERIES)
def test_json_inspection_functions_match_duckdb(duck, docs, q):
    assert_same(bt.sql(q, docs=docs).collect(), duck.sql(q))


def test_json_valid_is_false_for_text_that_is_not_json(duck):
    q = "SELECT json_valid('oops') AS r"
    assert_same(bt.sql(q).collect(), duck.sql(q))


def test_json_type_is_still_refused_because_it_names_a_different_thing():
    # DuckDB's json_type names the SQL type (UBIGINT); `.json.type_of` names the JSON
    # type (number). Answering one with the other is the failure this refusal prevents.
    with pytest.raises(NotImplementedError):
        bt.sql("SELECT json_type('{\"a\": 1}') AS r").collect()


TEMPORAL_QUERIES = [
    "SELECT strptime('2024-03-05 06:07:08', '%Y-%m-%d %H:%M:%S') AS r",
    "SELECT strptime('2024-03-05', '%Y-%m-%d') AS r",
    "SELECT try_strptime('2024-03-05', '%Y-%m-%d') AS r",
    "SELECT epoch_ms(1234567890) AS r",
    "SELECT epoch_ms(TIMESTAMP '2024-03-05 06:07:08') AS r",
    "SELECT make_timestamp_ms(1234567890) AS r",
    "SELECT make_timestamp_ns(1234567890000) AS r",
    "SELECT make_timestamp(2024, 3, 5, 6, 7, 8.0) AS r",
    "SELECT make_timestamp(1234567890000000) AS r",
    "SELECT time_bucket(INTERVAL 5 MINUTE, TIMESTAMP '2024-03-05 06:07:08') AS r",
    "SELECT time_bucket(INTERVAL 1 DAY, TIMESTAMP '2024-03-05 06:07:08') AS r",
    "SELECT time_bucket(INTERVAL 2 HOUR, TIMESTAMP '2024-03-05 06:07:08') AS r",
    "SELECT julian(DATE '1999-12-31') AS r",
    "SELECT julian(TIMESTAMP '2024-03-05 06:07:08') AS r",
    "SELECT era(DATE '2024-03-05') AS r",
    "SELECT era(TIMESTAMP '2024-03-05 06:07:08') AS r",
]


@pytest.mark.parametrize("q", TEMPORAL_QUERIES)
def test_temporal_constructors_match_duckdb(duck, q):
    assert_same(bt.sql(q).collect(), duck.sql(q))


def test_time_bucket_on_a_calendar_unit_is_refused_rather_than_misaligned():
    # DuckDB aligns month buckets to 2000-01-01; an epoch-aligned width would be off by
    # that origin, so the call is refused instead of answered.
    with pytest.raises(NotImplementedError):
        bt.sql("SELECT time_bucket(INTERVAL 1 MONTH, TIMESTAMP '2024-03-05') AS r").collect()


def test_make_timestamp_ns_truncates_to_the_microsecond_the_engine_stores(duck):
    # Engine timestamps are microseconds; DuckDB's `make_timestamp_ns` returns a
    # nanosecond timestamp. A sub-microsecond count therefore lands in the microsecond
    # that contains it. Pinned with both numbers so neither side can drift silently.
    q = "SELECT make_timestamp_ns(1234567890123) AS r"
    assert str(bt.sql(q).to_pydict()["r"][0]) == "1970-01-01 00:20:34.567890"
    text = "SELECT make_timestamp_ns(1234567890123)::VARCHAR AS r"
    assert duck.sql(text).fetchall()[0][0] == "1970-01-01 00:20:34.567890123"


def test_to_timestamp_is_the_same_instant_as_duckdbs(duck):
    # DuckDB's `to_timestamp` returns TIMESTAMPTZ, rendered in the session zone; the
    # engine's timestamps are tz-naive UTC. Same instant, different rendering — compared
    # as an epoch count so the comparison is about the value rather than the display.
    q = "SELECT epoch(to_timestamp(1234567890)) AS r"
    assert_same(bt.sql(q).collect(), duck.sql(q))


MISC_QUERIES = [
    "SELECT regexp_full_match('abc', 'a.c') AS r",
    "SELECT regexp_full_match('abcd', 'a.c') AS r",
    "SELECT regexp_full_match('ab', 'a|b') AS r",
    "SELECT constant_or_null(5, 3) AS r",
    "SELECT constant_or_null(5, NULL) AS r",
    "SELECT constant_or_null(5, 3, NULL) AS r",
    "SELECT grade_up([3, 1, 2]) AS r",
    "SELECT list_grade_up([10, 5, 7]) AS r",
    "SELECT array_grade_up([1, 2, 3]) AS r",
]


@pytest.mark.parametrize("q", MISC_QUERIES)
def test_misc_duckdb_names_match_duckdb(duck, q):
    assert_same(bt.sql(q).collect(), duck.sql(q))
