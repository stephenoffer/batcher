"""The Spark names this wave made reachable: lambdas, list edits, and composed scalars.

Same oracle and same rules as `test_sql_spark_dialect_names.py`: Spark's own
`@ExpressionDescription` examples, quoted here as the expected values, since there is no
JVM on this machine. These are unit tests rather than differential ones because DuckDB
does not have most of these names, let alone Spark's spelling of them.

The higher-order forms are the interesting half. `transform(xs, x -> x + 1)` binds the
lambda's parameter to `element()` and translates the body through the ordinary scalar
path, so anything the engine can compute is available inside a lambda — which is why the
tests below use arithmetic, a modulus and a null test inside one.
"""

from __future__ import annotations

import pytest

import batcher as bt

# (query, Spark's documented answer)
LAMBDA_CASES = [
    ("SELECT transform(array(1, 2, 3), x -> x + 1) AS r", [2, 3, 4]),
    ("SELECT transform(array(1, 2, 3), x -> x * x) AS r", [1, 4, 9]),
    ("SELECT filter(array(1, 2, 3), x -> x % 2 == 1) AS r", [1, 3]),
    ("SELECT filter(array(0, 2, 3), x -> x > 1) AS r", [2, 3]),
    ("SELECT exists(array(1, 2, 3), x -> x % 2 == 0) AS r", True),
    ("SELECT exists(array(1, 3, 5), x -> x % 2 == 0) AS r", False),
    ("SELECT forall(array(1, 2, 3), x -> x % 2 == 0) AS r", False),
    ("SELECT forall(array(2, 4, 8), x -> x % 2 == 0) AS r", True),
]

# The DuckDB spellings of the same two higher-order functions, which the same mechanism
# now serves.
DUCKDB_LAMBDA_CASES = [
    ("SELECT list_transform([1, 2, 3], x -> x * 2) AS r", [2, 4, 6]),
    ("SELECT array_transform([1, 2, 3], x -> x - 1) AS r", [0, 1, 2]),
    ("SELECT list_filter([1, 2, 3], x -> x > 1) AS r", [2, 3]),
    ("SELECT array_filter([1, NULL, 3], x -> x IS NOT NULL) AS r", [1, 3]),
]

LIST_CASES = [
    ("SELECT array_append(array('b', 'd', 'c', 'a'), 'd') AS r", ["b", "d", "c", "a", "d"]),
    ("SELECT array_prepend(array('b', 'd', 'c', 'a'), 'd') AS r", ["d", "b", "d", "c", "a"]),
    ("SELECT array_compact(array(1, 2, 3, null)) AS r", [1, 2, 3]),
    ("SELECT array_except(array(1, 2, 3), array(1, 3, 5)) AS r", [2]),
    ("SELECT array_remove(array(1, 2, 3, null, 3), 3) AS r", [1, 2, None]),
    ("SELECT arrays_overlap(array(1, 2, 3), array(3, 4, 5)) AS r", True),
    ("SELECT arrays_overlap(array(1, 2, 3), array(4, 5, 6)) AS r", False),
    ("SELECT get(array(1, 2, 3), 0) AS r", 1),
    ("SELECT get(array(1, 2, 3), 3) AS r", None),
    ("SELECT vector_inner_product(array(1.0, 2.0, 3.0), array(4.0, 5.0, 6.0)) AS r", 32.0),
    ("SELECT vector_norm(array(3.0, 4.0), 2.0) AS r", 5.0),
    ("SELECT array_repeat('123', 2) AS r", ["123", "123"]),
    ("SELECT array_repeat(5, 3) AS r", [5, 5, 5]),
]

SCALAR_CASES = [
    ("SELECT if(1 < 2, 'a', 'b') AS r", "a"),
    ("SELECT nvl2(NULL, 2, 1) AS r", 1),
    ("SELECT nvl2(3, 2, 1) AS r", 2),
    ("SELECT equal_null(3, 3) AS r", True),
    ("SELECT equal_null(NULL, NULL) AS r", True),
    ("SELECT equal_null(1, NULL) AS r", False),
    ("SELECT nullifzero(0) AS r", None),
    ("SELECT nullifzero(4) AS r", 4),
    ("SELECT zeroifnull(NULL) AS r", 0),
    ("SELECT zeroifnull(2) AS r", 2),
    ("SELECT pmod(10, 3) AS r", 1),
    ("SELECT pmod(-10, 3) AS r", 2),
    ("SELECT positive(1) AS r", 1),
    ("SELECT negative(1) AS r", -1),
    ("SELECT width_bucket(5.3, 0.2, 10.6, 5) AS r", 3),
    ("SELECT width_bucket(-2.1, 1.3, 3.4, 3) AS r", 0),
    ("SELECT width_bucket(8.1, 0.0, 5.7, 4) AS r", 5),
    ("SELECT btrim('    SparkSQL   ') AS r", "SparkSQL"),
    ("SELECT btrim('SSparkSQLS', 'SL') AS r", "parkSQ"),
    ("SELECT find_in_set('ab', 'abc,b,ab,c,def') AS r", 3),
    ("SELECT find_in_set('zz', 'abc,b,ab') AS r", 0),
    ("SELECT space(2) AS r", "  "),
    ("SELECT elt(1, 'scala', 'java') AS r", "scala"),
    ("SELECT elt(2, 'a', 'b') AS r", "b"),
    ("SELECT bit_get(11, 0) AS r", 1),
    ("SELECT bit_get(11, 2) AS r", 0),
    ("SELECT e() AS r", 2.718281828459045),
    ("SELECT regexp_count('abcabc', 'a') AS r", 2),
    ("SELECT regexp_substr('Steven Jones', 'Ste(v|ph)en') AS r", "Steven"),
    ("SELECT substring_index('www.apache.org', '.', 2) AS r", "www.apache"),
]

TEMPORAL_CASES = [
    ("SELECT add_months('2016-08-31', 1) AS r", "2016-09-30"),
    ("SELECT date_add('2016-07-30', 1) AS r", "2016-07-31"),
    ("SELECT date_sub('2016-07-30', 1) AS r", "2016-07-29"),
    ("SELECT unix_date(DATE('1970-01-02')) AS r", 1),
    ("SELECT unix_seconds(TIMESTAMP('1970-01-01 00:00:01')) AS r", 1),
    ("SELECT unix_millis(TIMESTAMP('1970-01-01 00:00:01')) AS r", 1000),
    ("SELECT unix_micros(TIMESTAMP('1970-01-01 00:00:01')) AS r", 1000000),
    ("SELECT timestamp_seconds(0) AS r", "1970-01-01 00:00:00"),
    ("SELECT timestamp_millis(1000) AS r", "1970-01-01 00:00:01"),
    ("SELECT timestamp_micros(1000000) AS r", "1970-01-01 00:00:01"),
]


def _one(query: str):
    return bt.sql(query, dialect="spark").to_pydict()["r"][0]


@pytest.mark.parametrize(("query", "want"), LAMBDA_CASES + DUCKDB_LAMBDA_CASES)
def test_higher_order_list_functions(query, want):
    assert _one(query) == want


@pytest.mark.parametrize(("query", "want"), LIST_CASES)
def test_list_edit_functions(query, want):
    assert _one(query) == want


@pytest.mark.parametrize(("query", "want"), SCALAR_CASES)
def test_composed_scalar_functions(query, want):
    got = _one(query)
    if isinstance(want, float):
        assert got == pytest.approx(want)
    else:
        assert got == want


@pytest.mark.parametrize(("query", "want"), TEMPORAL_CASES)
def test_temporal_functions(query, want):
    assert str(_one(query)) == str(want)


def test_a_two_parameter_lambda_is_refused_rather_than_half_translated():
    # `aggregate`/`zip_with` bind two parameters; the `.list` kernels have one
    # placeholder, so the call raises instead of silently dropping a parameter.
    with pytest.raises(NotImplementedError):
        bt.sql(
            "SELECT zip_with(array(1, 2), array(3, 4), (x, y) -> x + y) AS r", dialect="spark"
        ).collect()


def test_arrays_overlap_is_null_when_a_null_could_have_been_the_shared_element():
    # Spark's three-valued rule: no overlap found, but a null might have been it.
    assert _one("SELECT arrays_overlap(array(1, 2), array(3, NULL)) AS r") is None


def test_from_utc_timestamp_round_trips_through_to_utc_timestamp():
    query = (
        "SELECT to_utc_timestamp(from_utc_timestamp("
        "TIMESTAMP('2016-08-31 09:00:00'), 'Asia/Seoul'), 'Asia/Seoul') AS r"
    )
    assert str(_one(query)) == "2016-08-31 09:00:00"


def test_a_quoted_literal_section_is_emitted_without_its_quotes():
    # Java quotes a literal section (`yyyy'T'MM`) and sqlglot leaves the quotes in the
    # rewritten pattern, so they used to be printed: `1970'T'01` where Spark writes
    # `1970T01`.
    got = bt.sql("SELECT from_unixtime(0, \"yyyy'T'MM\") AS r", dialect="spark").to_pydict()
    assert got["r"] == ["1970T01"]


# --- the second batch: rounding, validity, structs and the zone fields ----------------

COMPOSED_CASES = [
    ("SELECT bround(2.5, 0) AS r", 2.0),
    ("SELECT bround(3.5, 0) AS r", 4.0),
    ("SELECT bround(-2.5, 0) AS r", -2.0),
    ("SELECT is_valid_utf8('Spark') AS r", True),
    ("SELECT validate_utf8('Spark') AS r", "Spark"),
    ("SELECT make_valid_utf8('Spark') AS r", "Spark"),
    ("SELECT try_validate_utf8('Spark') AS r", "Spark"),
]


@pytest.mark.parametrize(("query", "want"), COMPOSED_CASES)
def test_rounding_and_utf8_validity(query, want):
    assert _one(query) == want


def test_banker_rounding_differs_from_the_default_round():
    # `round` is half-away-from-zero (DuckDB's rule, which the engine follows);
    # `bround` is half-to-even. Both numbers are asserted so neither can drift.
    assert _one("SELECT round(2.5, 0) AS r") == 3.0
    assert _one("SELECT bround(2.5, 0) AS r") == 2.0


def test_named_struct_and_positional_struct():
    assert _one("SELECT named_struct('a', 1, 'b', 2) AS r") == {"a": 1, "b": 2}
    assert _one("SELECT struct(1, 2) AS r") == {"col1": 1, "col2": 2}


def test_the_duckdb_struct_literal_builds_the_same_thing():
    got = bt.sql("SELECT {'a': 1, 'b': 'x'} AS r").to_pydict()["r"][0]
    assert got == {"a": 1, "b": "x"}


def test_the_timezone_fields_of_a_naive_timestamp_are_zero():
    # Engine timestamps are tz-naive, so the offset is zero by construction — which is
    # also what DuckDB answers for a naive TIMESTAMP. Null propagates.
    assert bt.sql("SELECT timezone_hour(TIMESTAMP '2024-03-05 06:07:08') AS r").to_pydict() == {
        "r": [0]
    }
    assert bt.sql("SELECT timezone_minute(TIMESTAMP '2024-03-05 06:07:08') AS r").to_pydict() == {
        "r": [0]
    }


# --- the third batch: Java datetime patterns, weekday arithmetic, URLs -----------------

PATTERN_CASES = [
    ("SELECT date_format('2016-04-08', 'y') AS r", "2016"),
    ("SELECT date_format('2016-04-08', 'yyyy-MM-dd') AS r", "2016-04-08"),
    ("SELECT date_format(TIMESTAMP('2016-04-08 13:05:00'), 'HH:mm') AS r", "13:05"),
    ("SELECT to_date('2016-12-31', 'yyyy-MM-dd') AS r", "2016-12-31"),
    ("SELECT from_unixtime(0, 'yyyy-MM-dd HH:mm:ss') AS r", "1970-01-01 00:00:00"),
]

NEXT_DAY_CASES = [
    ("SELECT next_day('2015-01-14', 'TU') AS r", "2015-01-20"),
    ("SELECT next_day('2015-01-14', 'WE') AS r", "2015-01-21"),
    ("SELECT next_day('2015-01-14', 'Sun') AS r", "2015-01-18"),
    # Landing on the same weekday moves a whole week, not zero days.
    ("SELECT next_day('2015-01-14', 'Wednesday') AS r", "2015-01-21"),
]

URL_CASES = [
    ("SELECT parse_url('http://spark.apache.org/path?query=1', 'HOST') AS r", "spark.apache.org"),
    ("SELECT parse_url('http://spark.apache.org/path?query=1', 'PATH') AS r", "/path"),
    ("SELECT parse_url('http://spark.apache.org/path?query=1', 'QUERY') AS r", "query=1"),
    ("SELECT parse_url('http://spark.apache.org/path?query=1', 'PROTOCOL') AS r", "http"),
    ("SELECT parse_url('http://a.com/p#frag', 'REF') AS r", "frag"),
]


@pytest.mark.parametrize(("query", "want"), PATTERN_CASES + NEXT_DAY_CASES + URL_CASES)
def test_patterns_weekdays_and_urls(query, want):
    assert str(_one(query)) == want


def test_months_between_matches_sparks_documented_rounding():
    # Spark rounds to 8 decimal places unless told not to; both forms are pinned.
    assert _one("SELECT months_between('1997-02-28 10:30:00', '1996-10-30') AS r") == 3.94959677
    exact = _one("SELECT months_between('1997-02-28 10:30:00', '1996-10-30', false) AS r")
    assert exact == pytest.approx(3.9495967741935485)


def test_a_chrono_pattern_works_in_the_duckdb_dialect():
    # sqlglot rewrites a Spark pattern into a `%`-style one (with `strict` markers this
    # translator strips); a query written against DuckDB passes its own pattern through.
    got = bt.sql("SELECT strftime(TIMESTAMP '2016-04-08 00:00:00', '%Y/%m') AS r").to_pydict()
    assert got["r"] == ["2016/04"]


def test_the_three_argument_parse_url_is_declined():
    # Reading one query parameter needs the key escaped into the pattern; approximating
    # it would return a neighbouring parameter's value.
    with pytest.raises(NotImplementedError):
        bt.sql(
            "SELECT parse_url('http://a.com/p?q=1&qq=2', 'QUERY', 'q') AS r", dialect="spark"
        ).collect()


# --- the fourth batch: the "now" family and an array splice ----------------------------


def test_the_now_family_all_name_the_same_instant():
    # `now()`, `current_timestamp()` and `localtimestamp()` are one function here:
    # engine timestamps are tz-naive UTC, so there is no local/UTC distinction to draw.
    # The constant is bound once at plan-build time, which is what makes a query using it
    # deterministic across the morsels and partitions it runs on.
    import datetime as dt

    for query in (
        "SELECT now() AS r",
        "SELECT current_timestamp() AS r",
        "SELECT localtimestamp() AS r",
    ):
        got = _one(query)
        assert isinstance(got, dt.datetime)
        assert abs((got - dt.datetime.now()).total_seconds()) < 60


def test_now_is_one_value_for_the_whole_query():
    # Two references in one query must not read two different clocks.
    out = bt.sql("SELECT now() AS a, now() AS b", dialect="spark").to_pydict()
    assert out["a"] == out["b"]


def test_unix_timestamp_with_no_argument_is_the_current_second():
    import time

    got = _one("SELECT unix_timestamp() AS r")
    assert abs(got - time.time()) < 60


def test_current_timezone_is_utc_because_the_engine_stores_naive_utc():
    assert _one("SELECT current_timezone() AS r") == "UTC"


ARRAY_INSERT_CASES = [
    ("SELECT array_insert(array(1, 2, 3, 4), 2, 9) AS r", [1, 9, 2, 3, 4]),
    ("SELECT array_insert(array(1, 2, 3), 1, 0) AS r", [0, 1, 2, 3]),
    ("SELECT array_insert(array(1, 2), 3, 3) AS r", [1, 2, 3]),
]


@pytest.mark.parametrize(("query", "want"), ARRAY_INSERT_CASES)
def test_array_insert_splices_at_a_one_based_position(query, want):
    assert _one(query) == want


def test_a_negative_array_insert_position_is_declined():
    # Spark counts a negative position from the end; the two-slice splice cannot express
    # that, so it raises rather than inserting at the wrong end.
    with pytest.raises(NotImplementedError):
        bt.sql("SELECT array_insert(array(1, 2), -1, 9) AS r", dialect="spark").collect()
