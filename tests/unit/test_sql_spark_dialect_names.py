"""Spark SQL function names reachable from `bt.sql(dialect="spark")`.

Spark ships its own oracle in-source: every builtin carries an `@ExpressionDescription`
with `> SELECT _FUNC_(args);` and the expected output, and `FunctionRegistry.scala` maps
each expression class to its SQL name. Running all 527 documented examples through
`bt.sql` is a census that needs no JVM — which matters, because there is no Java runtime
on this machine. It put the reachable count at 75 of 312 probed; these are the names that
closed it to 100.

These are unit tests, not differential ones: DuckDB cannot parse Spark's syntax, so it is
not the oracle here. The expected values are Spark's own documented answers, quoted in
each case, **except** where Spark and DuckDB genuinely disagree on what a function means —
`dialect=` selects a parser, not a semantics, and the engine follows DuckDB. Those cases
are the last test, which pins the divergence so it cannot drift unnoticed.
"""

from __future__ import annotations

import pytest

import batcher as bt

SPARK_CASES = [
    # Math spellings sqlglot gives a typed node, each an existing `Expr` method.
    ("SELECT sec(0) AS r", 1.0),
    ("SELECT csc(1) AS r", 1.1883951057781212),
    ("SELECT rint(12.3456) AS r", 12.0),
    ("SELECT bit_count(0) AS r", 0),
    ("SELECT hypot(3, 4) AS r", 5.0),
    ("SELECT log1p(0) AS r", 0.0),
    ("SELECT expm1(0) AS r", 0.0),
    ("SELECT isnull(1) AS r", False),
    ("SELECT isnotnull(1) AS r", True),
    ("SELECT nanvl(cast('NaN' AS double), 123) AS r", 123.0),
    # List operations.
    ("SELECT flatten(array(array(1, 2), array(3, 4))) AS r", [1, 2, 3, 4]),
    ("SELECT array_sort(array(5, 6, 1)) AS r", [1, 5, 6]),
    ("SELECT array_join(array('hello', 'world'), ' ') AS r", "hello world"),
    ("SELECT array_position(array(312, 773, 708, 708), 708) AS r", 3),
    # `slice` is 1-based with a *length* second operand; `.list.slice` is 0-based with a
    # length, so both the base and the operand meaning have to be translated.
    ("SELECT slice(array(1, 2, 3, 4), 2, 2) AS r", [2, 3]),
    ("SELECT element_at(array(1, 2, 3), 2) AS r", 2),
    # Dates from a string, which Spark allows and the engine now does uniformly.
    ("SELECT year('2016-07-30') AS r", 2016),
    ("SELECT month('2016-07-30') AS r", 7),
    ("SELECT day('2009-07-30') AS r", 30),
    ("SELECT quarter('2016-08-31') AS r", 3),
    ("SELECT dayofyear('2016-04-09') AS r", 100),
    ("SELECT weekofyear('2008-02-20') AS r", 8),
    ("SELECT make_date(2013, 7, 15) AS r", __import__("datetime").date(2013, 7, 15)),
    # Strings.
    ("SELECT url_encode('a b') AS r", "a%20b"),
    ("SELECT url_decode('a%20b') AS r", "a b"),
    ("SELECT try_url_decode('a%20b') AS r", "a b"),
    ("SELECT try_mod(3, 2) AS r", 1),
]


@pytest.mark.unit
@pytest.mark.parametrize(("query", "expected"), SPARK_CASES)
def test_spark_name_resolves_and_answers(query, expected):
    got = bt.sql(query, dialect="spark").to_pydict()["r"][0]
    if isinstance(expected, float):
        assert got == pytest.approx(expected)
    else:
        assert got == expected


@pytest.mark.unit
def test_a_date_function_accepts_a_string_column_uniformly():
    """Which date functions worked on text used to be an accident of implementation.

    `dayname` and `last_day` cast to a timestamp as a side effect of how they computed,
    so they accepted a string column; `year`, `month` and `second` handed the array
    straight to Arrow's kernel and failed with "Year does not support: Utf8". The cast is
    now hoisted, so the whole family agrees.
    """
    ds = bt.from_pydict({"d": ["2016-07-30", "2024-02-29", None]})
    out = ds.select(
        y=bt.col("d").dt.year(),
        m=bt.col("d").dt.month(),
        s=bt.col("d").dt.second(),
        name=bt.col("d").dt.dayname(),
    ).to_pydict()
    assert out["y"] == [2016, 2024, None]
    assert out["m"] == [7, 2, None]
    assert out["s"] == [0, 0, None]
    assert out["name"] == ["Saturday", "Thursday", None]


@pytest.mark.unit
def test_duckdb_list_slice_bounds_are_inclusive_not_a_length():
    """`list_slice(l, begin, end)` and `slice(l, start, length)` are different functions.

    DuckDB's takes an inclusive 1-based `begin`..`end`; Spark's takes a 1-based start and
    a count. They cannot share a translation, and getting it wrong returns a plausible
    window one element along.
    """
    assert bt.sql("SELECT list_slice([1, 2, 3, 4], 2, 3) AS r").to_pydict()["r"] == [[2, 3]]
    assert bt.sql("SELECT array_slice([1, 2, 3, 4], 2, 4) AS r").to_pydict()["r"] == [[2, 3, 4]]
    assert bt.sql("SELECT slice(array(1, 2, 3, 4), 2, 3) AS r", dialect="spark").to_pydict()[
        "r"
    ] == [[2, 3, 4]]


@pytest.mark.unit
def test_two_argument_to_binary_is_refused_rather_than_answered():
    """Spark's `to_binary(s, charset)` encodes bytes; DuckDB's `to_binary(s)` is a bit string.

    Same name, different function. The two-argument form silently returned the bit string,
    which is the failure mode the census exists to catch, so it now raises.
    """
    with pytest.raises(NotImplementedError, match="bit-string"):
        bt.sql("SELECT to_binary('abc', 'utf-8') AS r", dialect="spark").to_pydict()
    # The one-argument DuckDB form is unaffected.
    assert bt.sql("SELECT to_binary('a') AS r").to_pydict()["r"] == ["01100001"]


@pytest.mark.unit
def test_spark_semantics_that_are_deliberately_not_adopted():
    """`dialect=` selects a parser, not a semantics — the engine follows DuckDB.

    Each of these runs and returns DuckDB's answer rather than Spark's. They are listed in
    the `migrate-from-spark` skill so a port can rewrite them; pinning them here means a
    later change cannot quietly flip one in either direction.
    """
    spark = {"dialect": "spark"}
    # Spark's `weekday` is Monday-based (Thursday is 3); DuckDB's is Sunday-based (4).
    assert bt.sql("SELECT weekday('2009-07-30') AS r", **spark).to_pydict()["r"] == [4]
    # Spark's `regexp_replace` replaces every match; DuckDB's replaces the first.
    assert bt.sql(r"SELECT regexp_replace('100-200', '(\d+)', 'num') AS r", **spark).to_pydict()[
        "r"
    ] == ["num-200"]
    # Spark abbreviates the day name; DuckDB spells it out.
    assert bt.sql("SELECT dayname('2008-02-20') AS r", **spark).to_pydict()["r"] == ["Wednesday"]
