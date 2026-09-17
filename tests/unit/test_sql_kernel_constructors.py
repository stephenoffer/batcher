"""The Spark-only constructors lifted out of the SQL front end, and `bt.sql_expr`/`call_function`.

DuckDB has none of `pmod`, `bit_get`, `elt`, `find_in_set`, `parse_url`, `regexp_substr`,
`try_url_decode`, `months_between`, `next_day` or `array_remove`, so the oracle is Spark's own
documentation: every expected value quoted below is the output printed in the docstring of the
same-named function in `pyspark/sql/functions/builtin.py` (the reference copy lives at
`/mnt/shared_storage/ref/spark/python/pyspark/sql/functions/builtin.py`). There is no JVM on
this machine, so the documented output is the strongest oracle available.

The names DuckDB does have are differential tests in
`tests/differential/test_diff_sql_kernel_constructors.py`.

Each constructor is also checked against the SQL spelling of the same call, since the point of
lifting a lowering is that the two cannot drift apart.
"""

from __future__ import annotations

import datetime as dt
import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher import PlanError

col = bt.col


def _rows(ds: bt.Dataset, name: str = "r") -> list:
    return ds.to_pydict()[name]


# --- numeric ----------------------------------------------------------------------------


def test_pmod_matches_the_documented_float_table():
    # builtin.py `pmod`: the nine (a, b) rows and their printed results, NaN included.
    a = [1.0, math.nan, 10.0, math.nan, -3.0, -10.0, -5.0, 7.0, 1.0]
    b = [math.nan, 2.0, 3.0, math.nan, 4.0, 3.0, -6.0, -8.0, 2.0]
    want = [math.nan, math.nan, 1.0, math.nan, 1.0, 2.0, -5.0, 7.0, 1.0]
    got = _rows(bt.from_pydict({"a": a, "b": b}).select(r=bt.pmod("a", "b")))
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert (math.isnan(g) and math.isnan(w)) or g == w


def test_pmod_on_integers_zero_divisor_and_nulls():
    ds = bt.from_pydict({"a": [-10, 10, 7, None, 5], "b": [3, -3, 0, 2, None]})
    assert _rows(ds.select(r=bt.pmod(col("a"), col("b")))) == [2, 1, None, None, None]


def test_bit_get_matches_both_documented_examples():
    ds = bt.from_pydict({"value": [1, 2, 3, None]})
    assert _rows(ds.select(r=bt.bit_get("value", bt.lit(1)))) == [0, 1, 1, None]
    ds = bt.from_pydict({"value": [1, 2, 3, None], "pos": [2, 1, None, 1]})
    assert _rows(ds.select(r=bt.bit_get(col("value"), col("pos")))) == [0, 1, None, None]


def test_pi_and_e_are_the_float_constants():
    ds = bt.from_pydict({"x": [0]}).select(p=bt.pi(), e=bt.e())
    assert ds.to_pydict() == {"p": [math.pi], "e": [math.e]}
    assert ds.schema.field("p").type == pa.float64()


def test_try_mod_is_the_percent_operator():
    # Spark `try_mod(-7, 2)` is -1 and a zero divisor is null; the engine's `%` is both.
    ds = bt.from_pydict({"a": [-7, 7], "b": [2, 0]})
    assert _rows(ds.select(r=col("a") % col("b"))) == [-1, None]


def test_positive_is_unary_plus():
    ds = bt.from_pydict({"x": [-1, None, 2]})
    assert _rows(ds.select(r=+col("x"))) == [-1, None, 2]


# --- elt / find_in_set / parse_url / regexp_substr / try_url_decode ----------------------


def test_elt_matches_the_documented_example_and_nulls_out_of_range():
    ds = bt.from_pydict({"a": [1, 2, 0, 3, None], "b": ["scala"] * 5, "c": ["java"] * 5})
    assert _rows(ds.select(r=bt.elt("a", col("b"), col("c")))) == [
        "scala",
        "java",
        None,
        None,
        None,
    ]
    assert _rows(ds.select(r=bt.elt(2, col("b"), col("c")))) == ["java"] * 5
    assert _rows(ds.select(r=bt.elt(7, col("b"), col("c")))) == [None] * 5


def test_elt_needs_a_candidate():
    with pytest.raises(PlanError, match="at least one value"):
        bt.elt(1)


def test_find_in_set_matches_the_documented_example():
    ds = bt.from_pydict({"b": ["abc,b,ab,c,def", "", None, "a,b"]})
    assert _rows(ds.select(r=col("b").str.find_in_set("ab"))) == [3, 0, None, 0]


def test_find_in_set_refuses_a_column_needle():
    with pytest.raises(PlanError, match="constant string"):
        col("b").str.find_in_set(col("a"))


URL = "https://spark.apache.org/path?query=1"


@pytest.mark.parametrize(
    ("part", "key", "want"),
    [
        ("QUERY", None, "query=1"),
        ("QUERY", "query", "1"),
        ("PROTOCOL", None, "https"),
        ("HOST", None, "spark.apache.org"),
        ("PATH", None, "/path"),
        ("FILE", None, "/path?query=1"),
        ("REF", None, None),
        ("host", None, "spark.apache.org"),
    ],
)
def test_parse_url_matches_the_documented_parts(part, key, want):
    ds = bt.from_pydict({"u": [URL, None]})
    assert _rows(ds.select(r=col("u").str.parse_url(part, key))) == [want, None]


def test_a_query_key_is_not_matched_by_a_longer_neighbour():
    ds = bt.from_pydict({"u": ["http://a.com/p?qq=2&q=1&a.b=3"]})
    got = ds.select(
        q=col("u").str.parse_url("QUERY", "q"),
        qq=col("u").str.parse_url("QUERY", "qq"),
        dotted=col("u").str.parse_url("QUERY", "a.b"),
        missing=col("u").str.parse_url("QUERY", "a"),
    ).to_pydict()
    assert got == {"q": ["1"], "qq": ["2"], "dotted": ["3"], "missing": [None]}


def test_parse_url_refuses_an_unknown_part_and_a_key_off_query():
    with pytest.raises(PlanError, match="part must be one of"):
        col("u").str.parse_url("PORT")
    with pytest.raises(PlanError, match="QUERY"):
        col("u").str.parse_url("HOST", "k")


def test_regexp_substr_is_extract_with_a_null_miss():
    # builtin.py `regexp_substr`: `\d+` finds "1", `mmm` finds nothing and prints NULL.
    ds = bt.from_pydict({"str": ["1a 2b 14m", None]})
    got = ds.select(
        hit=col("str").str.extract(r"\d+", 0, missing="null"),
        miss=col("str").str.extract("mmm", 0, missing="null"),
    ).to_pydict()
    assert got == {"hit": ["1", None], "miss": [None, None]}
    sql = bt.sql("SELECT regexp_substr('1a 2b 14m', 'mmm') AS r", dialect="spark")
    assert _rows(sql) == [None]


def test_try_url_decode_matches_both_documented_examples():
    ds = bt.from_pydict(
        {"url": ["https%3A%2F%2Fspark.apache.org", "https%3A%2F%2spark.apache.org", "a+b", None]}
    )
    spark = col("url").str.url_decode(form=True, malformed="null")
    assert _rows(ds.select(r=spark)) == ["https://spark.apache.org", None, "a b", None]
    sql = bt.sql("SELECT try_url_decode('https%3A%2F%2spark.apache.org') AS r", dialect="spark")
    assert _rows(sql) == [None]


def test_the_default_url_decode_still_keeps_a_malformed_escape():
    ds = bt.from_pydict({"s": ["100%", "%zz", "%4"]})
    assert _rows(ds.select(r=col("s").str.url_decode())) == ["100%", "%zz", "%4"]
    nulled = col("s").str.url_decode(malformed="null")
    assert _rows(ds.select(r=nulled)) == [None, None, None]


# --- lists --------------------------------------------------------------------------------


def test_array_append_and_prepend_match_the_documented_examples():
    ds = bt.from_pydict({"c1": [["b", "a", "c"]], "c2": ["c"]})
    assert _rows(ds.select(r=col("c1").list.append(col("c2")))) == [["b", "a", "c", "c"]]
    assert _rows(ds.select(r=col("c1").list.prepend(col("c2")))) == [["c", "b", "a", "c"]]
    data = bt.from_pydict({"data": [[1, 2, 3]]})
    assert _rows(data.select(r=col("data").list.append(4))) == [[1, 2, 3, 4]]
    assert _rows(data.select(r=col("data").list.prepend(4))) == [[4, 1, 2, 3]]
    assert _rows(data.select(r=col("data").list.append(None))) == [[1, 2, 3, None]]


def test_a_null_list_is_null_only_under_the_spark_rule():
    ds = bt.from_arrow(pa.table({"data": pa.array([None], pa.list_(pa.int64()))}))
    assert _rows(ds.select(r=col("data").list.append(4))) == [[4]]
    spark = col("data").list.append(4, propagate_nulls=True)
    assert _rows(ds.select(r=spark)) == [None]
    spark = col("data").list.prepend(4, propagate_nulls=True)
    assert _rows(ds.select(r=spark)) == [None]


def test_array_remove_matches_the_documented_examples():
    ds = bt.from_pydict({"data": [[1, 2, 3, 1, 1], [4, 5, 5, 4], [1, 1, 1]]})
    assert _rows(ds.select(r=col("data").list.remove(1))) == [[2, 3], [4, 5, 5, 4], []]
    assert _rows(ds.select(r=col("data").list.remove(5))) == [[1, 2, 3, 1, 1], [4, 4], [1, 1, 1]]


def test_array_remove_refuses_an_expression():
    with pytest.raises(PlanError, match="literal scalar"):
        col("data").list.remove(col("x"))


def test_arrays_overlap_matches_the_documented_examples():
    cases = [
        ([["a", "b"], ["a"]], [["b", "c"], ["b", "c"]], [True, False]),
        ([["a", None], ["a"]], [["b", None], ["b", "c"]], [None, False]),
        ([None, ["a"]], [["b", "c"], None], [None, None]),
        ([["a", "b"], ["a"]], [["a", "b"], ["a"]], [True, True]),
    ]
    for x, y, want in cases:
        tbl = pa.table(
            {"x": pa.array(x, pa.list_(pa.string())), "y": pa.array(y, pa.list_(pa.string()))}
        )
        overlap = col("x").list.has_any(col("y"), propagate_nulls=True)
        assert _rows(bt.from_arrow(tbl).select(r=overlap)) == want


# --- temporal -----------------------------------------------------------------------------


def test_months_between_matches_the_documented_examples():
    ds = bt.from_pydict(
        {"d1": [dt.datetime(1997, 2, 28, 10, 30)], "d2": [dt.datetime(1996, 10, 30)]}
    )
    assert _rows(ds.select(r=col("d1").dt.months_between("d2"))) == [3.94959677]
    assert _rows(ds.select(r=col("d2").dt.months_between("d1"))) == [-3.94959677]
    exact = _rows(ds.select(r=col("d1").dt.months_between("d2", round_off=False)))
    assert exact == [pytest.approx(3.9495967741935485, rel=1e-15)]


def test_months_between_is_whole_on_the_same_day_or_both_month_ends():
    # Spark's rule: equal days-of-month, or both last days of their months, ignore the
    # time of day and the 31-day fraction. 1997-02-28 and 1996-10-31 are both month ends.
    ds = bt.from_pydict(
        {
            "a": [
                dt.datetime(1997, 2, 28),
                dt.datetime(2024, 3, 15, 23),
                dt.datetime(2024, 3, 16),
                None,
            ],
            "b": [
                dt.datetime(1996, 10, 31),
                dt.datetime(2024, 1, 15),
                dt.datetime(2024, 1, 15),
                None,
            ],
        }
    )
    got = _rows(ds.select(r=col("a").dt.months_between("b")))
    assert got == [4.0, 2.0, round(2 + 1 / 31, 8), None]


def test_next_day_matches_the_documented_examples():
    ds = bt.from_pydict({"dt": [dt.date(2015, 7, 27), None]})
    assert _rows(ds.select(r=col("dt").dt.next_day("Sun"))) == [dt.date(2015, 8, 2), None]
    assert _rows(ds.select(r=col("dt").dt.next_day("Sat"))) == [dt.date(2015, 8, 1), None]
    # Landing on the same weekday moves a whole week: 2015-07-27 is a Monday.
    assert _rows(ds.select(r=col("dt").dt.next_day("MO"))) == [dt.date(2015, 8, 3), None]


def test_next_day_refuses_a_non_weekday():
    with pytest.raises(PlanError, match="weekday"):
        col("dt").dt.next_day("Funday")


def test_unix_date_is_partition_days():
    # builtin.py `unix_date`: 1970-01-02 is 1 and 2022-01-02 is 18994. Batcher already had
    # this function as the Iceberg day transform, so it is not given a second name.
    ds = bt.from_pydict({"dt": [dt.date(1970, 1, 2), dt.date(2022, 1, 2), None]})
    assert _rows(ds.select(r=bt.partition_days("dt"))) == [1, 18994, None]


# --- the SQL spelling agrees with the constructor ------------------------------------------


@pytest.mark.parametrize(
    ("query", "expr"),
    [
        ("pmod(a, b)", bt.pmod(col("a"), col("b"))),
        ("bit_get(a, 1)", bt.bit_get(col("a"), 1)),
        ("elt(a, 'x', 'y')", bt.elt(col("a"), bt.lit("x"), bt.lit("y"))),
        ("find_in_set('b', s)", col("s").str.find_in_set("b")),
        ("parse_url(u, 'QUERY', 'k')", col("u").str.parse_url("QUERY", "k")),
        ("array_append(xs, a)", col("xs").list.append(col("a"))),
        ("array_remove(xs, 2)", col("xs").list.remove(2)),
        ("arrays_overlap(xs, ys)", col("xs").list.has_any(col("ys"), propagate_nulls=True)),
        ("next_day(d, 'TU')", col("d").cast("date").dt.next_day("TU")),
        ("unix_date(d)", bt.partition_days(col("d"))),
        ("pi()", bt.pi()),
    ],
)
def test_the_sql_spelling_computes_what_the_constructor_does(query, expr):
    ds = bt.from_pydict(
        {
            "a": [-7, 2, None],
            "b": [3, -3, 0],
            "s": ["a,b", "b", None],
            "u": ["http://h/p?k=v&kk=w", None, "http://h"],
            "xs": [[1, 2], [None, 2], None],
            "ys": [[2], [3, None], [1]],
            "d": [dt.date(2015, 1, 14), dt.date(1969, 12, 31), None],
        }
    )
    via_sql = bt.sql(f"SELECT {query} AS r FROM t", dialect="spark", t=ds).to_pydict()
    assert via_sql == ds.select(r=expr).to_pydict()


# --- sql_expr -------------------------------------------------------------------------------


@pytest.fixture
def people():
    return bt.from_pydict({"x": [1, 2, None], "s": ["a", "B", None], "k": ["g", "g", "h"]})


def test_sql_expr_round_trips_to_the_expr_form(people):
    via_sql = people.select(bt.sql_expr("x + 1").alias("y")).to_pydict()
    assert via_sql == people.select((col("x") + 1).alias("y")).to_pydict()
    assert via_sql == {"y": [2, 3, None]}


def test_sql_expr_carries_its_alias_like_select_expr(people):
    got = people.select(bt.sql_expr("x * 2 AS doubled"), bt.sql_expr("lower(s) AS l"))
    assert got.to_pydict() == {"doubled": [2, 4, None], "l": ["a", "b", None]}


def test_sql_expr_builds_an_aggregate_for_agg(people):
    got = people.group_by("k").agg(bt.sql_expr("sum(x) AS total")).sort("k").to_pydict()
    assert got == {"k": ["g", "h"], "total": [3, None]}


def test_sql_expr_reads_a_dialect(people):
    got = people.select(bt.sql_expr("pmod(x, 2) AS m", dialect="spark"))
    assert got.to_pydict() == {"m": [1, 0, None]}


@pytest.mark.parametrize("text", ["x +", "a, b", "", "sum(("])
def test_an_invalid_expression_raises_plan_error(text):
    with pytest.raises(PlanError):
        bt.sql_expr(text)


@pytest.mark.parametrize(
    "text", ["SELECT x FROM t", "SELECT 1", "INSERT INTO t VALUES (1)", "DROP TABLE t"]
)
def test_a_full_query_is_refused(text):
    with pytest.raises(PlanError, match="one expression"):
        bt.sql_expr(text)


def test_a_subquery_is_refused():
    with pytest.raises(PlanError, match="subquery"):
        bt.sql_expr("x + (SELECT max(x) FROM t)")


def test_a_window_function_is_refused_with_a_plan_error():
    with pytest.raises(PlanError, match="sql_expr"):
        bt.sql_expr("row_number() OVER (ORDER BY x)")


def test_sql_expr_rejects_a_non_string():
    with pytest.raises(PlanError, match="SQL string"):
        bt.sql_expr(col("x"))


# --- call_function ------------------------------------------------------------------------


def test_call_function_reads_a_string_as_a_column_and_a_number_as_a_literal(people):
    got = people.select(r=bt.call_function("pmod", "x", 2, dialect="spark"))
    assert got.to_pydict() == {"r": [1, 0, None]}


def test_call_function_passes_a_lit_string_as_a_constant():
    ds = bt.from_pydict({"csv": ["a,b", "b,c", None]})
    got = ds.select(r=bt.call_function("find_in_set", bt.lit("b"), col("csv"), dialect="spark"))
    assert got.to_pydict() == {"r": [2, 1, None]}


def test_call_function_reaches_an_aggregate(people):
    # builtin.py `call_function`: `call_function("avg", col("id"))` is 2.0 over 1, 2, 3.
    ds = bt.from_pydict({"id": [1, 2, 3]})
    assert ds.agg(r=bt.call_function("avg", col("id"))).to_pydict() == {"r": [2.0]}


def test_call_function_matches_the_constructor_on_an_expression_argument(people):
    via_name = people.select(r=bt.call_function("upper", col("s").str.lower()))
    assert via_name.to_pydict() == people.select(r=col("s").str.lower().str.upper()).to_pydict()


@pytest.mark.parametrize("name", ["", "x; DROP TABLE t", "a b", "1abc"])
def test_call_function_refuses_a_non_identifier(name):
    with pytest.raises(PlanError, match="not a function name"):
        bt.call_function(name, col("x"))


def test_call_function_on_an_unknown_name_raises_plan_error():
    with pytest.raises(PlanError):
        bt.call_function("definitely_not_a_function", col("x"))
