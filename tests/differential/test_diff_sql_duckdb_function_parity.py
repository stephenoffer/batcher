"""DuckDB function-name parity for the SQL front-end — vs DuckDB.

Every case here is a DuckDB function the engine already implemented but `bt.sql` could
not reach: the translator had no row for the sqlglot node (`Cot`, `Levenshtein`, `MD5`),
or the name arrived as `exp.Anonymous` and was rejected outright (`product`, `sem`,
`gcd`, `century`), or — for the aggregates sqlglot does not model — was never even seen
as an aggregate. A differential census over all 478 DuckDB scalar/aggregate functions
put the supported count at 93; these are the 99 that closed.

The arguments are **columns**, not literals, on purpose: a constant-only query can be
answered by the optimizer's constant folding without the runtime kernel ever running, so
a literal-argument test would pass while the engine path stayed broken.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

# `f(col)` scalar calls over the numeric/string/date columns of `t`.
SCALAR_QUERIES = [
    # Math builtins sqlglot promotes to typed nodes that had no mapping.
    "SELECT atan(f) r FROM t",
    "SELECT asin(u) r FROM t",
    "SELECT acos(u) r FROM t",
    "SELECT sinh(f) r FROM t",
    "SELECT cosh(f) r FROM t",
    "SELECT tanh(f) r FROM t",
    "SELECT asinh(f) r FROM t",
    "SELECT acosh(f + 1) r FROM t",
    "SELECT atanh(u) r FROM t",
    "SELECT cot(f) r FROM t",
    "SELECT factorial(i::INTEGER) r FROM t",
    "SELECT atan2(f, u) r FROM t",
    # Math builtins that arrive anonymous.
    "SELECT bit_count(i) r FROM t",
    "SELECT gcd(i, j) r FROM t",
    "SELECT lcm(i, j) r FROM t",
    "SELECT greatest_common_divisor(i, j) r FROM t",
    "SELECT least_common_multiple(i, j) r FROM t",
    "SELECT isfinite(f) r FROM t",
    # The DuckDB function spellings of the arithmetic operators.
    "SELECT add(i, j) r FROM t",
    "SELECT subtract(i, j) r FROM t",
    "SELECT multiply(i, j) r FROM t",
    "SELECT divide(i, j) r FROM t",
    # String builtins.
    "SELECT md5(s) r FROM t",
    "SELECT sha1(s) r FROM t",
    "SELECT sha256(s) r FROM t",
    "SELECT hex(s) r FROM t",
    "SELECT base64(s::BLOB) r FROM t",
    "SELECT bit_length(s) r FROM t",
    "SELECT strlen(s) r FROM t",
    "SELECT unicode(s) r FROM t",
    "SELECT ord(s) r FROM t",
    "SELECT translate(s, 'ab', 'xy') r FROM t",
    "SELECT levenshtein(s, 'abc') r FROM t",
    "SELECT editdist3(s, 'abc') r FROM t",
    "SELECT damerau_levenshtein(s, 'abc') r FROM t",
    "SELECT jaro_similarity(s, 'abc') r FROM t",
    "SELECT jaro_winkler_similarity(s, 'abc') r FROM t",
    "SELECT split(s, 'b') r FROM t",
    "SELECT str_split(s, 'b') r FROM t",
    "SELECT string_split(s, 'b') r FROM t",
    "SELECT string_to_array(s, 'b') r FROM t",
    "SELECT regexp_extract_all(s, '[a-z]') r FROM t",
    "SELECT regexp_split_to_array(s, '[0-9]') r FROM t",
    # Date parts.
    "SELECT century(d) r FROM t",
    "SELECT decade(d) r FROM t",
    "SELECT millennium(d) r FROM t",
    "SELECT isoyear(d) r FROM t",
    "SELECT isodow(d) r FROM t",
    "SELECT weekday(d) r FROM t",
    "SELECT weekofyear(d) r FROM t",
    "SELECT dayname(d) r FROM t",
    "SELECT monthname(d) r FROM t",
    "SELECT yearweek(d) r FROM t",
    "SELECT epoch(ts) r FROM t",
    "SELECT epoch_ns(ts) r FROM t",
    "SELECT epoch_us(ts) r FROM t",
    "SELECT microsecond(ts) r FROM t",
    "SELECT millisecond(ts) r FROM t",
    "SELECT nanosecond(ts) r FROM t",
    "SELECT strftime(d, '%Y-%m') r FROM t",
    "SELECT make_date(i + 2000, 3, 5) r FROM t",
    # List builtins, both DuckDB prefixes.
    "SELECT list_extract(l, 2) r FROM t",
    "SELECT array_extract(l, 2) r FROM t",
    "SELECT list_element(l, 2) r FROM t",
    "SELECT list_unique(l) r FROM t",
    "SELECT array_unique(l) r FROM t",
    "SELECT list_intersect(l, [2, 3]) r FROM t",
    "SELECT array_intersect(l, [2, 3]) r FROM t",
    "SELECT list_reverse_sort(l) r FROM t",
    "SELECT array_reverse_sort(l) r FROM t",
]

# `agg(col)` over the whole table and per group. Each is a DuckDB aggregate the
# translator either had no row for, or (for the anonymous names) did not recognize as an
# aggregate at all — `SELECT product(x) FROM t` was not even treated as a grouped query.
AGG_QUERIES = [
    "SELECT product(f) r FROM t",
    "SELECT mean(f) r FROM t",
    "SELECT favg(f) r FROM t",
    "SELECT count_star() r FROM t",
    "SELECT stddev_pop(f) r FROM t",
    "SELECT var_pop(f) r FROM t",
    "SELECT skewness(f) r FROM t",
    "SELECT kurtosis(f) r FROM t",
    "SELECT approx_count_distinct(i) r FROM t",
    "SELECT count_if(f > 2) r FROM t",
    "SELECT countif(f > 2) r FROM t",
    "SELECT bit_and(i) r FROM t",
    "SELECT bit_or(i) r FROM t",
    "SELECT bit_xor(i) r FROM t",
    "SELECT corr(f, u) r FROM t",
    "SELECT covar_pop(f, u) r FROM t",
    "SELECT covar_samp(f, u) r FROM t",
    "SELECT arg_max(s, f) r FROM t",
    "SELECT arg_min(s, f) r FROM t",
    "SELECT argmax(s, f) r FROM t",
    "SELECT argmin(s, f) r FROM t",
    "SELECT max_by(s, f) r FROM t",
    "SELECT min_by(s, f) r FROM t",
    "SELECT regr_slope(u, f) r FROM t",
    "SELECT regr_intercept(u, f) r FROM t",
    "SELECT regr_r2(u, f) r FROM t",
    "SELECT regr_count(u, f) r FROM t",
    "SELECT regr_avgx(u, f) r FROM t",
    "SELECT regr_avgy(u, f) r FROM t",
    "SELECT regr_sxx(u, f) r FROM t",
    "SELECT regr_sxy(u, f) r FROM t",
    "SELECT regr_syy(u, f) r FROM t",
]

# The same aggregates under a GROUP BY: the ungrouped and grouped paths register
# aggregates differently, and the anonymous names were invisible to both.
GROUPED_QUERIES = [
    "SELECT g, product(f) r FROM t GROUP BY g",
    "SELECT g, mean(f) r FROM t GROUP BY g",
    "SELECT g, count_star() r FROM t GROUP BY g",
    "SELECT g, var_pop(f) r FROM t GROUP BY g",
    "SELECT g, stddev_pop(f) r FROM t GROUP BY g",
    "SELECT g, corr(f, u) r FROM t GROUP BY g",
    "SELECT g, arg_max(s, f) r FROM t GROUP BY g",
    "SELECT g, regr_slope(u, f) r FROM t GROUP BY g",
    "SELECT g, bit_xor(i) r FROM t GROUP BY g",
    "SELECT g, count_if(f > 2) r FROM t GROUP BY g",
    # An aggregate inside a larger scalar expression must still be hoisted.
    "SELECT g, product(f) + 1 r FROM t GROUP BY g",
]


@pytest.fixture
def table(duck):
    """A table with one column per argument shape the queries need.

    Every group has at least two rows so the sample statistics (`sem`, `corr`,
    `regr_*`) are defined; without that they are NULL in Batcher and NaN in DuckDB,
    which is a genuine divergence but not the one under test here.
    """
    t = pa.table(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "i": [6, 4, 9, 12, 8, 3],
            "j": [4, 6, 12, 8, 3, 9],
            "f": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "u": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "s": ["abc", "abd", "bcd", "xyz", "aab", "ba"],
            "d": pa.array(
                ["2024-03-05", "2023-12-31", "2024-01-01", "2022-06-15", "2021-02-28", "2020-01-02"]
            ).cast(pa.date32()),
            "ts": pa.array(
                [
                    "2024-03-05 06:07:08.123456",
                    "2023-12-31 23:59:59.999999",
                    "2024-01-01 00:00:00.000001",
                    "2022-06-15 12:30:45.500000",
                    "2021-02-28 01:02:03.040506",
                    "2020-01-02 11:22:33.445566",
                ]
            ).cast(pa.timestamp("us")),
            "l": [[1, 2, 2], [3, 1], [2], [5, 4, 4], [1], [2, 3]],
        }
    )
    duck.register("t", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize("query", SCALAR_QUERIES)
def test_scalar_function_matches_duckdb(duck, table, query):
    assert_same(bt.sql(query, t=table).collect(), duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize("query", AGG_QUERIES)
def test_aggregate_matches_duckdb(duck, table, query):
    assert_same(bt.sql(query, t=table).collect(), duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize("query", GROUPED_QUERIES)
def test_grouped_aggregate_matches_duckdb(duck, table, query):
    assert_same(bt.sql(query, t=table).collect(), duck.sql(query))


@pytest.mark.differential
def test_list_unique_is_the_distinct_count_not_the_distinct_list(duck, table):
    """`list_unique` counts; `list_distinct` collects. Both used to return the list."""
    unique = "SELECT list_unique(l) r FROM t"
    distinct = "SELECT list_distinct(l) r FROM t"
    assert_same(bt.sql(unique, t=table).collect(), duck.sql(unique))
    got = bt.sql(distinct, t=table).to_pydict()["r"]
    assert [sorted(x) for x in got] == [[1, 2], [1, 3], [2], [4, 5], [1], [2, 3]]


@pytest.mark.differential
def test_list_reverse_sort_is_descending(duck, table):
    """`list_reverse_sort` carries `asc=False`, which used to be dropped."""
    query = "SELECT list_reverse_sort(l) r FROM t"
    assert_same(bt.sql(query, t=table).collect(), duck.sql(query))
    assert bt.sql(query, t=table).to_pydict()["r"][0] == [2, 2, 1]


@pytest.mark.differential
def test_sem_uses_the_sample_stddev_where_duckdb_uses_the_population_one(duck, table):
    """A deliberate, pinned divergence: `sem` is `stddev / sqrt(n)`, sample stddev.

    DuckDB's `sem` divides the *population* stddev, so it reports a smaller number for
    the same data. The standard definition of the standard error of the mean uses the
    sample standard deviation, which is what `bt.sem`, `pandas.Series.sem` and
    `scipy.stats.sem` all compute — so the SQL spelling stays consistent with the
    DataFrame spelling rather than tracking DuckDB. Reaching `sem` from SQL at all is
    new; this test exists so the choice cannot drift unnoticed in either direction.
    """
    query = "SELECT sem(f) r FROM t"
    got = bt.sql(query, t=table).to_pydict()["r"][0]
    expected_sample = 1.8708286933869707 / 6**0.5  # stddev_samp(f) / sqrt(n)
    assert got == pytest.approx(expected_sample)
    duck_answer = duck.sql(query).fetchall()[0][0]
    assert duck_answer == pytest.approx(expected_sample * (5 / 6) ** 0.5)


@pytest.mark.differential
def test_sha2_rejects_a_digest_width_it_cannot_produce(table):
    """Only sha256 is implemented, so `sha2(s, 512)` must raise, not answer with sha256."""
    with pytest.raises(NotImplementedError, match="digest length"):
        bt.sql("SELECT sha2(s, 512) r FROM t", t=table).collect()
