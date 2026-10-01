"""Scalar filter / expression / string kernels vs DuckDB, over data that shards.

Each case here is a typed fast path in `bc-expr` (or the stacked-filter path in `bc-interp`)
that declines to a generic kernel on anything it cannot answer bit-for-bit, so the cases are
chosen to sit on both sides of every such boundary:

* calendar fields of dates inside and outside the 1900..2100 lookup table, across leap
  rules, before 1970 and far in the future;
* a fused `lo <= x <= hi` range over integers, dates (bounded by strings, too) and floats
  with NaN, `-0.0` and infinities;
* integer and date `IN` lists whose span is just inside and just past the bitmap limit,
  and short-string `IN` lists, including members of every length up to eight bytes;
* `x <> ''` / `x = ''`, answered from offsets alone;
* `LENGTH`/`UPPER`/`LOWER`/`SUBSTRING` on ASCII columns (whole-column kernels) and on
  multibyte ones (the per-row path), with empty strings and nulls;
* `NULLIF` against a literal and `COALESCE` with a literal fallback.

The table has 100,000 rows -- above `MIN_ROWS_TO_SHARD` (65,536) -- so the parallel
executor genuinely splits it into morsels, and each morsel is a slice of the column, which
is where an offset-based string kernel goes wrong if it reads the shared buffer.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

N = 100_000
EPOCH = dt.date(1970, 1, 1)

#: Days spanning 1500..2500 so both sides of the 1900..2100 table are hit, plus every day
#: around the century leap rules (1600, 1700, 1900, 2000, 2100) and the epoch.
_ANCHORS = [dt.date(y, 2, 27) for y in (1600, 1700, 1900, 2000, 2100, 2400)] + [
    dt.date(1969, 12, 30),
    dt.date(1899, 12, 30),
    dt.date(2099, 12, 30),
]


def _day(i: int) -> dt.date | None:
    if i % 17 == 0:
        return None
    if i < 9 * 7:
        return _ANCHORS[i // 7] + dt.timedelta(days=i % 7)
    # A stride coprime to the range walks 1500-01-01 .. ~2500 without clustering.
    return dt.date(1500, 1, 1) + dt.timedelta(days=(i * 7919) % 365_000)


_FLOATS = [0.0, -0.0, float("nan"), float("inf"), float("-inf"), 1.5, -2.25, 0.05, 0.07]
_ASCII = ["", "a", "MAIL", "SHIP", "AIR", "RAIL", "REG AIR", "TRUCKS12", "TRUCKS123", "the cat"]
#: No `ß`: Batcher upper-cases it to `SS` (Unicode full case mapping, as Spark and Polars do)
#: where DuckDB keeps `ß` (single-code-point mapping). That predates the ASCII kernels -- a
#: non-ASCII column never reaches them -- and is a semantic choice, not something this file tests.
_UNICODE = ["é", "日本語", "naïve café", "", "Ǆx", "plain", "ÅÉÎ"]


@pytest.fixture(scope="module")
def table() -> pa.Table:
    i = pa.array([None if k % 13 == 0 else (k * 31) % 20_011 - 10_000 for k in range(N)])
    return pa.table(
        {
            "k": pa.array(range(N), type=pa.int64()),
            "i": i,
            "d": pa.array([_day(k) for k in range(N)], type=pa.date32()),
            "f": pa.array(
                [
                    None if k % 11 == 0 else _FLOATS[k % len(_FLOATS)] * (1 + k % 3)
                    for k in range(N)
                ],
                type=pa.float64(),
            ),
            "s": pa.array(
                [None if k % 19 == 0 else _ASCII[(k * 7) % len(_ASCII)] for k in range(N)],
                type=pa.string(),
            ),
            "u": pa.array(
                [
                    None if k % 23 == 0 else _UNICODE[k % len(_UNICODE)] + str(k % 5)
                    for k in range(N)
                ],
                type=pa.string(),
            ),
        }
    )


@pytest.fixture(scope="module")
def session(table: pa.Table) -> bt.Session:
    s = bt.Session()
    s.register("t", table)
    return s


@pytest.fixture
def duck_t(duck, table):
    duck.register("t", table)
    return duck


DATE_PARTS = [
    "SELECT EXTRACT(YEAR FROM d) AS y, COUNT(*) AS n FROM t GROUP BY 1",
    "SELECT EXTRACT(MONTH FROM d) AS m, EXTRACT(DAY FROM d) AS dd, COUNT(*) AS n "
    "FROM t GROUP BY 1, 2",
    "SELECT EXTRACT(QUARTER FROM d) AS q, COUNT(*) AS n FROM t GROUP BY 1",
    "SELECT k, EXTRACT(YEAR FROM d) AS y, EXTRACT(MONTH FROM d) AS m, EXTRACT(DAY FROM d) AS dd "
    "FROM t WHERE k < 200",
    "SELECT SUM(dayofyear(d)) AS s, COUNT(dayofyear(d)) AS c FROM t",
]


@pytest.mark.parametrize("sql", DATE_PARTS)
def test_calendar_fields_match_duckdb(session, duck_t, sql):
    assert_same(session.sql(sql).collect(), duck_t.sql(sql))


RANGES = [
    "SELECT COUNT(*) AS n, SUM(i) AS s FROM t WHERE i BETWEEN -500 AND 4000",
    "SELECT COUNT(*) AS n FROM t WHERE i > -500 AND i < 4000",
    "SELECT COUNT(*) AS n FROM t WHERE 4000 >= i AND -500 < i",
    "SELECT COUNT(*) AS n, MIN(k) AS lo FROM t "
    "WHERE d BETWEEN DATE '1899-12-31' AND DATE '2000-02-29'",
    "SELECT COUNT(*) AS n FROM t WHERE d >= '1969-12-31' AND d < '2100-03-01'",
    "SELECT COUNT(*) AS n FROM t WHERE f >= -0.0 AND f <= 3.0",
    "SELECT COUNT(*) AS n FROM t WHERE f > -1e308 AND f < 1e308",
    "SELECT COUNT(*) AS n FROM t WHERE f > 0.0 AND f <= 'inf'::DOUBLE",
    # A range beside other conjuncts: the short-circuit path pairs the bounds into one unit.
    "SELECT COUNT(*) AS n, SUM(f) AS s FROM t WHERE i < 9000 AND d >= DATE '1950-01-01' "
    "AND s <> '' AND d < DATE '2050-01-01' AND i > -9000",
    "SELECT COUNT(*) AS n FROM t WHERE i BETWEEN 4000 AND -500",
]


@pytest.mark.parametrize("sql", RANGES)
def test_fused_ranges_match_duckdb(session, duck_t, sql):
    assert_same(session.sql(sql).collect(), duck_t.sql(sql))


IN_LISTS = [
    "SELECT COUNT(*) AS n, SUM(k) AS s FROM t WHERE i IN (-10000, -1, 0, 7, 42, 9999)",
    # A span of 65,536 (just past the bitmap) and 65,535 (just inside it).
    "SELECT COUNT(*) AS n FROM t WHERE i IN (-10000, 55536)",
    "SELECT COUNT(*) AS n FROM t WHERE i IN (-10000, 55535, 3)",
    "SELECT COUNT(*) AS n FROM t "
    "WHERE d IN (DATE '1600-02-29', DATE '1970-01-01', DATE '2000-02-29')",
    "SELECT COUNT(*) AS n FROM t WHERE s IN ('MAIL', 'SHIP', 'AIR')",
    "SELECT COUNT(*) AS n FROM t WHERE s IN ('', 'a', 'TRUCKS12', 'REG AIR')",
    "SELECT COUNT(*) AS n FROM t WHERE s IN ('TRUCKS123', 'AIR')",
    "SELECT COUNT(*) AS n FROM t WHERE u IN ('é1', '日本語2', 'ß0')",
    # The planner adds an implied `i <= max` bound as its own filter beneath the IN list.
    "SELECT COUNT(*) AS n, SUM(k) AS q FROM t WHERE s IN ('MAIL', 'SHIP', 'AIR') "
    "AND i IN (1, 7, 42, 99, 123, 256, 512, 1024, 2048, 4096, 8192)",
]


@pytest.mark.parametrize("sql", IN_LISTS)
def test_in_lists_match_duckdb(session, duck_t, sql):
    assert_same(session.sql(sql).collect(), duck_t.sql(sql))


STRINGS = [
    "SELECT COUNT(*) AS n FROM t WHERE s <> ''",
    "SELECT COUNT(*) AS n FROM t WHERE s = ''",
    "SELECT COUNT(*) AS n FROM t WHERE '' < s",
    "SELECT COUNT(*) AS n FROM t WHERE u <> ''",
    "SELECT SUM(LENGTH(s)) AS a, SUM(LENGTH(u)) AS b, COUNT(LENGTH(s)) AS c FROM t",
    "SELECT UPPER(s) AS m, COUNT(*) AS n FROM t GROUP BY 1",
    "SELECT LOWER(s) AS m, COUNT(*) AS n FROM t GROUP BY 1",
    "SELECT UPPER(u) AS m, LOWER(u) AS l, COUNT(*) AS n FROM t GROUP BY 1, 2",
    "SELECT SUBSTRING(s, 1, 4) AS p, COUNT(*) AS n FROM t GROUP BY 1",
    "SELECT SUBSTRING(s, 3) AS p, SUBSTRING(s, 2, 2) AS q, COUNT(*) AS n FROM t GROUP BY 1, 2",
    "SELECT SUBSTRING(u, 2, 3) AS p, COUNT(*) AS n FROM t GROUP BY 1",
    "SELECT COUNT(*) AS n FROM t WHERE s LIKE 'TR%'",
]


@pytest.mark.parametrize("sql", STRINGS)
def test_string_kernels_match_duckdb(session, duck_t, sql):
    assert_same(session.sql(sql).collect(), duck_t.sql(sql))


CONDITIONALS = [
    "SELECT SUM(COALESCE(NULLIF(f, 0.0), 1.0)) AS s FROM t",
    "SELECT COUNT(NULLIF(f, -0.0)) AS c, COUNT(NULLIF(i, 7)) AS ci FROM t",
    "SELECT SUM(COALESCE(i, 5)) AS s, SUM(COALESCE(i, 0.5)) AS sf FROM t",
    "SELECT COUNT(*) AS n FROM t WHERE COALESCE(d, DATE '2000-01-01') = DATE '2000-01-01'",
]


@pytest.mark.parametrize("sql", CONDITIONALS)
def test_conditionals_match_duckdb(session, duck_t, sql):
    assert_same(session.sql(sql).collect(), duck_t.sql(sql))


def test_empty_input_matches_duckdb(duck, table):
    empty = table.slice(0, 0)
    s = bt.Session()
    s.register("t", empty)
    duck.register("t", empty)
    for sql in (RANGES[0], IN_LISTS[0], STRINGS[0], STRINGS[4], CONDITIONALS[0], DATE_PARTS[0]):
        assert_same(s.sql(sql).collect(), duck.sql(sql))
