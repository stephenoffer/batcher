"""`IN (literal, …)` folded to the hash-set membership kernel matches DuckDB.

Covers the `fold_in_list` rule + the `eval_in_list` kernel: long lists fold to an
`InList`, short lists stay as comparisons, and both agree with DuckDB across int /
string / date values, nulls, and the empty/non-matching cases.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa

import batcher as bt


def _session_and_duck(duck, table: pa.Table) -> bt.Session:
    s = bt.Session()
    s.register("t", table)
    duck.register("t", table)
    return s


def test_int_in_list_large_with_nulls(duck):
    from conftest import assert_same

    t = pa.table({"x": [1, 2, 3, 5, 8, 13, 21, None, 5, 1], "g": list(range(10))})
    s = _session_and_duck(duck, t)
    q = "SELECT x, g FROM t WHERE x IN (1, 5, 8, 13, 21, 99)"  # 6 values → folds
    assert_same(s.sql(q).collect(), duck.sql(q))


def test_string_in_list(duck):
    from conftest import assert_same

    codes = ["13", "31", "23", "29", "30", "18", "17"]
    t = pa.table({"c": ["13", "31", "00", "17", None, "29", "99", "30", "23", "18"]})
    s = _session_and_duck(duck, t)
    vals = ", ".join(f"'{c}'" for c in codes)
    q = f"SELECT c FROM t WHERE c IN ({vals})"
    assert_same(s.sql(q).collect(), duck.sql(q))


def test_date_in_list(duck):
    from conftest import assert_same

    days = [dt.date(1995, 1, d) for d in range(1, 11)]
    t = pa.table({"d": pa.array(days, pa.date32())})
    s = _session_and_duck(duck, t)
    q = "SELECT d FROM t WHERE d IN (DATE '1995-01-02', DATE '1995-01-04', DATE '1995-01-06', "
    q += "DATE '1995-01-08', DATE '1995-01-10')"
    assert_same(s.sql(q).collect(), duck.sql(q))


def test_small_in_list_unfolded(duck):
    from conftest import assert_same

    t = pa.table({"x": [1, 2, 3, 4, 5, None]})
    s = _session_and_duck(duck, t)
    q = "SELECT x FROM t WHERE x IN (2, 4)"  # 2 values → stays a comparison chain
    assert_same(s.sql(q).collect(), duck.sql(q))


def test_in_list_combined_with_other_predicate(duck):
    from conftest import assert_same

    t = pa.table({"x": [1, 5, 8, 13, 21, 2, 7], "y": [10, 20, 30, 40, 50, 60, 70]})
    s = _session_and_duck(duck, t)
    q = "SELECT x, y FROM t WHERE x IN (1, 5, 8, 13, 21, 99) AND y < 45"
    assert_same(s.sql(q).collect(), duck.sql(q))
