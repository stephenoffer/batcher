"""`IN (literal, …)` folded to the hash-set membership kernel matches DuckDB.

Covers the `fold_in_list` rule + the `eval_in_list` kernel: long lists fold to an
`InList`, short lists stay as comparisons, and both agree with DuckDB across int /
string / date values, nulls, and the empty/non-matching cases.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa

import batcher as bt
from _harness import assert_same


def _session_and_duck(duck, table: pa.Table) -> bt.Session:
    s = bt.Session()
    s.register("t", table)
    duck.register("t", table)
    return s


def test_int_in_list_large_with_nulls(duck):
    t = pa.table({"x": [1, 2, 3, 5, 8, 13, 21, None, 5, 1], "g": list(range(10))})
    s = _session_and_duck(duck, t)
    q = "SELECT x, g FROM t WHERE x IN (1, 5, 8, 13, 21, 99)"  # 6 values → folds
    assert_same(s.sql(q).collect(), duck.sql(q))


def test_string_in_list(duck):
    codes = ["13", "31", "23", "29", "30", "18", "17"]
    t = pa.table({"c": ["13", "31", "00", "17", None, "29", "99", "30", "23", "18"]})
    s = _session_and_duck(duck, t)
    vals = ", ".join(f"'{c}'" for c in codes)
    q = f"SELECT c FROM t WHERE c IN ({vals})"
    assert_same(s.sql(q).collect(), duck.sql(q))


def test_date_in_list(duck):
    days = [dt.date(1995, 1, d) for d in range(1, 11)]
    t = pa.table({"d": pa.array(days, pa.date32())})
    s = _session_and_duck(duck, t)
    q = "SELECT d FROM t WHERE d IN (DATE '1995-01-02', DATE '1995-01-04', DATE '1995-01-06', "
    q += "DATE '1995-01-08', DATE '1995-01-10')"
    assert_same(s.sql(q).collect(), duck.sql(q))


def test_small_in_list_unfolded(duck):
    t = pa.table({"x": [1, 2, 3, 4, 5, None]})
    s = _session_and_duck(duck, t)
    q = "SELECT x FROM t WHERE x IN (2, 4)"  # 2 values → stays a comparison chain
    assert_same(s.sql(q).collect(), duck.sql(q))


def test_substring_in_list_folds_over_a_computed_target(duck):
    """The TPC-H q22 shape: the membership target is `substring(...)`, not a column.

    Folding here removes k-1 *evaluations of the substring*, so it fires from two members
    on. The result must be what the unfolded comparison chain (and DuckDB) produce,
    including for the null phone.
    """
    t = pa.table({"phone": ["13-111", "31-222", "99-333", None, "17-444", "23-555"]})
    s = _session_and_duck(duck, t)
    q = "SELECT phone FROM t WHERE substring(phone FROM 1 FOR 2) IN ('13', '17', '23')"
    assert_same(s.sql(q).collect(), duck.sql(q))


def test_two_member_computed_target_matches_unfolded(duck):
    t = pa.table({"s": ["ab", "cd", "ef", None, "AB"]})
    s = _session_and_duck(duck, t)
    q = "SELECT s FROM t WHERE upper(s) IN ('AB', 'EF')"
    assert_same(s.sql(q).collect(), duck.sql(q))


def test_timestamp_column_against_date_literals(duck):
    """A date-literal list over a *timestamp* column — the shape whose folded form has no
    typed kernel arm and falls back to the generic membership path."""
    stamps = [dt.datetime(1995, 1, d) for d in range(1, 11)]
    t = pa.table({"ts": pa.array(stamps, pa.timestamp("us"))})
    s = _session_and_duck(duck, t)
    q = "SELECT ts FROM t WHERE ts IN (DATE '1995-01-02', DATE '1995-01-04', "
    q += "DATE '1995-01-06', DATE '1995-01-08', DATE '1995-01-10')"
    assert_same(s.sql(q).collect(), duck.sql(q))


def test_in_list_combined_with_other_predicate(duck):
    t = pa.table({"x": [1, 5, 8, 13, 21, 2, 7], "y": [10, 20, 30, 40, 50, 60, 70]})
    s = _session_and_duck(duck, t)
    q = "SELECT x, y FROM t WHERE x IN (1, 5, 8, 13, 21, 99) AND y < 45"
    assert_same(s.sql(q).collect(), duck.sql(q))
