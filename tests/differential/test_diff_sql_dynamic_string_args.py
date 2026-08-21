"""A string function's parameters may be columns, not only constants.

The engine's string kernels take ``pattern``/``replacement``/``start``/``length`` as
plan-time constants, which is what the `.str` namespace is typed for. SQL does not require
that, and roughly thirty spellings were refused outright with *"requires a constant string
pattern"* — ``replace(s, old_col, new_col)``, ``substr(s, from_col, len_col)``,
``s LIKE pattern_col`` and the rest.

`StrFuncDyn` carries each parameter as a sub-expression; the engine groups rows by their
distinct parameter tuple and runs the *same* kernel per group, so there is one definition
of each function's semantics rather than two that can drift. These cases are checked
against DuckDB with a parameter column that varies per row (including nulls and empties),
which is the only thing that separates the two forms.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _table() -> pa.Table:
    """Parameters that differ per row, so a constant-folded answer cannot pass."""
    return pa.table(
        {
            "s": pa.array(["abc", "aaa", "", "x-y-z", None, "Mixed"], pa.string()),
            "p": pa.array(["a", "b", "", "-", None, "M"], pa.string()),
            "q": pa.array(["Z", "y", "", "+", "Z", ""], pa.string()),
            "n": pa.array([1, 2, 0, 3, 1, 2], pa.int64()),
            "m": pa.array([2, 1, 3, 2, 1, 3], pa.int64()),
            "lk": pa.array(["a%", "b", "", "%-%", None, "M%"], pa.string()),
        }
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT replace(s, p, q) AS r FROM t",
        "SELECT substr(s, n) AS r FROM t",
        "SELECT substr(s, n, m) AS r FROM t",
        "SELECT substring(s, n, m) AS r FROM t",
        "SELECT left(s, n) AS r FROM t",
        "SELECT right(s, n) AS r FROM t",
        "SELECT repeat(s, n) AS r FROM t",
        "SELECT starts_with(s, p) AS r FROM t",
        "SELECT ends_with(s, p) AS r FROM t",
        "SELECT contains(s, p) AS r FROM t",
        "SELECT strpos(s, p) AS r FROM t",
        "SELECT instr(s, p) AS r FROM t",
        "SELECT position(p IN s) AS r FROM t",
        "SELECT split_part(s, p, n) AS r FROM t",
        "SELECT string_split(s, p) AS r FROM t",
        "SELECT trim(s, p) AS r FROM t",
        "SELECT ltrim(s, p) AS r FROM t",
        "SELECT rtrim(s, p) AS r FROM t",
        "SELECT levenshtein(s, p) AS r FROM t",
        "SELECT translate(s, p, q) AS r FROM t",
        "SELECT regexp_replace(s, p, q) AS r FROM t",
        "SELECT regexp_replace(s, p, q, 'g') AS r FROM t",
        "SELECT regexp_matches(s, p) AS r FROM t",
        "SELECT regexp_extract(s, p) AS r FROM t",
        "SELECT s LIKE lk AS r FROM t",
        "SELECT s ILIKE lk AS r FROM t",
        "SELECT s NOT LIKE lk AS r FROM t",
        "SELECT s SIMILAR TO p AS r FROM t",
        "SELECT concat_ws(p, s, q) AS r FROM t",
        "SELECT concat_ws(p, s, NULL, q) AS r FROM t",
        "SELECT round(CAST(1.23456 AS DOUBLE), CAST(n AS INTEGER)) AS r FROM t",
        "SELECT trunc(CAST(1.23456 AS DOUBLE), CAST(n AS INTEGER)) AS r FROM t",
    ],
)
def test_a_parameter_may_be_a_column(duck, sql):
    table = _table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT replace(s, 'a', 'Z') AS r FROM t",
        "SELECT substr(s, 2, 2) AS r FROM t",
        "SELECT left(s, 2) AS r FROM t",
        "SELECT left(s, -1) AS r FROM t",
        "SELECT right(s, 2) AS r FROM t",
        "SELECT lpad(s, 5, '-') AS r FROM t",
        "SELECT rpad(s, 5, '-') AS r FROM t",
        "SELECT trim(s, 'a') AS r FROM t",
        "SELECT s LIKE 'a%' AS r FROM t",
        "SELECT regexp_matches(s, '^a') AS r FROM t",
        "SELECT split_part(s, '-', 2) AS r FROM t",
    ],
)
def test_the_constant_form_is_unchanged(duck, sql):
    """The constant path must keep taking the constant kernel, unchanged."""
    table = _table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_a_null_parameter_makes_the_row_null():
    """SQL: a function of a NULL argument is NULL, whichever argument it is."""
    table = _table()
    got = bt.sql("SELECT replace(s, p, q) AS r FROM t", t=table).collect().to_pydict()
    assert got["r"][4] is None  # `p` is null on that row


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT replace(s, p, q) AS r FROM t",
        "SELECT substr(s, n, m) AS r FROM t",
        "SELECT s LIKE lk AS r FROM t",
    ],
)
def test_the_per_row_form_is_independent_of_partitioning(sql):
    """The evaluator groups rows by their parameter tuple *within a batch*.

    That grouping is an optimization, so it must not be visible: splitting the input into
    four partitions changes which rows share a group and must change nothing else.
    """
    table = _table()
    whole = bt.sql(sql, t=table)
    assert whole.collect().to_pydict() == whole.repartition(4).collect().to_pydict()
