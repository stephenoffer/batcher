"""string_agg / group_concat / array_agg vs DuckDB.

Without ORDER BY the element order is arrival-dependent (as in DuckDB), so results
are compared as multisets of elements per group, not as ordered strings.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {"g": [1, 1, 2, 2, 2], "name": ["a", "b", "c", "d", "e"], "v": [10, 20, 30, 40, 50]}
    )
    duck.register("t", tbl)
    return tbl


def _as_sets(rows, sep):
    out = []
    for r in rows:
        norm = {}
        for k, v in r.items():
            if isinstance(v, str):
                norm[k] = tuple(sorted(v.split(sep)))
            elif isinstance(v, list):
                norm[k] = tuple(sorted(v))
            else:
                norm[k] = v
        out.append(tuple(sorted(norm.items())))
    return sorted(out)


@pytest.mark.parametrize(
    "sql,sep",
    [
        ("SELECT g, string_agg(name, ',') s FROM t GROUP BY g", ","),
        ("SELECT g, string_agg(name, '-') s FROM t GROUP BY g", "-"),
        ("SELECT string_agg(name, ',') s FROM t", ","),
        ("SELECT g, group_concat(name, ',') s FROM t GROUP BY g", ","),
        ("SELECT g, string_agg(v, ',') s FROM t GROUP BY g", ","),
    ],
)
def test_string_agg(duck, t, sql, sep):
    got = bt.sql(sql, t=t).collect().to_pylist()
    exp = duck.sql(sql).to_arrow_table().to_pylist()
    assert _as_sets(got, sep) == _as_sets(exp, sep)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT g, array_agg(v) a FROM t GROUP BY g",
        "SELECT g, array_agg(name) a FROM t GROUP BY g",
    ],
)
def test_array_agg(duck, t, sql):
    got = bt.sql(sql, t=t).collect().to_pylist()
    exp = duck.sql(sql).to_arrow_table().to_pylist()
    assert _as_sets(got, ",") == _as_sets(exp, ",")


def test_string_agg_over_empty_is_null(duck):
    # string_agg over zero rows is NULL in DuckDB, not the empty string "". It lowers to
    # array_agg (empty -> NULL) + list-join, and join(NULL) is NULL, so the fix flows
    # through. Previously Batcher returned "".
    empty = pa.table({"name": pa.array([], type=pa.string())})
    duck.register("e", empty)
    sql = "SELECT string_agg(name, ',') s FROM e"
    got = bt.sql(sql, e=empty).collect().to_pylist()
    exp = duck.sql(sql).to_arrow_table().to_pylist()
    assert got == exp == [{"s": None}]


def test_string_agg_null_separator_is_null(duck, t):
    # An explicit NULL separator makes the whole aggregate NULL in DuckDB (concatenating
    # through a NULL delimiter is NULL). A SQL NULL parses to `exp.Null`, not `exp.Literal`,
    # so Batcher previously fell through to the ',' default and wrongly joined the values.
    sql = "SELECT g, string_agg(name, NULL) s FROM t GROUP BY g ORDER BY g"
    got = bt.sql(sql, t=t).collect().to_pylist()
    exp = duck.sql(sql).to_arrow_table().to_pylist()
    assert got == exp
    assert all(r["s"] is None for r in got)
