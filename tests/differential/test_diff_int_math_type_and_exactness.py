"""Unary math on an `Int64` column: which functions stay integral, and which are exact.

Two properties travel together here and only one of them is visible to the usual harness.

**Exactness.** A function whose result on an integer *is* that integer must not round-trip
through `f64`, which silently drops the low bit above 2**53. `round` was fixed for this; the
identical bug sat in `trunc` immediately beside it — ``trunc(2**53 + 1)`` returned
``2**53`` — and survived because nothing compared the two.

**Type.** `sign` returned `Float64` for an integer column where DuckDB returns an integer type
and this engine's own `abs`/`round` preserve. The *values* were right, since -1, 0 and 1 are
exact in a double. That is exactly why no differential test caught it: `assert_same` is
int/float tolerant by design, so a column typed `double` instead of `int64` compares equal to
one that is correct. This module therefore asserts the **schema** as well as the values.

`floor` and `ceil` are the negative control. They genuinely do yield a double on an integer,
in DuckDB too, so a fix that swept them up would be wrong.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_TWO53 = 2**53
_VALUES = [_TWO53 + 1, _TWO53 + 3, 2**62 + 7, -(_TWO53) - 1, -7, 0, 9]

#: function -> whether an Int64 input keeps an integral output.
_INTEGRAL = {"abs": True, "round": True, "trunc": True, "sign": True, "floor": False, "ceil": False}


@pytest.fixture(scope="module")
def table() -> pa.Table:
    return pa.table({"x": pa.array(_VALUES, pa.int64())})


@pytest.fixture(scope="module")
def session(table):
    s = bt.Session()
    s.register("t", bt.from_arrow(table))
    return s


@pytest.mark.parametrize("fn,integral", sorted(_INTEGRAL.items()))
def test_the_output_type_matches_duckdb(fn, integral, session, table):
    """int-in/int-out for the integral family, double for floor and ceil — both engines."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.register("t", table)
    got = session.sql(f"SELECT {fn}(x) AS r FROM t").collect().schema.field(0).type
    want = str(con.sql(f"SELECT {fn}(x) AS r FROM t").types[0]).upper()
    if integral:
        assert pa.types.is_integer(got), f"{fn}(int64) came back {got}, expected an integer"
        assert "INT" in want, f"DuckDB no longer returns an integer for {fn}; revisit"
    else:
        assert pa.types.is_floating(got), f"{fn}(int64) came back {got}, expected a float"
        assert "DOUBLE" in want or "FLOAT" in want


@pytest.mark.parametrize("fn", ["abs", "round", "trunc"])
def test_the_integral_functions_are_exact_above_2_pow_53(fn, session, table):
    """The values an f64 round-trip would have corrupted.

    Compared as Python ints against DuckDB's own answer, so a float that merely *prints*
    right cannot pass.
    """
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.register("t", table)
    tbl = session.sql(f"SELECT {fn}(x) AS r FROM t").collect()
    got = [tbl.column(0)[i].as_py() for i in range(tbl.num_rows)]
    want = [r[0] for r in con.sql(f"SELECT {fn}(x) AS r FROM t").fetchall()]
    assert got == want
    assert all(isinstance(v, int) for v in got), f"{fn} produced a non-integer: {got}"


def test_trunc_of_an_integer_is_that_integer(session):
    """Stated without reference to DuckDB, because it is arithmetic rather than a convention."""
    tbl = session.sql("SELECT x, trunc(x) AS t FROM t").collect()
    xs = [tbl.column(0)[i].as_py() for i in range(tbl.num_rows)]
    ts = [tbl.column(1)[i].as_py() for i in range(tbl.num_rows)]
    assert xs == ts


def test_sign_is_minus_one_zero_or_one(session):
    """And a negative control that the column is not simply all zeros."""
    tbl = session.sql("SELECT sign(x) AS s FROM t").collect()
    vals = [tbl.column(0)[i].as_py() for i in range(tbl.num_rows)]
    assert set(vals) <= {-1, 0, 1}
    assert {-1, 0, 1} <= set(vals), "the fixture should exercise all three signs"
