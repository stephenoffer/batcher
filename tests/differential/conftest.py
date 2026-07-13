"""Differential-testing helpers.

The core correctness strategy (per the plan): run the same query through Batcher
and through a trusted oracle (DuckDB), and assert the results are equal. The
interpreter is deterministic and built on arrow's typed kernels, so any
divergence from DuckDB is a real bug — and once the JIT tiers land, each tier is
checked against this same oracle.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("batcher._native", reason="native engine not built")


def _normalize(table: pa.Table) -> list[tuple]:
    """Order-independent, type-tolerant view of a table for comparison.

    Rows are compared as tuples after sorting; integer/float that represent the
    same value compare equal (DuckDB may widen types).
    """
    cols = table.column_names
    rows = [tuple(_coerce(r[c]) for c in cols) for r in table.to_pylist()]
    return sorted(rows, key=lambda t: tuple((v is None, str(type(v)), v) for v in t))


#: Stand-in for NaN in a comparison. `nan != nan`, so a raw NaN in a result tuple makes the
#: comparison fail even when both engines agree — which silently made every NaN case
#: untestable, and left the float-key edges (where a real `-0.0` grouping bug lived) with no
#: differential coverage. SQL treats all NaNs as one value for grouping/equality, so a single
#: canonical sentinel is the right comparison semantics, not a fudge.
_NAN = "<nan>"


def _coerce(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return float(v)
    if isinstance(v, float):
        if v != v:  # NaN — any payload, any sign
            return _NAN
        if v == 0.0:  # -0.0 == 0.0 in SQL grouping, and they must compare equal here too
            return 0.0
        return round(v, 9)
    return v


def assert_same(batcher_table: pa.Table, duck_relation) -> None:
    """Assert a Batcher result equals a DuckDB relation (order-independent)."""
    duck_table = duck_relation.to_arrow_table()
    assert set(batcher_table.column_names) == set(duck_table.column_names), (
        f"column mismatch: {batcher_table.column_names} vs {duck_table.column_names}"
    )
    # Reorder DuckDB columns to match Batcher for tuple comparison.
    duck_table = duck_table.select(batcher_table.column_names)
    bat = _normalize(batcher_table)
    duck = _normalize(duck_table)
    assert bat == duck, f"\nBatcher: {bat}\nDuckDB:  {duck}"


def assert_same_ordered(batcher_table: pa.Table, duck_relation) -> None:
    """Assert equality preserving row order (for ORDER BY / LIMIT queries)."""
    duck_table = duck_relation.to_arrow_table().select(batcher_table.column_names)
    cols = batcher_table.column_names
    bat = [tuple(_coerce(r[c]) for c in cols) for r in batcher_table.to_pylist()]
    duck = [tuple(_coerce(r[c]) for c in cols) for r in duck_table.to_pylist()]
    assert bat == duck, f"\nBatcher: {bat}\nDuckDB:  {duck}"


def assert_tables_equal(actual: pa.Table, expected: pa.Table, *, ordered: bool = False) -> None:
    """Assert two Batcher results are equal — for comparing execution *paths* to each other.

    `assert_same` / `assert_same_ordered` compare against DuckDB; this compares Batcher to
    Batcher (`collect()` vs `collect(spill=True)` vs `iter_batches()`), which is how invariant
    #7 is checked. It goes through the same `_coerce` normalization, so a NaN compares equal to
    a NaN — a plain `to_pydict() ==` cannot express that (`nan != nan`) and silently reports a
    false mismatch on any float column carrying one.

    Args:
        actual: The table produced by the path under test.
        expected: The table produced by the oracle path.
        ordered: Whether row order is part of the contract (sorts) or not.
    """
    assert actual.column_names == expected.column_names, (
        f"column mismatch: {actual.column_names} vs {expected.column_names}"
    )
    a = [tuple(_coerce(r[c]) for c in actual.column_names) for r in actual.to_pylist()]
    e = [tuple(_coerce(r[c]) for c in expected.column_names) for r in expected.to_pylist()]
    if not ordered:
        key = lambda t: tuple((v is None, str(type(v)), str(v)) for v in t)  # noqa: E731
        a, e = sorted(a, key=key), sorted(e, key=key)
    assert a == e, f"\nactual:   {a}\nexpected: {e}"


@pytest.fixture
def duck():
    """A fresh in-memory DuckDB connection."""
    con = duckdb.connect()
    yield con
    con.close()
