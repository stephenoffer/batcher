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


def _coerce(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return float(v)
    if isinstance(v, float):
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


@pytest.fixture
def duck():
    """A fresh in-memory DuckDB connection."""
    con = duckdb.connect()
    yield con
    con.close()
