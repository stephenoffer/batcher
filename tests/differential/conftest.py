"""Differential-testing helpers.

The core correctness strategy (per the plan): run the same query through Batcher
and through a trusted oracle (DuckDB), and assert the results are equal. The
interpreter is deterministic and built on arrow's typed kernels, so any
divergence from DuckDB is a real bug — and once the JIT tiers land, each tier is
checked against this same oracle.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("batcher._native", reason="native engine not built")


def _sort_key(v) -> tuple:
    """A total order over coerced values that sorts numbers by numeric *value*.

    Sorting by ``str(type(v))`` first (the old key) put ints before floats regardless of
    magnitude, so a column mixing the two — which now happens because integral values
    canonicalize to ``int`` and fractional ones stay ``float`` — sorted into a different
    order than an oracle result of a single numeric type, and the row-by-row comparison
    then failed on multisets that were actually equal. Numbers therefore share one bucket
    and sort by value; the exact ``repr`` breaks ties so two distinct large ints that share
    a float image still order deterministically and identically on both sides.
    """
    if v is None:
        return (0, "")
    if isinstance(v, bool):
        return (1, repr(v))
    if isinstance(v, (int, float)):
        f = v if math.isfinite(v) else (math.inf if v > 0 else -math.inf)
        return (2, float(f), repr(v))
    return (3, str(type(v)), str(v))


def _normalize(table: pa.Table) -> list[tuple]:
    """Order-independent, type-tolerant view of a table for comparison.

    Rows are compared as tuples after sorting; integer/float that represent the
    same value compare equal (DuckDB may widen types).
    """
    cols = table.column_names
    rows = [tuple(_coerce(r[c]) for c in cols) for r in table.to_pylist()]
    return sorted(rows, key=lambda t: tuple(_sort_key(v) for v in t))


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
        # Keep integers EXACT. Coercing to float (the old behaviour) collapsed any two
        # int64 values that share a float64 image — e.g. 2^53 and 2^53+1 — so a differential
        # test over large integers could not see an off-by-one. The int/float divide is
        # bridged from the float side below (integral floats canonicalize to int), never by
        # degrading the int side.
        return v
    if isinstance(v, (float, Decimal)):
        if isinstance(v, float) and v != v:  # NaN — any payload, any sign
            return _NAN
        if not math.isfinite(v):  # ±inf: keep as float, never int() it
            return float(v)
        r = round(float(v), 9)
        # Canonicalize every integral value (float or DuckDB Decimal) to int: it makes int↔
        # float↔decimal widening compare equal (1 vs 1.0 vs Decimal('1.0')), and it must be
        # *uniform* — including ±0.0 → int 0 — or a column mixes numeric types across rows.
        # A genuinely fractional value (1.5) stays float and still differs from an int.
        if r == int(r):  # True for -0.0/0.0 too (-0.0 == 0), unifying signed zero
            return int(r)
        return r
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
