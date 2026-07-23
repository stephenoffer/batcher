"""Shared result-comparison helpers for the whole suite.

These live in a module of their own — rather than in `tests/differential/conftest.py`,
where they started — because a `conftest` is imported under the bare name ``conftest``.
Two directories each holding one means ``from conftest import assert_same`` binds to
whichever was imported *first*, so any pytest selection spanning `tests/differential`
and `tests/integration` (e.g. ``pytest tests/ -k sql``) resolved the wrong module and
failed with `ImportError`. A uniquely-named module is unambiguous from anywhere on
`sys.path`, which pytest guarantees for both `tests/` and each test's own directory.

The comparison semantics are the load-bearing part; see `assert_same` and `_coerce`.
`tests/differential/conftest.py` re-exports these names, so it stays the documented
home of the differential oracle and the `duck` fixture.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pyarrow as pa

__all__ = [
    "assert_same",
    "assert_same_ordered",
    "assert_tables_equal",
    "duck_materialize",
]

#: Stand-in for NaN in a comparison. `nan != nan`, so a raw NaN in a result tuple makes the
#: comparison fail even when both engines agree — which silently made every NaN case
#: untestable, and left the float-key edges (where a real `-0.0` grouping bug lived) with no
#: differential coverage. SQL treats all NaNs as one value for grouping/equality, so a single
#: canonical sentinel is the right comparison semantics, not a fudge.
_NAN = "<nan>"


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


def _rows(table: pa.Table, cols: list[str]) -> list[tuple]:
    """Every row as a coerced tuple, in table order."""
    return [tuple(_coerce(r[c]) for c in cols) for r in table.to_pylist()]


def _normalize(table: pa.Table) -> list[tuple]:
    """Order-independent, type-tolerant view of a table for comparison.

    Rows are compared as tuples after sorting; integer/float that represent the
    same value compare equal (DuckDB may widen types).
    """
    return sorted(
        _rows(table, table.column_names),
        key=lambda t: tuple(_sort_key(v) for v in t),
    )


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
    cols = batcher_table.column_names
    duck_table = duck_relation.to_arrow_table().select(cols)
    bat, duck = _rows(batcher_table, cols), _rows(duck_table, cols)
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
    a = _rows(actual, actual.column_names)
    e = _rows(expected, expected.column_names)
    if not ordered:
        key = lambda t: tuple((v is None, str(type(v)), str(v)) for v in t)  # noqa: E731
        a, e = sorted(a, key=key), sorted(e, key=key)
    assert a == e, f"\nactual:   {a}\nexpected: {e}"


def duck_materialize(con, name: str, table) -> None:
    """Register `table` as `name` by **copying it into DuckDB's own storage**.

    Use this instead of `con.register(name, table)` whenever the query compares a FLOAT
    column that may hold a NaN. Registering hands DuckDB an Arrow scan, and DuckDB pushes
    the filter *into* that scan, where it is evaluated with **IEEE** semantics — every
    comparison with NaN false. Its own executor instead ranks NaN above every number
    (`SELECT 'nan'::DOUBLE > 1` is `true`, and so is `'nan' = 'nan'`), which is its
    documented behavior and what Batcher matches. So on `WHERE f > 1` over `[1.5, NaN]` the
    same DuckDB answers `[1.5]` through a registered Arrow table and `[1.5, NaN]` through a
    real one — measured on duckdb 1.5.4.

    That makes a registered Arrow table an unreliable oracle for exactly the values these
    tests exist to pin. Copying to a real table removes the Arrow scan, so the comparison
    runs in DuckDB's executor and the oracle states DuckDB's actual semantics.

    (Signed zero is unaffected — both paths agree `-0.0 = 0.0` — so the ordinary
    `register` is fine for a float column without NaN.)
    """
    con.register(f"_arrow_{name}", table)
    con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "_arrow_{name}"')
    con.unregister(f"_arrow_{name}")
