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
    "assert_same_for_query",
    "assert_same_ordered",
    "assert_tables_equal",
    "duck_materialize",
    "has_outermost_order_by",
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


#: Decimal places the float grid keeps for a value of order 1 or larger. This is the
#: absolute precision the comparison has always had; `_round_sig` only ever *adds* places
#: below that, so no comparison this suite already makes is loosened.
_ABS_DECIMALS = 9

#: Significant figures kept for a value below 1, where `_ABS_DECIMALS` alone would round
#: away the whole number.
_SIG_DIGITS = 9


def _round_sig(v: float) -> float:
    """Snap `v` onto a grid that is never coarser than `round(v, 9)` and never zeroes it.

    The grid used to be ``round(v, 9)`` alone: a fixed *absolute* width, which means a
    different proportion at every magnitude. Above 1 that is fine and is what the whole
    differential suite has been calibrated against. Below 1 it decays, and past 5e-10 it
    stops existing:

        >>> round(1e-10, 9), round(9e-11, 9)
        (0.0, 0.0)

    Those two differ by 10%, and the oracle could not tell them apart — nor either from a
    true zero, which is the shape a dropped term or a fully-cancelled sum actually takes.
    Every column whose values live near zero (a probability, a residual, a normalized
    score, `exp` of anything negative, the variance of near-constant data) was therefore
    compared against DuckDB at no precision at all, while reading as covered.

    So the grid keeps whichever of the two rules is *finer*: nine decimal places, or nine
    significant figures. For ``|v| >= 1`` that is the nine decimals it always was, so this
    tightens the comparison and can never loosen it.
    """
    if v == 0.0 or not math.isfinite(v):
        return v
    # `Decimal(repr(v)).adjusted()` is the decimal exponent without a `log10` round-trip,
    # so a value that is exactly a power of ten cannot be pushed into the next bucket by a
    # representation error.
    exponent = Decimal(repr(abs(v))).adjusted()
    return round(v, max(_ABS_DECIMALS, _SIG_DIGITS - 1 - exponent))


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
        r = _round_sig(float(v))
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
    """Assert a Batcher result equals a DuckDB relation, as a row multiset.

    Neither row order nor **column order** is checked here, and the second one is a
    deliberate reversal worth recording, because the obvious tightening is wrong.

    Making the column comparison positional looked free: the whole SQL slice of the suite
    (2,334 tests) passes unchanged under it, so every explicit `SELECT a, b` agrees with
    DuckDB on ordering. It then failed **49 join tests**, all one divergence:

        SELECT * FROM emp JOIN dept USING (dept_id)
        duckdb  -> ['id', 'dept_id', 'dept']      # the key keeps its left-table position
        batcher -> ['dept_id', 'id', 'dept']      # the key comes first

    Batcher is the one following the specification. SQL:2016 §7.7 puts a `USING` join's
    coalesced columns first, then the left table's remainder, then the right's, and
    PostgreSQL does the same. So a positional check here would not have been enforcing a
    contract — it would have been enforcing *DuckDB's* deviation from one, and the harness
    already records what that costs: on TPC-H q6 a comparator's wrong answer was reported
    against Batcher until the oracle was pinned down.

    The order of an explicit select list *is* a promise, and it is checked — in
    `assert_same_for_query`, which sees the query and can tell `SELECT a, b` from
    `SELECT *`. The divergence above is pinned in
    `tests/differential/test_diff_join_column_order.py` so a change to *Batcher's* ordering
    still fails something.
    """
    duck_table = duck_relation.to_arrow_table()
    assert set(batcher_table.column_names) == set(duck_table.column_names), (
        f"column mismatch: {batcher_table.column_names} vs {duck_table.column_names}"
    )
    # Reorder DuckDB's columns to match Batcher's for the tuple comparison.
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


def has_outermost_order_by(sql: str) -> bool:
    """Whether `sql` ends in an `ORDER BY` that is a promise about the result's row order.

    Only an `ORDER BY` at paren depth zero counts. A subquery's constrains nothing about
    the statement's result and every engine is free to discard it, which is the same
    distinction `benchmarks/harness/order.py` makes for the same reason.
    """
    depth = 0
    low = sql.lower()
    for i, char in enumerate(low):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and low.startswith("order by", i):
            return True
    return False


class _Materialized:
    """Wraps an already-read Arrow table in the tiny interface the assert helpers use.

    `assert_same` and `assert_same_ordered` take "a DuckDB relation" and call
    `to_arrow_table()` on it. Handing them this instead lets a caller that has already read
    the result pass it on without a second read.
    """

    __slots__ = ("_table",)

    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def to_arrow_table(self) -> pa.Table:
        return self._table


def _materialized(duck_relation) -> pa.Table:
    """The Arrow table behind a DuckDB relation, a cursor, or an already-read table."""
    if isinstance(duck_relation, pa.Table):
        return duck_relation
    return duck_relation.to_arrow_table()


def _selects_star(query: str) -> bool:
    """Whether the query's outermost projection includes a `*`, so it names no column order.

    A `*` has to be an entire select item to count. Merely containing one does not: the
    first version of this returned True for ``SELECT count(*) AS n`` and for ``SELECT a * b``,
    which would have silently switched the column-order check off for every aggregate and
    every multiplication in the suite — the check reporting itself as applied while applying
    to nothing.
    """
    lowered = query.lower()
    start = lowered.find("select")
    if start == -1:
        return True  # not a shape this can reason about; assume the weaker contract
    end = lowered.find(" from ", start)
    projection = query[start + len("select") : end if end != -1 else len(query)]

    items, depth, current = [], 0, []
    for char in projection:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            items.append("".join(current))
            current = []
            continue
        current.append(char)
    items.append("".join(current))
    return any(item.strip() == "*" or item.strip().endswith(".*") for item in items)


def assert_same_for_query(batcher_table: pa.Table, duck_relation, query: str) -> None:
    """Compare against DuckDB, checking row order exactly when `query` asked for one.

    A parametrized SQL test runs a *list* of queries through one assertion, and some of
    them end in `ORDER BY` while others do not. Picking `assert_same` for the whole list —
    which is what these tests did — means the ordered ones are compared as multisets, so
    the order they asked for is the one property never checked. Picking
    `assert_same_ordered` for the whole list is worse: it would demand an order from the
    queries that never asked for one, where any row order is correct and both engines are
    free to disagree.

    So the query decides, which also means a query *added* to one of those lists later
    gets the right comparison without anyone remembering to think about it. That is the
    point: the previous arrangement was correct only for as long as nobody appended an
    `ORDER BY` to the list, and three of them already had.

    Args:
        batcher_table: The result Batcher produced.
        duck_relation: The DuckDB relation for the same query.
        query: The SQL both ran. Read only for its outermost `ORDER BY`.
    """
    # Materialize ONCE and pass the table onward. `to_arrow_table()` is not idempotent for
    # every DuckDB handle: a *relation* from `duck.sql(...)` can be read repeatedly, but a
    # *cursor* from `duck.execute(...)` is consumed by the first read and answers `None`
    # afterwards. Calling it here and again inside `assert_same` therefore worked for the
    # three callers this helper shipped with and broke on the fourth — found by rewriting
    # 1,153 call sites in a sandbox and running them, which is the only reason it surfaced
    # before it was widely adopted rather than after.
    duck_table = _materialized(duck_relation)

    # An explicit select list states its own column order, so check it. `SELECT *` does not:
    # over a `USING` join the two engines legitimately disagree about where the key goes
    # (see `assert_same`), and holding a star to DuckDB's answer would enforce DuckDB's
    # deviation from SQL:2016 rather than any contract of ours.
    if not _selects_star(query):
        assert batcher_table.column_names == duck_table.column_names, (
            f"column ORDER mismatch for an explicit select list: "
            f"{batcher_table.column_names} vs {duck_table.column_names}"
        )
    if has_outermost_order_by(query):
        assert_same_ordered(batcher_table, _Materialized(duck_table))
    else:
        assert_same(batcher_table, _Materialized(duck_table))


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
