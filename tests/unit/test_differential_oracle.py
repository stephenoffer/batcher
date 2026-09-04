"""The differential oracle, tested. `tests/_harness.py` had no test of its own.

Every one of the ~2,800 tests under `tests/differential/` reaches its verdict through
`assert_same`, `assert_same_ordered` or `assert_tables_equal`, and through the `_coerce`
canonicalization beneath them. Nothing checked those. A bug there does not produce a
failure anywhere — it produces a *pass* everywhere, which is the single highest-leverage
false green this repository can have, and the one thing `just lint-tests` cannot see because
the tests it is judging all look fine.

So this file asserts the two halves an oracle needs, and the second matters more:

**What it must call equal.** DuckDB widens types, so `1`, `1.0` and `Decimal('1.0')` are one
value; NaN equals NaN because SQL groups them together; `-0.0` equals `0.0` for the same
reason. Each is a deliberate tolerance with a reason, and each is a place a stricter
comparison would produce false failures nobody could act on.

**What it must call different — the tests that prove the oracle can fail at all.** A helper
that always passes is indistinguishable from a working one until the day it matters. So
every tolerance above is paired with a case just outside it, and every ordering guarantee is
paired with a permutation that must be rejected.

Two of these are regression tests for defects the oracle actually had:

- `_coerce` used `round(v, 9)`, an *absolute* grid, so **every value below 5e-10 became 0**
  and any two of them compared equal. `1e-10` and `9e-11` differ by 10% and were
  indistinguishable from each other and from a true zero — the shape a dropped term or a
  fully-cancelled sum takes. Every column living near zero (a probability, a residual, a
  normalized score, the variance of near-constant data) was compared at no precision at all
  while reading as covered.
- `assert_same` compared column names as a **set** and then reordered DuckDB's columns to
  match Batcher's, so `SELECT a, b` answered as `b, a` passed, as did any rewrite that
  permuted a projection's output.
- `assert_same_for_query` read the DuckDB handle **twice** — once for the column-order check,
  then again inside `assert_same`. A relation from `duck.sql(...)` tolerates that; a cursor
  from `duck.execute(...)` is consumed by the first read and answers `None`.

## Why coverage could not have found the third one

That last defect is worth stating as a shape rather than a bug, because no coverage number
would have flagged it: **every line was executed, and the helper was correct for 100% of its
callers.** All three shipped callers passed `duck.sql(...)`, so the cursor form — which the
signature accepts and the docstring invites — was structurally unreachable from its own test
set. The test surface was narrower than the accepted input types, and line coverage measures
the first while saying nothing about the second.

It is the same error as a sample that feels like evidence and has the wrong shape. Tightening
`assert_same` to positional column order was validated against **2,334 passing SQL tests**,
which contained no `SELECT *` over a join — the only shape where the engines differ — and the
full suite then failed 49 of them. One is about callers, one is about queries; both are a
large green number that could not have gone red.

The defect surfaced only from rewriting 1,153 call sites in a throwaway sandbox and running
them, which is the argument for measuring a mechanical expansion rather than landing it: the
expansion itself found nothing and was dropped, and it still paid for itself.
"""

from __future__ import annotations

import math
import struct
from decimal import Decimal
from random import Random

import pyarrow as pa
import pytest

from _harness import (
    _coerce,
    _selects_star,
    _sort_key,
    assert_same,
    assert_same_for_query,
    assert_same_ordered,
    assert_tables_equal,
    has_outermost_order_by,
)

pytestmark = pytest.mark.unit


class _FakeRelation:
    """Stands in for a DuckDB relation, which the helpers only ever `to_arrow_table()`."""

    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def to_arrow_table(self) -> pa.Table:
        return self._table


def _t(**cols) -> pa.Table:
    return pa.table({k: pa.array(v) for k, v in cols.items()})


# --------------------------------------------------------------------------- #
# What the oracle must call EQUAL
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("left", "right", "why"),
    [
        (1, 1.0, "DuckDB widens an int64 sum to double"),
        (1, Decimal("1.0"), "DuckDB returns Decimal for a sum over integers"),
        (1.0, Decimal("1.00"), "trailing zeros carry no information"),
        (-0.0, 0.0, "SQL groups signed zeros together"),
        (2.0000000001, 2.0000000002, "agreement past the ninth decimal is float noise"),
    ],
)
def test_values_the_oracle_must_treat_as_one(left, right, why):
    assert _coerce(left) == _coerce(right), why


def test_nan_equals_nan_because_sql_groups_them():
    """`nan != nan` in IEEE, so a raw NaN in a result makes any tuple comparison fail.

    Without canonicalizing it, every NaN case in the suite would be untestable — which is
    how the float-key edges, where a real `-0.0` grouping bug lived, went uncovered.
    """
    assert _coerce(float("nan")) == _coerce(float("nan"))
    assert_same(_t(x=[float("nan")]), _FakeRelation(_t(x=[float("nan")])))


def test_row_order_is_not_part_of_an_unordered_comparison():
    """A query that asked for no order may return its rows in any order."""
    assert_same(_t(a=[1, 2, 3]), _FakeRelation(_t(a=[3, 1, 2])))


# --------------------------------------------------------------------------- #
# What the oracle must call DIFFERENT — the half that proves it can fail
# --------------------------------------------------------------------------- #


def test_a_wrong_row_is_rejected():
    """The baseline. If this ever passes, nothing else in this file means anything."""
    with pytest.raises(AssertionError):
        assert_same(_t(a=[1, 2, 3]), _FakeRelation(_t(a=[1, 2, 4])))


def test_a_missing_row_is_rejected():
    with pytest.raises(AssertionError):
        assert_same(_t(a=[1, 2, 3]), _FakeRelation(_t(a=[1, 2])))


def test_a_null_is_not_a_zero():
    with pytest.raises(AssertionError):
        assert_same(_t(a=[None, 1]), _FakeRelation(_t(a=[0, 1])))


@pytest.mark.parametrize(
    ("mine", "theirs"),
    [
        (1e-10, 9e-11),  # 10% apart, both were 0 under the old absolute grid
        (1.2e-12, 0.0),  # a cancelled sum against a true zero
        (-3e-10, 3e-10),  # opposite signs, both were 0
    ],
)
def test_small_magnitudes_are_still_distinguishable(mine, theirs):
    """Regression: an absolute `round(v, 9)` grid rounded all of these to zero.

    These are not exotic values. They are what a residual, a probability of a rare event, or
    `exp` of anything comfortably negative looks like, and the oracle could not see any
    difference between them.
    """
    assert _coerce(mine) != _coerce(theirs)
    with pytest.raises(AssertionError):
        assert_same(_t(x=[mine]), _FakeRelation(_t(x=[theirs])))


def test_a_permuted_explicit_projection_is_rejected():
    """`SELECT a, b` answered as `b, a` must fail — the order is the query's own promise."""
    with pytest.raises(AssertionError, match="column ORDER"):
        assert_same_for_query(
            _t(b=[1], a=[2]), _FakeRelation(_t(a=[2], b=[1])), "SELECT a, b FROM t"
        )


def test_a_star_projection_is_not_held_to_duckdbs_column_order():
    """The counterpart, and the reason the check is on the *query* rather than the tables.

    `SELECT * FROM emp JOIN dept USING (dept_id)` puts the key first in Batcher and leaves it
    in its left-table position in DuckDB. SQL:2016 §7.7 and PostgreSQL agree with Batcher, so
    a positional check over a star would enforce DuckDB's deviation rather than a contract —
    and it did: it failed 49 join tests, every one of them on this difference.
    """
    assert_same_for_query(
        _t(dept_id=[10], id=[1]), _FakeRelation(_t(id=[1], dept_id=[10])), "SELECT * FROM t"
    )


class _ConsumedOnce:
    """A DuckDB *cursor*: readable once, `None` thereafter. `duck.execute(...)` behaves so."""

    def __init__(self, table: pa.Table) -> None:
        self._table: pa.Table | None = table

    def to_arrow_table(self) -> pa.Table | None:
        table, self._table = self._table, None
        return table


@pytest.mark.parametrize(
    "query",
    ["SELECT a, b FROM t", "SELECT a, b FROM t ORDER BY a", "SELECT * FROM t"],
)
def test_the_oracle_handle_is_read_exactly_once(query):
    """Regression: `assert_same_for_query` read the relation twice and broke on a cursor.

    `to_arrow_table()` is not idempotent across DuckDB handle types — a relation from
    `duck.sql(...)` can be read repeatedly, a cursor from `duck.execute(...)` is consumed by
    the first read and answers `None` after. The helper checked column order from one read
    and then handed the *handle* to `assert_same`, which read it again.

    It worked for all three callers it shipped with, because all three used `duck.sql`. It
    was found by rewriting 1,153 call sites in a throwaway sandbox and running them, which
    turned a latent trap into ten loud failures before anyone adopted it — the reason to
    measure a mechanical expansion instead of landing it.
    """
    table = _t(a=[1, 2], b=[3, 4])
    assert_same_for_query(table, _ConsumedOnce(table), query)


def test_a_single_read_handle_still_rejects_a_permutation():
    """The control for the test above: reading once must not mean checking nothing."""
    with pytest.raises(AssertionError, match="column ORDER"):
        assert_same_for_query(
            _t(b=[3, 4], a=[1, 2]), _ConsumedOnce(_t(a=[1, 2], b=[3, 4])), "SELECT a, b FROM t"
        )


def test_only_an_explicit_select_list_carries_a_column_order():
    assert _selects_star("SELECT * FROM t")
    assert _selects_star("SELECT t.* FROM t")
    assert not _selects_star("SELECT a, b FROM t")
    assert not _selects_star("SELECT count(*) AS n FROM t"), (
        "a `*` inside an aggregate argument is not a star projection"
    )


def test_a_large_integer_is_not_collapsed_onto_its_float_image():
    """`2**53` and `2**53 + 1` share a float64 image; the oracle must not.

    Coercing integers to float (which it once did) made any off-by-one over a large int64
    invisible.
    """
    assert _coerce(2**53) != _coerce(2**53 + 1)
    with pytest.raises(AssertionError):
        assert_same(_t(n=[2**53]), _FakeRelation(_t(n=[2**53 + 1])))


# --------------------------------------------------------------------------- #
# Ordering: the guarantee `assert_same` deliberately does not make
# --------------------------------------------------------------------------- #


def test_assert_same_ordered_rejects_what_assert_same_accepts():
    """The pair that makes the ordered/unordered distinction real.

    `CLAUDE.md`'s loudest warning is that a spilled `descending` sort returned unsorted data
    while every gate passed, because the test compared the result as a multiset. This is
    that difference, asserted.
    """
    mine, theirs = _t(a=[1, 2, 3]), _t(a=[3, 1, 2])
    assert_same(mine, _FakeRelation(theirs))  # accepted: no order was promised
    with pytest.raises(AssertionError):
        assert_same_ordered(mine, _FakeRelation(theirs))


def test_assert_tables_equal_checks_order_only_when_asked():
    mine, theirs = _t(a=[1, 2, 3]), _t(a=[3, 1, 2])
    assert_tables_equal(mine, theirs)
    with pytest.raises(AssertionError):
        assert_tables_equal(mine, theirs, ordered=True)


@pytest.mark.parametrize(
    ("sql", "ordered"),
    [
        ("SELECT a FROM t", False),
        ("SELECT a FROM t ORDER BY a", True),
        ("SELECT a FROM t ORDER BY a DESC LIMIT 2", True),
        # A subquery's order constrains nothing about the statement's result.
        ("SELECT a FROM (SELECT a FROM t ORDER BY a)", False),
        ("SELECT (SELECT max(b) FROM u ORDER BY b) AS m FROM t", False),
    ],
)
def test_only_an_outermost_order_by_is_a_promise(sql, ordered):
    assert has_outermost_order_by(sql) is ordered


def test_assert_same_for_query_dispatches_on_the_query():
    """The helper must be strict exactly when the query asked for an order, and no more."""
    mine, shuffled = _t(a=[1, 2, 3]), _t(a=[3, 1, 2])
    assert_same_for_query(mine, _FakeRelation(shuffled), "SELECT a FROM t")
    with pytest.raises(AssertionError):
        assert_same_for_query(mine, _FakeRelation(shuffled), "SELECT a FROM t ORDER BY a")


# --------------------------------------------------------------------------- #
# Totality: `_coerce` runs on every value of every comparison
# --------------------------------------------------------------------------- #


def test_coerce_is_total_and_sortable_over_arbitrary_doubles():
    """Fuzzed over random bit patterns, so subnormals, infinities and NaNs all appear.

    `_coerce` sits on the hot path of every differential assertion and `_sort_key` orders
    whatever it returns. Either raising would turn a correctness failure into a confusing
    error inside the harness, which is a worse outcome than a plain mismatch.
    """
    rng = Random(0)
    values = [
        0.0,
        -0.0,
        5e-324,  # the smallest subnormal
        1e-308,
        1.7976931348623157e308,
        -1.7976931348623157e308,
        float("inf"),
        float("-inf"),
        float("nan"),
        Decimal("1E+30"),
        Decimal("1E-30"),
    ]
    values += [struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0] for _ in range(2_000)]

    coerced = [_coerce(v) for v in values]
    sorted(coerced, key=_sort_key)  # must be a total order over the mixed result types

    # And the property the small-magnitude regression turns on, over the whole fuzz corpus:
    # no finite nonzero may become zero.
    collapsed = [
        v
        for v, c in zip(values, coerced, strict=True)
        if isinstance(v, float) and v != 0.0 and math.isfinite(v) and c == 0
    ]
    assert not collapsed, f"{len(collapsed)} nonzero values collapsed to 0, e.g. {collapsed[:3]}"
