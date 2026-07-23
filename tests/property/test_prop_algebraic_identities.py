"""Property: relational-algebra identities that must hold for *any* input.

These need no oracle — they are laws of the algebra, so a counterexample is unambiguously
a Batcher bug (not an oracle disagreement). Hypothesis searches for the input that breaks
one:

  * **Idempotence** — ``distinct(distinct(x)) == distinct(x)``; ``sort(x)`` twice is
    ``sort(x)`` (a re-sort of sorted data must not perturb it).
  * **Filter split** — ``filter(a & b) == filter(a).filter(b)`` (conjunction distributes
    over sequential filtering; the null-drop semantics must agree).
  * **Set-op identity/annihilation** — ``union(x, ∅) == x`` (Batcher ``union`` is UNION
    ALL; ``distinct=True`` makes it dedup to ``distinct(x)``); ``x.except_(x) == ∅``;
    ``x.intersect(x) == distinct(x)``.
  * **Round-trip** — ``from_arrow(x).collect()`` reproduces ``x`` value-for-value (schema
    and cell values), the FFI boundary neither drops nor mangles a row.

Includes ``-0.0``/``NaN``/nulls/empties: these are Batcher-vs-Batcher equalities, so the
signed-zero total-order question (ledger B26) does not enter — both sides use the same
comparator, and the laws must hold whatever that comparator is.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher import col

pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = [pytest.mark.property, pytest.mark.integration]


def _coerce(v: object) -> object:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if v != v:
            return "<nan>"
        if not math.isfinite(v):
            return float(v)
        r = round(v, 9)
        return int(r) if r == int(r) else r
    return v


def _rowset(table: pa.Table) -> list[tuple]:
    cols = table.column_names
    rows = [tuple(_coerce(r[c]) for c in cols) for r in table.to_pylist()]
    return sorted(rows, key=lambda t: tuple((v is None, str(type(v)), str(v)) for v in t))


_SCHEMA = pa.schema([("k", pa.int64()), ("i", pa.int64()), ("f", pa.float64())])
_ints = st.one_of(st.none(), st.sampled_from([-3, -1, 0, 1, 2, 3, 7]))
_floats = st.one_of(st.none(), st.sampled_from([0.0, -0.0, 1.5, -1.5, float("nan")]))


@st.composite
def _table(draw: st.DrawFn) -> pa.Table:
    n = draw(st.integers(min_value=0, max_value=30))
    return pa.table(
        {
            "k": pa.array(draw(st.lists(st.integers(0, 3), min_size=n, max_size=n)), pa.int64()),
            "i": pa.array(draw(st.lists(_ints, min_size=n, max_size=n)), pa.int64()),
            "f": pa.array(draw(st.lists(_floats, min_size=n, max_size=n)), pa.float64()),
        },
        schema=_SCHEMA,
    )


_PROP = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@_PROP
@given(_table())
def test_distinct_is_idempotent(table: pa.Table) -> None:
    """distinct(distinct(x)) == distinct(x)."""
    ds = bt.from_arrow(table)
    once = _rowset(ds.distinct().collect())
    twice = _rowset(ds.distinct().distinct().collect())
    assert once == twice, f"distinct not idempotent:\n once={once}\n twice={twice}"


@_PROP
@given(_table())
def test_sort_is_idempotent(table: pa.Table) -> None:
    """Re-sorting already-sorted data reproduces it exactly (ordered)."""
    ds = bt.from_arrow(table).sort("k", "i", "f")
    once = ds.collect()
    twice = ds.sort("k", "i", "f").collect()
    a = [tuple(_coerce(r[c]) for c in once.column_names) for r in once.to_pylist()]
    b = [tuple(_coerce(r[c]) for c in twice.column_names) for r in twice.to_pylist()]
    assert a == b, f"sort not idempotent:\n once={a}\n twice={b}"


@_PROP
@given(_table(), st.integers(-1, 2), st.integers(-1, 2))
def test_filter_conjunction_splits(table: pa.Table, va: int, vb: int) -> None:
    """filter(a & b) == filter(a).filter(b)."""
    ds = bt.from_arrow(table)
    a = col("i") > bt.lit(va)
    b = col("k") <= bt.lit(vb)
    combined = _rowset(ds.filter(a & b).collect())
    chained = _rowset(ds.filter(a).filter(b).collect())
    assert combined == chained, f"filter split broke:\n combined={combined}\n chained={chained}"


@_PROP
@given(_table())
def test_union_empty_identity(table: pa.Table) -> None:
    """union(x, ∅) == x (UNION ALL); with distinct=True == distinct(x)."""
    ds = bt.from_arrow(table)
    empty = bt.from_arrow(table.slice(0, 0))
    all_rows = _rowset(ds.union(empty).collect())
    base_all = _rowset(ds.collect())
    assert all_rows == base_all, (
        f"UNION ALL with empty changed x:\n union={all_rows}\n base={base_all}"
    )

    dedup = _rowset(ds.union(empty, distinct=True).collect())
    base_distinct = _rowset(ds.distinct().collect())
    assert dedup == base_distinct, (
        f"UNION(distinct) with empty != distinct(x):\n union={dedup}\n base={base_distinct}"
    )


@_PROP
@given(_table())
def test_except_self_is_empty(table: pa.Table) -> None:
    """x EXCEPT x == ∅, and x INTERSECT x == distinct(x)."""
    ds = bt.from_arrow(table)
    diff = _rowset(ds.except_(bt.from_arrow(table)).collect())
    assert diff == [], f"x EXCEPT x not empty: {diff}"
    inter = _rowset(ds.intersect(bt.from_arrow(table)).collect())
    base = _rowset(ds.distinct().collect())
    assert inter == base, f"x INTERSECT x != distinct(x):\n inter={inter}\n base={base}"


@_PROP
@given(_table())
def test_from_arrow_round_trips(table: pa.Table) -> None:
    """from_arrow(x).collect() reproduces x — schema and every value."""
    out = bt.from_arrow(table).collect()
    assert out.schema.types == table.schema.types, f"schema drift: {out.schema} vs {table.schema}"
    # Row order is preserved by a bare scan, so compare positionally.
    got = [tuple(_coerce(r[c]) for c in table.column_names) for r in out.to_pylist()]
    want = [tuple(_coerce(r[c]) for c in table.column_names) for r in table.to_pylist()]
    assert got == want, f"round-trip changed values:\n got={got}\n want={want}"
