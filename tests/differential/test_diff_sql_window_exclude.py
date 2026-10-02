"""Window frame `EXCLUDE` vs DuckDB.

The SQL front-end answers a frame exclusion by splitting the frame around what it excludes
and folding the aggregate over the pieces (`_sql/parser/windowing/derived.py`). The shortcut
it replaces in the docs, subtracting the current row from the unexcluded aggregate, is wrong
on three inputs this fixture carries on purpose: a NULL current row (``sum - NULL`` is NULL),
a frame the exclusion leaves empty (``x - x`` is 0 where SQL says NULL) and a non-finite float
(``inf - inf`` is NaN). The fixture also has ties on the ORDER BY key, which is what separates
``EXCLUDE CURRENT ROW`` from ``GROUP`` and ``TIES`` under a ``GROUPS`` frame.

Results are compared positionally on the unique `id`, with a NaN-aware float tolerance.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_T = pa.table(
    {
        "id": list(range(10)),
        "g": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "o": [1, 2, 2, 3, 1, 1, 2, 1, 2, 3],
        "v": pa.array([10, None, 5, 7, None, None, 3, 4, 4, -1], pa.int64()),
        "f": pa.array([1.5, math.inf, None, 2.0, math.nan, 1.0, None, 0.5, -2.0, 3.0]),
        "b": pa.array([True, False, None, True, None, None, True, False, True, True]),
    }
)

_FUNCS = [
    "sum(v)",
    "count(v)",
    "count(*)",
    "avg(v)",
    "min(v)",
    "max(f)",
    "min(f)",
    "sum(f)",
    "avg(f)",
    "bool_and(b)",
    "bool_or(b)",
]

#: (frame, exclusion) pairs the rewrite answers. ROWS frames order on the unique `(o, id)`
#: so the frame is defined; GROUPS and peer-bounded RANGE frames order on the tied `o`.
_SUPPORTED = [
    ("ORDER BY o, id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING", "CURRENT ROW"),
    ("ORDER BY o, id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW", "CURRENT ROW"),
    ("ORDER BY o, id ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING", "CURRENT ROW"),
    ("ORDER BY o, id ROWS BETWEEN CURRENT ROW AND CURRENT ROW", "NO OTHERS"),
    ("ORDER BY o ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING", "GROUP"),
    ("ORDER BY o GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING", "CURRENT ROW"),
    ("ORDER BY o GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING", "GROUP"),
    ("ORDER BY o GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING", "TIES"),
    ("ORDER BY o GROUPS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING", "TIES"),
    ("ORDER BY o RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW", "CURRENT ROW"),
    ("ORDER BY o RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW", "GROUP"),
    ("ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING", "CURRENT ROW"),
]


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b or abs(a - b) <= 1e-9 * max(1.0, abs(a))
    return type(a) is type(b) and a == b


@pytest.mark.parametrize(("frame", "exclude"), _SUPPORTED)
def test_frame_exclusion_matches_duckdb(duck, frame, exclude):
    funcs = [f for f in _FUNCS if not (f == "count(*)" and not frame.startswith("ORDER"))]
    cols = ", ".join(
        f"{fn} OVER (PARTITION BY g {frame} EXCLUDE {exclude}) AS c{i}"
        for i, fn in enumerate(funcs)
    )
    query = f"SELECT id, {cols} FROM t ORDER BY id"
    duck.register("t", _T)
    want = duck.sql(query).fetchall()
    got = [tuple(r.values()) for r in bt.sql(query, t=_T).collect().to_pylist()]
    for i, fn in enumerate(funcs, start=1):
        assert all(_same(g[i], w[i]) for g, w in zip(got, want, strict=True)), (
            fn,
            [g[i] for g in got],
            [w[i] for w in want],
        )


def test_exclusion_changes_the_answer(duck):
    """Positive control: on this fixture excluding the current row is not a no-op."""
    base = "sum(v) OVER (PARTITION BY g ORDER BY o, id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING{})"
    query = f"SELECT id, {base.format('')} AS a, {base.format(' EXCLUDE CURRENT ROW')} AS b FROM t"
    out = bt.sql(query, t=_T).to_pydict()
    assert out["a"] != out["b"]


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id, 1 + max(f) OVER (PARTITION BY g ORDER BY o GROUPS BETWEEN 1 PRECEDING AND "
        "1 FOLLOWING EXCLUDE TIES) AS m FROM t ORDER BY id",
        "SELECT id, f AS m FROM t QUALIFY count(v) OVER (PARTITION BY g ORDER BY o, id "
        "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING EXCLUDE CURRENT ROW) > 0 ORDER BY id",
    ],
    ids=["nested", "qualify"],
)
def test_exclusion_inside_an_expression_and_qualify(duck, query):
    duck.register("t", _T)
    want = duck.sql(query).fetchall()
    got = [tuple(r.values()) for r in bt.sql(query, t=_T).collect().to_pylist()]
    assert [g[0] for g in got] == [w[0] for w in want]
    assert all(_same(g[1], w[1]) for g, w in zip(got, want, strict=True))


@pytest.mark.parametrize(
    "window",
    [
        # Peers under a bounded ROWS frame, and a value-offset RANGE frame, have no exact split.
        "sum(v) OVER (ORDER BY o ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING EXCLUDE TIES)",
        "sum(v) OVER (ORDER BY o RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING EXCLUDE GROUP)",
        # An aggregate with no fold over pieces.
        "median(v) OVER (ORDER BY o, id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING "
        "EXCLUDE CURRENT ROW)",
    ],
)
def test_an_exclusion_with_no_exact_split_is_refused(window):
    with pytest.raises(NotImplementedError, match="EXCLUDE"):
        bt.sql(f"SELECT {window} AS x FROM t", t=_T).collect()


def test_a_window_sharing_text_with_a_grouped_aggregate_counts_the_groups(duck):
    """`count(*) OVER ()` beside `count(*)` in a GROUP BY is a window over the groups.

    The grouped-aggregate substitution matched the window's *own* `count(*)` by its SQL
    text and replaced it with the grouped column, so the query failed with "unsupported
    window function: column". The `LIMIT ... PERCENT` rewrite adds exactly this window.
    """
    table = pa.table({"k": [1, 1, 2, 3], "v": [1, 2, 3, 4]})
    query = "SELECT k, count(*) AS c, count(*) OVER () AS n FROM t GROUP BY k ORDER BY k"
    duck.register("t", table)
    got = bt.sql(query, t=table).collect()
    assert [tuple(r.values()) for r in got.to_pylist()] == duck.sql(query).fetchall()
