"""The SQL window vocabulary must be the engine's window vocabulary, not a subset of it.

Two defects motivate this file, and they are opposite failure modes of the same seam between
`_sql/parser/windowing/frame.py` and the engine's `WINDOW_AGGREGATES`.

**A capability that existed and was unreachable.** `_window_func` mapped exactly five names
(`sum`/`avg`/`min`/`max`/`count`) from sqlglot's node names to the engine's tags, so
``bit_or(x) OVER (...)`` failed with "unsupported window function: bitwiseoragg" — the name
sqlglot gives a typed aggregate node, which only the *aggregate* front-end translated. The
engine computes all of these and the DataFrame spelling ``col("x").bit_or().over(...)``
returned DuckDB's own answers throughout, so nothing was missing but the mapping.

**A frame that was silently dropped.** `_build.py` reduced the frame to `None` for any
function outside `WINDOW_FRAMEABLE`. That is right for a function SQL gives no frame either
(ranking, `lag`/`lead`, the fills) and wrong for an aggregate, where SQL defines the framed
answer: ``window(frame=(-1, 0), functions={"w": ("stddev", "f")})`` returned the *running*
standard deviation, which is a wrong result rather than a missing feature. Exactly three
functions are in that gap — `stddev`, `var`, `count_distinct` — and they now refuse.

The frame cases compare against DuckDB *with* the frame and are also pinned against the
running answer, because the defect was returning a real number that happened to answer the
other question: a test that only checked "not an error" would have passed throughout.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

pytestmark = pytest.mark.differential

_FRAME = "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW"


def _table() -> pa.Table:
    """One column per aggregate, each chosen so the framed answer *differs* from the running
    one — which `test_the_frame_actually_narrows_the_answer` enforces, and which a single
    shared column cannot deliver. Disjoint powers of two make `bit_and` zero either way, and
    a single `True` anywhere makes `bool_or` true either way, so both need their own values.
    """
    return pa.table(
        {
            "i": pa.array([1, 2, 3, 4, 5], pa.int64()),
            # Disjoint bits: OR and XOR accumulate, so the frame visibly narrows them.
            "x": pa.array([1, 2, 4, 8, 16], pa.int64()),
            # Overlapping bits, so AND has something left to differ on.
            "y": pa.array([7, 6, 5, 12, 8], pa.int64()),
            "f": pa.array([1.0, 2.0, 4.0, 8.0, 16.0], pa.float64()),
            # A True followed by a run of False: the running OR latches true, the framed one
            # falls back. `bool_and` needs the mirror image, hence a second column: a leading
            # False latches the running AND false while the framed one recovers.
            "b": pa.array([True, False, False, True, False]),
            "b2": pa.array([False, True, True, True, True]),
        }
    )


#: (SQL name, column) for every aggregate the SQL front-end could not reach.
_NEWLY_REACHABLE = [
    ("bit_or", "x"),
    ("bit_and", "y"),
    ("bit_xor", "x"),
    ("bool_and", "b2"),
    ("bool_or", "b"),
    ("stddev", "f"),
    ("variance", "f"),
]

#: The subset of the above the engine can also compute over an explicit frame.
_FRAMEABLE = [(fn, c) for fn, c in _NEWLY_REACHABLE if fn not in {"stddev", "variance"}]


@pytest.mark.parametrize(("fn", "col"), _NEWLY_REACHABLE)
def test_running_window_aggregate_matches_duckdb(duck, fn, col):
    """`agg(x) OVER (ORDER BY i)` — the running form, which SQL could not reach at all."""
    t = _table()
    duck.register("t", t)
    sql = f"SELECT i, {fn}({col}) OVER (ORDER BY i) AS w FROM %s ORDER BY i"
    assert_same_ordered(bt.from_arrow(t).sql(sql % "self").collect(), duck.sql(sql % "t"))


@pytest.mark.parametrize(("fn", "col"), _NEWLY_REACHABLE)
def test_whole_partition_window_aggregate_matches_duckdb(duck, fn, col):
    """`agg(x) OVER ()` — no ORDER BY, so every row carries the partition's aggregate."""
    t = _table()
    duck.register("t", t)
    sql = f"SELECT i, {fn}({col}) OVER () AS w FROM %s ORDER BY i"
    assert_same_ordered(bt.from_arrow(t).sql(sql % "self").collect(), duck.sql(sql % "t"))


@pytest.mark.parametrize(("fn", "col"), _FRAMEABLE)
def test_framed_window_aggregate_matches_duckdb(duck, fn, col):
    """An explicit `ROWS` frame is honoured, not silently widened to the running form."""
    t = _table()
    duck.register("t", t)
    sql = f"SELECT i, {fn}({col}) OVER (ORDER BY i {_FRAME}) AS w FROM %s ORDER BY i"
    assert_same_ordered(bt.from_arrow(t).sql(sql % "self").collect(), duck.sql(sql % "t"))


@pytest.mark.parametrize(("fn", "col"), _FRAMEABLE)
def test_the_frame_actually_narrows_the_answer(duck, fn, col):
    """The framed answer differs from the running one, so the frame is provably applied.

    Without this, `test_framed_window_aggregate_matches_duckdb` could pass on a function
    whose framed and running answers coincide, which is how a dropped frame hides.
    """
    t = _table()
    duck.register("t", t)
    framed = duck.sql(f"SELECT {fn}({col}) OVER (ORDER BY i {_FRAME}) AS w FROM t").to_arrow_table()
    running = duck.sql(f"SELECT {fn}({col}) OVER (ORDER BY i) AS w FROM t").to_arrow_table()
    assert framed.column("w").to_pylist() != running.column("w").to_pylist(), (
        f"{fn} framed == running on this data, so the frame case proves nothing"
    )


@pytest.mark.parametrize("fn", ["stddev", "var", "count_distinct"])
def test_an_unframeable_aggregate_refuses_a_frame_instead_of_ignoring_it(fn):
    """The regression guard: these silently returned the *running* answer to a framed query.

    `assert_same`-style checks cannot catch this — the returned column is a perfectly good
    number, just the answer to the question without the frame.
    """
    ds = bt.from_arrow(_table())
    with pytest.raises(bt.PlanError, match="does not support an explicit frame"):
        ds.window(order_by=["i"], frame=(-1, 0), functions={"w": (fn, "f")}).collect()


@pytest.mark.parametrize("fn", ["stddev", "var"])
def test_the_running_form_of_an_unframeable_aggregate_still_works(duck, fn):
    """Refusing the frame must not cost the form that was always correct."""
    t = _table()
    duck.register("t", t)
    duck_name = {"stddev": "stddev", "var": "variance"}[fn]
    got = bt.from_arrow(t).window(order_by=["i"], functions={"w": (fn, "f")}).collect()
    want = duck.sql(f"SELECT *, {duck_name}(f) OVER (ORDER BY i) AS w FROM t")
    assert_same_ordered(got.select(["w"]), want.select("w"))


def test_ranking_functions_still_ignore_a_frame():
    """SQL gives ranking functions no frame, so dropping one there stays correct."""
    ds = bt.from_arrow(_table())
    out = ds.window(order_by=["i"], frame=(-1, 0), functions={"w": "row_number"}).collect()
    assert out.column("w").to_pylist() == [1, 2, 3, 4, 5]


def test_the_sql_window_table_names_only_tags_the_engine_has():
    """Every tag the SQL mapping emits must be one `WINDOW_AGGREGATES` declares.

    The two drifting apart is the defect this file exists for; a mapping entry naming a tag
    the engine dropped would fail at execution, far from the table that caused it.
    """
    from batcher._sql.parser.windowing import _WINDOW_AGGS
    from batcher.plan.ir_tags import WINDOW_AGGREGATES

    unknown = sorted(set(_WINDOW_AGGS.values()) - WINDOW_AGGREGATES)
    assert not unknown, f"SQL window mapping names tags the engine does not have: {unknown}"
