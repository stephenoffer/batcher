"""Set-operation edge cases vs DuckDB: float key folding (-0.0/0.0/NaN), NULL set
equality, multiset (ALL) multiplicity, empty operands — plus a regression that a
numeric-type-mismatched set op raises a *clean* error instead of aborting the process
with a Rust panic (the `distinct_dense` heterogeneous-batch bug).
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from conftest import assert_same


def _fa():
    return pa.table(
        {
            "x": pa.array(
                [0.0, -0.0, float("nan"), float("nan"), 1.0, 1.0, None, None, -0.0], pa.float64()
            )
        }
    )


def _fb():
    return pa.table({"x": pa.array([0.0, float("nan"), 2.0, None, 1.0], pa.float64())})


# --- float key folding: -0.0 == 0.0, all NaNs one, NULL == NULL for set ops -----------
@pytest.mark.differential
def test_union_distinct_float_folds_zero_nan_null(duck):
    from conftest import assert_same

    a, b = _fa(), _fb()
    duck.register("a", a)
    duck.register("b", b)
    got = bt.from_arrow(a).union(bt.from_arrow(b), distinct=True).collect()
    assert_same(got, duck.sql("SELECT x FROM a UNION SELECT x FROM b"))


@pytest.mark.differential
def test_union_all_float(duck):
    from conftest import assert_same

    a, b = _fa(), _fb()
    duck.register("a", a)
    duck.register("b", b)
    got = bt.from_arrow(a).union(bt.from_arrow(b)).collect()
    assert_same(got, duck.sql("SELECT x FROM a UNION ALL SELECT x FROM b"))


@pytest.mark.differential
@pytest.mark.parametrize("distinct", [True, False])
def test_intersect_float(duck, distinct):
    from conftest import assert_same

    a, b = _fa(), _fb()
    duck.register("a", a)
    duck.register("b", b)
    got = bt.from_arrow(a).intersect(bt.from_arrow(b), distinct=distinct).collect()
    kw = "INTERSECT" if distinct else "INTERSECT ALL"
    assert_same(got, duck.sql(f"SELECT x FROM a {kw} SELECT x FROM b"))


@pytest.mark.differential
@pytest.mark.parametrize("distinct", [True, False])
def test_except_float(duck, distinct):
    from conftest import assert_same

    a, b = _fa(), _fb()
    duck.register("a", a)
    duck.register("b", b)
    got = bt.from_arrow(a).except_(bt.from_arrow(b), distinct=distinct).collect()
    kw = "EXCEPT" if distinct else "EXCEPT ALL"
    assert_same(got, duck.sql(f"SELECT x FROM a {kw} SELECT x FROM b"))


# --- multiset (ALL) multiplicity across signed-zero / NaN copies ----------------------
@pytest.mark.differential
@pytest.mark.parametrize("op,kw", [("intersect", "INTERSECT ALL"), ("except_", "EXCEPT ALL")])
def test_setop_all_multiplicity_float(duck, op, kw):
    from conftest import assert_same

    a = pa.table(
        {"x": pa.array([0.0, 0.0, 0.0, -0.0, float("nan"), float("nan"), 1.0, 1.0], pa.float64())}
    )
    b = pa.table({"x": pa.array([0.0, -0.0, float("nan"), 1.0, 1.0, 1.0, 2.0], pa.float64())})
    duck.register("a", a)
    duck.register("b", b)
    got = getattr(bt.from_arrow(a), op)(bt.from_arrow(b), distinct=False).collect()
    assert_same(got, duck.sql(f"SELECT x FROM a {kw} SELECT x FROM b"))


# --- multi-column DISTINCT with all-NULL rows and signed zero, chunked & spilled ------
@pytest.mark.differential
def test_distinct_multicol_null_signed_zero(duck):
    from conftest import assert_same

    t = pa.table(
        {
            "a": pa.array([1.0, 1.0, None, None, 2.0, 2.0, 1.0], pa.float64()),
            "b": pa.array([0.0, -0.0, None, None, float("nan"), float("nan"), 0.0], pa.float64()),
        }
    )
    duck.register("t", t)
    got = bt.from_arrow(t).distinct().collect()
    assert_same(got, duck.sql("SELECT DISTINCT a, b FROM t"))


@pytest.mark.differential
def test_distinct_float_chunked_matches_duckdb(duck):
    from conftest import assert_same

    rng = np.random.default_rng(0)
    x = np.concatenate(
        [
            np.zeros(1, "float64"),
            -np.zeros(1, "float64"),
            rng.integers(0, 50, 5000).astype("float64"),
        ]
    )
    t = pa.table({"x": pa.array(x, pa.float64())})
    duck.register("t", t)
    got = bt.from_arrow(t.to_batches(max_chunksize=37)).distinct().collect()
    assert_same(got, duck.sql("SELECT DISTINCT x FROM t"))


# --- empty operands -------------------------------------------------------------------
@pytest.mark.differential
def test_setops_empty_operands(duck):
    from conftest import assert_same

    empty = pa.table({"x": pa.array([], pa.int64())})
    ne = pa.table({"x": pa.array([1, 2, 3], pa.int64())})
    duck.register("empty", empty)
    duck.register("ne", ne)
    E, N = bt.from_arrow(empty), bt.from_arrow(ne)
    assert_same(N.except_(N).collect(), duck.sql("SELECT x FROM ne EXCEPT SELECT x FROM ne"))
    assert_same(
        N.intersect(E).collect(), duck.sql("SELECT x FROM ne INTERSECT SELECT x FROM empty")
    )
    assert_same(E.except_(N).collect(), duck.sql("SELECT x FROM empty EXCEPT SELECT x FROM ne"))
    assert_same(
        E.union(E, distinct=True).collect(),
        duck.sql("SELECT x FROM empty UNION SELECT x FROM empty"),
    )


# --- count(distinct) folds signed zero / NaN, excludes NULL ---------------------------
@pytest.mark.differential
def test_count_distinct_float_folding(duck):
    from conftest import assert_same

    a = _fa()
    duck.register("a", a)
    got = bt.from_arrow(a).group_by().agg(n=col("x").n_unique()).collect()
    assert_same(got, duck.sql("SELECT count(DISTINCT x) AS n FROM a"))


# --- REGRESSION: a numeric-promotable set op must COERCE, not panic ------------------
# `distinct_dense` validated only the first batch's dtype, then downcast every batch to
# Int64 — so a UNION of an Int64 branch and a Float64 branch (which arrive as
# differently-typed single-column batches) panicked with "primitive array". Two fixes
# landed: `distinct_dense` now declines a heterogeneous batch (no panic), and the Union
# executor promotes int64+float64 to a common Float64 supertype before dedup — matching
# DuckDB, which coerces `int UNION float` to DOUBLE. So the query now SUCCEEDS with a
# coerced double result rather than raising.
@pytest.mark.differential
@pytest.mark.parametrize(
    "build,sql",
    [
        (lambda A, B: A.union(B, distinct=True), "SELECT x FROM a UNION SELECT x FROM b"),
        (lambda A, B: A.intersect(B), "SELECT x FROM a INTERSECT SELECT x FROM b"),
        (lambda A, B: A.except_(B), "SELECT x FROM a EXCEPT SELECT x FROM b"),
    ],
)
def test_int_float_setop_coerces_to_double_like_duckdb(build, sql, duck):
    a = pa.table({"x": pa.array([1, 2, 3], pa.int64())})
    b = pa.table({"x": pa.array([2.0, 3.5, 4.0], pa.float64())})
    duck.register("a", a)
    duck.register("b", b)
    assert_same(build(bt.from_arrow(a), bt.from_arrow(b)).collect(), duck.sql(sql))


def test_genuinely_incompatible_setop_raises_cleanly_not_panics():
    # A pair that has no common supertype (int64 vs string) must surface a *catchable*
    # error, never a Rust PanicException (which derives from BaseException).
    A = bt.from_arrow(pa.table({"x": pa.array([1, 2, 3], pa.int64())}))
    B = bt.from_arrow(pa.table({"x": pa.array(["a", "b", "c"], pa.string())}))
    with pytest.raises(Exception):  # noqa: B017 - a *catchable* error, never a BaseException panic
        A.union(B, distinct=True).collect()


# --- REGRESSION: INTERSECT / EXCEPT must not crash on the out-of-core (spill) path ----
# `intersect`/`except_` lower to `Aggregate(bool_or) over Union(left, right)` — an
# aggregate whose input spans TWO sources. The spilling aggregate executor
# (`dist/spill.py::execute_spilling_aggregate`) assumed a single-source, map-only input
# and called `_relabel_single_source`, which asserts `len(sources) == 1`. So *any*
# `intersect`/`except_` routed out-of-core — an explicit `collect(spill=True)`, or a
# large set op tripping Carbonite's spill estimate under a tight memory envelope — died
# with `AssertionError: expected a single-source subplan` instead of completing. The
# fix mirrors the Join path's `supports_spilling_join` decline: a multi-source aggregate
# input declines the spill path and falls back to the (correct, mergeable) in-memory
# engine. UNION ALL / UNION DISTINCT already declined cleanly (they never build an
# aggregate), so only INTERSECT/EXCEPT regressed.
@pytest.mark.differential
@pytest.mark.parametrize("op,kw", [("intersect", "INTERSECT"), ("except_", "EXCEPT")])
@pytest.mark.parametrize("distinct", [True, False])
def test_setop_spill_path_does_not_crash(duck, op, kw, distinct):
    from conftest import assert_same

    a = pa.table({"x": pa.array([1, 1, 2, 2, 3, 3, 4, None], pa.int64())})
    b = pa.table({"x": pa.array([2, 3, 3, 5, None], pa.int64())})
    duck.register("a", a)
    duck.register("b", b)
    got = getattr(bt.from_arrow(a), op)(bt.from_arrow(b), distinct=distinct).collect(spill=True)
    sql_kw = kw if distinct else f"{kw} ALL"
    assert_same(got, duck.sql(f"SELECT x FROM a {sql_kw} SELECT x FROM b"))
