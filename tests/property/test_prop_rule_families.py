"""Property: every new Kyber rule family preserves results vs DuckDB.

Each of the ~150 rules lives in a *family* (`kyber/rules/extra/*.py`): boolean algebra,
sargable-predicate normalization, arithmetic reassociation, predicate inference, set-op
rewrites, extra aggregate rewrites, temporal sargability, window rewrites, empty-relation
folding, top-N/limit rewrites, join rewrites. The example-based differential suite pins
one shape per rule; this file adds a *property* per family — Hypothesis generates a random
table and a query deliberately shaped to trip that family, runs it through the FULL
optimizer (``.collect()``), and asserts the answer matches DuckDB over the same data.

Because the query is randomized (constants, null density, row count, empties), a rule that
is result-preserving only on the hand-picked example but wrong on a null/overflow/empty
edge shows up here as a counterexample — which is a real correctness bug in that rule.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher import col, count, lit

pytest.importorskip("batcher._native", reason="native engine not built")
duckdb = pytest.importorskip("duckdb")

from _optmeta_common import ordered_rows, rowset  # noqa: E402  (test-dir helper)

# Importing the extra-rule package runs its `@rule` decorators, registering every family
# into DEFAULT_REGISTRY so the `.collect()` path below actually exercises them.
import batcher.kyber.rules.extra  # noqa: E402

pytestmark = [pytest.mark.property, pytest.mark.integration]


@pytest.fixture(autouse=True)
def _stable_morsel_tuning():
    """Pin adaptive morsel-size tuning off (a result-invariant Carbonite concern)."""
    from batcher.config import active_config, set_config

    prev = active_config()
    set_config(
        prev.replace(execution=dataclasses.replace(prev.execution, adaptive_morsel_sizing=False))
    )
    yield
    set_config(prev)


_INT = st.integers(min_value=-10, max_value=10)
_NULL_INT = st.one_of(st.none(), _INT)
_NULL_BOOL = st.one_of(st.none(), st.booleans())
_NULL_STR = st.one_of(st.none(), st.sampled_from(["a", "b", "c"]))
_DATES = [dt.date(2019, 5, 1), dt.date(2020, 1, 1), dt.date(2020, 12, 31), dt.date(2021, 7, 9)]
_NULL_DATE = st.one_of(st.none(), st.sampled_from(_DATES))

_SCHEMA = pa.schema(
    [
        ("a", pa.int64()),
        ("b", pa.int64()),
        ("g", pa.int64()),
        ("flag", pa.bool_()),
        ("d", pa.date32()),
        ("s", pa.string()),
    ]
)


@st.composite
def _table(draw: st.DrawFn) -> pa.Table:
    """A random table on `_SCHEMA` (0..24 rows; nulls/empty/single-row reachable)."""
    n = draw(st.integers(min_value=0, max_value=24))
    return pa.table(
        {
            "a": draw(st.lists(_NULL_INT, min_size=n, max_size=n)),
            "b": draw(st.lists(_NULL_INT, min_size=n, max_size=n)),
            "g": draw(st.lists(st.integers(0, 3), min_size=n, max_size=n)),
            "flag": draw(st.lists(_NULL_BOOL, min_size=n, max_size=n)),
            "d": draw(st.lists(_NULL_DATE, min_size=n, max_size=n)),
            "s": draw(st.lists(_NULL_STR, min_size=n, max_size=n)),
        },
        schema=_SCHEMA,
    )


_PROP = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
    ],
)


def _ds(table: pa.Table) -> bt.Dataset:
    return bt.from_arrow(table.to_batches() or table)


def _match(table: pa.Table, ds: bt.Dataset, sql: str, *, ordered: bool = False) -> None:
    """Assert ``ds.collect()`` matches DuckDB's ``sql`` over the same table (aliased ``t``)."""
    con = duckdb.connect()
    try:
        con.register("t", table)
        expected = con.sql(sql).to_arrow_table()
        got = ds.collect()
        assert set(got.column_names) == set(expected.column_names), (
            f"column mismatch: batcher={got.column_names} duckdb={expected.column_names}\n{sql}"
        )
        expected = expected.select(got.column_names)
        if ordered:
            g, e = ordered_rows(got), ordered_rows(expected)
        else:
            g, e = rowset(got), rowset(expected)
        assert g == e, (
            f"RULE FAMILY CHANGED THE RESULT vs DuckDB:\nSQL: {sql}\nbatcher: {g}\nduck: {e}"
        )
    finally:
        con.close()


# --- boolean_algebra --------------------------------------------------------
# Annihilators, idempotence, absorption, complementation, NOT-through-compare,
# bool==literal, IN dedup — the NORMALIZE boolean rewrites.
@_PROP
@given(
    _table(),
    _INT,
    st.sampled_from(["dup_and", "compl_or", "eq_true", "not_cmp", "in_dedup", "absorb"]),
)
def test_boolean_algebra(table: pa.Table, x: int, shape: str) -> None:
    ds = _ds(table)
    if shape == "dup_and":  # p AND p == p (idempotence / remove-duplicate-conjuncts)
        pred, sql = (col("a") > x) & (col("a") > x), f"(a > {x}) AND (a > {x})"
    elif shape == "compl_or":  # p OR NOT p == TRUE (complementation, total order)
        pred, sql = (col("a") > x) | ~(col("a") > x), f"(a > {x}) OR NOT (a > {x})"
    elif shape == "eq_true":  # flag == TRUE == flag
        pred, sql = (col("flag") == lit(True)), "flag = TRUE"
    elif shape == "not_cmp":  # NOT (a < b) -> a >= b
        pred, sql = ~(col("a") < col("b")), "NOT (a < b)"
    elif shape == "in_dedup":  # IN with duplicate / single values
        pred, sql = col("a").is_in([x, x, x + 1]), f"a IN ({x}, {x}, {x + 1})"
    else:  # absorption: p AND (p OR q)
        pred = (col("a") > x) & ((col("a") > x) | (col("b") < x))
        sql = f"(a > {x}) AND ((a > {x}) OR (b < {x}))"
    _match(table, ds.filter(pred).select("a", "b"), f"SELECT a, b FROM t WHERE {sql}")


# --- sargable ---------------------------------------------------------------
# Peel constant arithmetic off the column so a predicate becomes `col OP literal`.
@_PROP
@given(_table(), st.integers(-5, 5), _INT)
def test_sargable(table: pa.Table, k: int, m: int) -> None:
    ds = _ds(table)
    shape = k % 3
    if shape == 0:  # col + k = lit  (additive bijection, equality only)
        pred, sql = (col("a") + lit(k)) == lit(m), f"(a + {k}) = {m}"
    elif shape == 1:  # col - k <> lit
        pred, sql = (col("a") - lit(k)) != lit(m), f"(a - {k}) <> {m}"
    else:  # lit OP col  (comparison flip canonicalization)
        pred, sql = lit(m) < col("a"), f"{m} < a"
    _match(table, ds.filter(pred).select("a"), f"SELECT a FROM t WHERE {sql}")


# --- arith_algebra ----------------------------------------------------------
# Integer constant reassociation / factoring in a projection (wrapping i64 ring).
@_PROP
@given(_table(), st.integers(-6, 6), st.integers(-6, 6))
def test_arith_algebra(table: pa.Table, c1: int, c2: int) -> None:
    ds = _ds(table)
    out = ds.select(
        "g",
        x=(col("a") + lit(c1)) + lit(c2),  # fold_add_sub_constants
        y=(col("a") * lit(c1)) * lit(c2),  # fold_mul_constants
        z=col("a") * lit(c1) + col("b") * lit(c1),  # factor_common_mul
    )
    sql = (
        f"SELECT g, (a + {c1}) + {c2} AS x, (a * {c1}) * {c2} AS y, "
        f"(a * {c1}) + (b * {c1}) AS z FROM t"
    )
    _match(table, out, sql)


# --- predicate_infer --------------------------------------------------------
# Redundant / contradictory conjuncts and IN-list refinement from sibling constraints.
@_PROP
@given(
    _table(), _INT, _INT, st.sampled_from(["redundant", "contradiction", "in_refine", "transitive"])
)
def test_predicate_infer(table: pa.Table, lo: int, hi: int, shape: str) -> None:
    ds = _ds(table)
    if shape == "redundant":  # a > lo AND a > hi  (weaker conjunct dropped)
        pred, sql = (col("a") > lo) & (col("a") > hi), f"a > {lo} AND a > {hi}"
    elif shape == "contradiction":  # a > hi AND a < lo  (empty when hi >= lo)
        pred, sql = (col("a") > hi) & (col("a") < lo), f"a > {hi} AND a < {lo}"
    elif shape == "in_refine":  # a IN (...) AND a > lo
        pred = col("a").is_in([lo, lo + 1, lo + 2, hi]) & (col("a") > lo)
        sql = f"a IN ({lo}, {lo + 1}, {lo + 2}, {hi}) AND a > {lo}"
    else:  # a < b AND b < lit  → transitive a < lit (closure kept, result identical)
        pred, sql = (col("a") < col("b")) & (col("b") < lit(hi)), f"a < b AND b < {hi}"
    _match(table, ds.filter(pred).select("a", "b"), f"SELECT a, b FROM t WHERE {sql}")


# --- temporal_sargable ------------------------------------------------------
# year()/decade() extraction comparisons → half-open date ranges on the raw column.
@_PROP
@given(_table(), st.sampled_from([2019, 2020, 2021]), st.sampled_from(["==", ">=", "<"]))
def test_temporal_sargable(table: pa.Table, year: int, op: str) -> None:
    ds = _ds(table)
    y = col("d").dt.year()
    pred = {"==": y == year, ">=": y >= year, "<": y < year}[op]
    _match(
        table,
        ds.filter(pred).select("d"),
        f"SELECT d FROM t WHERE year(d) {op if op != '==' else '='} {year}",
    )


# --- setops -----------------------------------------------------------------
# UNION / UNION ALL rewrites (flatten, dedup branches, distinct folding, pushdown).
@_PROP
@given(_table(), _table(), st.booleans())
def test_setops_union(a: pa.Table, b: pa.Table, distinct: bool) -> None:
    """A filtered UNION / UNION ALL of two tables matches the same set-op in DuckDB."""
    got = (
        _ds(a)
        .select("a", "g")
        .union(_ds(b).select("a", "g"), distinct=distinct)
        .filter(col("g") >= lit(0))
        .collect()
    )
    con = duckdb.connect()
    try:
        con.register("ta", a)
        con.register("tb", b)
        kw = "UNION" if distinct else "UNION ALL"
        expected = (
            con.sql(
                f"SELECT a, g FROM (SELECT a, g FROM ta {kw} SELECT a, g FROM tb) u WHERE g >= 0"
            )
            .to_arrow_table()
            .select(got.column_names)
        )
        assert rowset(got) == rowset(expected), (
            f"setops union ({kw}) != DuckDB:\nbatcher: {rowset(got)}\nduck: {rowset(expected)}"
        )
    finally:
        con.close()


# --- agg_extra --------------------------------------------------------------
# Aggregates that simplify against the group key (count of key, distinct of key, etc.).
@_PROP
@given(_table())
def test_agg_extra(table: pa.Table) -> None:
    ds = _ds(table)
    out = ds.group_by("g").agg(
        n=count(),
        cg=col("g").count(),  # count_of_group_key → count(*)
        ug=col("g").n_unique(),  # count_distinct_of_group_key → 1 per group
        sa=col("a").sum(),
    )
    _match(
        table,
        out,
        "SELECT g, count(*) AS n, count(g) AS cg, count(DISTINCT g) AS ug, sum(a) AS sa "
        "FROM t GROUP BY g",
    )


# --- window_rules -----------------------------------------------------------
# Drop constant/duplicate partition & order keys; dead-window elimination.
@_PROP
@given(_table())
def test_window_rules(table: pa.Table) -> None:
    from batcher import row_number

    ds = _ds(table)
    # A constant partition key + duplicate order keys — the rules prune them without
    # changing the row_number assignment.
    out = ds.with_columns(
        rn=row_number().over(partition_by=["g"], order_by=["a", "a", "b"])
    ).select("g", "a", "b", "rn")
    _match(
        table,
        out,
        "SELECT g, a, b, row_number() OVER (PARTITION BY g ORDER BY a, a, b) AS rn FROM t",
    )


# --- empty_relation ---------------------------------------------------------
# Provably-empty inputs fold: filter(FALSE), limit(0), project/aggregate over empty.
@_PROP
@given(_table(), st.sampled_from(["false_filter", "limit0", "agg_empty", "project_empty"]))
def test_empty_relation(table: pa.Table, shape: str) -> None:
    ds = _ds(table)
    if shape == "false_filter":
        _match(table, ds.filter(lit(False)).select("a", "g"), "SELECT a, g FROM t WHERE FALSE")
    elif shape == "limit0":
        _match(table, ds.select("a", "g").limit(0), "SELECT a, g FROM t LIMIT 0")
    elif shape == "agg_empty":  # aggregate over an empty (contradiction) input
        _match(
            table,
            ds.filter((col("a") > lit(5)) & (col("a") < lit(-5)))
            .group_by("g")
            .agg(s=col("a").sum()),
            "SELECT g, sum(a) AS s FROM t WHERE a > 5 AND a < -5 GROUP BY g",
        )
    else:
        _match(
            table,
            ds.filter(lit(False)).select("g", x=col("a") + col("b")),
            "SELECT g, a + b AS x FROM t WHERE FALSE",
        )


# --- topn_limit -------------------------------------------------------------
# Sort+Limit fusion, offset past cardinality, inert/ redundant limits (ordered compare).
@_PROP
@given(_table(), st.integers(0, 12), st.integers(0, 6))
def test_topn_limit(table: pa.Table, n: int, offset: int) -> None:
    ds = _ds(table)
    # Total order on all columns so LIMIT/OFFSET slices deterministically vs DuckDB.
    out = ds.sort("a", "b", "g", "d", "s", "flag").limit(n, offset=offset).select("a", "b", "g")
    _match(
        table,
        out,
        "SELECT a, b, g FROM t ORDER BY a NULLS LAST, b NULLS LAST, g NULLS LAST, "
        f"d NULLS LAST, s NULLS LAST, flag NULLS LAST LIMIT {n} OFFSET {offset}",
        ordered=True,
    )


# --- join_extra -------------------------------------------------------------
# Empty-side folding, dedup join keys, semi/anti short-circuits.
_LEFT = pa.schema([("id", pa.int64()), ("x", pa.int64())])
_RIGHT = pa.schema([("id", pa.int64()), ("y", pa.int64())])


@st.composite
def _join_tables(draw: st.DrawFn) -> tuple[pa.Table, pa.Table]:
    ln = draw(st.integers(0, 16))
    rn = draw(st.integers(0, 16))
    left = pa.table(
        {
            "id": draw(st.lists(st.integers(0, 4), min_size=ln, max_size=ln)),
            "x": draw(st.lists(_NULL_INT, min_size=ln, max_size=ln)),
        },
        schema=_LEFT,
    )
    right = pa.table(
        {
            "id": draw(st.lists(st.integers(0, 4), min_size=rn, max_size=rn)),
            "y": draw(st.lists(_NULL_INT, min_size=rn, max_size=rn)),
        },
        schema=_RIGHT,
    )
    return left, right


@_PROP
@given(_join_tables(), st.sampled_from(["inner", "left", "semi", "anti"]))
def test_join_extra(tables: tuple[pa.Table, pa.Table], how: str) -> None:
    left, right = tables
    lds = bt.from_arrow(left.to_batches() or left)
    rds = bt.from_arrow(right.to_batches() or right)
    got = lds.join(rds, on="id", how=how).collect()
    con = duckdb.connect()
    try:
        con.register("l", left)
        con.register("r", right)
        if how == "inner":
            sql = "SELECT l.id AS id, l.x AS x, r.y AS y FROM l JOIN r ON l.id = r.id"
        elif how == "left":
            sql = "SELECT l.id AS id, l.x AS x, r.y AS y FROM l LEFT JOIN r ON l.id = r.id"
        elif how == "semi":
            sql = "SELECT l.id AS id, l.x AS x FROM l WHERE l.id IN (SELECT id FROM r)"
        else:  # anti
            sql = (
                "SELECT l.id AS id, l.x AS x FROM l "
                "WHERE l.id NOT IN (SELECT id FROM r WHERE id IS NOT NULL)"
            )
        expected = con.sql(sql).to_arrow_table().select(got.column_names)
        assert rowset(got) == rowset(expected), (
            f"join_extra ({how}) != DuckDB:\nbatcher: {rowset(got)}\nduck: {rowset(expected)}"
        )
    finally:
        con.close()
