"""Differential guard on Kyber's *join-and-set-op* rules — optimized result == DuckDB.

The existing property test (``tests/property/test_prop_optimizer_result_invariance``)
searches single-table pipelines (filter / with_columns / group-by / distinct / sort /
limit / union), but exercises **no join**. This module fills that gap with a randomized
but *seeded* (deterministic) sweep over the surface where a Kyber rule most plausibly
changes a result:

* predicate pushdown through **outer** joins (a predicate wrongly sunk to a
  null-producing side turns a LEFT/RIGHT join into an inner join — a silent row loss);
* cost-based **join reordering** over 3-table chain/star graphs (must preserve the
  answer no matter how the interior is reshaped);
* **aggregate-through-join** pushdown (``eager_aggregation`` /
  ``pre_aggregation_through_join`` — a partial aggregate pushed below a join is only
  correct when the fan-out is accounted for);
* **complementary-filter** union merge and other set-op rewrites.

Each case runs the full optimizer (``collect()``) and asserts the result matches DuckDB
on the same input (order-independent multiset compare). A regression in any of the above
rules — the exact class of bug an example-based test misses — fails here.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pytest

from conftest import assert_same

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")
duckdb = pytest.importorskip("duckdb")

from batcher import col, count  # noqa: E402

_INT = pa.int64()


def _tbl(rng: random.Random, cols: list[str], *, domain=(None, 0, 1, 2, 3), n_max=7) -> pa.Table:
    n = rng.randint(0, n_max)
    return pa.table(
        {c: [rng.choice(domain) for _ in range(n)] for c in cols},
        schema=pa.schema([(c, _INT) for c in cols]),
    )


# --- predicate pushdown through joins (incl. OUTER) -------------------------


@pytest.mark.parametrize("how", ["inner", "left", "right", "full", "semi", "anti"])
@pytest.mark.parametrize("seed", range(12))
def test_join_then_filter_matches_duckdb(how: str, seed: int) -> None:
    """`L join R` then a filter over the join output == DuckDB, for every join type.

    Predicate pushdown must not sink a conjunct onto an outer join's null-producing
    side (which would drop the null-extended rows the join is defined to keep).
    """
    rng = random.Random(seed * 131 + len(how) * 17 + ord(how[0]))
    L = _tbl(rng, ["k", "a"])
    R = _tbl(rng, ["k", "b"])
    ds = bt.from_arrow(L.to_batches() or L).join(
        bt.from_arrow(R.to_batches() or R), on="k", how=how, suffix="_r"
    )
    outcols = list(ds.columns)
    # A random 0-2 conjunct predicate over the join's output columns.
    conj_bt, conj_sql = [], []
    for _ in range(rng.randint(0, 2)):
        c = rng.choice(outcols)
        op = rng.choice(["<", "<=", ">", ">=", "==", "!="])
        v = rng.randint(-1, 4)
        conj_bt.append(
            {
                "<": col(c) < v,
                "<=": col(c) <= v,
                ">": col(c) > v,
                ">=": col(c) >= v,
                "==": col(c) == v,
                "!=": col(c) != v,
            }[op]
        )
        conj_sql.append(f"{c} {'=' if op == '==' else op} {v}")
    if conj_bt:
        pred = conj_bt[0]
        for e in conj_bt[1:]:
            pred = pred & e
        ds = ds.filter(pred)

    con = duckdb.connect()
    con.register("L", L)
    con.register("R", R)
    if how in ("semi", "anti"):
        exists = "EXISTS" if how == "semi" else "NOT EXISTS"
        base = f"SELECT L.k AS k, L.a AS a FROM L WHERE {exists} (SELECT 1 FROM R WHERE R.k = L.k)"
    else:
        key = {"inner": "L.k", "left": "L.k", "right": "R.k", "full": "COALESCE(L.k, R.k)"}[how]
        jt = {
            "inner": "JOIN",
            "left": "LEFT JOIN",
            "right": "RIGHT JOIN",
            "full": "FULL OUTER JOIN",
        }[how]
        base = f"SELECT {key} AS k, L.a AS a, R.b AS b FROM L {jt} R ON L.k = R.k"
    sql = base if not conj_sql else f"SELECT * FROM ({base}) t WHERE " + " AND ".join(conj_sql)
    assert_same(ds.collect(), con.execute(sql))


# --- 3-table join reordering ------------------------------------------------


@pytest.mark.parametrize("shape", ["chain", "star"])
@pytest.mark.parametrize("seed", range(16))
def test_three_way_inner_join_reorder_matches_duckdb(shape: str, seed: int) -> None:
    """A 3-table inner-join (chain or star) result is DuckDB-identical after reordering."""
    rng = random.Random(seed * 977 + (0 if shape == "chain" else 1))
    if shape == "star":
        A, B, C = (_tbl(rng, ["ka", "x"]), _tbl(rng, ["ka", "y"]), _tbl(rng, ["ka", "z"]))
        ds = (
            bt.from_arrow(A.to_batches() or A)
            .join(bt.from_arrow(B.to_batches() or B), on="ka", suffix="_b")
            .join(bt.from_arrow(C.to_batches() or C), on="ka", suffix="_c")
        )
        sql = (
            "SELECT A.ka AS ka, A.x AS x, B.y AS y, C.z AS z "
            "FROM A JOIN B ON A.ka=B.ka JOIN C ON A.ka=C.ka"
        )
    else:
        A, B, C = (_tbl(rng, ["ka", "x"]), _tbl(rng, ["ka", "kb"]), _tbl(rng, ["kb", "z"]))
        ds = (
            bt.from_arrow(A.to_batches() or A)
            .join(bt.from_arrow(B.to_batches() or B), on="ka", suffix="_b")
            .join(bt.from_arrow(C.to_batches() or C), on="kb", suffix="_c")
        )
        sql = (
            "SELECT A.ka AS ka, A.x AS x, B.kb AS kb, C.z AS z "
            "FROM A JOIN B ON A.ka=B.ka JOIN C ON B.kb=C.kb"
        )
    con = duckdb.connect()
    con.register("A", A)
    con.register("B", B)
    con.register("C", C)
    assert_same(ds.collect(), con.execute(sql))


# --- aggregate through join -------------------------------------------------


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("agg", ["min", "max", "sum", "count", "nunique"])
@pytest.mark.parametrize("seed", range(6))
def test_aggregate_over_join_matches_duckdb(how: str, agg: str, seed: int) -> None:
    """`GROUP BY g, AGG(measure)` over a join == DuckDB — the pre-aggregation-pushdown guard.

    Larger tables (so the estimator sees a row reduction and the cost-gated pushdown
    actually fires) with both a key-unique and a key-duplicated R (fan-out).
    """
    rng = random.Random(seed * 41 + len(agg) + (0 if how == "inner" else 100))
    n = rng.randint(40, 120)
    kd = rng.randint(2, 6)
    L = pa.table(
        {
            "k": [rng.randrange(kd) for _ in range(n)],
            "g": [rng.choice([None, 0, 1, 2]) for _ in range(n)],
            "x": [rng.choice([None, -2, 0, 1, 3, 7]) for _ in range(n)],
        },
        schema=pa.schema([("k", _INT), ("g", _INT), ("x", _INT)]),
    )
    if rng.random() < 0.5:  # R unique on k
        ks = list(range(kd))
        R = pa.table(
            {"k": ks, "y": [rng.choice([None, 1, 2, 5]) for _ in ks]},
            schema=pa.schema([("k", _INT), ("y", _INT)]),
        )
    else:  # R with fan-out
        m = rng.randint(3, 10)
        R = pa.table(
            {
                "k": [rng.randrange(kd) for _ in range(m)],
                "y": [rng.choice([None, 1, 2, 5]) for _ in range(m)],
            },
            schema=pa.schema([("k", _INT), ("y", _INT)]),
        )
    mc = rng.choice(["x", "y"])
    gk = rng.choice(["g", "k"])
    a = {
        "min": col(mc).min(),
        "max": col(mc).max(),
        "sum": col(mc).sum(),
        "count": count(),
        "nunique": col(mc).n_unique(),
    }[agg]
    ds = (
        bt.from_arrow(L.to_batches() or L)
        .join(bt.from_arrow(R.to_batches() or R), on="k", how=how, suffix="_r")
        .group_by(gk)
        .agg(r=a)
    )
    con = duckdb.connect()
    con.register("L", L)
    con.register("R", R)
    jt = {"inner": "JOIN", "left": "LEFT JOIN"}[how]
    sa = {
        "min": f"MIN({mc})",
        "max": f"MAX({mc})",
        "sum": f"SUM({mc})",
        "count": "COUNT(*)",
        "nunique": f"COUNT(DISTINCT {mc})",
    }[agg]
    sql = f"SELECT L.{gk} AS {gk}, {sa} AS r FROM L {jt} R ON L.k=R.k GROUP BY L.{gk}"
    assert_same(ds.collect(), con.execute(sql))


# --- set operations ---------------------------------------------------------


@pytest.mark.parametrize("setop", ["union", "unionall", "intersect", "except"])
@pytest.mark.parametrize("seed", range(10))
def test_set_operation_matches_duckdb(setop: str, seed: int) -> None:
    """union / union all / intersect / except over two tables == DuckDB."""
    rng = random.Random(seed * 613 + len(setop))
    A = _tbl(rng, ["a", "b"])
    B = _tbl(rng, ["a", "b"])
    dsA, dsB = bt.from_arrow(A.to_batches() or A), bt.from_arrow(B.to_batches() or B)
    ds, kw = {
        "union": (dsA.union(dsB, distinct=True), "UNION"),
        "unionall": (dsA.union(dsB, distinct=False), "UNION ALL"),
        "intersect": (dsA.intersect(dsB), "INTERSECT"),
        "except": (dsA.except_(dsB), "EXCEPT"),
    }[setop]
    con = duckdb.connect()
    con.register("A", A)
    con.register("B", B)
    assert_same(ds.collect(), con.execute(f"SELECT a,b FROM A {kw} SELECT a,b FROM B"))


@pytest.mark.parametrize("seed", range(20))
def test_complementary_filter_union_matches_duckdb(seed: int) -> None:
    """`filter(x, p) UNION filter(x, NOT p)` == `DISTINCT x` == DuckDB.

    Exercises ``merge_distinct_union_of_complementary_filters`` (which folds the pair to
    ``Distinct(x)``) over both a total ``IS NULL`` / ``IS NOT NULL`` split (valid for any
    column) and an ordinary comparison over NOT-NULL columns (valid only there).
    """
    rng = random.Random(seed * 17 + 3)
    non_null = rng.random() < 0.5
    field = pa.field("a", _INT, nullable=not non_null)
    n = rng.randint(0, 8)
    dom = (0, 1, 2, 3) if non_null else (None, 0, 1, 2, 3)
    tbl = pa.table({"a": [rng.choice(dom) for _ in range(n)]}, schema=pa.schema([field]))
    ds = bt.from_arrow(tbl.to_batches() or tbl)
    if non_null and rng.random() < 0.5:
        v = rng.randint(0, 3)
        p, notp = col("a") > v, col("a") <= v
        p_sql, notp_sql = f"a > {v}", f"a <= {v}"
    else:
        p, notp = col("a").is_null(), col("a").is_not_null()
        p_sql, notp_sql = "a IS NULL", "a IS NOT NULL"
    ds = ds.filter(p).union(ds.filter(notp), distinct=True)
    con = duckdb.connect()
    con.register("t", tbl)
    sql = f"SELECT a FROM t WHERE {p_sql} UNION SELECT a FROM t WHERE {notp_sql}"
    assert_same(ds.collect(), con.execute(sql))


# --- nullability-driven rewrites (NOT NULL source columns) ------------------


@pytest.mark.parametrize("seed", range(8))
def test_coalesce_of_not_null_column_preserves_type(seed: int) -> None:
    """`coalesce(a, 1.5)` over a NOT-NULL int column stays DOUBLE (no type narrowing).

    ``drop_coalesce_of_non_nullable_first_arg`` may reduce ``coalesce(a, 1.5)`` to ``a``
    only if the type is preserved; a bug there would narrow the DOUBLE back to BIGINT.
    """
    rng = random.Random(seed * 29 + 7)
    n = rng.randint(1, 8)
    tbl = pa.table(
        {"a": [rng.randint(-3, 5) for _ in range(n)]},
        schema=pa.schema([pa.field("a", _INT, nullable=False)]),
    )
    ds = bt.from_arrow(tbl.to_batches() or tbl).with_columns(r=bt.coalesce(col("a"), bt.lit(1.5)))
    got = ds.collect()
    assert got.schema.field("r").type == pa.float64()
    con = duckdb.connect()
    con.register("t", tbl)
    assert_same(got, con.execute("SELECT a, COALESCE(a, 1.5) AS r FROM t"))
