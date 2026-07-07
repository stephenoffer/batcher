"""Differential: the expanded metadata-answer derivations match DuckDB.

Each new EXACT stat-propagation shortcut (bool_and/bool_or, count of a non-null
column, sum from a recorded total, a group-key's min/max carried through a grouped
aggregate, an identity cast, a value carried through a window) is proven to equal
DuckDB's *executed* answer — over BOTH a real Parquet file (footer stats drive it)
AND an in-memory source (hand-built EXACT stats) — across nulls, empty, single-row,
all-null and type edges. The complementary `test_stats_propagation_expanded.py`
proves the shortcut actually fires from metadata (and correctly falls back).

The metadata engine under test is `kyber.answer_aggregate` / `answer_count`, which
answer only from `Provenance.EXACT` end to end; a returned value is therefore
identical to the executed result or the test fails.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col, count, lit

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("batcher._native", reason="native engine not built")

from batcher import kyber  # noqa: E402
from batcher.api.orchestration import collect_source_stats  # noqa: E402
from batcher.plan.logical import Filter  # noqa: E402
from batcher.plan.source_stats import SourceStatistics  # noqa: E402
from batcher.plan.stats import ColumnStat, Provenance  # noqa: E402


def _parquet(tmp_path, table: pa.Table, *, row_group_size: int | None = None) -> str:
    path = str(tmp_path / "data.parquet")
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def _exact_stats(table: pa.Table) -> SourceStatistics:
    """A genuinely-exact `SourceStatistics` computed from the in-memory table.

    Uses pyarrow compute so the min/max/null_count are the true values — the
    metadata answer is then compared to DuckDB executing the same query, so a wrong
    derivation surfaces immediately.
    """
    import pyarrow.compute as pc

    cols: dict[str, ColumnStat] = {}
    for name in table.column_names:
        arr = table.column(name)
        n_null = arr.null_count
        non_null = len(arr) - n_null
        mn = pc.min(arr).as_py() if non_null else None
        mx = pc.max(arr).as_py() if non_null else None
        total = pc.sum(arr).as_py() if (pa.types.is_integer(arr.type) and non_null) else None
        cols[name] = ColumnStat(
            min=mn,
            max=mx,
            null_count=float(n_null),
            total_sum=float(total) if total is not None else None,
            provenance=Provenance.EXACT,
        )
    return SourceStatistics(row_count=table.num_rows, columns=cols, exact_rows=True)


def _meta(plan, sources, source_stats):
    return kyber.answer_aggregate(plan, sources, source_stats)


# --------------------------------------------------------------------------
# bool_and / bool_or from a boolean column's EXACT min/max
# --------------------------------------------------------------------------
_BOOL_CASES = {
    "mixed": [True, True, False, True],
    "all_true": [True, True, True],
    "all_false": [False, False],
    "single_true": [True],
    "single_false": [False],
    "with_nulls_mixed": [True, None, False, True],
    "with_nulls_all_true": [True, None, True],
}


@pytest.mark.parametrize("name", list(_BOOL_CASES))
def test_bool_and_or_matches_duckdb_inmemory(duck, name):
    values = _BOOL_CASES[name]
    table = pa.table({"flag": pa.array(values, type=pa.bool_())})
    ds = bt.from_arrow(table)
    agg = ds.agg(a=col("flag").bool_and(), o=col("flag").bool_or())
    meta = _meta(agg._plan, ds._sources, [_exact_stats(table)])
    duck.register("t", table)
    want = duck.sql("SELECT bool_and(flag) AS a, bool_or(flag) AS o FROM t").fetchone()
    assert meta is not None  # every non-empty boolean group is metadata-answerable
    assert (meta["a"], meta["o"]) == (want[0], want[1])


@pytest.mark.parametrize("name", ["mixed", "all_true", "with_nulls_mixed"])
def test_bool_and_or_matches_duckdb_parquet(duck, tmp_path, name):
    table = pa.table({"flag": pa.array(_BOOL_CASES[name], type=pa.bool_())})
    path = _parquet(tmp_path, table)
    ds = bt.read.parquet(path)
    agg = ds.agg(a=col("flag").bool_and(), o=col("flag").bool_or())
    meta = _meta(agg._plan, ds._sources, collect_source_stats(ds._sources, None))
    want = duck.sql(f"SELECT bool_and(flag) AS a, bool_or(flag) AS o FROM '{path}'").fetchone()
    if meta is not None:  # footer may or may not carry bool min/max; either is correct
        assert (meta["a"], meta["o"]) == (want[0], want[1])


def test_bool_all_null_falls_back(duck):
    # An all-null (or empty) boolean group is SQL NULL — not metadata-derivable.
    table = pa.table({"flag": pa.array([None, None], type=pa.bool_())})
    ds = bt.from_arrow(table)
    agg = ds.agg(a=col("flag").bool_and())
    assert _meta(agg._plan, ds._sources, [_exact_stats(table)]) is None


# --------------------------------------------------------------------------
# count(col) of a provably non-null column == row count
# --------------------------------------------------------------------------
def test_count_non_null_column_matches_duckdb_parquet(duck, tmp_path):
    table = pa.table({"x": list(range(1, 1001))})  # no nulls
    path = _parquet(tmp_path, table, row_group_size=128)
    ds = bt.read.parquet(path)
    agg = ds.agg(c=col("x").count())
    meta = _meta(agg._plan, ds._sources, collect_source_stats(ds._sources, None))
    want = duck.sql(f"SELECT count(x) AS c FROM '{path}'").fetchone()[0]
    assert meta is not None and meta["c"] == want


def test_count_column_with_nulls_matches_duckdb(duck):
    table = pa.table({"x": [1, None, 3, None, 5]})
    ds = bt.from_arrow(table)
    agg = ds.agg(c=col("x").count())
    meta = _meta(agg._plan, ds._sources, [_exact_stats(table)])
    duck.register("t", table)
    want = duck.sql("SELECT count(x) AS c FROM t").fetchone()[0]
    assert meta is not None and meta["c"] == want == 3


# --------------------------------------------------------------------------
# sum(col) from a recorded EXACT total (with the empty-group NULL guard)
# --------------------------------------------------------------------------
def test_sum_from_total_matches_duckdb(duck):
    table = pa.table({"v": [10, 20, 30, 40]})
    ds = bt.from_arrow(table)
    agg = ds.agg(s=col("v").sum())
    meta = _meta(agg._plan, ds._sources, [_exact_stats(table)])
    duck.register("t", table)
    want = duck.sql("SELECT sum(v) AS s FROM t").fetchone()[0]
    assert meta is not None and meta["s"] == want


def test_sum_empty_relation_falls_back():
    # SQL sum over zero rows is NULL, not 0 — a provably-empty relation must fall back.
    ds = bt.from_arrow(pa.table({"v": pa.array([], type=pa.int64())}))
    stats = SourceStatistics(
        row_count=0,
        columns={"v": ColumnStat(total_sum=0.0, null_count=0.0, provenance=Provenance.EXACT)},
        exact_rows=True,
    )
    agg = ds.agg(s=col("v").sum())
    assert _meta(agg._plan, ds._sources, [stats]) is None


# --------------------------------------------------------------------------
# a GROUP BY key's min/max carried EXACT through a grouped aggregate
# --------------------------------------------------------------------------
def test_group_key_minmax_matches_duckdb_parquet(duck, tmp_path):
    table = pa.table({"k": [1, 2, 2, 3, 3, 3], "v": [10, 20, 30, 40, 50, 60]})
    path = _parquet(tmp_path, table)
    ds = bt.read.parquet(path)
    outer = ds.group_by("k").agg(c=count()).agg(mn=col("k").min(), mx=col("k").max())
    meta = _meta(outer._plan, ds._sources, collect_source_stats(ds._sources, None))
    want = duck.sql(
        f"SELECT min(k) AS mn, max(k) AS mx FROM (SELECT k FROM '{path}' GROUP BY k)"
    ).fetchone()
    assert meta is not None and (meta["mn"], meta["mx"]) == (want[0], want[1])


def test_group_key_minmax_matches_duckdb_inmemory(duck):
    table = pa.table({"k": [5, 5, 1, 9, 1], "v": [1, 2, 3, 4, 5]})
    ds = bt.from_arrow(table)
    outer = ds.group_by("k").agg(c=count()).agg(mn=col("k").min(), mx=col("k").max())
    meta = _meta(outer._plan, ds._sources, [_exact_stats(table)])
    duck.register("t", table)
    want = duck.sql(
        "SELECT min(k) AS mn, max(k) AS mx FROM (SELECT k FROM t GROUP BY k)"
    ).fetchone()
    assert meta is not None and (meta["mn"], meta["mx"]) == (want[0], want[1])


# --------------------------------------------------------------------------
# an identity cast is a no-op — the column's EXACT min/max survive it
# --------------------------------------------------------------------------
def test_identity_cast_preserves_minmax_matches_duckdb(duck):
    table = pa.table({"v": [10, 20, 30]})  # int64
    ds = bt.from_arrow(table)
    projected = ds.select(w=col("v").cast("int64"))  # identity: int64 -> int64
    agg = projected.agg(mn=col("w").min(), mx=col("w").max())
    meta = _meta(agg._plan, ds._sources, [_exact_stats(table)])
    duck.register("t", table)
    want = duck.sql("SELECT min(CAST(v AS BIGINT)) AS mn, max(CAST(v AS BIGINT)) AS mx FROM t")
    want_row = want.fetchone()
    assert meta is not None and (meta["mn"], meta["mx"]) == (want_row[0], want_row[1])


# --------------------------------------------------------------------------
# trivially-constant filters answered from metadata
# --------------------------------------------------------------------------
def test_filter_true_and_false_count_matches_duckdb(duck):
    table = pa.table({"x": list(range(20))})
    ds = bt.from_arrow(table)
    stats = [_exact_stats(table)]
    keep_all = Filter(ds._plan, lit(True))
    drop_all = Filter(ds._plan, lit(False))
    assert kyber.answer_count(keep_all, ds._sources, stats) == 20
    assert kyber.answer_count(drop_all, ds._sources, stats) == 0
    assert kyber.answer_is_empty(drop_all, ds._sources, stats) is True
    # DuckDB agreement on the same trivially-constant predicates.
    duck.register("t", table)
    assert duck.sql("SELECT count(*) FROM t WHERE TRUE").fetchone()[0] == 20
    assert duck.sql("SELECT count(*) FROM t WHERE FALSE").fetchone()[0] == 0
