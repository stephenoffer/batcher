"""Property: the mergeable algebra is partition-independent (single-node == many chunks).

Stateful operators are built as ``partial → combine → finalize`` with an
associative-commutative ``combine`` (``.claude/rules/rust-engine.md``). That is the
one invariant that lets the *same* implementation serve one core, many cores, and many
machines: an aggregate/distinct computed over one morsel must equal the same computed
over any random chunking of the input, and both must equal DuckDB. ``tests/property/
test_prop_partition_invariant.py`` pins the basic case; this generalizes it to a wider
aggregate set (sum/count/min/max/mean/std/var/median/count-distinct, multi-key),
distinct, and sort-limit, and — because the native distributed primitives are directly
callable — also drives ``combine_finalize(partial(pₖ))`` over the raw Rust kernels and
asserts it equals the single-node result. A counterexample is a distribution-
correctness bug (``combine`` is not truly mergeable).
"""

from __future__ import annotations

import json

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher import col, count
from batcher.config import active_config

nat = pytest.importorskip("batcher._native", reason="native engine not built")
duckdb = pytest.importorskip("duckdb")

pytestmark = [pytest.mark.property, pytest.mark.integration]

_vals = st.integers(min_value=-40, max_value=40)
_nullable = st.one_of(st.none(), _vals)
_SCHEMA = pa.schema([("g", pa.int64()), ("h", pa.int64()), ("v", pa.int64())])


def _coerce(v: object) -> object:
    if isinstance(v, bool):
        return v
    try:
        return round(float(v), 9)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return v


def _rowset(table: pa.Table) -> list[tuple]:
    cols = table.column_names
    rows = [tuple(_coerce(r[c]) for c in cols) for r in table.to_pylist()]
    return sorted(rows, key=lambda t: tuple((x is None, str(x)) for x in t))


@st.composite
def _case(draw: st.DrawFn) -> tuple[pa.Table, int]:
    """A (two-key grouped table, chunk count) pair with a nullable value column."""
    n = draw(st.integers(min_value=0, max_value=90))
    g = draw(st.lists(st.integers(min_value=0, max_value=5), min_size=n, max_size=n))
    h = draw(st.lists(st.integers(min_value=0, max_value=3), min_size=n, max_size=n))
    v = draw(st.lists(_nullable, min_size=n, max_size=n))
    n_chunks = draw(st.integers(min_value=1, max_value=8))
    return pa.table({"g": g, "h": h, "v": v}, schema=_SCHEMA), n_chunks


def _chunks(table: pa.Table, n: int) -> list[pa.RecordBatch]:
    rows = table.num_rows
    size = max(1, (rows + n - 1) // n)
    return table.combine_chunks().to_batches(max_chunksize=size)


def _load(batches: list[pa.RecordBatch], table: pa.Table) -> bt.Dataset:
    return bt.from_arrow(batches) if batches else bt.from_arrow(table)


def _aggregate(ds: bt.Dataset) -> bt.Dataset:
    """The multi-key, multi-aggregate stress query (every mergeable moment kind)."""
    return ds.group_by("g", "h").agg(
        s=col("v").sum(),
        n=count(),
        c=col("v").count(),
        lo=col("v").min(),
        hi=col("v").max(),
        a=col("v").mean(),
        sd=col("v").std(),
        vv=col("v").var(),
        md=col("v").median(),
        nd=col("v").n_unique(),
    )


def _duckdb_aggregate(table: pa.Table) -> list[tuple]:
    con = duckdb.connect()
    try:
        con.register("t", table)
        out = con.sql(
            "SELECT g, h, SUM(v) s, COUNT(*) n, COUNT(v) c, MIN(v) lo, MAX(v) hi, "
            "AVG(v) a, STDDEV_SAMP(v) sd, VAR_SAMP(v) vv, MEDIAN(v) md, "
            "COUNT(DISTINCT v) nd FROM t GROUP BY g, h"
        ).to_arrow_table()
    finally:
        con.close()
    return _rowset(out)


def _native_chunked_aggregate(table: pa.Table, chunks: list[pa.RecordBatch]) -> list[tuple]:
    """`combine_finalize(partial(pₖ))` driven directly over the Rust primitives.

    Builds the same group-keys/aggregates IR the streaming folder builds, runs each
    chunk through ``partial_aggregate``, and merges the partials with a single
    ``combine_finalize`` — the exact mergeable path the distributed executor composes.
    """
    agg = _aggregate(_load([], table))._plan
    gk = json.dumps([{"expr": k.expr.to_ir(), "alias": k.alias} for k in agg.group_keys])
    ag = json.dumps([spec.agg.to_ir(spec.alias) for spec in agg.aggregates])
    input_ir = json.dumps(agg.input.to_ir())  # a bare scan of source 0
    cfg = active_config().engine_config_json()
    partials = []
    for batch in chunks:
        if batch.num_rows == 0:
            continue
        rows = nat.execute_plan(input_ir, [[batch]], cfg)
        if rows and sum(b.num_rows for b in rows):
            partials.append(nat.partial_aggregate(gk, ag, rows))
    if not partials:
        return []
    merged = nat.combine_finalize(gk, ag, partials)
    return _rowset(pa.Table.from_batches([merged]) if merged.num_rows else _empty_like(merged))


def _empty_like(batch: pa.RecordBatch) -> pa.Table:
    return pa.Table.from_batches([], schema=batch.schema)


_PROP = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@_PROP
@given(_case())
def test_aggregate_chunk_invariant_and_oracle(case: tuple[pa.Table, int]) -> None:
    """Aggregate over 1 chunk == over N random chunks == DuckDB."""
    table, n_chunks = case
    one = _rowset(_aggregate(_load(table.combine_chunks().to_batches(), table)).collect())
    many = _rowset(_aggregate(_load(_chunks(table, n_chunks), table)).collect())
    assert one == many, f"chunking changed the aggregate:\n1: {one}\nN: {many}"
    assert one == _duckdb_aggregate(table), f"aggregate != DuckDB:\nbatcher: {one}"


@_PROP
@given(_case())
def test_native_mergeable_primitive(case: tuple[pa.Table, int]) -> None:
    """`combine_finalize(partial(pₖ))` over the raw Rust kernels == single-node."""
    table, n_chunks = case
    single = _rowset(_aggregate(_load(table.combine_chunks().to_batches(), table)).collect())
    native = _native_chunked_aggregate(table, _chunks(table, n_chunks))
    assert native == single, (
        f"native combine_finalize(partial(pₖ)) != single-node:\nnative: {native}\nsingle: {single}"
    )


@_PROP
@given(_case())
def test_distinct_chunk_invariant_and_oracle(case: tuple[pa.Table, int]) -> None:
    """Multi-column distinct over 1 chunk == over N chunks == DuckDB."""
    table, n_chunks = case

    def _distinct(ds: bt.Dataset) -> list[tuple]:
        return _rowset(ds.select("g", "v").distinct().collect())

    one = _distinct(_load(table.combine_chunks().to_batches(), table))
    many = _distinct(_load(_chunks(table, n_chunks), table))
    assert one == many, f"chunking changed distinct:\n1: {one}\nN: {many}"
    con = duckdb.connect()
    try:
        con.register("t", table)
        expected = _rowset(con.sql("SELECT DISTINCT g, v FROM t").to_arrow_table())
    finally:
        con.close()
    assert one == expected, f"distinct != DuckDB:\nbatcher: {one}\nduckdb : {expected}"


@_PROP
@given(_case(), st.integers(min_value=0, max_value=20))
def test_sort_limit_chunk_invariant(case: tuple[pa.Table, int], k: int) -> None:
    """A total-order sort + limit is chunk-invariant (ordered comparison).

    Sorting on all columns gives a total order over distinct rows; tied rows are
    byte-identical, so an unstable tie-break cannot change the row *values* — the
    ordered result is deterministic regardless of how the input was chunked.
    """
    table, n_chunks = case

    def _sorted(ds: bt.Dataset) -> list[tuple]:
        out = ds.sort("g", "h", "v").limit(k).collect()
        cols = out.column_names
        return [tuple(_coerce(r[c]) for c in cols) for r in out.to_pylist()]

    one = _sorted(_load(table.combine_chunks().to_batches(), table))
    many = _sorted(_load(_chunks(table, n_chunks), table))
    assert one == many, f"chunking changed sort-limit:\n1: {one}\nN: {many}"
