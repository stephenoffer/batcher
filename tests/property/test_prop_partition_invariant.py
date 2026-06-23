"""Property: results never depend on how the input is chunked.

Partition-independence is the load-bearing invariant for distribution — if a query
gives the same answer over one morsel and over many, splitting the input across
partitions/actors is safe. ``tests/integration/test_partition_independence.py`` pins
this with hand-picked cases; here Hypothesis searches the space, generating random
tables, group counts, null densities, and chunk counts and asserting the answer is
chunk-invariant. A counterexample is a distribution-correctness bug.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher import col, count

pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = [pytest.mark.property, pytest.mark.integration]

# Small but varied: enough rows to span several chunks, bounded so the suite stays fast.
_values = st.integers(min_value=-50, max_value=50)
_nullable = st.one_of(st.none(), _values)


# Explicit schema so a zero-row draw yields typed-empty columns (not Null-typed,
# which pa.table infers from empty Python lists and which no aggregate accepts).
_SCHEMA = pa.schema([("g", pa.int64()), ("v", pa.int64()), ("w", pa.int64())])


@st.composite
def _grouped_table(draw: st.DrawFn) -> tuple[pa.Table, int]:
    """A (table, n_chunks) pair: a `g` group key, a dense `v`, and a nullable `w`."""
    n = draw(st.integers(min_value=0, max_value=80))
    n_groups = draw(st.integers(min_value=1, max_value=6))
    g = draw(st.lists(st.integers(min_value=0, max_value=n_groups - 1), min_size=n, max_size=n))
    v = draw(st.lists(_values, min_size=n, max_size=n))
    w = draw(st.lists(_nullable, min_size=n, max_size=n))
    n_chunks = draw(st.integers(min_value=1, max_value=7))
    return pa.table({"g": g, "v": v, "w": w}, schema=_SCHEMA), n_chunks


def _load(table: pa.Table, batches: list[pa.RecordBatch]):
    """Build a Dataset from batches, falling back to the (schema-bearing) empty Table."""
    return bt.from_arrow(batches) if batches else bt.from_arrow(table)


def _chunks(table: pa.Table, n: int) -> list[pa.RecordBatch]:
    rows = table.num_rows
    size = max(1, (rows + n - 1) // n)
    return table.combine_chunks().to_batches(max_chunksize=size)


def _rowset(table: pa.Table) -> list[tuple]:
    cols = table.column_names
    rows = [tuple(r[c] for c in cols) for r in table.to_pylist()]
    return sorted(rows, key=lambda t: tuple((x is None, str(x)) for x in t))


def _assert_chunk_invariant(build, table: pa.Table, n_chunks: int) -> None:
    one = _rowset(build(_load(table, table.combine_chunks().to_batches())).collect())
    many = _rowset(build(_load(table, _chunks(table, n_chunks))).collect())
    assert one == many, f"\n1-chunk:  {one}\nN-chunk:  {many}"


# Hypothesis re-draws per example; the engine setup per example is the slow part, so
# cap the example count and silence the function-scoped-fixture health check (there
# are no fixtures here — the importorskip is module-level).
_PROP = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@_PROP
@given(_grouped_table())
def test_aggregate_chunk_invariant(case: tuple[pa.Table, int]) -> None:
    table, n_chunks = case
    _assert_chunk_invariant(
        lambda ds: ds.group_by("g").agg(
            s=col("v").sum(),
            n=count(),
            a=col("v").mean(),
            lo=col("v").min(),
            hi=col("v").max(),
            nd=col("v").approx_n_unique(),
        ),
        table,
        n_chunks,
    )


@_PROP
@given(_grouped_table())
def test_filter_project_chunk_invariant(case: tuple[pa.Table, int]) -> None:
    table, n_chunks = case
    _assert_chunk_invariant(
        lambda ds: ds.filter(col("v") >= 0).select("g", t=col("v") + col("g")),
        table,
        n_chunks,
    )


@_PROP
@given(_grouped_table())
def test_distinct_chunk_invariant(case: tuple[pa.Table, int]) -> None:
    table, n_chunks = case
    _assert_chunk_invariant(lambda ds: ds.select("g", "v").distinct(), table, n_chunks)
