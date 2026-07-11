"""A `Dataset` is consumable by any Arrow-speaking engine, with no `to_arrow()` and no copy.

`__arrow_c_stream__` is the bridge out of Batcher. Without it, handing a result to Polars
or DuckDB means materializing the whole thing through `to_arrow()` first — which is both
a copy and a memory ceiling. With it, the consumer pulls batches from the lazy plan, so a
result larger than memory streams into DuckDB rather than landing in it.

These pin the protocol itself (a real PyCapsule, exported lazily) and the three consumers
a data engineer actually reaches for.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


@pytest.fixture
def ds():
    return bt.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_the_export_is_an_arrow_pycapsule(ds):
    capsule = ds.__arrow_c_stream__()
    assert type(capsule).__name__ == "PyCapsule"


def test_pyarrow_consumes_a_dataset_directly(ds):
    table = pa.table(ds)
    assert table.num_rows == 3
    assert table.column_names == ["a", "b"]


def test_polars_consumes_a_dataset_directly(ds):
    pl = pytest.importorskip("polars")
    assert pl.DataFrame(ds).to_dict(as_series=False) == {"a": [1, 2, 3], "b": ["x", "y", "z"]}


def test_duckdb_queries_a_dataset_directly(ds):
    duckdb = pytest.importorskip("duckdb")
    assert duckdb.sql("SELECT sum(a) AS s FROM ds").fetchone()[0] == 6


def test_duckdb_queries_a_lazy_plan_without_materializing_it():
    """The plan has not executed until the consumer pulls; no `to_arrow()` anywhere."""
    duckdb = pytest.importorskip("duckdb")
    lazy = bt.range(0, 10_000).filter(bt.col("value") % 7 == 0)  # noqa: F841 - read by SQL
    assert duckdb.sql("SELECT count(*) FROM lazy").fetchone()[0] == 1429


def test_the_stream_is_pulled_incrementally_not_materialized_up_front():
    """A consumer that reads one batch and stops must not have paid for the whole result.

    This is the difference between the capsule and `to_arrow()`: the reader holds the
    plan's batch iterator, so `read_next_batch` pulls exactly one morsel.
    """
    source = pa.table({"a": list(range(1000))}).to_batches(max_chunksize=100)
    reader = pa.RecordBatchReader.from_stream(bt.from_arrow(source))
    first = reader.read_next_batch()
    assert first.num_rows == 100, "one morsel, not the whole 1000-row result"
    reader.close()


def test_the_exported_schema_matches_the_dataset_schema(ds):
    assert pa.table(ds).schema == ds.schema


def test_an_empty_result_exports_an_empty_table_with_the_right_schema(ds):
    empty = ds.filter(bt.col("a") > 99)
    table = pa.table(empty)
    assert table.num_rows == 0
    assert table.column_names == ["a", "b"]


def test_a_derived_lazy_plan_exports_its_computed_columns(ds):
    table = pa.table(ds.select("a", doubled=bt.col("a") * 2))
    assert table.column("doubled").to_pylist() == [2, 4, 6]


def test_the_round_trip_through_another_engine_is_lossless(ds):
    pl = pytest.importorskip("polars")
    back = bt.from_arrow(pl.DataFrame(ds).to_arrow())
    assert back.to_pydict() == ds.to_pydict()
