"""A `map_batches` pipeline feeding a breaker distributes, instead of refusing to run.

The UDF pipeline distributes on its own and every relational breaker distributes on its own,
but nothing composed them: the map executors ship a whole sub-plan to each worker, which is
sound only for a map-only one, so `map_batches(...).sort(...)` — and `.distinct()`,
`.limit()`, a partitioned window — matched no dispatch branch and raised `PlanError` on
splittable data. That is the shape of most batch-inference work once the model has run: read,
infer, then order or deduplicate the output.

`_stage_map_prefix` makes it a stage boundary. Each worker writes its own partition of the
UDF output to shared scratch and only file locators come back, so the post-inference rows
never pass through the driver; the breaker above then sees an ordinary splittable Parquet
source and takes the route it always had.

The source must be a real file for any of this to be under test. Over an in-memory source
`_unsupported` runs the plan on one node and returns the right answer, so the gap this closes
is invisible — which is exactly why it survived the in-memory distributed matrix.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")
pytest.importorskip("ray", reason="the distributed path needs Ray")

_W = 2
_N = 200


@pytest.fixture(scope="module")
def parquet_path(tmp_path_factory):
    """A multi-row-group Parquet file, so the source really splits."""
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "k": pa.array([i % 7 if i % 13 else None for i in range(_N)], pa.int64()),
            "v": pa.array([float(i % 11) * 1.5 for i in range(_N)], pa.float64()),
            "g": pa.array([f"g{i % 5}" for i in range(_N)]),
            "w": pa.array([(i * 37) % 101 for i in range(_N)], pa.int64()),
        }
    )
    path = tmp_path_factory.mktemp("mapstage") / "t.parquet"
    pq.write_table(table, path, row_group_size=32)
    return str(path)


def _mapped(path: str):
    """Read, then a batch UDF — the prefix of any inference job.

    The UDF *adds* a column, so a staged plan that landed the pre-UDF rows would be caught on
    the schema rather than only on row order. `output_columns` declares that column to the
    planner, which a UDF's opaque output otherwise cannot state.
    """
    return bt.read_parquet(path).map_batches(
        lambda b: b.append_column("d", pa.array([v * 2 for v in b.column("v").to_pylist()])),
        output_columns=["k", "v", "g", "w", "d"],
    )


def test_the_source_really_splits(parquet_path):
    """Without this the file is vacuous: a non-splittable source hides every missing route."""
    from batcher.dist.executor import _is_splittable_source

    assert _is_splittable_source(bt.read_parquet(parquet_path)._sources[0])


def test_map_then_sort_matches_single_node(parquet_path):
    """Ordered comparison: a sort is exactly what an order-independent assertion cannot see."""

    def build():
        return _mapped(parquet_path).sort("w", "k").select("w", "k", "d")

    assert_tables_equal(
        build().collect(distributed=True, num_workers=_W), build().collect(), ordered=True
    )


def test_map_then_distinct_matches_duckdb(duck, parquet_path):
    duck.register("t", bt.read_parquet(parquet_path).collect())
    ds = _mapped(parquet_path).select("k").distinct()
    got = ds.collect(distributed=True, num_workers=_W)
    assert_same(got, duck.sql("SELECT DISTINCT k FROM t"))


def test_map_then_window_matches_duckdb(duck, parquet_path):
    duck.register("t", bt.read_parquet(parquet_path).collect())
    ds = (
        _mapped(parquet_path)
        .with_columns(s=bt.col("v").sum().over(partition_by="g"))
        .select("g", "v", "s")
    )
    got = ds.collect(distributed=True, num_workers=_W)
    assert_same(got, duck.sql("SELECT g, v, sum(v) OVER (PARTITION BY g) AS s FROM t"))


def test_map_then_limit_returns_the_row_count_single_node_does(parquet_path):
    """A `limit` over a UDF: the row *count* is the contract, not which rows.

    Which rows a bare `LIMIT` returns is unspecified over an unordered relation, and the
    staged path reads its input back from scratch files, so pinning the identities here would
    be pinning undefined behavior. The count is what must agree.
    """
    ds = _mapped(parquet_path).limit(17)
    assert ds.collect(distributed=True, num_workers=_W).num_rows == ds.collect().num_rows == 17


def test_the_udf_output_column_survives_the_stage(parquet_path):
    """The staged scratch must carry the UDF's *output* schema, not the source's.

    Landing the pre-UDF rows would still produce the right row count and pass a count-only
    assertion, so this asserts the column the UDF added and its values.
    """
    got = _mapped(parquet_path).sort("k", "w").collect(distributed=True, num_workers=_W)
    assert "d" in got.column_names
    assert got.column("d").to_pylist() == [v * 2 for v in got.column("v").to_pylist()]
