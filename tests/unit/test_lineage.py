"""Column-level lineage: which source columns each output column derives from.

Two properties carry the design and both are asserted here:

* **Data flow, not control flow.** Filtering on `ssn` does not put `ssn` into the lineage
  of the surviving columns — its values never reach the output.
* **Over-approximate, never under-approximate.** An operator the analysis does not model
  (`map_batches`) is assumed to derive every output column from every input column. A
  false "this might carry PII" costs a review; a false "it cannot" costs a breach.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.governance import column_lineage
from batcher.plan.expr_ir import Col, count, lit
from batcher.plan.logical import (
    Aggregate,
    AggregateSpec,
    Filter,
    Join,
    JoinOutputCol,
    Project,
    Projection,
    RowId,
    Scan,
    Union,
)
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

T = "people.parquet"
U = "orders.parquet"
_LEFT = SchemaRef.from_arrow(
    pa.schema([("id", pa.int64()), ("ssn", pa.string()), ("age", pa.int64())])
)
_RIGHT = SchemaRef.from_arrow(pa.schema([("id", pa.int64()), ("amt", pa.int64())]))


def _lin(plan, tables=(T, U)):
    return column_lineage(plan, list(tables))


def test_a_scan_column_originates_in_itself():
    assert _lin(Scan(0, _LEFT)) == {
        "id": frozenset({(T, "id")}),
        "ssn": frozenset({(T, "ssn")}),
        "age": frozenset({(T, "age")}),
    }


def test_a_derived_column_carries_every_column_it_reads():
    plan = Project(Scan(0, _LEFT), (Projection("both", Col("ssn").str.len() + Col("age")),))
    assert _lin(plan)["both"] == frozenset({(T, "ssn"), (T, "age")})


def test_a_literal_column_has_no_origin():
    plan = Project(Scan(0, _LEFT), (Projection("one", lit(1)),))
    assert _lin(plan)["one"] == frozenset()


def test_a_generated_row_index_has_no_origin():
    assert _lin(RowId(Scan(0, _LEFT), "idx"))["idx"] == frozenset()


def test_filtering_on_a_column_does_not_put_it_in_the_lineage():
    """Control flow, not data flow: `ssn`'s values never reach `age`."""
    plan = Project(Filter(Scan(0, _LEFT), Col("ssn") == "x"), (Projection("age", Col("age")),))
    assert _lin(plan)["age"] == frozenset({(T, "age")})


def test_an_aggregate_traces_its_group_keys_and_its_inputs():
    plan = Aggregate(
        Scan(0, _LEFT),
        (Projection("id", Col("id")),),
        (AggregateSpec("total", Col("age").sum()), AggregateSpec("n", count())),
    )
    got = _lin(plan)
    assert got["id"] == frozenset({(T, "id")})
    assert got["total"] == frozenset({(T, "age")})
    assert got["n"] == frozenset()  # count(*) reads no column


def test_a_join_traces_each_output_column_to_its_own_side():
    plan = Join(
        Scan(0, _LEFT),
        Scan(1, _RIGHT),
        ("id",),
        ("id",),
        "inner",
        (JoinOutputCol("left", "ssn", "ssn"), JoinOutputCol("right", "amt", "amt")),
    )
    got = _lin(plan)
    assert got["ssn"] == frozenset({(T, "ssn")})
    assert got["amt"] == frozenset({(U, "amt")})


def test_a_union_merges_the_origins_of_each_positional_column():
    left = Project(Scan(0, _LEFT), (Projection("v", Col("age")),))
    right = Project(Scan(1, _RIGHT), (Projection("v", Col("amt")),))
    assert _lin(Union((left, right)))["v"] == frozenset({(T, "age"), (U, "amt")})


def test_an_unmodelled_operator_over_approximates(tmp_path):
    """`map_batches` is opaque, so every output column inherits every input origin."""
    path = str(tmp_path / "people.parquet")
    bt.from_pydict({"id": [1], "ssn": ["x"], "age": [3]}).write(path, format="parquet")
    got = bt.read.parquet(path).map_batches(lambda b: b).lineage()
    everything = sorted(f"{path}.{c}" for c in ("age", "id", "ssn"))
    assert got["id"] == everything
    assert got["ssn"] == everything


# --- The public `Dataset.lineage()` view ---------------------------------------
def test_lineage_names_the_table_by_the_path_it_is_read_from(tmp_path):
    path = str(tmp_path / "people.parquet")
    bt.from_pydict({"first": ["a"], "last": ["b"]}).write(path, format="parquet")
    ds = bt.read.parquet(path).select(name=bt.concat(bt.col("first"), bt.col("last")))
    assert ds.lineage() == {"name": sorted([f"{path}.first", f"{path}.last"])}


def test_an_in_memory_source_has_no_table_name_so_it_is_labelled_positionally():
    ds = bt.from_pydict({"a": [1]}).select(b=bt.col("a"))
    assert ds.lineage() == {"b": ["<source 0>.a"]}


def test_lineage_executes_nothing(tmp_path):
    """It is a plan analysis: a dataset that would fail to run still reports lineage."""
    path = str(tmp_path / "missing.parquet")
    bt.from_pydict({"a": [1]}).write(path, format="parquet")
    ds = bt.read.parquet(path).select(b=bt.col("a"))
    import os

    os.remove(path)
    assert ds.lineage() == {"b": [f"{path}.a"]}


def test_a_pii_column_can_be_traced_to_every_downstream_column_that_carries_it(tmp_path):
    """The governance question this exists to answer."""
    path = str(tmp_path / "customers.parquet")
    bt.from_pydict({"id": [1], "ssn": ["x"], "city": ["NYC"]}).write(path, format="parquet")
    derived = bt.read.parquet(path).select(
        key=bt.hmac_sha256(bt.col("ssn"), key="k"),
        city=bt.col("city"),
        pair=bt.concat(bt.col("ssn"), bt.col("city")),
    )
    pii = f"{path}.ssn"
    carrying = {col for col, origins in derived.lineage().items() if pii in origins}
    assert carrying == {"key", "pair"}
