"""`list.sum` preserves the element type, matching DuckDB and Batcher's own execution.

This pins a schema-inference fix. `Dataset.schema` infers a projection's output types
without running it (so a plan can be described cheaply), and that inference had
`list.sum` classified as always-`double`. The engine actually returns the element type
(an Int list sums to Int64, as DuckDB's ``list_sum`` does), so the inferred schema
disagreed with the executed one — the exact silent divergence
`tests/unit/test_available_schema.py` guards. The value is asserted against DuckDB here
so the "sum of ints is an int" contract can't quietly regress to float.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential

_DATA = pa.table({"arr": [[1, 2, 3], [4, 5, 6], [10, 20, 30]]})


def test_list_sum_of_ints_stays_integer_vs_duckdb(duck):
    duck.register("t", _DATA)
    out = bt.from_arrow(_DATA).with_columns(s=col("arr").list.sum()).select("s").collect()
    assert_same(out, duck.sql("SELECT list_sum(arr) AS s FROM t"))


def test_inferred_schema_matches_execution_for_list_sum():
    # The inference path (no execution) must agree with the executed schema — the
    # property that broke. Both must report an integer type for an Int list's sum.
    ds = bt.from_arrow(_DATA).with_columns(s=col("arr").list.sum())
    inferred = {f.name: str(f.type) for f in ds.schema}
    executed = {f.name: str(f.type) for f in ds.collect().schema}
    assert inferred == executed
    assert inferred["s"] == "int64"


def test_list_mean_stays_float_vs_duckdb(duck):
    # The counter-case: mean is genuinely float even for an Int list, and the fix must
    # not have swept it into the element-preserving bucket.
    duck.register("t", _DATA)
    out = bt.from_arrow(_DATA).with_columns(m=col("arr").list.mean()).select("m").collect()
    assert_same(out, duck.sql("SELECT list_avg(arr) AS m FROM t"))
    ds = bt.from_arrow(_DATA).with_columns(m=col("arr").list.mean())
    assert {f.name: str(f.type) for f in ds.schema}["m"] == "double"
