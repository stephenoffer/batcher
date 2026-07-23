"""An empty result carries the same column *types* whichever executor produced it.

A query returning zero rows still has a schema, and it must be the one a matching run
would have produced. `api`, `dist`, and `core` each need that schema, and — being unable
to import `api` (the import matrix forbids it) — two of them grew their own null-typed
spelling. The result was that the same empty query returned `i: null` or `i: int64`
depending purely on which executor ran, which breaks `concat`, `write_parquet`, and any
typed projection downstream.

These tests pin the shared neutral helper and the absence of the old spelling.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.logical import empty_result_schema
from batcher.plan.schema import placeholder_schema


def _no_match_plan():
    """A plan whose result is empty but whose types are fully inferable."""
    return bt.from_pydict({"i": [1, 2, 3], "v": [1.5, 2.5, 3.5]}).filter(bt.col("i") > 99)._plan


@pytest.mark.unit
def test_empty_result_keeps_inferred_types_not_null():
    # The regression: the old `pa.table({c: [] for c in ...})` spelling typed both null.
    schema = empty_result_schema(_no_match_plan(), ["i", "v"])
    assert schema.field("i").type == pa.int64()
    assert schema.field("v").type == pa.float64()


@pytest.mark.unit
def test_placeholder_is_the_documented_fallback_only():
    # Null placeholders are still correct when the caller's expected columns disagree
    # with what the plan can infer — the guard that keeps inference strictly safer.
    schema = empty_result_schema(_no_match_plan(), ["i", "v", "unexpected"])
    assert schema == placeholder_schema(["i", "v", "unexpected"])


@pytest.mark.unit
def test_adaptive_staging_agrees_with_the_relational_executor():
    # `staging._table` is the copy that diverged; assert it now matches the shared helper
    # rather than re-deriving null-typed columns.
    from batcher.api.adaptive.staging import _table

    plan = _no_match_plan()
    assert _table([], plan).schema == empty_result_schema(plan, plan.available_columns())


@pytest.mark.unit
def test_api_private_aliases_point_at_the_neutral_helper():
    # `api` keeps its private names, but they must not be a second implementation.
    from batcher.api._join_helpers import _empty_result_schema, _empty_schema

    assert _empty_result_schema is empty_result_schema
    assert _empty_schema is placeholder_schema


@pytest.mark.unit
def test_empty_collect_is_typed_end_to_end():
    # The user-visible contract: an empty collect() is not all-null.
    ds = bt.from_pydict({"i": [1, 2, 3], "v": [1.5, 2.5, 3.5]})
    table = ds.filter(bt.col("i") > 99).collect()
    assert table.num_rows == 0
    assert table.schema.field("i").type == pa.int64()
    assert table.schema.field("v").type == pa.float64()
