"""Column-lineage completeness for the join family and streaming row-set operators.

Lineage is a governance answer: "if ``customers.ssn`` is PII, which downstream columns
carry it?" The safe error direction is *over*-approximation — a false "carries PII" costs
a review; a false "does not" costs a breach. These pin two regressions where the analysis
was wrong in the *unsafe* direction, or wrong outright:

* ``AsofJoin`` / ``WatermarkStreamJoin`` were not modeled, so they fell through to the
  opaque catch-all. That catch-all merges each child's lineage with ``dict |=``, so when
  both sides carry a same-named column the later side *overwrites* the earlier one — a
  left-derived output column then reported only right-side origins and **omitted its true
  left origin** (a lineage gap), while also reporting origins it does not have. Both joins
  carry the same ``JoinOutputCol(side, name, alias)`` output as a plain ``Join`` and must
  resolve each output column to the one column of the one side it is fed by.
* ``WatermarkDedup`` (streaming ``distinct``) is a pure row-set operator — it changes which
  rows survive, never a value — yet the catch-all reported every output column as derived
  from every input column (a plain ``id`` marked as carrying an ``ssn`` PII tag).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.governance import column_lineage
from batcher.plan.logical import Scan
from batcher.plan.logical.join import AsofJoin, JoinOutputCol, WatermarkStreamJoin
from batcher.plan.logical.relational import WatermarkDedup
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

_SCHEMA = SchemaRef.from_arrow(pa.schema([("t", pa.int64()), ("value", pa.float64())]))


def _asof() -> AsofJoin:
    out = (
        JoinOutputCol("left", "t", "t"),
        JoinOutputCol("left", "value", "lval"),
        JoinOutputCol("right", "value", "rval"),
    )
    return AsofJoin(Scan(0, _SCHEMA), Scan(1, _SCHEMA), "t", "t", (), (), "backward", out)


def test_asof_join_lineage_attributes_each_side_precisely() -> None:
    lin = column_lineage(_asof(), ["L", "R"])
    # A left-derived output names ONLY its left origin — before the fix it named the
    # right side's columns and dropped ('L', 'value') entirely (a lineage gap).
    assert lin["lval"] == frozenset({("L", "value")})
    assert lin["rval"] == frozenset({("R", "value")})
    assert lin["t"] == frozenset({("L", "t")})


def test_asof_join_left_value_is_not_omitted() -> None:
    # The breach-direction assertion stated plainly: the true origin must be present.
    lin = column_lineage(_asof(), ["L", "R"])
    assert ("L", "value") in lin["lval"]
    assert ("R", "value") not in lin["lval"]


def test_watermark_stream_join_lineage_attributes_each_side() -> None:
    out = (
        JoinOutputCol("left", "value", "lval"),
        JoinOutputCol("right", "value", "rval"),
    )
    node = WatermarkStreamJoin(
        Scan(0, _SCHEMA), Scan(1, _SCHEMA), ("t",), ("t",), out, "t", "t", 1000, 0
    )
    lin = column_lineage(node, ["L", "R"])
    assert lin["lval"] == frozenset({("L", "value")})
    assert lin["rval"] == frozenset({("R", "value")})


def test_watermark_dedup_is_a_row_set_operator() -> None:
    schema = SchemaRef.from_arrow(
        pa.schema([("id", pa.int64()), ("ssn", pa.string()), ("t", pa.int64())])
    )
    node = WatermarkDedup(Scan(0, schema), ("id",), "t", 0)
    lin = column_lineage(node, ["T"])
    # Each column keeps exactly its own origin — `id` does NOT inherit the `ssn` origin.
    assert lin["id"] == frozenset({("T", "id")})
    assert lin["ssn"] == frozenset({("T", "ssn")})
    assert lin["t"] == frozenset({("T", "t")})
