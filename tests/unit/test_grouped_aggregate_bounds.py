"""What is known about a grouped aggregate's *value* outputs.

A grouped aggregate's outputs vary by group, so none is the provable constant a global
aggregate produces — but that is not the same as knowing nothing. A `HAVING` clause is a
predicate on exactly these columns, and with no statistics it falls to a flat constant.
"""

from __future__ import annotations

import pytest

from batcher.kyber.stats.aggregate_columns import grouped_aggregate_columns
from batcher.plan.stats import ColumnStat, Provenance, RelStats

pytestmark = pytest.mark.unit


def _child() -> RelStats:
    return RelStats(
        1000.0,
        Provenance.EXACT,
        {
            "k": ColumnStat(min=1, max=9, provenance=Provenance.EXACT),
            "v": ColumnStat(min=0, max=1000, provenance=Provenance.EXACT),
        },
    )


def _agg(func: str, alias: str = "out"):
    """A `GROUP BY k AGG f(v)` node over a two-column scan."""
    import pyarrow as pa

    from batcher.plan.expr_ir import AggExpr, Col
    from batcher.plan.logical import Aggregate, AggregateSpec, Projection, Scan
    from batcher.plan.schema import SchemaRef

    schema = SchemaRef(pa.schema([("k", pa.int64()), ("v", pa.int64())]))
    agg = AggExpr(func, None if func == "count_star" else Col("v"))
    return Aggregate(
        input=Scan(source_id=0, schema=schema),
        group_keys=(Projection(expr=Col("k"), alias="k"),),
        aggregates=(AggregateSpec(alias=alias, agg=agg),),
    )


def test_an_order_bounded_aggregate_inherits_its_column_bounds():
    """`min`/`max`/`avg` of a column return a value inside that column's own range.

    So `HAVING max(v) > 10^6` over a column whose maximum is 1,000 is provably empty, and a
    `HAVING` on any of them can interpolate instead of falling to a flat constant.
    """
    for func in ("min", "max", "mean"):
        cols = grouped_aggregate_columns(_agg(func), _child())
        assert "out" in cols, f"{func} produced no bounds"
        assert (cols["out"].min, cols["out"].max) == (0, 1000)
        # Bounds on a *set* of per-group values — never exact.
        assert cols["out"].provenance is Provenance.DEFAULT


def test_a_counting_aggregate_is_bounded_by_one_and_the_input_rows():
    cols = grouped_aggregate_columns(_agg("count_star"), _child())
    assert "out" in cols
    assert cols["out"].min == 1
    assert cols["out"].max == 1000


def test_the_group_key_still_carries_its_exact_bounds():
    cols = grouped_aggregate_columns(_agg("max"), _child())
    assert (cols["k"].min, cols["k"].max) == (1, 9)
    assert cols["k"].provenance is Provenance.EXACT
