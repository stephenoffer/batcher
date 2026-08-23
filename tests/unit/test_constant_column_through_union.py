"""A comma join's `__cross_key` must stay recognizable across a `UNION ALL`.

`constant_column_value` proves an output column holds one literal in every row, and
`is_cartesian_key_pair` uses that to tell a comma/cross join's synthetic `__cross_key` from a
genuine equi-key. `drop_redundant_cross_key` then removes the pseudo-key once a real one
drives the join, which is what lets the join take the single-key fast path.

The tracer walked `Project`, `Filter`/`Sort`/`Limit`/`Sample`/`Distinct` and inner/semi/anti
`Join` — but not `Union`, and projection pushdown moves `__cross_key = lit(1)` *into* each
branch, so above a union the column is a bare `Col` with no `Project` left to prove it. The
pseudo-key then read as a real one and survived, leaving the join on a composite
`(__cross_key, real_key)`: a two-column key, so no dense direct map, plus a constant column
materialized for every probe row. Measured on TPC-DS sf1, the same 15-row-build join over
`store_sales`: 5.7 ms with a plain scan on the probe side against 48.5 ms with a two-branch
`UNION ALL`.

TPC-DS reaches this on every sales/returns union — q5, q77, q80 and more — because its
queries are written with comma joins.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.plan.expr_ir import Lit, col
from batcher.plan.logical import Project, Projection, Scan, Union
from batcher.plan.logical.transforms import constant_column_value, is_cartesian_key_pair
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit


def _leaf(value: int, source_id: int = 0):
    """A scan with a `k` column, projected to add a constant `c` — one union branch."""
    schema = SchemaRef.from_arrow(pa.schema([pa.field("k", pa.int64())]))
    scan = Scan(source_id=source_id, schema=schema)
    return Project(
        scan,
        (
            Projection("k", col("k")),
            Projection("c", Lit(value)),
        ),
    )


def _is_constant(plan, column: str, want: object) -> bool:
    got = constant_column_value(plan, column)
    return got == want


def test_a_union_of_branches_holding_the_same_constant_is_constant():
    """Every branch proves the same literal, so the union's output column holds it too."""
    u = Union((_leaf(1, 0), _leaf(1, 1)))
    assert _is_constant(u, "c", 1)


def test_a_union_whose_branches_disagree_is_not_constant():
    """Different literals per branch means the output column is not one value."""
    u = Union((_leaf(1, 0), _leaf(2, 1)))
    assert constant_column_value(u, "c") != 1
    assert constant_column_value(u, "c") != 2


def test_a_union_with_one_unprovable_branch_is_not_constant():
    """A branch that only passes the column through proves nothing, so neither does the union
    — a partial proof is not a proof."""
    schema = SchemaRef.from_arrow(pa.schema([pa.field("k", pa.int64()), pa.field("c", pa.int64())]))
    passthrough = Scan(source_id=2, schema=schema)
    u = Union((_leaf(1, 0), passthrough))
    assert constant_column_value(u, "c") != 1


def test_the_cross_key_pair_is_recognized_across_a_union():
    """The whole point: a pseudo-edge whose left side is a union is still a pseudo-edge.

    Without the union arm this returned False and `drop_redundant_cross_key` left the join on
    a composite key.
    """
    left = Union((_leaf(1, 0), _leaf(1, 1)))
    right = _leaf(1, 2)
    assert is_cartesian_key_pair(left, "c", right, "c")


def test_a_real_key_across_a_union_is_not_a_pseudo_edge():
    """The other direction, which is what keeps this from deleting genuine join keys: `k` is
    not constant in either branch, so the pair is a real edge."""
    left = Union((_leaf(1, 0), _leaf(1, 1)))
    right = _leaf(1, 2)
    assert not is_cartesian_key_pair(left, "k", right, "k")
