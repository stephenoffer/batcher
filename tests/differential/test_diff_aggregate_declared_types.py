"""Every aggregate's *declared* output type is the one the engine actually produces.

`Aggregate.available_schema` is the control plane's static answer for what a `GROUP BY`
returns, and it is not a convenience. `Dataset.schema` is answered from it; an empty result
is *typed* from it (`plan.logical.empty_result_schema`), so a query that matches no rows
returns `null`-typed columns wherever the inference declines; the device tier holds every GPU
result against it; and the sort/window predicates that decide whether a plan can spill or
distribute read the key's type out of it, so an unknown type is a declined bounded-memory
path rather than a cosmetic gap.

Nine of the 39 members of `AGG_FNS` had no classification at all — `any_value`, `entropy`,
`histogram`, `kahan_sum`, `kurtosis_pop`, `list_agg`, `mad`, `quantile_disc` and
`approx_top_k`. Nothing failed. `group_by(k).agg(v=col.mad())` simply reported no schema, and
the consequences arrived somewhere else entirely: `collect()` on an empty relation returned
`k: int64, v: double` while `collect(spill=True)` returned `k: null, v: null`, because the
spill path types its empty result from the declaration and the in-memory path gets its types
from the engine.

This file is the reconciliation that keeps the table honest. It does not restate the rules —
restating them is what produced the drift. It asks the *engine* for the type, on every
aggregate the vocabulary names crossed with every input type it accepts, and requires the
declaration to agree or to be explicitly, individually admitted as unknown. A new member of
`AGG_FNS` therefore fails here until someone classifies it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.expr_ir import AggExpr
from batcher.plan.ir_tags import AGG_FNS

pytestmark = pytest.mark.differential

#: One column per input type family an aggregate might be handed. `z` (the all-null column)
#: is not padding: a `null`-typed input is the one case where the value-preserving aggregates
#: do not preserve their input, and it is where the declaration is most easily wrong.
_COLUMNS: dict[str, pa.Array] = {
    "i32": pa.array([1, 2, 3, 4], pa.int32()),
    "i64": pa.array([1, 2, 3, 4], pa.int64()),
    "f32": pa.array([1.0, 2.0, 3.0, 4.0], pa.float32()),
    "f64": pa.array([1.0, 2.0, 3.0, 4.0], pa.float64()),
    "s": pa.array(["a", "b", "a", "c"]),
    "ts": pa.array([1, 2, 3, 4], pa.timestamp("us")),
    "d": pa.array([1, 2, 3, 4], pa.date32()),
    "bo": pa.array([True, False, True, True], pa.bool_()),
    "z": pa.array([None] * 4, pa.null()),
}

#: Aggregates whose output type the control plane is allowed not to know, each with the reason
#: it cannot. **Empty today, and that is the point** — every entry costs a `null`-typed empty
#: result and a declined spill on any plan that sorts the output. An addition here is a
#: decision to surface, not a way to make this file pass.
_UNDECLARED: dict[str, str] = {}


@pytest.fixture(scope="module")
def rows() -> pa.Table:
    return pa.table({**_COLUMNS, "g": pa.array([1, 1, 2, 2], pa.int64())})


def _pairs(rows: pa.Table, func: str) -> list[tuple[str, pa.DataType, pa.DataType | None]]:
    """Every `(column, engine type, declared type)` this aggregate accepts."""
    out = []
    for name in _COLUMNS:
        for second in (None, bt.col("f64")):
            try:
                # Building is inside the guard, not just collecting: a pair the accumulators
                # cannot mean anything over (`SUM` of a string) is refused by the plan's own
                # domain validation at `agg()` time, before there is a dataset to collect.
                agg = AggExpr(func, bt.col(name), input2=second)
                dataset = bt.from_arrow(rows).group_by("g").agg(v=agg)
                actual = dataset.collect().schema.field("v").type
            except Exception:
                # The unary form first, then the binary one. Trying both rather than listing
                # which aggregates take a second argument is what keeps a newly added binary
                # aggregate visible here instead of silently unreachable — this file's whole
                # value is that a new member of the vocabulary cannot slip past it.
                continue
            schema = dataset._plan.available_schema()
            out.append((name, actual, schema.arrow.field("v").type if schema else None))
            break
    return out


@pytest.mark.parametrize("func", sorted(AGG_FNS))
def test_the_declared_type_is_the_engine_type(rows, func):
    accepted = _pairs(rows, func)
    assert accepted, f"{func} accepted none of the input types — the fixture cannot see it"
    for name, actual, declared in accepted:
        if declared is None:
            assert func in _UNDECLARED, (
                f"{func} over {name!r} declares no output type (the engine returns {actual}); "
                "classify it in `plan.logical.aggregate`, or admit it in `_UNDECLARED` with "
                "the reason — an unknown type costs a null-typed empty result and a declined "
                "out-of-core sort on anything that orders by it"
            )
            continue
        assert declared == actual, (
            f"{func} over {name!r}: declared {declared}, engine returns {actual}"
        )


def test_the_fixture_reaches_every_input_family(rows):
    """Guard against a vacuous sweep.

    Every assertion above is "declared == actual for the pairs the engine accepts". If the
    fixture stopped producing usable columns — a renamed column, a type the reader no longer
    builds — each parametrization would find nothing to compare and the file would pass while
    checking nothing. `sum` accepts the numeric families and refuses the rest, so it is a
    positive control for both halves.
    """
    accepted = {name for name, _actual, _declared in _pairs(rows, "sum")}
    assert {"i32", "i64", "f32", "f64"} <= accepted
    assert "s" not in accepted


def test_every_aggregate_is_classified():
    """No member of the vocabulary may be silently unclassified.

    The per-function test above only sees a gap on an input the engine accepts. This states
    the invariant directly: the union of the classification sets covers `AGG_FNS`, so adding
    a function to the vocabulary without giving it an output type fails here immediately
    rather than at the first empty result someone happens to spill.
    """
    from batcher.plan.logical.aggregate import (
        _AGG_BOOL,
        _AGG_FLOAT,
        _AGG_INPUT,
        _AGG_INT,
        _AGG_LIST_OF_INPUT,
        _AGG_MAP_COUNT_OF_INPUT,
        _AGG_WIDEN_INPUT,
    )

    classified = (
        _AGG_INT
        | _AGG_FLOAT
        | _AGG_BOOL
        | _AGG_INPUT
        | _AGG_WIDEN_INPUT
        | _AGG_LIST_OF_INPUT
        | _AGG_MAP_COUNT_OF_INPUT
    )
    assert sorted(AGG_FNS - classified) == sorted(_UNDECLARED)
    assert not classified - AGG_FNS, "a classification names a function the vocabulary lost"
