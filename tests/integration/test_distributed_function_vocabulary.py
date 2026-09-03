"""Every aggregate and window function, distributed, against the single-node answer.

`tests/differential/test_diff_function_vocabulary_paths.py` crosses the two function
vocabularies with `collect()` / `collect(spill=True)` / `iter_batches()`. This asks the third
question, the one that needs a cluster: does the same function, on the same rows, give the
same answer when the work is fanned across workers?

It is the question the per-operator distributed tests cannot answer. They run one aggregate
and one window function, which is the right economy for testing *routing*; a gap in the
function vocabulary hides underneath a perfectly routed operator. The shape that motivated
this file is exactly that: `percent_rank`, `cume_dist`, `ntile` and `last_value` had no
ordered-bucket decomposition, so a global window carrying one raised `PlanError` on
distributed data while `window` itself distributed fine.

Two properties are asserted, and the second matters as much as the first:

* every function that has a distributed path returns the single-node answer;
* the functions that do **not** have one are enumerated, and each raises rather than running
  the whole relation on one node behind the user's back. That list is the engine's honest
  remaining gap, and holding it exactly means a function that *gains* a path fails here until
  it is moved -- which is how `lag` was found to have left the list.

The source is a multi-file Parquet directory on purpose: `dist.executor._unsupported` runs an
in-memory source on one node by design, so a missing route over `bt.from_arrow` would be
indistinguishable from the right answer.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import AggExpr
from batcher.plan.ir_tags import AGG_FNS
from batcher.plan.logical.window import WINDOW_FUNCS

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="ray not installed")

_N = 3000
_WORKERS = 2

#: The global-window functions with no ordered-bucket decomposition, and why each resists one.
#: A rolling scalar or a bounded boundary tail covers the rest; these need something neither
#: gives. Entries are removed when a decomposition lands, never added to make a run pass.
_NO_GLOBAL_DECOMPOSITION = {
    "lead": "reads the ordered bucket the offset walk has not reached",
    "backward_fill": "same direction as `lead`: the value comes from later rows",
    "interpolate": "needs the bounding value on both sides of a gap",
    "count_distinct": "a running distinct count is not recoverable from per-bucket counts",
    "median": "a running median needs the whole prefix, not a summary of it",
    "rle_id": "a run continues across a cut only if the boundary values match",
    "product": (
        "deliberately excluded on numerical grounds: over a few thousand values it overflows "
        "to inf and underflows to 0, and the two association orders disagree on inf * 0"
    ),
    "forward_fill": "carries the last non-null across a cut; no exchange written for it yet",
}


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def splittable(cluster_scratch) -> str:
    table = pa.table(
        {
            "rid": pa.array(range(_N), pa.int64()),
            "g": pa.array([i % 7 for i in range(_N)], pa.int64()),
            # Duplicated, so `ORDER BY o` has peer groups a bucket cut could straddle.
            "o": pa.array([(i * 37) % 1009 for i in range(_N)], pa.int64()),
            "k": pa.array([None if i % 9 == 0 else (i * 37) % 101 for i in range(_N)], pa.int64()),
            "f": pa.array([float(i % 29) for i in range(_N)], pa.float64()),
            "s": pa.array([None if i % 11 == 0 else f"v{i % 17}" for i in range(_N)]),
        }
    )
    directory = cluster_scratch("dist_vocabulary")
    for part in range(3):
        pq.write_table(table, directory / f"p{part}.parquet")
    return str(directory)


def _canonical(table: pa.Table, order: str) -> tuple:
    def scalar(value):
        if isinstance(value, float):
            return "nan" if math.isnan(value) else (0.0 if value == 0.0 else round(value, 9))
        if isinstance(value, (list, tuple)):
            return tuple(sorted((scalar(v) for v in value), key=repr))
        if isinstance(value, dict):
            return tuple(sorted(((scalar(k), scalar(v)) for k, v in value.items()), key=repr))
        return value

    names = sorted(table.column_names)
    types = [str(table.schema.field(n).type) for n in names]
    if order in table.column_names:
        table = table.sort_by([(order, "ascending")])
    data = table.to_pydict()
    rows = [tuple(scalar(data[n][i]) for n in names) for i in range(table.num_rows)]
    if order not in table.column_names:
        rows.sort(key=lambda row: tuple(repr(v) for v in row))
    return (names, types, rows)


#: The engine's own words for "that pair is not a call", as opposed to "that pair is broken".
#: Three families, and this sweep meets all three because it crosses *every* function with a
#: fixed set of columns: an arity it cannot satisfy (`arg_max` wants a value *and* an ordering
#: key, and the one-argument `AggExpr` built here is not a call it has), and a required
#: parameter it does not supply (`ewm_mean` needs an alpha or a half-life). The third -- a type
#: refusal, `bit_and` over a double or `bool_and` over a string -- is matched by shape instead,
#: so a type the analyzer learns to name later needs no edit here.
_DECLINED = ("requires an input column", "requires exactly one of")


def _decline_reason(exc: BaseException) -> str | None:
    """The refusal `exc` states, or None when it is a failure this sweep must not absorb.

    Standing down on a combination the engine declines is the design; standing down on
    *anything at all* would convert a regression into a silently missing case, and the run
    would stay green with the function untested. So the recognised refusals are matched by
    message and everything else is handed back to the caller to re-raise.

    An earlier revision enumerated the type refusals as literal strings -- "needs integer
    input", "needs numeric input" -- and thereby missed "needs boolean input", turning six
    legitimate `bool_and`/`bool_or` declines into failures. Enumerating a set the engine is
    free to extend is the wrong shape for this test; matching the sentence it builds is not.
    """
    if not isinstance(exc, (PlanError, RuntimeError)):
        return None
    text = str(exc)
    refuses_type = "needs " in text and " input" in text
    if refuses_type or any(marker in text for marker in _DECLINED):
        return text
    return None


def _agrees(build, order: str) -> None:
    single = _canonical(build().collect(), order)
    fanned = _canonical(build().collect(distributed=True, num_workers=_WORKERS), order)
    assert fanned[0] == single[0], f"columns {fanned[0]} vs {single[0]}"
    assert fanned[1] == single[1], f"types {fanned[1]} vs {single[1]}"
    assert fanned[2] == single[2], "the distributed rows differ from the single-node rows"


@pytest.mark.parametrize("column", ["k", "f", "s"])
@pytest.mark.parametrize("func", sorted(AGG_FNS))
def test_a_grouped_aggregate_distributes(splittable, func, column):
    def build():
        return bt.read.parquet(splittable).group_by("g").agg(v=AggExpr(func, bt.col(column)))

    try:
        build().collect()
    except (PlanError, RuntimeError) as exc:
        reason = _decline_reason(exc)
        if reason is None:
            raise
        pytest.skip(f"{func} does not accept a {column!r} column: {reason}")
    _agrees(build, "g")


@pytest.mark.parametrize("scope", ["partitioned", "global"])
@pytest.mark.parametrize("func", sorted(WINDOW_FUNCS))
def test_a_window_function_distributes(splittable, func, scope):
    keys = {"partition_by": ["g"]} if scope == "partitioned" else {}

    def build(spec):
        return bt.read.parquet(splittable).window(order_by=["o"], functions={"w": spec}, **keys)

    for spec in (func, (func, bt.col("f"))):
        try:
            build(spec).collect()
        except (PlanError, RuntimeError) as exc:
            if _decline_reason(exc) is None:
                raise
            continue
        if scope == "global" and func in _NO_GLOBAL_DECOMPOSITION:
            # The contract is that it says so loudly rather than running the whole relation on
            # one node behind the user's back, and that the message names the operator.
            with pytest.raises(PlanError, match="global window"):
                build(spec).collect(distributed=True, num_workers=_WORKERS)
            return
        _agrees(lambda spec=spec: build(spec), "rid")
        return
    pytest.skip(f"{func} is not expressible through window(order_by=...)")


def test_the_undecomposed_list_names_only_real_window_functions():
    """A stale entry would silently exempt a function from the agreement assertion.

    The list is consulted by name, so a rename or removal upstream would leave an entry that
    matches nothing -- and the function it was meant to describe would quietly start being
    asserted, or worse, a live function would go on being expected to raise.
    """
    assert set(_NO_GLOBAL_DECOMPOSITION) <= WINDOW_FUNCS
    # And it must not have grown to cover the vocabulary: if most functions were exempt, the
    # agreement half of this file would be asserting almost nothing.
    assert len(_NO_GLOBAL_DECOMPOSITION) < len(WINDOW_FUNCS) / 3
