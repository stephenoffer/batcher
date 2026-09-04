"""A distributed task must read the columns single-node reads, not its side's whole table.

Column pruning is decided once, by Kyber, as `required_columns_per_source`. Single-node asks
it of the *whole plan*; every distributed operator has to ask it too, because what a worker
reads off storage is chosen on the driver before the task is sent. Asking the wrong node gets
a correct answer to a different question — and a correct answer, which is why nothing else
catches it.

There are two wrong nodes and they are wrong for the same reason.

One is a join's *side*. A side is typically a bare `Scan` (or a `Filter` over one),
which requires every column it has; what narrows the read is the operator above it, and Kyber
prunes a join's own `output` list rather than inserting a `Project` below it. So
`source_pushdown(join.left, ...)` answers "all of them" while `source_pushdown(join, ...)`
answers what single-node reads. On a 13-column fact table joined to a dimension and projected
to three outputs, that is 13 columns against 2 — the dominant IO of the query, paid on every
worker, with the right rows coming back.

The other is a pass-through breaker's *map prefix*. A sort emits every column it is given and
a window emits those plus its aliases, so the breaker and its prefix both correctly answer "all
of them" — what narrows the read is the projection **above** the breaker, which the dispatcher
is already holding in `above` to re-apply on the driver once the stage lands. `df.sort("k")
.select("k", "c3")` over the same table read 2 columns single-node and 13 on every worker.

The same question is asked once per execution mode, and all three had to learn the same
answer: the distributed dispatcher (`stage_pushdown`), the out-of-core collect and the
streaming iterator (`dist.spill.narrow_to_stage`, which pushes the projection into the plan so
every path below derives from it rather than being told separately).

The engine already knew this in two of its four join paths: the out-of-core join asks the join
node (`map_projection(join, sid)`), and the Flight join pre-projects each side
(`flight_join._project_join_side`). The disk shuffle, both ASOF paths, and every sort/window
stage did neither.

Asserted as *parity with single-node* rather than against a hand-written column list, so the
test states the contract ("distribution changes where a plan runs, never what it reads") and
cannot drift as the pruner improves.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit

#: Wide enough that reading it whole is unmistakably different from reading the two columns
#: the query needs.
_WIDE = pa.table(
    {f"c{i}": pa.array([1, 2, 3], pa.int64()) for i in range(12)}
    | {
        "k": pa.array([1, 2, 3], pa.int64()),
        "t": pa.array([10, 20, 30], pa.int64()),
    }
)
_DIM = pa.table(
    {
        "k": pa.array([1, 2], pa.int64()),
        "t": pa.array([5, 25], pa.int64()),
        "w": pa.array(["x", "y"]),
    }
)

_QUERIES = {
    "equi_join": lambda left, right: left.join(right, on="k").select("k", "c3", "w"),
    "equi_join_filtered": lambda left, right: (
        left.filter(bt.col("c7") > 0).join(right, on="k").select("k", "c3", "w")
    ),
    "left_join": lambda left, right: left.join(right, on="k", how="left").select("k", "c3", "w"),
    "asof_join_by": lambda left, right: left.join_asof(right, on="t", by="k").select(
        "k", "c3", "w"
    ),
    "asof_join_keyless": lambda left, right: (
        left.sort("t").join_asof(right.sort("t"), on="t").select("t", "c3", "w")
    ),
    # A join whose left side computes a column the projection then keeps. The read must
    # narrow to the source columns that *derived* column depends on, not to the derived name
    # (which no source has). `stage_pushdown` gets this for free because it asks about the
    # whole stage, but only if it is genuinely asked about the whole stage.
    "join_over_a_derived_column": lambda left, right: (
        left.with_columns(m=bt.col("c3") * 3).join(right, on="k").select("m", "w")
    ),
}


#: Single-input stages, where the narrowing comes from `above` rather than from the operator.
#: Each pairs a query with the breaker its distributed stage is built around.
_STAGES = {
    "sort_then_select": (lambda left: left.sort("k").select("k", "c3"), "Sort"),
    "sort_then_filter_select": (
        lambda left: left.sort("k").filter(bt.col("c3") > 0).select("k", "c3"),
        "Sort",
    ),
    "window_then_select": (
        lambda left: left.window(
            partition_by=["k"], order_by=["c0"], functions={"r": "row_number"}
        ).select("k", "r"),
        "Window",
    ),
    "distinct_then_select": (lambda left: left.select("k", "c3").distinct(), "Distinct"),
    # The breaker keys on a column the plan *derived*, and the projection above drops it.
    # `required_columns_per_source` answers about the **source**, so subtracting its answer
    # from the breaker's input columns removes the derived one — and that is not a missed
    # optimization, it is a broken plan: the breaker re-validates its keys against the
    # narrowed input and raises `ColumnNotFoundError`. Caught by exactly this shape while the
    # narrowing was being written, which is why all four spellings of it are here.
    "sort_on_derived_key": (
        lambda left: left.with_columns(m=bt.col("k") * 2).sort("m").select("k", "c3"),
        "Sort",
    ),
    "window_order_on_derived_key": (
        lambda left: (
            left.with_columns(m=bt.col("k") * 2)
            .window(partition_by=["k"], order_by=["m"], functions={"r": "row_number"})
            .select("k", "r")
        ),
        "Window",
    ),
    "window_partition_on_derived_key": (
        lambda left: (
            left.with_columns(m=bt.col("k") % 4)
            .window(partition_by=["m"], order_by=["c0"], functions={"r": "row_number"})
            .select("c0", "r")
        ),
        "Window",
    ),
    # The control: nothing above narrows, so the stage must still read the whole table. A fix
    # that narrowed unconditionally would drop columns the caller asked for.
    "sort_alone": (lambda left: left.sort("k"), "Sort"),
}


def _binary_node(plan):
    """The join / ASOF node in an optimized plan, or None."""
    from batcher.plan.logical import AsofJoin, Join, RangeJoin

    stack = [plan]
    while stack:
        node = stack.pop()
        if isinstance(node, (Join, AsofJoin, RangeJoin)):
            return node
        stack.extend(
            c
            for c in (
                getattr(node, "input", None),
                getattr(node, "left", None),
                getattr(node, "right", None),
            )
            if c is not None
        )
    return None


@pytest.mark.parametrize("query", sorted(_QUERIES))
def test_a_join_task_reads_what_the_whole_plan_reads(query):
    from batcher.dist.executors.partition_io import source_pushdown
    from batcher.kyber.optimizer import Optimizer
    from batcher.kyber.rules.projections import required_columns_per_source

    ds = _QUERIES[query](bt.from_arrow(_WIDE), bt.from_arrow(_DIM))
    optimized = Optimizer(sources=ds._sources).logical_rewrite(ds._plan)
    single_node = required_columns_per_source(optimized)
    node = _binary_node(optimized)
    assert node is not None, "the query under test must contain a join"

    for source_id, expected in single_node.items():
        actual, _ = source_pushdown(node, source_id)
        assert actual == expected, (
            f"{query}: the distributed task for source {source_id} would read {actual}, "
            f"where single-node reads {expected}"
        )


@pytest.mark.parametrize("query", ["equi_join", "asof_join_by"])
def test_asking_the_side_instead_of_the_operator_is_what_loses_the_narrowing(query):
    """Pin the mechanism, so a regression says *why* rather than only *that*.

    Without this, a future change that made both spellings equally wide would leave the test
    above green (parity holds — at every column) while the pruning quietly stopped working.
    """
    from batcher.dist.executors.partition_io import source_pushdown
    from batcher.kyber.optimizer import Optimizer

    ds = _QUERIES[query](bt.from_arrow(_WIDE), bt.from_arrow(_DIM))
    node = _binary_node(Optimizer(sources=ds._sources).logical_rewrite(ds._plan))
    from_operator, _ = source_pushdown(node, 0)
    from_side, _ = source_pushdown(node.left, 0)
    assert from_operator is not None
    assert len(from_operator) < len(from_side), (
        "the wide side's own subtree should require every column; if it no longer does, this "
        "test is measuring nothing"
    )
    assert set(from_operator) < set(from_side)


@pytest.mark.parametrize("stage", sorted(_STAGES))
def test_a_single_input_stage_reads_what_the_whole_plan_reads(stage):
    """Includes `sort_alone`, where the answer is the whole table — the control that a fix
    narrowing unconditionally would fail."""
    from batcher.dist.executors.partition_io import stage_pushdown
    from batcher.kyber.optimizer import Optimizer
    from batcher.kyber.rules.projections import required_columns_per_source
    from batcher.plan.logical import Distinct, Sort, Window

    build, breaker_name = _STAGES[stage]
    ds = build(bt.from_arrow(_WIDE))
    optimized = Optimizer(sources=ds._sources).logical_rewrite(ds._plan)
    expected = required_columns_per_source(optimized).get(0)

    # Walk down to the breaker exactly as the dispatcher's `_split_at` does, collecting the
    # operators above it — which is the list `stage_pushdown` needs.
    above: list = []
    node = optimized
    while not isinstance(node, (Sort, Window, Distinct)):
        above.append(node)
        node = node.input
    assert type(node).__name__ == breaker_name

    actual, _ = stage_pushdown(above, node, 0)
    assert actual == expected, (
        f"{stage}: the distributed stage would read {actual}, where single-node reads {expected}"
    )


@pytest.mark.parametrize("stage", sorted(_STAGES))
def test_the_out_of_core_stage_narrows_to_the_same_columns(stage):
    """`narrow_to_stage` is the spilling/streaming counterpart of `stage_pushdown`.

    It rewrites the plan rather than passing a projection alongside it, so what is asserted is
    the *rewritten plan's* source requirement — which is what every path below (the source
    read, the bucket write, the reducer, the empty-result schema) then derives from.
    """
    from batcher.dist.spill import narrow_to_stage
    from batcher.kyber.optimizer import Optimizer
    from batcher.kyber.rules.projections import required_columns_per_source
    from batcher.plan.logical import Distinct, Sort, Window

    build, breaker_name = _STAGES[stage]
    ds = build(bt.from_arrow(_WIDE))
    optimized = Optimizer(sources=ds._sources).logical_rewrite(ds._plan)
    expected = required_columns_per_source(optimized).get(0)

    above: list = []
    node = optimized
    while not isinstance(node, (Sort, Window, Distinct)):
        above.append(node)
        node = node.input
    assert type(node).__name__ == breaker_name

    narrowed = narrow_to_stage(above, node)
    actual = required_columns_per_source(narrowed).get(0)
    assert actual == expected, (
        f"{stage}: the out-of-core stage would read {actual}, where single-node reads {expected}"
    )
