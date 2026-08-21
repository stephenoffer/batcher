"""Splitting a plan into resource stages must not re-validate it against the stage boundary.

The defect this pins only ever appeared under ``distributed=True``: `map_batches(...)` alone
distributed fine, but `.select(...)` or `.filter(...)` above it raised
``ColumnNotFoundError: ... available: []`` while the identical pipeline worked single-node.
That is the ordinary shape of batch inference — score, then narrow — so `ds.ml.predict`,
`generate` and `embed` all hit it, as did every SQL ``SELECT <cols> FROM AI_GENERATE(...)``.

Both halves of the cause used `dataclasses.replace`, which re-runs `__post_init__`:
`dist.executors.plan_analysis._rebuild_stage` on the driver, and
`core.udf.execute.prebuild_factories` inside the worker. Each rebuilds a node onto a boundary
`Scan` that carries an empty schema — a stand-in for the upstream stage's published morsels,
not a description of them, because a `MapBatches` cannot report its output *types* through an
opaque `fn`. Re-validating against a stand-in asks a question it cannot answer.

**These are unit tests on purpose.** CI installs no Ray, so the integration coverage in
`tests/integration/test_distributed_map_batches_projection.py` never runs in the PR gate —
which is why this survived. Both code paths are ordinary functions over a `LogicalPlan`, so
the regression is catchable with no cluster at all, and that is what these do.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError
from batcher.core.udf.execute import prebuild_factories
from batcher.dist.executors.plan_analysis import split_into_resource_stages
from batcher.plan.visitor import reparent_unvalidated

pytestmark = pytest.mark.unit


class Upper:
    """A load-once class UDF, which is what puts its stage in a resource pool of its own."""

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        values = [s.upper() for s in batch.column("body").to_pylist()]
        return batch.append_column("resp", pa.array(values, pa.string()))


DECLARED = ["id", "body", "resp"]


def _scored():
    return bt.from_pydict({"id": [1, 2], "body": ["a", "b"]}).ml.map_batches(
        Upper, output_columns=DECLARED
    )


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("select", lambda ds: ds.select("id", "resp")),
        ("filter", lambda ds: ds.filter(bt.col("id") > 1)),
        ("filter_then_select", lambda ds: ds.filter(bt.col("id") > 1).select("id", "resp")),
    ],
)
def test_a_plan_with_work_above_map_batches_splits_into_stages(name: str, build) -> None:
    """This raised ``ColumnNotFoundError ... available: []`` for every one of these shapes."""
    stages = split_into_resource_stages(build(_scored())._plan)
    assert stages is not None and len(stages) >= 2


def test_the_split_puts_the_pool_stage_first_and_the_projection_above_it() -> None:
    stages = split_into_resource_stages(_scored().select("id", "resp")._plan)
    assert [s.wants_pool for s in stages] == [True, False]


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.select("id", "resp"),
        lambda ds: ds.filter(bt.col("id") > 1),
    ],
)
def test_each_stage_survives_the_workers_factory_prebuild(build) -> None:
    """The second half of the cause, which runs inside the Ray actor rather than the driver."""
    for stage in split_into_resource_stages(build(_scored())._plan):
        assert prebuild_factories(stage.sub_plan) is not None


def test_map_batches_alone_still_splits() -> None:
    """The case that always worked, kept so a fix cannot regress it into a single stage."""
    assert split_into_resource_stages(_scored()._plan) is None


# --- the helper itself ----------------------------------------------------------------


def test_reparent_copies_every_other_field() -> None:
    plan = _scored().filter(bt.col("id") > 1)._plan
    other = bt.from_pydict({"id": [9], "body": ["z"], "resp": ["Z"]})._plan
    moved = reparent_unvalidated(plan, input=other)
    assert moved.input is other
    assert moved.predicate is plan.predicate  # `==` on two Exprs builds an Expr
    assert type(moved) is type(plan)


def test_skipping_validation_does_not_disable_it_where_it_belongs() -> None:
    """The check still runs when the user builds the plan, which is the input that can answer it."""
    with pytest.raises(ColumnNotFoundError, match="resp"):
        bt.from_pydict({"id": [1], "body": ["a"]}).ml.map_batches(Upper).select("id", "resp")
