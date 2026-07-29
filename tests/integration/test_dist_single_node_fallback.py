"""The distributed executor's single-node fallback must run a UDF plan.

`_single_node` is where every `collect(distributed=True)` lands when Ray is unavailable, the
cluster is one node, or resources are too tight to place the workers. It lowered the plan
through `kyber.optimize`, and `MapBatches.to_ir()` raises by design — so the fallback could
not run a `map_batches` pipeline **at all**, failing with
``NotImplementedError: map_batches is executed in Python, not lowered to the engine IR``.
That is an internal message about a wire contract, raised for the batch-inference workload
most likely to be run under `distributed=True` in the first place.

These call the fallback directly rather than through Ray: the bug is in the fallback, and
driving it from a real cluster makes the test slow and its failure mode ambiguous.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.dist.executors.ray_runtime.lifecycle import _single_node

pytestmark = pytest.mark.integration


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"k": ["a", "b", "a", "b", "a"], "x": [1, 2, 3, 4, 5]})


def _fallback(dataset: bt.Dataset) -> pa.Table:
    return _single_node(dataset._plan, dataset._sources)


def _sorted(mapping: dict) -> dict:
    return {name: sorted(values, key=str) for name, values in mapping.items()}


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("plain", lambda d: d.map_batches(lambda b: b)),
        (
            "after_aggregate",
            lambda d: d.group_by("k").agg(s=bt.col("x").sum()).map_batches(lambda b: b),
        ),
        (
            "map_groups",
            lambda d: d.group_by("k").map_groups(
                lambda g: {"k": [g.column("k")[0].as_py()], "n": [g.num_rows]},
                output_columns=["k", "n"],
            ),
        ),
        (
            "declared_input_columns",
            lambda d: d.map_batches(
                lambda b: {"y": [v * 2 for v in b.column("x").to_pylist()]},
                input_columns=["x"],
                output_columns=["y"],
            ),
        ),
    ],
)
def test_fallback_matches_local_execution(ds: bt.Dataset, name: str, build) -> None:
    plan = build(ds)
    assert _sorted(_fallback(plan).to_pydict()) == _sorted(plan.to_pydict())


def test_fallback_still_handles_a_plan_without_udfs(ds: bt.Dataset) -> None:
    """The non-UDF route is untouched, so pin it alongside."""
    plan = ds.filter(bt.col("x") > 1)
    assert _sorted(_fallback(plan).to_pydict()) == _sorted(plan.to_pydict())


def test_fallback_keeps_the_schema_of_an_empty_udf_result(ds: bt.Dataset) -> None:
    plan = ds.filter(bt.col("x") > 99).map_batches(lambda b: b)
    table = _fallback(plan)
    assert table.num_rows == 0
    assert table.schema.names == ["k", "x"]
