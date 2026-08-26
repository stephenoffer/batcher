"""The distributed single-node fallback must plan the query `collect()` would plan.

`dist.executors.ray_runtime.lifecycle._single_node` is where **every** distributed run lands
when Ray is unavailable, the cluster turns out to be one node, or resources are too tight to
place workers. It is a fallback in *scheduling* only: the plan it runs should be the plan the
single-node path runs, because that is the whole claim of "single-node == distributed".

It was not. The call read `kyber.optimize(plan)` -- no `sources`, no hub -- so the cardinality
estimator had nothing to estimate from and every cost-based choice quietly took its default.
A 20,000-row/50-row join came out `strategy="hash"` here and `strategy="broadcast"` on the
single-node path, for the identical query, and `optimize_full` returned an empty
`BuildSideDecision` list rather than a measured one.

Nothing caught it because the fallback returns the *right rows*: a hash join of a big and a
tiny table is correct, just needlessly expensive, and the correctness suites compare rows.
The `_single_node_with_udfs` branch three lines above already passed `sources`, which is what
makes this a slip rather than a design choice.

Blast radius, measured rather than assumed: of the 13 shapes in `tests/_engagement_shapes.py`
only `join` plans differently with and without `sources`. Build-side selection is the main
cardinality-driven choice, so a source-less plan is identical for the other twelve. That bound
is fixture-limited -- it says those shapes do not depend on cardinality *at this data size*,
not that they never could -- but it does mean the defect was a join defect, not a general one.

The assertion is deliberately not "the strategy is broadcast". That would re-encode today's
cost model and break the next time it is tuned. What must hold is that the two paths *agree*,
whatever they decide -- so the test optimizes the same plan both ways and compares.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


@pytest.fixture
def skewed_join(tmp_path):
    """A join whose build side is only obvious if you can see the row counts."""
    pq.write_table(pa.table({"k": list(range(20000)), "v": [1] * 20000}), tmp_path / "big.parquet")
    pq.write_table(pa.table({"k": list(range(50)), "w": [2] * 50}), tmp_path / "small.parquet")
    return bt.read.parquet(str(tmp_path / "big.parquet")).join(
        bt.read.parquet(str(tmp_path / "small.parquet")), on="k"
    )


def _strategies(physical) -> list[str]:
    """Every join strategy in a physical plan, innermost first."""
    found: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            if "strategy" in node:
                found.append(node["strategy"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(physical.to_json()))
    return found


def test_the_fallback_optimizes_with_the_sources_it_was_given(skewed_join, monkeypatch):
    """The captured plan must match what the single-node path plans for the same query."""
    from batcher import kyber
    from batcher.dist.executors.ray_runtime import lifecycle

    captured = {}
    original = kyber.optimize

    def spy(plan, *args, **kwargs):
        physical = original(plan, *args, **kwargs)
        captured["physical"] = physical
        return physical

    monkeypatch.setattr(kyber, "optimize", spy)
    lifecycle._single_node(skewed_join._plan, skewed_join._sources)
    assert "physical" in captured, "the fallback no longer routes through kyber.optimize"

    expected, _logical, decisions = kyber.optimize_full(
        skewed_join._plan, None, skewed_join._sources, None
    )
    assert _strategies(captured["physical"]) == _strategies(expected), (
        "the distributed fallback planned a different join strategy than the single-node "
        "path for the identical query -- it optimized without `sources`, so the estimator "
        "had no cardinalities and the cost-based choice defaulted"
    )
    assert decisions, "the control is void: single-node made no build-side decision either"


def test_planning_without_sources_really_does_change_the_plan(skewed_join):
    """The control.

    Without this, the test above could pass because the two calls agree on everything for
    some reason unrelated to `sources` -- and the diagnosis in the module docstring would
    rest on nothing. This is the measurement that showed the defect in the first place.
    """
    from batcher import kyber

    with_sources, _, decisions = kyber.optimize_full(
        skewed_join._plan, None, skewed_join._sources, None
    )
    without, _, blind_decisions = kyber.optimize_full(skewed_join._plan, None, None, None)

    assert _strategies(with_sources) != _strategies(without), (
        "planning with and without `sources` now agrees on this shape, so this fixture can "
        "no longer detect the defect -- pick a shape whose plan depends on cardinality"
    )
    assert decisions and not blind_decisions


def test_the_fallback_returns_the_right_rows(skewed_join):
    """Bounding the defect: it was a cost bug, never a wrong answer.

    Worth pinning, because "the distributed fallback planned it differently" would be a far
    more serious finding if the rows had differed too.
    """
    from batcher.dist.executors.ray_runtime import lifecycle

    got = lifecycle._single_node(skewed_join._plan, skewed_join._sources)
    expected = skewed_join.collect()
    assert got.num_rows == expected.num_rows == 50
    assert sorted(got.column("k").to_pylist()) == sorted(expected.column("k").to_pylist())
