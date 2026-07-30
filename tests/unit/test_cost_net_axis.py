"""The `net` cost axis — what a plan costs to move across a cluster.

`Cost.net` and `CostWeights.net` existed from the start and **nothing ever wrote to
them**, so every plan Kyber ranked was ranked as though the network were free. On one
machine that is right; on a cluster it is the term that decides the plan. These tests pin
both halves: that a single-node model is byte-for-byte unchanged, and that a distributed
one charges volume, fan-out, and the co-partitioning that avoids both.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel
from batcher.kyber.cost.shuffle import fanout_bytes, shuffle_bytes

pytestmark = pytest.mark.unit


def _models(ds, workers: int) -> tuple[CostModel, CostModel]:
    """A single-node and a `workers`-wide model over the same estimator inputs."""
    return (
        CostModel(CardinalityEstimator(ds._sources)),
        CostModel(CardinalityEstimator(ds._sources), workers=workers),
    )


def _frame():
    return bt.from_pydict({"k": list(range(1000)), "v": [i % 7 for i in range(1000)]})


# --- the shape of the two terms ---------------------------------------------------


def test_a_shuffle_keeps_the_share_that_hashes_to_its_own_node():
    # 1/W of a hash repartition lands in the bucket already on the producing node, so a
    # two-worker shuffle moves half its data and a ten-thousand-worker shuffle moves
    # essentially all of it. A model without the discount over-charges small clusters.
    assert shuffle_bytes(1000, 10, 2) == pytest.approx(1000 * 10 * 0.5)
    assert shuffle_bytes(1000, 10, 4) == pytest.approx(1000 * 10 * 0.75)
    assert shuffle_bytes(1000, 10, 10_000) == pytest.approx(1000 * 10 * (1 - 1e-4))
    # And nothing crosses a network that does not exist.
    assert shuffle_bytes(1000, 10, 1) == 0.0


def test_fanout_is_quadratic_in_the_fleet_for_an_all_to_all():
    # The term that actually stops a shuffle-heavy plan reaching ten thousand nodes:
    # P x R fragments to open, frame, and drain before a useful byte moves.
    assert fanout_bytes(100, 100) == 10 * fanout_bytes(10, 100)
    assert fanout_bytes(10_000, 10_000) > 1e10


def test_a_broadcast_fanout_is_linear_where_a_repartition_is_quadratic():
    # This is why a broadcast join is not merely "a cheaper shuffle": its advantage over
    # a repartition grows with the *square* of the cluster, not with a constant.
    for workers in (10, 1000):
        assert fanout_bytes(1, workers) * workers == fanout_bytes(workers, workers)


# --- single-node is unchanged -----------------------------------------------------


def test_single_node_charges_no_net_anywhere():
    # The safety property for every existing ranking: with one worker the axis is
    # identically zero, so no plan Kyber ranked before is re-ranked by this change.
    ds = _frame().group_by("v").agg(s=col("k").sum()).sort("s")
    solo, _ = _models(ds, 1)
    node = ds._plan
    while node is not None:
        assert solo.op_cost(node).net == 0.0
        node = getattr(node, "input", None)


# --- per-operator volume ----------------------------------------------------------


def test_an_aggregate_shuffles_its_partials_not_its_input():
    # The mergeable `partial -> combine` form is what makes a wide aggregate cheap: each
    # worker reduces locally, so what crosses the wire is bounded by `groups x workers`,
    # not by the input. A two-group aggregate over a billion rows must not be costed as a
    # billion-row shuffle, which is what a model with no notion of pre-aggregation does.
    ds = _frame()
    est = CardinalityEstimator(ds._sources)
    dist = CostModel(est, workers=8)
    agg = ds.group_by("v").agg(s=col("k").sum())._plan
    in_rows = est.estimate(agg.input).rows
    groups = est.estimate(agg).rows
    assert groups * 8 < in_rows  # the bound is the *binding* one here, not the input
    full_input_shuffle = shuffle_bytes(in_rows, dist.row_bytes(agg), 8)
    assert 0.0 < dist.op_cost(agg).net < full_input_shuffle + fanout_bytes(8, 8)


def test_an_aggregate_partial_can_never_exceed_its_input():
    # The other half of the bound: past `workers = input/groups` every worker is already
    # emitting one partial per row it saw, so the volume saturates at the input rather
    # than growing without limit as the fleet does.
    ds = _frame()
    est = CardinalityEstimator(ds._sources)
    agg = ds.group_by("v").agg(s=col("k").sum())._plan
    in_rows = est.estimate(agg.input).rows
    huge = CostModel(est, workers=100_000)
    volume = huge.op_cost(agg).net - fanout_bytes(100_000, 100_000)
    assert volume <= in_rows * huge.row_bytes(agg)


def test_a_top_n_sort_gathers_only_its_limit():
    # The reason to keep a limit fused into a sort survives into the network axis: each
    # worker forwards its own k rows rather than range-partitioning the whole relation.
    ds = _frame()
    full = ds.sort("k")
    top = ds.sort("k").limit(10)
    _, dist = _models(ds, 16)
    assert dist.op_cost(top._plan).net < dist.op_cost(full._plan).net


def test_a_window_moves_its_input_whether_or_not_it_is_partitioned():
    # A window is a shuffle-bearing operator either way, and both shapes must be visible:
    # a partitioned window repartitions by its PARTITION BY keys, and an *unpartitioned*
    # one gathers the whole relation onto a single worker because its frame spans
    # everything. The second is the operator that will not scale, and costing it at zero
    # is what would let the optimizer treat a global `row_number()` as free.
    ds = _frame()
    _, dist = _models(ds, 16)
    partitioned = ds.with_columns(col("k").sum().over(["v"]).alias("w"))._plan
    global_frame = ds.with_columns(col("k").sum().over([]).alias("w"))._plan
    assert dist.op_cost(partitioned).net > 0.0
    assert dist.op_cost(global_frame).net > 0.0


def test_map_only_operators_move_nothing():
    ds = _frame()
    filtered = ds.filter(col("k") > 5)
    projected = ds.select(col("k"))
    _, dist = _models(ds, 64)
    assert dist.op_cost(filtered._plan).net == 0.0
    assert dist.op_cost(projected._plan).net == 0.0
    assert dist.op_cost(ds._plan).net == 0.0


# --- the co-partitioning that avoids a shuffle entirely ---------------------------


def test_an_aggregate_on_a_join_key_needs_no_second_shuffle():
    # The property `dist` already relies on, now visible to the cost model: a hash join
    # leaves its output partitioned by the join key, so an aggregate grouping on a
    # superset of that key computes complete groups locally. Charging it a full shuffle
    # would make the optimizer avoid exactly the plan shape it should prefer.
    left = bt.from_pydict({"k": list(range(100)), "a": list(range(100))})
    right = bt.from_pydict({"k": list(range(100)), "b": list(range(100))})
    joined = left.join(right, on="k")
    co_partitioned = joined.group_by("k").agg(s=col("a").sum())
    reshuffled = joined.group_by("b").agg(s=col("a").sum())
    _, dist = _models(left, 32)
    assert dist.op_cost(co_partitioned._plan).net == 0.0
    assert dist.op_cost(reshuffled._plan).net > 0.0


def test_net_grows_with_the_fleet_for_an_all_to_all_stage():
    ds = _frame()
    agg = ds.group_by("k").agg(s=col("v").sum())
    small = CostModel(CardinalityEstimator(ds._sources), workers=4)
    large = CostModel(CardinalityEstimator(ds._sources), workers=4096)
    assert large.op_cost(agg._plan).net > small.op_cost(agg._plan).net
