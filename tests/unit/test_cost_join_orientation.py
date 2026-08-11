"""Pricing a join at the orientation it will run must equal pricing that orientation.

`join_op_cost` exists because `JOIN_REORDER` ranks orders *before* `SELECTION` picks a build
side, so the reorder must cost the orientation SELECTION will choose rather than the one the
node happens to spell. An inner join is commutative, so that is a well-defined quantity: it is
what the mirrored join costs as written.

It was not what the code computed. The swapped estimate was assembled by adjusting the
as-written cost by a build/probe *row* difference, and the two sides of that adjustment did
not describe the same thing:

* the subtracted term omitted the probe's `cache_factor`, which the as-written cost included,
  so the swapped estimate kept a residue of the **as-written** build side's cache residency;
* the added term carried no cache residency of its own, so the orientation actually chosen was
  priced as if its hash table were always resident;
* the `io` axis was not recomputed at all, so a swap whose whole point is to build the side
  that fits in memory was still charged for the other side's spill.

Every one of those errors scales with the build side's size — the quantity the terms exist to
price — and the choice between the two orientations was made on the same partial arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import batcher as bt
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel
from batcher.plan.logical import Join, JoinOutputCol, LogicalPlan

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _SmallNode:
    """A worker whose cache and memory are small enough that both terms are live.

    The cache knee and the spill threshold are the two machine-shaped multipliers a join's
    cost carries, and on the host running the tests a few thousand rows clear neither. Naming
    a tiny node puts both in range without materializing a table that would.
    """

    l3_cache_bytes: int = 1 << 10
    memory_bytes: int = 1 << 12
    storage_class: str = ""


def _join_node(plan: LogicalPlan) -> Join:
    while not isinstance(plan, Join):
        children = [v for v in vars(plan).values() if isinstance(v, LogicalPlan)]
        plan = children[0]
    return plan


def _mirrored(join: Join) -> Join:
    """The same inner join written the other way round — build side on the other input."""
    flipped = tuple(
        JoinOutputCol("right" if o.side == "left" else "left", o.name, o.alias) for o in join.output
    )
    return Join(join.right, join.left, join.right_keys, join.left_keys, "inner", flipped)


def _model_and_join() -> tuple[CostModel, Join]:
    small = bt.from_pydict({"k": list(range(8))})
    big = bt.from_pydict({"k": [i % 8 for i in range(4_000)], "v": list(range(4_000))})
    ds = small.join(big, on="k")
    model = CostModel(CardinalityEstimator(ds._sources), hardware=_SmallNode())
    return model, _join_node(ds._plan)


def test_the_swapped_orientation_is_priced_as_the_mirrored_join():
    """The contract, stated as an identity rather than as a formula.

    The small side is on the left, so building it is cheaper and `join_op_cost` must report
    the mirrored join's cost — on every axis, not just `cpu`.
    """
    model, join = _model_and_join()
    as_written = model.op_cost(join)
    mirrored = model.op_cost(_mirrored(join))
    assert mirrored.total() < as_written.total(), "the fixture must actually provoke a swap"

    chosen = model.join_op_cost(join)
    assert chosen.cpu == pytest.approx(mirrored.cpu)
    assert chosen.mem == pytest.approx(mirrored.mem)
    assert chosen.io == pytest.approx(mirrored.io)


def test_a_swap_stops_paying_the_other_side_s_spill():
    """The `io` axis follows the side that is actually built.

    Building the 8-row side spills nothing; building the 4,000-row side does. Reporting the
    latter's spill for a plan that builds the former is the single largest cost error a plan
    can contain, by `cost.terms`' own account.
    """
    model, join = _model_and_join()
    assert model.op_cost(join).io > 0.0
    assert model.join_op_cost(join).io < model.op_cost(join).io


def test_the_cheaper_orientation_is_never_costed_above_the_written_one():
    """A commutative join can only be priced at the better of its two orientations."""
    model, join = _model_and_join()
    assert model.join_op_cost(join).total() <= model.op_cost(join).total()


def test_a_non_inner_join_keeps_its_written_build_side():
    """Only an inner join is commutative; anything else is priced exactly as written."""
    small = bt.from_pydict({"k": list(range(8))})
    big = bt.from_pydict({"k": [i % 8 for i in range(4_000)], "v": list(range(4_000))})
    ds = small.join(big, on="k", how="left")
    model = CostModel(CardinalityEstimator(ds._sources), hardware=_SmallNode())
    join = _join_node(ds._plan)
    assert model.join_op_cost(join) == model.op_cost(join)
