"""The decision surface across the modality x scale cross-product.

Every other test here pins one decision. This pins the *ensemble*, because the failures
this file exists for were never in one rule — they were a set of rules that each looked
reasonable and together sized a pipeline by three orders of magnitude wrong. A width, a
memory envelope, a morsel, a task count, and a shuffle cost all read the same data, and a
change to any one of them can silently put it out of step with the rest.

`CLAUDE.md` asks for the cross-product `{collect, spill, iter_batches, distributed}` x
`{nulls, empty, one row, duplicates, -0.0/NaN, descending}` for *results*. This is its
analogue for *decisions*: `{narrow, embedding, image, video} x {one node, small cluster,
fleet}`, asserting the invariants that must hold in every cell rather than a number in one.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.carbonite.memory.estimator import peak_operator_bytes
from batcher.carbonite.memory.pressure import PressureLevel
from batcher.carbonite.policies.morsel import morsel_target
from batcher.config import active_config
from batcher.kyber.annotate import annotate_ops
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel
from batcher.plan.physical import PhysicalPlan

pytestmark = pytest.mark.unit

_ROWS = 64

#: One row of each modality, by the bytes it really occupies.
_MODALITIES = {
    "narrow": 16,  # two int64 keys
    "embedding": 768 * 4,  # a float32 vector
    "image": 224 * 224 * 3,  # a decoded RGB frame
    "video_frame": 480 * 270 * 3,  # a small decoded video frame
}

#: Fleet sizes spanning single node to the scale the engine claims.
_FLEETS = (1, 8, 1024)


def _frame(modality: str):
    """A relation whose rows really are `_MODALITIES[modality]` bytes wide."""
    if modality == "narrow":
        return bt.from_arrow(pa.table({"k": pa.array(range(_ROWS)), "v": pa.array(range(_ROWS))}))
    if modality == "embedding":
        values = pa.array(np.zeros(_ROWS * 768, dtype="float32"))
        return bt.from_arrow(
            pa.table(
                {
                    "k": pa.array(range(_ROWS)),
                    "e": pa.FixedSizeListArray.from_arrays(values, 768),
                }
            )
        )
    shape = (224, 224, 3) if modality == "image" else (480, 270, 3)
    arr = pa.FixedShapeTensorArray.from_numpy_ndarray(np.zeros((_ROWS, *shape), dtype="uint8"))
    return bt.from_arrow(pa.table({"k": pa.array(range(_ROWS)), "t": arr}))


def _annotated(ds):
    est = CardinalityEstimator(ds._sources)
    ops = annotate_ops(ds._plan, est, active_config(), CostModel(est))
    return est, ops, PhysicalPlan(ir={}, output_schema=None, ops=ops)


@pytest.mark.parametrize("modality", list(_MODALITIES))
def test_the_width_estimate_is_within_an_order_of_magnitude_of_the_truth(modality):
    """The number every other decision is derived from.

    Not "bigger than before" but *close to the truth*: each of these was off by three to
    four orders of magnitude when the width came from a type prior that could not see
    through an extension label.
    """
    ds = _frame(modality)
    est = CardinalityEstimator(ds._sources)
    estimated = est.row_width(ds._plan, active_config().optimizer.row_bytes)
    truth = ds.collect().nbytes / _ROWS
    assert 0.1 <= estimated / truth <= 10.0, f"{modality}: {estimated} vs {truth}"


@pytest.mark.parametrize("modality", list(_MODALITIES))
def test_a_morsel_never_exceeds_its_byte_budget(modality):
    """The streaming working set is bounded whatever a row weighs.

    The row floor used to override the byte bound for anything past 1,024 B/row, which is
    below every modality here except the narrow one.
    """
    cfg = active_config()
    ds = _frame(modality)
    target = morsel_target(cfg, PressureLevel.NORMAL, None, None, ds._plan)
    width = _MODALITIES[modality]
    rows = cfg.execution.morsel_rows if target is None else target[0]
    # Inside the budget, or a single row when one row alone already exceeds it.
    assert rows == 1 or rows * width <= cfg.execution.morsel_bytes, f"{modality}: {rows} rows"


@pytest.mark.parametrize("modality", list(_MODALITIES))
def test_a_task_never_holds_more_than_the_byte_target(modality):
    """Fan-out follows the bytes a task will hold, not only the row count."""
    cfg = active_config()
    ds = _frame(modality).sort("k")
    _, ops, _ = _annotated(ds)
    breaker = next((o for o in ops if o.bounds.n_max_parallelism > 0), None)
    if breaker is None:  # pragma: no cover - a plan shape with no sized breaker
        pytest.skip("no breaker with a parallelism request")
    per_task = breaker.bounds.m_max_bytes / breaker.bounds.n_max_parallelism
    assert per_task <= cfg.optimizer.target_bytes_per_task * 1.01, modality


@pytest.mark.parametrize("modality", list(_MODALITIES))
def test_the_envelope_grows_with_the_modality(modality):
    """A breaker over wider rows must be budgeted for more memory, monotonically."""
    _, _, narrow = _annotated(_frame("narrow").sort("k"))
    _, _, this = _annotated(_frame(modality).sort("k"))
    if modality == "narrow":
        return
    assert peak_operator_bytes(this) > peak_operator_bytes(narrow), modality


# --- the scale axis ---------------------------------------------------------------


@pytest.mark.parametrize("modality", list(_MODALITIES))
def test_a_single_node_plan_is_charged_no_network(modality):
    """The safety property behind the whole `net` axis, in every cell."""
    ds = _frame(modality).group_by("k").agg(n=bt.col("k").count())
    est = CardinalityEstimator(ds._sources)
    solo = CostModel(est)
    node = ds._plan
    while node is not None:
        assert solo.op_cost(node).net == 0.0, modality
        node = getattr(node, "input", None)


@pytest.mark.parametrize("modality", list(_MODALITIES))
def test_shuffle_cost_is_monotone_in_the_fleet(modality):
    """More workers can only mean more on the wire for an all-to-all stage.

    Both terms are monotone -- the volume's `1 - 1/W` discount shrinks and the `W^2`
    fan-out grows -- so a non-monotone result means one of them is mis-signed.
    """
    ds = _frame(modality).group_by("k").agg(n=bt.col("k").count())
    est = CardinalityEstimator(ds._sources)
    nets = [CostModel(est, workers=w).op_cost(ds._plan).net for w in _FLEETS]
    assert nets == sorted(nets), f"{modality}: {nets}"
    assert nets[0] == 0.0


@pytest.mark.parametrize("workers", _FLEETS)
def test_a_wider_relation_never_costs_less_to_shuffle(workers):
    """At a fixed fleet, the network cost must track the bytes, not the row count."""
    per_modality = {}
    for modality in _MODALITIES:
        ds = _frame(modality).group_by("k").agg(n=bt.col("k").count())
        est = CardinalityEstimator(ds._sources)
        per_modality[modality] = CostModel(est, workers=workers).op_cost(ds._plan).net
    ordered = [per_modality[m] for m in ("narrow", "embedding", "image")]
    assert ordered == sorted(ordered), per_modality
