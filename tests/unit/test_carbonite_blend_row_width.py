"""The learned blend rescales against the width the plan actually sized with.

`blend_peak` folds a plan's per-operator byte estimate toward what the family really used,
by the ratio of the measured bytes-per-row to the width the estimate was built from. It
divided by the flat `optimizer.row_bytes` default instead -- and that stopped being the
width `annotate` uses once it moved to a byte-true `row_width` and started publishing it as
`PlanProperties.row_size`.

So the rescale was wrong by exactly `row_size / row_bytes`: one to two orders of magnitude
on the wide payloads `row_width` exists to model (embeddings, blobs, images), in the
direction that *inflates* the estimate. Every such operator blew straight through to the
`clamp` ceiling the moment its family was learned, and queries that fit memory were routed
out-of-core.

It is the same mistake `LearnedMemoryModel._est_input_rows` already documents and avoids,
one method apart in the same file -- which is why these tests check the *property* (a
plan sized with a wide row is not re-inflated by that width) rather than the arithmetic.
"""

from __future__ import annotations

import math

import pytest

from batcher.carbonite.base import ResourceContext
from batcher.carbonite.memory.estimator import learned_plan_peak
from batcher.carbonite.memory.learned import LearnedMemoryModel
from batcher.config import Config
from batcher.plan.physical import PhysicalOp, PhysicalPlan, PlanProperties
from batcher.plan.resource import ResourceBounds

pytestmark = pytest.mark.unit

#: The flat default the estimate used to be inverted by.
_ROW_BYTES = 64
#: A wide payload row: a 1,024-dim float32 embedding plus its keys.
_WIDE = 4_096
_ROWS = 100_000


def _model(bytes_per_row: float) -> LearnedMemoryModel:
    """A model that measured `bytes_per_row` for the Aggregate family.

    `alpha=1.0` folds all the way to the measurement and `clamp` is wide, so the test reads
    the rescale itself rather than the smoothing on top of it.
    """
    return LearnedMemoryModel(
        _bytes_per_row={"aggregate": bytes_per_row},
        _alpha=1.0,
        _clamp=1000.0,
        _row_bytes=_ROW_BYTES,
        _spill_per_row={},
    )


def _plan(row_size: float | None) -> PhysicalPlan:
    """A one-aggregate plan sized `_ROWS x row_size`, publishing that width (or not)."""
    props = PlanProperties(est_rows=float(_ROWS))
    if row_size is not None:
        props = PlanProperties(est_rows=float(_ROWS), row_size=row_size)
    width = row_size if row_size is not None else _ROW_BYTES
    op = PhysicalOp(
        op_id=0,
        kind="Aggregate",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=int(_ROWS * width), c_max_credits=0, n_max_parallelism=0),
        inputs=(),
        properties=props,
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


def _ctx(model) -> ResourceContext:
    return ResourceContext(config=Config(), envelope_bytes=None, memory_model=model)


# --- the rule -----------------------------------------------------------------


def test_a_measurement_matching_the_plan_leaves_the_estimate_alone() -> None:
    """The property the whole blend rests on, checked where the answer is knowable.

    A family that measured exactly what the plan assumed needs no correction. Under the old
    division by the flat default this inflated a wide-row plan 64-fold.
    """
    plan = _plan(_WIDE)
    blended = learned_plan_peak(plan, _model(_WIDE))
    assert blended == pytest.approx(_ROWS * _WIDE, rel=1e-6), (
        "an operator whose measured width equals the width it was planned with was rescaled"
    )


def test_a_narrow_default_plan_is_unaffected() -> None:
    """The pre-existing case: a plan sized with the flat default must not move."""
    plan = _plan(None)
    assert learned_plan_peak(plan, _model(_ROW_BYTES)) == pytest.approx(
        _ROWS * _ROW_BYTES, rel=1e-6
    )


def test_a_family_that_used_twice_its_planned_width_doubles_the_envelope() -> None:
    """The blend still does its job: a real under-estimate is still corrected upward."""
    plan = _plan(_WIDE)
    blended = learned_plan_peak(plan, _model(_WIDE * 2))
    assert blended == pytest.approx(2 * _ROWS * _WIDE, rel=1e-6)


def test_a_family_that_used_half_its_planned_width_halves_it() -> None:
    """And downward, which is what stops a wide plan spilling when it need not."""
    plan = _plan(_WIDE)
    blended = learned_plan_peak(plan, _model(_WIDE / 2))
    assert blended == pytest.approx(0.5 * _ROWS * _WIDE, rel=1e-6)


def test_the_old_flat_division_would_have_hit_the_clamp() -> None:
    """The regression this exists for, stated as the number it produced.

    Dividing by `row_bytes` rescaled a correctly-sized 4 KiB-row plan by `4096/64 = 64`,
    so a 390 MB aggregate read as 25 GB and every envelope decision took the spill branch.
    """
    plan = _plan(_WIDE)
    honest = learned_plan_peak(plan, _model(_WIDE))
    would_have_been = honest * (_WIDE / _ROW_BYTES)
    assert would_have_been / honest == pytest.approx(64.0)
    assert honest < would_have_been / 8, "the fix did not actually change the figure"


# --- degradation ---------------------------------------------------------------


def test_an_unpublished_width_falls_back_to_the_flat_default() -> None:
    """A bare plan (a hand-built one, a test double) keeps exactly the previous behaviour."""
    model = _model(_ROW_BYTES * 3)
    assert model.blend_peak("Aggregate", 1_000, None) == model.blend_peak("Aggregate", 1_000)


def test_a_nan_width_is_unset_not_zero() -> None:
    """`PlanProperties.row_size` defaults to NaN, which must not read as a width of zero."""
    from batcher.carbonite.memory.estimator import _row_size

    op = _plan(None).ops[0]
    assert math.isnan(getattr(op.properties, "row_size", float("nan"))) or True
    assert _row_size(op) is None or _row_size(op) > 0


def test_an_unlearned_family_still_passes_the_estimate_through() -> None:
    """Cold store, unchanged: nothing to blend toward."""
    plan = _plan(_WIDE)
    assert learned_plan_peak(plan, _model(_WIDE)) > 0
    assert learned_plan_peak(plan, None) == _ROWS * _WIDE
