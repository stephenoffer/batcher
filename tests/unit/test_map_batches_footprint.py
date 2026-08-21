"""`map_batches` is the operator at the centre of every inference pipeline, and it was
invisible to both halves of the resource loop.

* Its **width** fell back to the flat 64 B/row constant, because it is executed in Python
  and publishes no output schema. The rows flowing through it are the widest in the engine.
* Its **memory** was budgeted at one morsel, though a stage carrying an explicit
  `batch_size` re-batches to exactly that many rows regardless of the morsel it was handed.

Together those made a stage really holding gigabytes report one megabyte to Carbonite.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.annotate import annotate_ops
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel

pytestmark = pytest.mark.unit

_IMAGE_BYTES = 224 * 224 * 3


def _images(rows: int = 8):
    arr = pa.FixedShapeTensorArray.from_numpy_ndarray(np.zeros((rows, 224, 224, 3), dtype="uint8"))
    return bt.from_arrow(pa.table({"img": arr}))


def _annotate(ds):
    est = CardinalityEstimator(ds._sources)
    return annotate_ops(ds._plan, est, active_config(), CostModel(est)), est


# --- the width -------------------------------------------------------------------


def test_a_passthrough_stage_is_priced_at_its_input_width():
    # `MapBatches.available_columns` already implements the operator's stated contract --
    # "if `output_columns` is omitted, the input columns are assumed to pass through" -- so
    # pricing the same assumption is not a new guess, it is the existing one costed.
    ds = _images()
    est = CardinalityEstimator(ds._sources)
    staged = ds.ml.map_batches(lambda b: b)
    assert est.row_width(staged._plan, 64.0) == pytest.approx(_IMAGE_BYTES)


def test_a_declared_output_schema_keeps_the_flat_default():
    # A declared `output_columns` means the shape genuinely changed, and the plan knows
    # nothing about the columns the UDF invented. Claiming the input's width for them would
    # be a fabricated number rather than a costed contract.
    ds = _images()
    est = CardinalityEstimator(ds._sources)
    staged = ds.ml.map_batches(lambda b: b, output_columns=("score",))
    assert est.row_width(staged._plan, 64.0) == 64.0


def test_the_width_is_not_asserted_as_a_schema():
    # Deliberately *not* fixed by giving `MapBatches` an `available_schema`: that method
    # feeds type inference and expression validation, where claiming the input's types
    # survive a UDF that may rewrite them turns an estimate into a wrong answer.
    staged = _images().ml.map_batches(lambda b: b)
    assert staged._plan.available_schema() is None


# --- the memory ------------------------------------------------------------------


def test_an_explicit_batch_size_is_what_the_stage_holds():
    # A stage re-batches to `batch_size` rows regardless of the morsel it was handed, so the
    # morsel byte cap does not bound it. At the batch sizes `kyber/gpu/sizing.py` seeds from
    # VRAM headroom, that is gigabytes on an image column.
    #
    # The budget is `min(batch_size, rows) x width`: `_streaming_bytes` also caps every
    # in-flight estimate at the rows that actually exist, because an operator cannot hold
    # more rows than its input has. So the fixture is sized so the *batch* is what binds —
    # a batch larger than the relation would measure the row cap instead, and eight rows of
    # decoded image already clear the morsel byte budget several times over, which is the
    # property under test.
    ds = _images(rows=8)
    ops, _ = _annotate(ds.ml.map_batches(lambda b: b, batch_size=8))
    stage = next(o for o in ops if o.kind == "MapBatches")
    assert stage.bounds.m_max_bytes == pytest.approx(8 * _IMAGE_BYTES, rel=0.01)
    assert stage.bounds.m_max_bytes > active_config().execution.morsel_bytes


def test_a_stage_with_no_batch_size_is_still_one_morsel():
    # The safety property: without an explicit batch size the stage really does stream a
    # morsel at a time, and its budget must not move.
    ds = _images()
    ops, _ = _annotate(ds.ml.map_batches(lambda b: b))
    stage = next(o for o in ops if o.kind == "MapBatches")
    assert stage.bounds.m_max_bytes <= active_config().execution.morsel_bytes


def test_a_narrow_stage_is_unchanged_by_either_fix():
    narrow = bt.from_pydict({"x": list(range(64))})
    ops, est = _annotate(narrow.ml.map_batches(lambda b: b))
    stage = next(o for o in ops if o.kind == "MapBatches")
    assert stage.bounds.m_max_bytes <= active_config().execution.morsel_bytes
    # And the width is the input's real (narrow) width, not an inflated one.
    assert est.row_width(narrow.ml.map_batches(lambda b: b)._plan, 64.0) <= 64.0


def test_the_batch_size_budget_scales_with_the_batch():
    # Both batch sizes stay at or under the relation's row count, so the batch is what binds
    # in each case and the ratio is the batch ratio rather than the row cap's.
    ds = _images(rows=8)
    small, _ = _annotate(ds.ml.map_batches(lambda b: b, batch_size=2))
    large, _ = _annotate(ds.ml.map_batches(lambda b: b, batch_size=8))

    def mem(ops):
        return next(o for o in ops if o.kind == "MapBatches").bounds.m_max_bytes

    assert mem(large) == pytest.approx(4 * mem(small), rel=0.01)
