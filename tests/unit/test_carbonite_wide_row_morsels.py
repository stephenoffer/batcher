"""A morsel must fit its byte budget even when one row is bigger than the budget.

Two coupled gaps made the byte budget unenforceable on exactly the columns that need it:

* `MIN_MORSEL_ROWS` (1,024) was applied unconditionally, so it **overrode** the byte bound
  for any row wider than `morsel_bytes / 1024` — 1,024 bytes at the defaults, which is
  below essentially every unstructured or multimodal column.
* the width cap was read only from the *learned* memory model, which is empty on a cold
  store — and the first run of a multimodal pipeline is the one with no measurement and the
  one that OOMs.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.carbonite.memory.pressure import PressureLevel
from batcher.carbonite.policies.morsel import (
    MIN_MORSEL_ROWS,
    morsel_target,
    planned_row_cap,
    row_floor,
)
from batcher.config import active_config

pytestmark = pytest.mark.unit


def _tensor_frame(shape=(224, 224, 3), rows: int = 4):
    """A frame carrying a canonical `arrow.fixed_shape_tensor` column — a decoded image."""
    arr = pa.FixedShapeTensorArray.from_numpy_ndarray(np.zeros((rows, *shape), dtype="uint8"))
    return bt.from_arrow(pa.table({"img": arr}))


# --- the floor must not override the budget it accompanies ------------------------


def test_the_row_floor_holds_on_narrow_rows():
    # The floor's whole reason to exist: per-batch overhead on rows small enough that a
    # 1,024-row batch is still well inside the budget. Unchanged.
    budget = active_config().execution.morsel_bytes
    assert row_floor(budget, 8.0) == MIN_MORSEL_ROWS
    assert row_floor(budget, 64.0) == MIN_MORSEL_ROWS


def test_the_row_floor_yields_to_the_byte_budget_on_wide_rows():
    # At the defaults the crossover is 1,024 B/row. Past it, insisting on 1,024 rows is
    # insisting on a morsel over budget by the ratio — 147x for an image, 6,000x for a
    # 1080p frame — which is the opposite of what a floor is for.
    budget = active_config().execution.morsel_bytes
    embedding, image, frame = 768 * 4.0, 224.0 * 224 * 3, 1920.0 * 1080 * 3
    for width in (embedding, image, frame):
        rows = row_floor(budget, width)
        # Inside the budget, or a single row when one row alone already exceeds it — which
        # is the case for a 1080p frame and is the only morsel that exists for it.
        assert rows == 1 or rows * width <= budget


def test_a_row_wider_than_the_whole_budget_gets_one_row():
    # A single 1080p frame is ~5.9 MiB against a 1 MiB budget. One row is the only morsel
    # that exists; a floor that demands more is demanding an OOM.
    assert row_floor(1 << 20, 6_220_800.0) == 1


# --- the cold-start width, from the schema ---------------------------------------


def test_a_narrow_plan_is_left_completely_alone():
    # The safety property. A plan whose rows are no wider than assumed must produce no cap
    # and no recommendation at all, so the unpressured common case is untouched.
    ds = bt.from_pydict({"a": list(range(100)), "b": list(range(100))})
    cfg = active_config()
    assert planned_row_cap(cfg, ds._plan) is None
    assert morsel_target(cfg, PressureLevel.NORMAL, None, None, ds._plan) is None


def test_a_tensor_column_is_sized_from_its_schema_before_anything_is_measured():
    # No learned model, no pressure — and the morsel is still cut to fit, because the width
    # of a fixed-shape tensor is in its type and was knowable all along.
    cfg = active_config()
    ds = _tensor_frame()
    cap = planned_row_cap(cfg, ds._plan)
    assert cap is not None
    assert cap * (224 * 224 * 3) <= cfg.execution.morsel_bytes
    rows, _ = morsel_target(cfg, PressureLevel.NORMAL, None, None, ds._plan)
    assert rows == cap
    assert rows < MIN_MORSEL_ROWS  # the flat floor would have been 170x over budget


def test_a_wider_tensor_gets_a_smaller_morsel():
    # The cap tracks the width rather than snapping to a constant.
    cfg = active_config()
    small = planned_row_cap(cfg, _tensor_frame(shape=(32, 32, 3))._plan)
    large = planned_row_cap(cfg, _tensor_frame(shape=(224, 224, 3))._plan)
    assert large is not None
    assert small is None or small > large


def test_the_more_binding_of_the_two_widths_wins():
    # The two signals cover each other's blind spots and neither may suppress the other.
    # The learned width is keyed by operator *family*, so a narrow measurement from an
    # earlier query must not stand down a cap this plan's own schema demands; and a learned
    # measurement of a wide variable-length payload must still bind when the schema (which
    # can only price a varlen column by prior) says nothing.
    cfg = active_config()

    class _Model:
        def __init__(self, width):
            self._width = width

        def max_bytes_per_row(self, families=None):
            return self._width

    ds = _tensor_frame()
    schema_only = morsel_target(cfg, PressureLevel.NORMAL, None, None, ds._plan)
    # A narrow earlier measurement does not undo the schema's exact tensor width.
    assert morsel_target(cfg, PressureLevel.NORMAL, _Model(8.0), None, ds._plan) == schema_only
    # A wider measurement binds harder than the schema.
    wide = morsel_target(cfg, PressureLevel.NORMAL, _Model(4_000_000.0), None, ds._plan)
    assert wide[0] < schema_only[0]

    # And on a narrow plan, a wide measurement is still the thing that saves it: the schema
    # prices a string column by prior and cannot know it holds megabytes.
    narrow = bt.from_pydict({"blob": ["x"] * 10})
    assert morsel_target(cfg, PressureLevel.NORMAL, None, None, narrow._plan) is None
    assert (
        morsel_target(cfg, PressureLevel.NORMAL, _Model(4_000_000.0), None, narrow._plan)[0]
        < MIN_MORSEL_ROWS
    )


def test_pressure_and_width_compose_without_the_floor_undoing_either():
    cfg = active_config()
    ds = _tensor_frame()
    normal = morsel_target(cfg, PressureLevel.NORMAL, None, None, ds._plan)
    pressed = morsel_target(cfg, PressureLevel.CRITICAL, None, None, ds._plan)
    assert pressed[0] <= normal[0]
    assert pressed[1] < normal[1]
