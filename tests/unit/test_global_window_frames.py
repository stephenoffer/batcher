"""The frame classification `dist/global_window/frames.py` routes on.

`frames` states its function sets as literals rather than importing them from `admission`,
because `admission` imports `frames`. A literal can drift from the table it mirrors, and a
function listed here that `admission` has no offset for would be *admitted with no
correction*: split into buckets, windowed per bucket, and returned wrong. These pin the two
tables together, and pin which frames count as which shape.
"""

from __future__ import annotations

import pytest

from batcher.dist.global_window import admission
from batcher.dist.global_window.frames import (
    ROWS_RUNNING_FUNCS,
    TRAILING_FUNCS,
    frame_for_helpers,
    is_rows_running_frame,
    trailing_rows_distance,
)
from batcher.plan.expr_ir import Col
from batcher.plan.logical.window import WindowFrame, WindowFuncSpec

pytestmark = pytest.mark.unit


def _fn(func: str, frame: WindowFrame | None) -> WindowFuncSpec:
    return WindowFuncSpec(func=func, input=Col("x"), alias="w", frame=frame)


def test_rows_running_funcs_are_offsettable():
    assert set(ROWS_RUNNING_FUNCS) <= set(admission._OFFSETTABLE)
    assert set(TRAILING_FUNCS) <= set(ROWS_RUNNING_FUNCS)


@pytest.mark.parametrize(
    ("frame", "running", "distance"),
    [
        (WindowFrame(None, 0, "rows"), True, None),
        (WindowFrame(-3, 0, "rows"), False, 3),
        (WindowFrame(0, 0, "rows"), False, None),  # CURRENT ROW reads no prior bucket
        (WindowFrame(-3, 1, "rows"), False, None),  # a FOLLOWING edge
        (WindowFrame(-3, -1, "rows"), False, None),  # ends before the current row
        (WindowFrame(None, 0, "range"), False, None),  # the default frame, spelled out
        (WindowFrame(-3, 0, "range"), False, None),  # a value offset, not a row count
        (WindowFrame(0, None, "rows"), False, None),  # a reverse running frame
    ],
)
def test_frame_shapes(frame, running, distance):
    fn = _fn("sum", frame)
    assert is_rows_running_frame(fn) is running
    assert trailing_rows_distance(fn) == distance
    assert frame_for_helpers(fn) == (frame if running or distance else None)


def test_a_moment_takes_the_running_frame_but_not_the_trailing_one():
    assert is_rows_running_frame(_fn("var", WindowFrame(None, 0, "rows")))
    assert trailing_rows_distance(_fn("var", WindowFrame(-3, 0, "rows"))) is None
