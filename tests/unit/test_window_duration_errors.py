"""`window()` rejects a bad duration with the project's typed error, and says the right thing.

`_duration_micros` raises `PlanError` for a calendar unit and for a non-positive duration,
but an *unparseable* string used to escape as the bare `ValueError` from `parse_offset` —
whose advice names the ``y``/``mo`` units that `window()` goes on to refuse. So the one
message a user reached by mistyping a duration pointed them at a unit that cannot work.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.mark.unit
@pytest.mark.parametrize("duration", ["250ms", "1 fortnight", "hour", "", "5", "1h 30m"])
def test_an_unparseable_duration_raises_plan_error(duration):
    with pytest.raises(PlanError):
        bt.window(bt.col("ts"), duration)


@pytest.mark.unit
@pytest.mark.parametrize("slide", ["250ms", "hour", ""])
def test_an_unparseable_slide_raises_plan_error(slide):
    with pytest.raises(PlanError):
        bt.window(bt.col("ts"), "1h", slide=slide)


@pytest.mark.unit
def test_the_message_names_the_argument_and_only_fixed_units():
    with pytest.raises(PlanError) as excinfo:
        bt.window(bt.col("ts"), "250ms")
    message = str(excinfo.value)
    assert "window duration" in message, "the message names which argument was wrong"
    assert "'250ms'" in message, "the message quotes the value it could not parse"
    # The guidance must not recommend a unit this function rejects.
    assert "w/d/h/m/s" in message
    assert "y/mo/w/d/h/m/s" not in message, "y and mo are calendar units window() refuses"


@pytest.mark.unit
def test_the_slide_message_names_the_slide():
    with pytest.raises(PlanError, match="window slide"):
        bt.window(bt.col("ts"), "1h", slide="250ms")


@pytest.mark.unit
@pytest.mark.parametrize("duration", ["1mo", "1y", "2mo3d"])
def test_a_calendar_unit_is_still_rejected_as_before(duration):
    with pytest.raises(PlanError, match="calendar unit"):
        bt.window(bt.col("ts"), duration)


@pytest.mark.unit
@pytest.mark.parametrize("duration", ["0s", "-1h"])
def test_a_non_positive_duration_is_still_rejected_as_before(duration):
    with pytest.raises(PlanError, match="positive duration"):
        bt.window(bt.col("ts"), duration)


@pytest.mark.unit
@pytest.mark.parametrize("duration", ["1h", "30m", "1h30m", "1d", "1w", "45s"])
def test_a_fixed_length_duration_is_accepted(duration):
    assert bt.window(bt.col("ts"), duration) is not None
