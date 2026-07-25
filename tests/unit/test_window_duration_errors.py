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
@pytest.mark.parametrize("duration", ["1 fortnight", "hour", "", "5", "1h 30m", "h1"])
def test_an_unparseable_duration_raises_plan_error(duration):
    with pytest.raises(PlanError):
        bt.window(bt.col("ts"), duration)


@pytest.mark.unit
@pytest.mark.parametrize("slide", ["1 fortnight", "hour", ""])
def test_an_unparseable_slide_raises_plan_error(slide):
    with pytest.raises(PlanError):
        bt.window(bt.col("ts"), "1h", slide=slide)


@pytest.mark.unit
def test_the_message_names_the_argument_and_only_fixed_units():
    with pytest.raises(PlanError) as excinfo:
        bt.window(bt.col("ts"), "1 fortnight")
    message = str(excinfo.value)
    assert "window duration" in message, "the message names which argument was wrong"
    assert "'1 fortnight'" in message, "the message quotes the value it could not parse"
    # The guidance must not recommend a unit this function rejects.
    assert "w/d/h/m/s" in message
    assert "y/mo/w/d/h/m/s" not in message, "y and mo are calendar units window() refuses"


@pytest.mark.unit
def test_the_slide_message_names_the_slide():
    with pytest.raises(PlanError, match="window slide"):
        bt.window(bt.col("ts"), "1h", slide="1 fortnight")


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


# --------------------------------------------------------------------------- #
# A window width and a watermark delay are written in the same pipeline, so the
# two duration parsers must accept the same spellings
# --------------------------------------------------------------------------- #
_SHARED_SPELLINGS = [
    "1h",
    "1 hour",
    "30m",
    "30 minutes",
    "1d",
    "1 day",
    "1w",
    "2 weeks",
    "2h30m",
    "1s",
    "10 seconds",
    "500ms",
]


@pytest.mark.unit
@pytest.mark.parametrize("duration", _SHARED_SPELLINGS)
def test_a_window_and_a_watermark_accept_the_same_durations(duration):
    """Seven of twelve spellings used to work on exactly one of the two.

    `parse_offset` reads the compact combinable form and the streaming module reads the
    spelled-out one, and neither fell back to the other: `"1d"` sized a window but could not
    delay a watermark, and `"10 seconds"` did the reverse.
    """
    from batcher.plan.streaming.spec import Watermark

    assert bt.window(bt.col("ts"), duration) is not None
    assert Watermark.of("ts", duration).lateness_micros >= 0


@pytest.mark.unit
@pytest.mark.parametrize("duration", _SHARED_SPELLINGS)
def test_the_two_parsers_agree_on_the_value(duration):
    from batcher.plan.functions.temporal import _duration_micros
    from batcher.plan.streaming.spec import parse_interval_seconds

    assert _duration_micros(duration, arg="window duration") == round(
        parse_interval_seconds(duration) * 1_000_000
    )


@pytest.mark.unit
@pytest.mark.parametrize("duration", ["1mo", "1y", "1 fortnight", "garbage"])
def test_neither_parser_accepts_a_calendar_or_nonsense_duration(duration):
    from batcher.plan.streaming.spec import Watermark

    with pytest.raises(PlanError):
        bt.window(bt.col("ts"), duration)
    with pytest.raises(PlanError):
        Watermark.of("ts", duration)


@pytest.mark.unit
def test_a_sub_second_window_is_accepted_and_exact():
    """`500ms` is 500,000 microseconds, not a rounded second."""
    from batcher.plan.functions.temporal import _duration_micros

    assert _duration_micros("500ms", arg="window duration") == 500_000
    assert _duration_micros("250ms", arg="window duration") == 250_000
