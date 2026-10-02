"""The derived fields of `StreamingQueryProgress`: lateness, late rows and the watermark.

These three are what an operator reads to decide whether a stream is healthy, and each one
folds several raw numbers into one. The folds are where they can be wrong without looking
wrong: a watermark taken as the *maximum* across operators reports a window closed while
another operator still admits rows into it, and a late-row count read off one operator
hides the drops in the others. So each fold is pinned against more than one operator, and
against the empty case.
"""

from __future__ import annotations

import pytest

from batcher.plan.streaming import StateOperatorProgress, StreamingQueryProgress

pytestmark = pytest.mark.unit


def _progress(*ops: StateOperatorProgress, behind_by_ms: float = 0.0) -> StreamingQueryProgress:
    return StreamingQueryProgress(
        0, 10, 10, 100.0, 0.0, behind_by_ms=behind_by_ms, state_operators=ops
    )


@pytest.mark.parametrize(("behind", "expected"), [(0.0, False), (0.001, True), (150.0, True)])
def test_is_behind_is_strictly_positive_lag(behind, expected):
    assert _progress(behind_by_ms=behind).is_behind is expected


def test_num_late_rows_sums_every_stateful_operator():
    p = _progress(
        StateOperatorProgress("windowed_aggregate", num_late_inputs_dropped=2),
        StateOperatorProgress("stream_join", num_late_inputs_dropped=5),
        StateOperatorProgress("dedup"),
    )
    assert p.num_late_rows == 7


def test_num_late_rows_is_zero_for_a_stateless_pipeline():
    assert _progress().num_late_rows == 0


def test_the_watermark_is_the_minimum_across_operators_ignoring_unset_ones():
    p = _progress(
        StateOperatorProgress("a", watermark_micros=900),
        StateOperatorProgress("b", watermark_micros=None),
        StateOperatorProgress("c", watermark_micros=300),
        StateOperatorProgress("d", watermark_micros=600),
    )
    assert p.event_time_watermark_micros == 300


def test_the_watermark_is_none_until_some_operator_has_one():
    assert _progress().event_time_watermark_micros is None
    assert _progress(StateOperatorProgress("a")).event_time_watermark_micros is None


def test_a_zero_watermark_is_a_watermark_not_an_absence():
    """The epoch is a legal event time; a truthiness test would drop it."""
    p = _progress(StateOperatorProgress("a", watermark_micros=0))
    assert p.event_time_watermark_micros == 0
