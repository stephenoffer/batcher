"""The learned reducer count sizes from a high quantile, not a mean.

The errors are asymmetric, and that is what picks the statistic. The count is clamped to
`[1, workers]`, so it only ever *reduces* the fan-out: an under-estimate hands each reducer
more state than it can hold and spills, while an over-estimate is capped at the worker count
the query would have used anyway.

A family's history is also routinely multi-modal — the same aggregate runs over a day and
over a year — and a mean sits between the modes describing neither, which is exactly the
reasoning `kyber.cpu_shares` already applies to CPU utilization.
"""

from __future__ import annotations

import pytest

from batcher.dist.adaptive_sizing.sizing import _sizing_quantile

pytestmark = pytest.mark.unit


def test_empty_history_is_zero():
    assert _sizing_quantile([]) == 0.0


def test_a_single_sample_is_itself():
    assert _sizing_quantile([100.0]) == 100.0


def test_uniform_history_returns_that_value():
    assert _sizing_quantile([50.0] * 10) == 50.0


def test_a_bimodal_family_sizes_for_its_large_mode():
    """A mean would sit between the modes and under-provision every large run."""
    history = [10.0] * 8 + [10_000.0] * 2
    assert _sizing_quantile(history) == 10_000.0
    assert sum(history) / len(history) < 10_000.0  # the mean would not


def test_a_short_history_takes_its_maximum():
    """With little to go on, the conservative reading is the largest run seen."""
    assert _sizing_quantile([1.0, 2.0, 3.0]) == 3.0


def test_a_lone_outlier_does_not_dominate_a_long_history():
    """The quantile is not the max: one freak run must not pin the fan-out forever."""
    history = [10.0] * 100 + [1_000_000.0]
    assert _sizing_quantile(history) == 10.0


def test_the_result_is_always_an_observed_value():
    """Nearest-rank, so the number is a volume the family was actually measured at rather
    than a point between two modes that no run ever produced."""
    history = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert _sizing_quantile(history) in history


def test_order_does_not_matter():
    assert _sizing_quantile([5.0, 1.0, 3.0]) == _sizing_quantile([1.0, 3.0, 5.0])
