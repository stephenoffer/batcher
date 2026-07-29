"""Training on sorted data with shuffling off is a silent convergence bug.

The model sees a contiguous run of the sort key per step — all of one class, then the next
— and learns the ordering. Nothing raises; the loss curve just looks disappointing. The
field guides file it under "training loss not decreasing" and "non-deterministic results",
where the listed causes are the learning rate and the data, never the ordering.

`shuffle=False` is the right default for `to_torch_dataloader` because it is
`torch.utils.data.DataLoader`'s, and a user reaching for that method comes from torch. So
the default stands and the *plan* is inspected instead: a `sort` the user wrote, feeding a
loader with no shuffling, is a combination almost nobody wants.
"""

from __future__ import annotations

import warnings

import pytest

import batcher as bt
from batcher._internal.errors import PerformanceWarning
from batcher.api.dataset.ml import _warn_if_training_on_sorted_data

pytestmark = pytest.mark.unit


def _sorted_plan():
    return bt.from_pydict({"label": [1, 0, 1], "x": [1.0, 2.0, 3.0]}).sort("label")._plan


def _plain_plan():
    return bt.from_pydict({"label": [1, 0, 1], "x": [1.0, 2.0, 3.0]})._plan


def test_a_sorted_plan_without_shuffling_warns():
    with pytest.warns(PerformanceWarning, match="explicitly sorted"):
        _warn_if_training_on_sorted_data(_sorted_plan(), shuffle=False, window=None)


def test_shuffling_silences_it():
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        _warn_if_training_on_sorted_data(_sorted_plan(), shuffle=True, window=None)


def test_an_explicit_window_silences_it():
    """`local_shuffle_buffer_size` is shuffling by another spelling."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        _warn_if_training_on_sorted_data(_sorted_plan(), shuffle=False, window=4096)


def test_an_unsorted_plan_is_silent():
    """A corpus that was never sorted is the common case and must stay quiet — advice on
    every loader is how a reader learns to ignore it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        _warn_if_training_on_sorted_data(_plain_plan(), shuffle=False, window=None)


def test_it_says_when_the_order_is_legitimate():
    """A sequence model wants the order preserved; the advice must not read as unconditional."""
    with pytest.warns(PerformanceWarning, match="sequence model"):
        _warn_if_training_on_sorted_data(_sorted_plan(), shuffle=False, window=None)


def test_a_sort_nested_below_other_operators_is_still_found():
    """The sort need not be the last thing the user wrote."""
    plan = (
        bt.from_pydict({"label": [1, 0, 1], "x": [1.0, 2.0, 3.0]})
        .sort("label")
        .filter(bt.col("x") > 0.0)
        ._plan
    )
    with pytest.warns(PerformanceWarning, match="explicitly sorted"):
        _warn_if_training_on_sorted_data(plan, shuffle=False, window=None)
