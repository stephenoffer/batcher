"""`num_workers` says what it takes when given something else.

It accepts ``"auto"`` or an integer, but the resolver reached `int()` directly: anything
`int()` could parse got through, and anything it could not raised `int()`'s own message —
``invalid literal for int() with base 10: 'AUTO'`` — naming neither the parameter nor the
two things it takes. ``"AUTO"`` is worth catching by name, being a plausible spelling of the
default that failed as a *parse* error rather than an unrecognized option.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import PlanError
from batcher.ml.gpu import resolve_num_workers

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("value", "expected"), [("auto", None), (4, 4), ("4", 4), (4.0, 4)])
def test_accepted_values_still_resolve(value, expected) -> None:
    """A float was never documented but always worked, so it must keep working."""
    resolved = resolve_num_workers(value, 0.0)
    if expected is not None:
        assert resolved == expected
    else:
        assert resolved >= 1


def test_auto_on_a_gpu_stage_keeps_one_context() -> None:
    assert resolve_num_workers("auto", 1.0) == 1


@pytest.mark.parametrize("value", ["lots", "", "AUTO"])
def test_an_unparseable_string_names_the_parameter(value: str) -> None:
    with pytest.raises(PlanError, match="num_workers must be 'auto' or an integer"):
        resolve_num_workers(value, 0.0)


def test_a_near_miss_of_auto_is_suggested() -> None:
    with pytest.raises(PlanError, match="Did you mean 'auto'"):
        resolve_num_workers("AUTO", 0.0)


@pytest.mark.parametrize("value", [None, True, [], {}])
def test_a_wrong_type_names_its_type(value) -> None:
    with pytest.raises(PlanError, match="num_workers must be 'auto'"):
        resolve_num_workers(value, 0.0)


def test_a_fractional_worker_count_is_rejected_not_truncated() -> None:
    """Silently making 2.5 workers into 2 is the surprise; 4.0 is not."""
    with pytest.raises(PlanError, match="truncated to 2"):
        resolve_num_workers(2.5, 0.0)


@pytest.mark.parametrize("value", [0, -3])
def test_a_non_positive_count_still_floors_at_one(value: int) -> None:
    """Pre-existing behavior, pinned so the new validation cannot change it."""
    assert resolve_num_workers(value, 0.0) == 1
