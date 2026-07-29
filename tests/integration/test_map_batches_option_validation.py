"""Every `map_batches` option says what it takes when given something else.

The retry options were range-checked but not *type*-checked, so a string reached the
comparison and raised Python's own ``'<' not supported between instances of 'str' and
'int'``. Two more were not checked at all: `model_memory_gb` feeds the resource layer and
Kyber's cost model, and `resources` goes straight to Ray's scheduler — both accepted
negatives, strings, and `None` in silence, which is a request that can never be satisfied.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"x": [1, 2, 3]})


def _run(ds: bt.Dataset, **options) -> None:
    ds.map_batches(lambda b: b, **options).to_pydict()


@pytest.mark.parametrize("option", ["max_errored_rows", "max_retries"])
@pytest.mark.parametrize("value", ["many", None, -1])
def test_whole_number_options_reject_bad_input(ds: bt.Dataset, option: str, value) -> None:
    with pytest.raises(PlanError, match=option):
        _run(ds, **{option: value})


@pytest.mark.parametrize("option", ["max_errored_rows", "max_retries"])
def test_whole_number_options_reject_a_fraction(ds: bt.Dataset, option: str) -> None:
    with pytest.raises(PlanError, match="whole number"):
        _run(ds, **{option: 2.5})


@pytest.mark.parametrize("option", ["timeout", "retry_backoff", "model_memory_gb"])
@pytest.mark.parametrize("value", ["slow", None, -1.0])
def test_numeric_options_reject_bad_input(ds: bt.Dataset, option: str, value) -> None:
    with pytest.raises(PlanError, match=option):
        _run(ds, **{option: value})


def test_a_fractional_numeric_option_is_still_allowed(ds: bt.Dataset) -> None:
    """`timeout=0.5` and `model_memory_gb=1.5` are meaningful; only whole-count options are not."""
    _run(ds, timeout=0.5, retry_backoff=0.25, model_memory_gb=1.5)


@pytest.mark.parametrize(
    "value", [{"TPU": -1}, {"TPU": "x"}, {"TPU": 0}, {1: 2}, {"": 4}, "notadict", 5]
)
def test_resources_rejects_a_request_ray_could_never_satisfy(ds: bt.Dataset, value) -> None:
    with pytest.raises(PlanError, match="resources"):
        _run(ds, resources=value)


def test_resources_normalizes_to_the_declared_field_type(ds: bt.Dataset) -> None:
    """The node declares `tuple[tuple[str, float], ...]`; it used to store whatever came in."""
    plan = ds.map_batches(lambda b: b, resources={"TPU": 4})
    assert plan._plan.resources == (("TPU", 4.0),)
    assert all(isinstance(amount, float) for _name, amount in plan._plan.resources)


def test_no_resources_stays_empty(ds: bt.Dataset) -> None:
    assert ds.map_batches(lambda b: b)._plan.resources == ()


def test_the_valid_option_set_still_runs(ds: bt.Dataset) -> None:
    out = ds.map_batches(
        lambda b: b,
        max_errored_rows=5,
        max_retries=3,
        retry_backoff=0.5,
        timeout=1.0,
        model_memory_gb=2.0,
        resources={"TPU": 4},
    )
    assert out.to_pydict() == {"x": [1, 2, 3]}
