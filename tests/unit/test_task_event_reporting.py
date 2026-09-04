"""Per-task Ray event reporting, and when a stage is too wide to be worth it.

Ray emits a running and a finished event per task to the GCS. That is what puts a task in
`ray list tasks`, the Dashboard, and the State API, and at ordinary fan-out it is worth its
cost. A hundred-thousand-partition map stage is not ordinary fan-out: it puts a hundred
thousand events onto a control plane every driver in the fleet shares, to fill a table nobody
can read at that size.

The trade is observability, never scheduling — the tasks run identically either way — so the
risk this file guards is the opposite of a performance one: a default that quietly hides
ordinary queries from the Dashboard. Every assertion below is about *where the line falls*,
and the boundary cases are asserted on both sides so the cap cannot silently become "always
off" or "always on".
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher._internal.errors import ConfigError
from batcher.config import Config, config_context
from batcher.dist.executors.ray_runtime import task_event_options

pytestmark = pytest.mark.unit


def _cfg(**kwargs) -> Config:
    cfg = Config()
    return cfg.replace(distributed=dataclasses.replace(cfg.distributed, **kwargs))


def test_an_ordinary_fanout_keeps_its_tasks_visible():
    """The default must not cost a normal query its Dashboard entry."""
    assert task_event_options(1) == {}
    assert task_event_options(64) == {}


def test_the_cap_is_inclusive_on_both_sides():
    """A stage *at* the cap still reports; one task past it does not.

    Asserted as a pair because either half alone is satisfied by a constant.
    """
    with config_context(_cfg(task_events_fanout_cap=1_000)):
        assert task_event_options(1_000) == {}
        assert task_event_options(1_001) == {"enable_task_events": False}


def test_always_keeps_reporting_however_wide_the_stage():
    with config_context(_cfg(task_events="always", task_events_fanout_cap=1)):
        assert task_event_options(10_000_000) == {}


def test_never_stops_reporting_however_narrow_the_stage():
    with config_context(_cfg(task_events="never")):
        assert task_event_options(1) == {"enable_task_events": False}


def test_the_option_is_one_ray_actually_accepts():
    """The positive control. `enable_task_events` is newer than the rest of the task API, and
    an emitted keyword Ray rejects would fail every submission in the stage it was meant to
    make cheaper — while every assertion above still passed."""
    ray = pytest.importorskip("ray")
    with config_context(_cfg(task_events="never")):
        opts = task_event_options(1)
    ray.remote(lambda: None).options(**opts)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"task_events": "sometimes"}, id="unknown-mode"),
        pytest.param({"task_events_fanout_cap": 0}, id="zero-cap"),
    ],
)
def test_a_bad_setting_is_refused_at_config_time(kwargs):
    """A typo here fails open — the option is simply never emitted — so it has to be caught
    where it is written rather than where it is read."""
    with pytest.raises(ConfigError):
        _cfg(**kwargs).validate()
