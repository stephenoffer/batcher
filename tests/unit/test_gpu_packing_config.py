"""The GPU packing tunables are refused when set out of range, not clamped in silence.

Each of these is clamped where it is used, which is what makes an out-of-range value dangerous
rather than loud: it never crashes, it becomes a different number. A `gpu_task_fraction` of
`1.5` asks Ray for one and a half devices per shard and pends the fan-out forever; a
`gpu_shard_expansion` below one claims a shard's intermediate is smaller than the shard, which
packs a device to twice what it holds and turns every run into a subdivision storm. Neither
reads as a configuration error at the point it bites.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher._internal.device_share import MAX_COTENANTS
from batcher._internal.errors import ConfigError
from batcher.config import active_config, set_config

pytestmark = pytest.mark.unit


@pytest.fixture
def restore_config():
    saved = active_config()
    yield
    set_config(saved)


def _apply(**kw):
    cfg = active_config()
    set_config(cfg.replace(distributed=dataclasses.replace(cfg.distributed, **kw)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_task_fraction", 1.5),
        ("gpu_task_fraction", -0.1),
        ("gpu_max_tasks_per_device", 0),
        ("gpu_max_tasks_per_device", MAX_COTENANTS + 1),
        ("gpu_shard_expansion", 0.5),
        ("gpu_merge_wave", -1),
    ],
)
def test_an_out_of_range_packing_knob_is_refused(field, value, restore_config) -> None:
    with pytest.raises(ConfigError, match=field):
        _apply(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_task_fraction", 0.0),
        ("gpu_task_fraction", 1.0),
        ("gpu_max_tasks_per_device", 1),
        ("gpu_max_tasks_per_device", MAX_COTENANTS),
        ("gpu_shard_expansion", 1.0),
        ("gpu_merge_wave", 0),
        ("gpu_merge_wave", 1024),
    ],
)
def test_the_boundary_values_are_accepted(field, value, restore_config) -> None:
    _apply(**{field: value})
    assert getattr(active_config().distributed, field) == value


def test_the_shipped_defaults_validate(restore_config) -> None:
    """A default that its own validator rejects would fail every process at import."""
    d = active_config().distributed
    assert 0.0 <= d.gpu_task_fraction <= 1.0
    assert 1 <= d.gpu_max_tasks_per_device <= MAX_COTENANTS
    assert d.gpu_shard_expansion >= 1.0
    assert d.gpu_merge_wave >= 0
    assert d.gpu_pack_shards is True
