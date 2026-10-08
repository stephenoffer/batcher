"""A drop in free RAM that is the engine's own retained arena does not re-key the config.

`api.orchestration.autoconfig.resolve_auto_config` senses the memory envelope from live free
RAM, and the config it returns is part of every plan-memo and reuse-verdict key. After a large
query mimalloc keeps the pages it freed for its purge delay, so the next query sensed less
free RAM and re-planned under a smaller budget (TPC-DS sf10 q4: 0.6 s -> 2.1 s). Pinned here:
a drop matched by the engine's resident-set growth, inside the retention window, hands back
the same config object; a drop the engine's growth does not account for, or one sensed after
the window, is honored.
"""

from __future__ import annotations

import pytest

from batcher.api.orchestration import autoconfig
from batcher.config import Config

GB = 1 << 30


@pytest.fixture
def box(monkeypatch):
    state = {"sensed": 15 * GB, "held": 27 * GB, "now": 1000.0}
    monkeypatch.setattr(
        "batcher.carbonite.memory.pressure.PressureMonitor.envelope_bytes",
        lambda self: state["sensed"],
    )
    monkeypatch.setattr(autoconfig, "_engine_held_bytes", lambda: state["held"])
    monkeypatch.setattr(autoconfig.time, "monotonic", lambda: state["now"])
    monkeypatch.setattr(autoconfig, "_RESOLVED", None)
    monkeypatch.setattr(autoconfig, "_HELD_AT_SENSE", None)
    monkeypatch.setattr(autoconfig, "_LAST_END", None)
    return state


def _after_a_query(state, *, sensed, held, seconds_later):
    autoconfig._LAST_END = state["now"]
    state.update(sensed=sensed, held=held, now=state["now"] + seconds_later)


def test_the_engines_retained_arena_keeps_the_config(box):
    cfg = Config()
    # A staged run left 5.6 GB resident; free RAM fell by 6 GB (the 64 GB TPC-DS box).
    box.update(sensed=int(14.7 * GB), held=int(28.7 * GB))
    first = autoconfig.resolve_auto_config(cfg)
    _after_a_query(box, sensed=int(8.7 * GB), held=int(34.3 * GB), seconds_later=1.0)
    assert autoconfig.resolve_auto_config(cfg) is first
    # Free RAM has partly come back and the arena is partly purged: the envelope stays near
    # where it was (14.2 GB), not the 10.5 GB the free-RAM reading alone says.
    _after_a_query(box, sensed=int(10.5 * GB), held=int(32.4 * GB), seconds_later=1.0)
    assert autoconfig.resolve_auto_config(cfg).memory.max_memory_bytes >= 14 * GB


def test_a_larger_retention_is_credited_into_a_new_envelope(box):
    cfg = Config()
    first = autoconfig.resolve_auto_config(cfg)
    _after_a_query(box, sensed=7 * GB, held=32 * GB, seconds_later=1.0)
    second = autoconfig.resolve_auto_config(cfg)
    assert second.memory.max_memory_bytes == 12 * GB, "8 GB drop, 5 GB of it the engine's"
    assert second is not first


def test_a_drop_the_engine_did_not_cause_is_pressure(box):
    cfg = Config()
    first = autoconfig.resolve_auto_config(cfg)
    # Another process took 6 GB; this one's resident set did not move.
    _after_a_query(box, sensed=9 * GB, held=27 * GB, seconds_later=1.0)
    second = autoconfig.resolve_auto_config(cfg)
    assert second is not first
    assert second.memory.max_memory_bytes == 9 * GB


def test_memory_still_held_after_the_window_is_live(box):
    cfg = Config()
    first = autoconfig.resolve_auto_config(cfg)
    _after_a_query(
        box,
        sensed=int(8.7 * GB),
        held=int(32.4 * GB),
        seconds_later=autoconfig._RETENTION_WINDOW_S + 1,
    )
    second = autoconfig.resolve_auto_config(cfg)
    assert second is not first
    assert second.memory.max_memory_bytes == int(8.7 * GB)


def test_only_the_growth_is_credited(box):
    cfg = Config()
    first = autoconfig.resolve_auto_config(cfg)
    # 6 GB less free RAM, of which only 1 GB is the engine's growth: still real pressure.
    _after_a_query(box, sensed=9 * GB, held=28 * GB, seconds_later=1.0)
    assert autoconfig.resolve_auto_config(cfg) is not first
