"""Adaptive morsel sizing: shrink the morsel target under memory pressure.

A morsel only batches data, so its size never changes a query's result — these tests
assert the pressure→size policy and that a query is byte-identical whether the morsel
target is the default or a pressure-shrunk one (the result-invariance the feature
rests on).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import Config, col, config_context
from batcher.carbonite import ResourceManager
from batcher.carbonite.memory.pressure import PressureLevel
from batcher.config.config import ExecutionConfig


@pytest.mark.parametrize(
    ("level", "factor"),
    [
        (PressureLevel.NORMAL, None),  # no pressure → keep the configured target
        (PressureLevel.ELEVATED, 0.5),
        (PressureLevel.SPILL, 0.25),
        (PressureLevel.CRITICAL, 0.25),
    ],
)
def test_recommend_morsel_target_scales_with_pressure(level, factor, monkeypatch):
    rm = ResourceManager()
    monkeypatch.setattr(rm._pressure, "level", lambda: level)
    base = rm._config.execution
    got = rm.recommend_morsel_target()
    if factor is None:
        assert got is None
    else:
        assert got == (int(base.morsel_rows * factor), int(base.morsel_bytes * factor))


def test_recommend_morsel_target_floors_tiny_morsels(monkeypatch):
    # A tiny configured morsel under heavy pressure never shrinks below the floors.
    cfg = Config().replace(execution=ExecutionConfig(morsel_rows=2000, morsel_bytes=100_000))
    with config_context(cfg):
        rm = ResourceManager()
        monkeypatch.setattr(rm._pressure, "level", lambda: PressureLevel.CRITICAL)
        rows, nbytes = rm.recommend_morsel_target()
        assert rows == 1024  # _MIN_MORSEL_ROWS, not 500
        assert nbytes == 64 * 1024  # _MIN_MORSEL_BYTES, not 25_000


def test_recommended_config_carries_scaled_morsel(monkeypatch):
    rm = ResourceManager()
    monkeypatch.setattr(rm._pressure, "level", lambda: PressureLevel.SPILL)
    adapted = rm.recommended_config()
    assert adapted is not None
    assert adapted.execution.morsel_rows == int(rm._config.execution.morsel_rows * 0.25)
    # Everything else is preserved (only the morsel target changes).
    assert adapted.memory == rm._config.memory


def test_adaptive_morsel_sizing_is_result_invariant(monkeypatch):
    # The contract: a shrunk morsel produces an identical result. Force a small target
    # and assert the aggregate matches the default-morsel run row-for-row.
    t = pa.table({"k": [i % 7 for i in range(5000)], "v": list(range(5000))})

    def query():
        return bt.from_arrow(t).group_by("k").agg(s=col("v").sum()).collect()

    baseline = query()

    monkeypatch.setattr(ResourceManager, "recommend_morsel_target", lambda self: (1024, 64 * 1024))
    with config_context(Config().replace(execution=ExecutionConfig(adaptive_morsel_sizing=True))):
        adapted = query()

    def rows(tbl):
        return sorted(tuple(r.values()) for r in tbl.to_pylist())

    assert rows(adapted) == rows(baseline)
