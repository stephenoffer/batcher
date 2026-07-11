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
    # `recommend_morsel_target` is a pure *reader* of pressure (`classify`), not its sampler.
    monkeypatch.setattr(rm._pressure, "classify", lambda: level)
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
        monkeypatch.setattr(rm._pressure, "classify", lambda: PressureLevel.CRITICAL)
        rows, nbytes = rm.recommend_morsel_target()
        assert rows == 1024  # _MIN_MORSEL_ROWS, not 500
        assert nbytes == 64 * 1024  # _MIN_MORSEL_BYTES, not 25_000


def test_recommended_config_carries_scaled_morsel(monkeypatch):
    rm = ResourceManager()
    monkeypatch.setattr(rm._pressure, "classify", lambda: PressureLevel.SPILL)
    adapted = rm.recommended_config()
    assert adapted is not None
    assert adapted.execution.morsel_rows == int(rm._config.execution.morsel_rows * 0.25)
    # Everything else is preserved (only the morsel target changes).
    assert adapted.memory == rm._config.memory


def test_morsel_cap_restricted_to_plan_families(monkeypatch):
    # C10: a wide family (a big aggregate) learned in an earlier query must not throttle a
    # narrow scan/filter plan. Passing that plan's own families leaves its morsel untouched;
    # passing the wide family (or nothing) tightens it.
    from batcher.carbonite.memory.learned import LearnedMemoryModel

    rm = ResourceManager()
    monkeypatch.setattr(rm._pressure, "classify", lambda: PressureLevel.NORMAL)
    ex = rm._config.execution
    # A learned aggregate width that fills a morsel to 4× the byte budget.
    wide = LearnedMemoryModel(
        _bytes_per_row={"aggregate": (4.0 * ex.morsel_bytes) / ex.morsel_rows},
        _alpha=0.5,
        _clamp=4.0,
        _row_bytes=8,
        _spill_per_row={},
    )
    monkeypatch.setattr(rm, "_mem_model", wide)
    # A plan touching only scan/filter is unaffected by the aggregate's width.
    assert rm.recommend_morsel_target(["Scan", "Filter"]) is None
    # A plan that includes the aggregate (or the global default) is tightened.
    tightened = rm.recommend_morsel_target(["Aggregate"])
    assert tightened is not None and tightened[0] < ex.morsel_rows
    assert rm.recommend_morsel_target()[0] < ex.morsel_rows  # global default still tightens


def test_adaptive_morsel_sizing_is_result_invariant(monkeypatch):
    # The contract: a shrunk morsel produces an identical result. Force a small target
    # and assert the aggregate matches the default-morsel run row-for-row.
    t = pa.table({"k": [i % 7 for i in range(5000)], "v": list(range(5000))})

    def query():
        return bt.from_arrow(t).group_by("k").agg(s=col("v").sum()).collect()

    baseline = query()

    monkeypatch.setattr(
        ResourceManager,
        "recommend_morsel_target",
        lambda self, families=None: (1024, 64 * 1024),
    )
    with config_context(Config().replace(execution=ExecutionConfig(adaptive_morsel_sizing=True))):
        adapted = query()

    def rows(tbl):
        return sorted(tuple(r.values()) for r in tbl.to_pylist())

    assert rows(adapted) == rows(baseline)
