"""Adaptive throughput batch-size controller (B1) — the auto-tuning Ray Data lacks."""

from __future__ import annotations

import pytest

from batcher.ml.autobatch import ThroughputController
from batcher.ml.gpu import max_actors_per_gpu, recommend_gpu_fraction


def test_climbs_toward_throughput_plateau():
    # Throughput model: rises linearly with size up to a 1024-row plateau.
    c = ThroughputController(min_rows=16, max_rows=8192, initial=64, vram_cap=0.9)
    size = c.current()
    seen = [size]
    for _ in range(40):
        throughput = min(size, 1024) * 1.0
        size = c.update(throughput, vram_fraction=0.5)
        seen.append(size)
    assert max(seen) >= 1024  # it explored past the knee
    assert c.current() >= 700  # and settled in the high-throughput region
    assert c.current() <= 8192


def test_vram_cap_forces_shrink():
    c = ThroughputController(min_rows=16, max_rows=8192, initial=4096, vram_cap=0.85)
    smaller = c.update(1000.0, vram_fraction=0.95)  # over the cap
    assert smaller < 4096


def test_stays_within_bounds():
    c = ThroughputController(min_rows=32, max_rows=128, initial=64)
    for _ in range(20):
        s = c.update(1e9)  # always "improving" → would grow unboundedly
        assert 32 <= s <= 128


def test_nan_observation_is_ignored():
    c = ThroughputController(min_rows=16, max_rows=8192, initial=256)
    before = c.current()
    assert c.update(float("nan")) == before


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        ThroughputController(min_rows=0)
    with pytest.raises(ValueError):
        ThroughputController(grow=1.0)
    with pytest.raises(ValueError):
        ThroughputController(shrink=1.5)


def test_recovers_and_regrows_after_vram_backoff():
    # After a VRAM-forced shrink, a now-safe high throughput should let it grow again.
    c = ThroughputController(min_rows=16, max_rows=8192, initial=2048, vram_cap=0.85)
    after_cap = c.update(500.0, vram_fraction=0.95)
    grown = c.update(900.0, vram_fraction=0.5)
    assert grown >= after_cap


# --- fractional-GPU packing math (C2/C3) ------------------------------------


def test_small_model_packs_many_actors_per_gpu():
    # all-MiniLM (~0.08 GB) on an A10G (24 GB): packs many; fraction floored at 0.25.
    assert max_actors_per_gpu(0.08, 24.0) >= 4
    assert recommend_gpu_fraction(0.08, 24.0) == 0.25


def test_large_model_gets_whole_gpu():
    # A 14 GB (7B) model does not fit twice on 24 GB → one actor → whole GPU.
    assert max_actors_per_gpu(14.0, 24.0) == 1
    assert recommend_gpu_fraction(14.0, 24.0) == 1.0


def test_context_overhead_reduces_density():
    many = max_actors_per_gpu(0.1, 24.0, context_overhead_gb=0.0)
    fewer = max_actors_per_gpu(0.1, 24.0, context_overhead_gb=2.0)
    assert fewer < many  # per-process device context cuts how many fit


def test_packing_always_at_least_one_and_bounded():
    assert max_actors_per_gpu(0.0, 24.0) == 1  # unknown size → whole GPU
    assert max_actors_per_gpu(100.0, 24.0) == 1  # too big → still 1 (no zero)
    assert 0.25 <= recommend_gpu_fraction(1.0, 24.0) <= 1.0


# --- predictive VRAM guard (correctness audit: no multiplicative overshoot) ---


def test_growth_halts_before_vram_overshoot():
    # At vram just below the cap, a 1.5x grow would push the next batch over the cap.
    # The predictive guard must hold the size instead of growing into an OOM.
    c = ThroughputController(min_rows=16, max_rows=8192, initial=1000, vram_cap=0.85, grow=1.5)
    # vram=0.7 → 0.7*1.5 = 1.05 > 0.85 → must NOT grow even though throughput improves.
    nxt = c.update(1000.0, vram_fraction=0.7)
    assert nxt == 1000  # held at the safe ceiling, not grown to 1500


def test_growth_proceeds_when_predicted_vram_safe():
    c = ThroughputController(min_rows=16, max_rows=8192, initial=1000, vram_cap=0.85, grow=1.5)
    # vram=0.5 → 0.5*1.5 = 0.75 <= 0.85 → safe to grow.
    nxt = c.update(1000.0, vram_fraction=0.5)
    assert nxt > 1000
