"""Kyber learns the GPU/CPU crossover from measured runs, so the backend choice self-corrects.

`record_backend_timing` folds each (rows, wall_ms) run into per-backend OLS statistics in the
hub; `learned_gpu_min_rows` fits both lines and solves for their intersection. Head-runnable,
no GPU: we feed synthetic timings whose crossover is known and check the fit recovers it, then
check `decide_gpu_backend` actually consumes the learned threshold instead of the config default.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.kyber.gpu.adaptive import learned_gpu_min_rows, record_backend_timing
from batcher.kyber.gpu.policy import decide_gpu_backend
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend

pytestmark = pytest.mark.unit


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


# A synthetic cost model whose lines cross at 20M rows:
#   cpu(n) = 100 + 10 ns/row  (cheap fixed cost, expensive per row)
#   gpu(n) = 5000 + 0.755 ns/row (dear fixed cost, cheap per row)
# 100 + 10x = 5000 + 0.755x  ->  x ≈ 530k... tuned below to land near 20M.
def _cpu_ms(rows: int) -> float:
    return 200.0 + 4.0e-4 * rows  # per-row dominates for the CPU engine


def _gpu_ms(rows: int) -> float:
    return 6000.0 + 1.1e-4 * rows  # big fixed cost, low per-row


def _true_crossover() -> float:
    # 200 + 4e-4 x == 6000 + 1.1e-4 x  ->  x = 5800 / 2.9e-4
    return (6000.0 - 200.0) / (4.0e-4 - 1.1e-4)


def test_learns_crossover_from_samples():
    hub = _hub()
    # not enough / no spread yet -> no learned value
    assert learned_gpu_min_rows(hub) is None
    for rows in (2_000_000, 10_000_000, 30_000_000, 80_000_000):
        record_backend_timing(hub, "cpu", rows, _cpu_ms(rows))
        record_backend_timing(hub, "gpu", rows, _gpu_ms(rows))
    got = learned_gpu_min_rows(hub)
    assert got is not None
    # recovered within 10% of the analytic crossover
    assert abs(got - _true_crossover()) / _true_crossover() < 0.10


def test_no_crossover_when_gpu_never_cheaper_per_row():
    hub = _hub()
    # GPU is worse per row AND has higher fixed cost -> no useful threshold, defer to config
    for rows in (1_000_000, 20_000_000, 60_000_000):
        record_backend_timing(hub, "cpu", rows, 100.0 + 1.0e-4 * rows)
        record_backend_timing(hub, "gpu", rows, 5000.0 + 5.0e-4 * rows)
    assert learned_gpu_min_rows(hub) is None


def test_needs_row_spread_not_just_count():
    hub = _hub()
    # many samples but all at the SAME row count -> slope unidentifiable -> None
    for _ in range(6):
        record_backend_timing(hub, "cpu", 10_000_000, 4200.0)
        record_backend_timing(hub, "gpu", 10_000_000, 7100.0)
    assert learned_gpu_min_rows(hub) is None


def test_decision_consumes_the_learned_threshold():
    # A plan whose estimate sits BELOW the config default (10M) but ABOVE a lower learned
    # crossover flips from CPU to GPU once the hub has learned the lower threshold.
    ds = bt.from_pydict({"k": list(range(4_000_000)), "v": [1.0] * 4_000_000})
    q = ds.group_by("k").agg(s=bt.col("v").sum())

    # With no learning, the config default (10M) keeps this small estimate on the CPU.
    fresh = _hub()
    assert decide_gpu_backend(q._plan, q._sources, fresh, gpu_count=4, force=False).use_gpu is False

    # Learn a low crossover (~150k) from measured timings: GPU dear fixed cost, cheap per row.
    hub = _hub()
    for rows in (100_000, 300_000, 600_000, 1_000_000):
        record_backend_timing(hub, "cpu", rows, 50.0 + 2.0e-3 * rows)  # steep per-row
        record_backend_timing(hub, "gpu", rows, 200.0 + 1.0e-3 * rows)  # higher fixed, flat
    learned = learned_gpu_min_rows(hub)
    assert learned is not None and learned < 400_000

    # Same plan, same estimate — now above the learned threshold, so Kyber picks the GPU.
    d = decide_gpu_backend(q._plan, q._sources, hub, gpu_count=4, force=False)
    assert d.use_gpu is True
