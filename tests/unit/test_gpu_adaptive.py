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


# eight spread-out sizes so the fit clears the _MIN_SAMPLES=8 confidence gate.
_SAMPLE_ROWS = (
    2_000_000,
    6_000_000,
    12_000_000,
    18_000_000,
    26_000_000,
    40_000_000,
    60_000_000,
    80_000_000,
)


def test_learns_crossover_from_samples():
    hub = _hub()
    # not enough / no spread yet -> no learned value
    assert learned_gpu_min_rows(hub) is None
    for rows in _SAMPLE_ROWS:
        record_backend_timing(hub, "cpu", rows, _cpu_ms(rows))
        record_backend_timing(hub, "gpu", rows, _gpu_ms(rows))
    got = learned_gpu_min_rows(hub)
    assert got is not None
    # the analytic crossover (~20M) sits inside the band around the 10M default, so it is
    # recovered within 10% (not clamped).
    assert abs(got - _true_crossover()) / _true_crossover() < 0.10


def test_no_crossover_when_gpu_never_cheaper_per_row():
    hub = _hub()
    # GPU is worse per row AND has higher fixed cost -> no useful threshold, defer to config
    for rows in _SAMPLE_ROWS:
        record_backend_timing(hub, "cpu", rows, 100.0 + 1.0e-4 * rows)
        record_backend_timing(hub, "gpu", rows, 5000.0 + 5.0e-4 * rows)
    assert learned_gpu_min_rows(hub) is None


def test_needs_row_spread_not_just_count():
    hub = _hub()
    # many samples but all at the SAME row count -> slope unidentifiable -> None
    for _ in range(10):
        record_backend_timing(hub, "cpu", 10_000_000, 4200.0)
        record_backend_timing(hub, "gpu", 10_000_000, 7100.0)
    assert learned_gpu_min_rows(hub) is None


def test_learned_crossover_is_clamped_to_a_band_around_the_default():
    # A learned crossover far below the default is clamped to default/8, so a noisy early fit
    # can only nudge the threshold within a bounded range.
    hub = _hub()
    for rows in _SAMPLE_ROWS:
        record_backend_timing(hub, "cpu", rows, 50.0 + 2.0e-3 * rows)  # very steep per-row
        record_backend_timing(hub, "gpu", rows, 200.0 + 1.0e-3 * rows)  # true crossover ~150k
    learned = learned_gpu_min_rows(hub)
    from batcher.config import active_config

    default = active_config().distributed.gpu_min_rows
    assert learned == default // 8  # clamped up to the band floor


def test_decision_consumes_the_learned_threshold(monkeypatch):
    # Kyber's decision must use the learned crossover, not the config default. Pin the estimate
    # so the test turns only on the threshold: 5M input sits below the 10M default (→ CPU) but
    # above a learned+clamped crossover (→ GPU).
    from batcher.kyber.gpu import policy

    monkeypatch.setattr(policy, "_estimate", lambda *a, **k: (5_000_000, 0.08))
    ds = bt.from_pydict({"k": [1, 2], "v": [1.0, 2.0]})
    q = ds.group_by("k").agg(s=bt.col("v").sum())

    fresh = _hub()  # no learning -> config default 10M -> 5M stays on CPU
    assert decide_gpu_backend(q._plan, q._sources, fresh, gpu_count=4, force=False).use_gpu is False

    hub = _hub()  # learn a low crossover (clamps to default/8 = 1.25M) -> 5M flips to GPU
    for rows in _SAMPLE_ROWS:
        record_backend_timing(hub, "cpu", rows, 50.0 + 2.0e-3 * rows)
        record_backend_timing(hub, "gpu", rows, 200.0 + 1.0e-3 * rows)
    assert learned_gpu_min_rows(hub) < 5_000_000
    assert decide_gpu_backend(q._plan, q._sources, hub, gpu_count=4, force=False).use_gpu is True


def test_record_cpu_crossover_feeds_the_learner_when_a_gpu_is_present(monkeypatch):
    # The executor's CPU-side hook records a sample only on a GPU cluster (else it's a no-op that
    # never calls the estimator). Mock the GPU count so the gated path runs without a cluster.
    from batcher.api.terminal import gpu_backend

    monkeypatch.setattr(gpu_backend, "_cluster_gpu_count", lambda: 4)
    hub = _hub()
    ds = bt.from_pydict({"k": [1, 1, 2, 3], "v": [1.0, 2.0, 3.0, 4.0]})
    q = ds.group_by("k").agg(s=bt.col("v").sum())
    gpu_backend.record_cpu_crossover(q._plan, q._sources, hub, wall_ms=12.3)
    # a "cpu" bucket now exists in the crossover namespace
    from batcher.kyber.gpu.adaptive import _NS

    assert (hub.get_keyed_param(_NS, "cpu") or {}).get("n", 0) == 1


def test_record_cpu_crossover_is_a_noop_without_a_gpu(monkeypatch):
    from batcher.api.terminal import gpu_backend
    from batcher.kyber.gpu.adaptive import _NS

    monkeypatch.setattr(gpu_backend, "_cluster_gpu_count", lambda: 0)
    hub = _hub()
    ds = bt.from_pydict({"k": [1, 2], "v": [1.0, 2.0]})
    q = ds.group_by("k").agg(s=bt.col("v").sum())
    gpu_backend.record_cpu_crossover(q._plan, q._sources, hub, wall_ms=5.0)
    assert hub.get_keyed_param(_NS, "cpu") is None  # nothing recorded on a CPU-only cluster
