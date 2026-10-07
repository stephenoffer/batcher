"""The shared provider quota behind `bt.ml.ProviderLimit` (AP-392), without Ray.

`QuotaState` is the whole policy and runs on an injected clock, so the rate, concurrency and
lease-expiry rules are pinned exactly. `LocalQuota` and an engine with ``shared_limit=`` are
then driven from many threads, the single-process stand-in for many workers. The cluster
path is `tests/integration/test_shared_provider_limit.py`.
"""

from __future__ import annotations

import threading
import time

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.dist.limits import LocalQuota, QuotaConfig, QuotaState

pytestmark = pytest.mark.unit


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_requests_per_minute_turn_into_queued_send_delays():
    clock = _Clock()
    state = QuotaState(QuotaConfig(requests_per_minute=60, burst=2 / 60), clock)
    delays = [state.try_acquire()[1] for _ in range(4)]
    assert delays == pytest.approx([0.0, 0.0, 1.0, 2.0])  # two in the bucket, then 1/s
    clock.now = 10.0
    assert state.try_acquire()[1] == pytest.approx(0.0)  # refilled while idle


def test_tokens_per_minute_bind_independently_and_cap_one_request_at_a_bucket():
    clock = _Clock()
    state = QuotaState(QuotaConfig(tokens_per_minute=600), clock)
    assert state.try_acquire(600)[1] == pytest.approx(0.0)
    assert state.try_acquire(60)[1] == pytest.approx(6.0)  # 10 tokens/s refill
    assert state.try_acquire(10**9)[1] == pytest.approx(66.0)  # capped at one bucket


def test_concurrency_slots_are_leased_released_and_reclaimed():
    clock = _Clock()
    state = QuotaState(QuotaConfig(max_concurrency=2, lease_seconds=30), clock)
    a, _ = state.try_acquire()
    b, _ = state.try_acquire()
    assert a and b
    assert state.try_acquire()[0] is None  # full: nothing is charged, ask again
    state.release(a)
    c, _ = state.try_acquire()
    assert c is not None
    clock.now = 31.0  # b and c were never released: their leases expire
    assert state.in_flight == 0


def test_the_latest_configuration_wins():
    clock = _Clock()
    state = QuotaState(QuotaConfig(max_concurrency=1), clock)
    assert state.try_acquire()[0] is not None
    assert state.try_acquire()[0] is None
    state.configure(QuotaConfig(max_concurrency=2))
    assert state.try_acquire()[0] is not None


def test_local_quota_holds_concurrency_across_threads():
    quota = LocalQuota(QuotaConfig(max_concurrency=3))
    lock = threading.Lock()
    live = {"now": 0, "peak": 0}

    def worker() -> None:
        lease, _wait = quota.acquire(0, quota.state.config)
        with lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.01)
        with lock:
            live["now"] -= 1
        quota.release(lease)

    threads = [threading.Thread(target=worker) for _ in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert live["peak"] == 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": ""}, "non-empty"),
        ({"name": "p"}, "limits nothing"),
        ({"name": "p", "requests_per_minute": 0}, "positive"),
        ({"name": "p", "max_concurrency": 2.5}, "integer"),
    ],
)
def test_provider_limit_validation(kwargs, match):
    with pytest.raises(PlanError, match=match):
        bt.ml.ProviderLimit(**kwargs)


def test_engines_built_separately_share_one_quota_by_name(monkeypatch):
    """Two engines (two "workers") naming one quota stay under its concurrency together."""
    import batcher.ml.serving.http as http_mod

    lock = threading.Lock()
    live = {"now": 0, "peak": 0}

    def fake(url, body, **kw):
        with lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.01)
        with lock:
            live["now"] -= 1
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(http_mod, "post_json", fake)
    quota = bt.ml.ProviderLimit("test-shared-by-name", max_concurrency=3)
    try:
        engines = [
            bt.ml.http_engine("http://x/v1", "m", concurrency=8, shared_limit=quota)()
            for _ in range(2)
        ]
        results: list = []
        threads = [
            threading.Thread(target=lambda e=e: results.append(e(["p"] * 16))) for e in engines
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for engine in engines:
            engine.close()
    finally:
        quota.close()
    assert results == [["ok"] * 16] * 2
    assert live["peak"] == 3  # 2 workers x 8 slots each, held to the quota's 3


def test_a_custom_client_can_hold_a_lease():
    quota = bt.ml.ProviderLimit("test-custom-client", requests_per_minute=6000)
    try:
        with quota.lease(estimated_tokens=10) as waited:
            assert waited == 0.0
    finally:
        quota.close()
