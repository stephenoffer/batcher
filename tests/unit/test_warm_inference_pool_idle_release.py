"""A finished inference query gives its devices back; a running one keeps them.

The session-warm inference pool holds one actor per device so a model loads once per
session. Held for the whole session that is a deadlock rather than an optimization on any
cluster with a second tenant: measured with a Batcher arm and a Ray Data arm in one
benchmark process on eight T4s, Batcher's pool finished its query, kept all eight devices,
and Ray Data's own eight-actor pool then sat `pending` and never placed. Neither engine
reported anything -- the second one simply never ran.

Both halves need pinning and the second is the one that decays quietly. A release that
fires while a stage is still running would kill the actors underneath it, and the query
would fail rather than slow down.

No cluster: `_shutdown_pools` is the single teardown point, so recording it accounts for
every actor the timer would kill.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors import map as M

pytestmark = pytest.mark.unit


@pytest.fixture
def warm(monkeypatch):
    """A populated `_SESSION_POOLS`, a clean timer, and every teardown recorded."""
    torn: list = []
    monkeypatch.setattr(M, "_SESSION_POOLS", {("sig", ()): ["actor"]})
    monkeypatch.setattr(M, "_INFER_IDLE_TIMER", [])
    monkeypatch.setattr(M, "_INFER_IN_USE", [])
    monkeypatch.setattr(M, "_shutdown_pools", lambda registry: torn.append(registry))
    return torn


def test_an_idle_pool_is_released(warm, monkeypatch):
    with M._inference_pool_in_use():
        pass
    # The timer is armed rather than fired; run its callback directly so the test does not
    # sleep, which is what makes the *negative* case below meaningful rather than flaky.
    assert M._INFER_IDLE_TIMER, "leaving a stage must arm the release"
    M._release_inference_pools_if_idle()
    assert warm == [M._SESSION_POOLS]


def test_it_does_not_fire_while_a_stage_is_running(warm):
    # The positive control for the test above: the same callback, with the lease held.
    with M._inference_pool_in_use():
        M._release_inference_pools_if_idle()
        assert warm == [], "a running stage's own actors must not be killed underneath it"
    M._release_inference_pools_if_idle()
    assert warm == [M._SESSION_POOLS]


def test_a_second_stage_cancels_the_pending_release(warm):
    with M._inference_pool_in_use():
        pass
    armed = list(M._INFER_IDLE_TIMER)
    assert armed
    with M._inference_pool_in_use():
        assert not M._INFER_IDLE_TIMER, "entering a stage must cancel the pending release"
    assert all(not t.is_alive() for t in armed)


def test_zero_disables_the_release(warm, monkeypatch):
    import dataclasses

    from batcher.config import active_config, set_config

    base = active_config()
    set_config(
        base.replace(distributed=dataclasses.replace(base.distributed, warm_inference_idle_s=0.0))
    )
    try:
        with M._inference_pool_in_use():
            pass
        assert not M._INFER_IDLE_TIMER, "0 must mean whole-session residency"
    finally:
        set_config(base)


def test_release_inference_pools_cancels_the_timer(warm):
    with M._inference_pool_in_use():
        pass
    assert M._INFER_IDLE_TIMER
    M.release_inference_pools()
    assert not M._INFER_IDLE_TIMER
