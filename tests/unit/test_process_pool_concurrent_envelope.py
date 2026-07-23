"""The shared memory envelope must not shrink under a pipeline already running in it.

`process_pool` is one buffer pool per process, deliberately shared so concurrent queries
and the transfer layer account against a single budget. Reconciling it to each caller's
budget is right *between* queries — an autoscaler or a differently-sized query should be
able to resize the envelope. It is wrong while another pipeline holds reservations inside
it: the smaller cap applies to work Carbonite already admitted under the larger one.
"""

from __future__ import annotations

import pytest

import batcher.carbonite.memory.pool as poolmod


@pytest.fixture(autouse=True)
def _fresh_pool(monkeypatch):
    """Each test gets its own process pool, so ordering cannot leak an envelope."""
    monkeypatch.setattr(poolmod, "_process_pool", None, raising=False)
    yield
    monkeypatch.setattr(poolmod, "_process_pool", None, raising=False)


def test_a_concurrent_pipeline_cannot_shrink_a_reserved_envelope():
    """A cheap query's budget must not retune the envelope an expensive one is inside."""
    pool = poolmod.process_pool(8_000_000)
    with pool.reserve(4_000_000) as ok:
        assert ok, "the reservation under test did not get admitted"
        poolmod.process_pool(1_000_000)  # a second, cheaper pipeline arrives
        assert pool.limit == 8_000_000, "a concurrent pipeline shrank the live envelope"


def test_the_envelope_still_shrinks_once_idle():
    """The deferral must not become a one-way ratchet — an idle pool still resizes down."""
    pool = poolmod.process_pool(8_000_000)
    with pool.reserve(4_000_000):
        poolmod.process_pool(1_000_000)  # deferred
    poolmod.process_pool(1_000_000)  # pool now idle → applies

    assert pool.limit == 1_000_000


def test_growth_applies_immediately_even_under_reservations():
    """Capacity the autoscaler just added must reach a running query without waiting."""
    pool = poolmod.process_pool(1_000_000)
    with pool.reserve(500_000):
        poolmod.process_pool(9_000_000)
        assert pool.limit == 9_000_000
