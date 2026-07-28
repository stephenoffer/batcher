"""The one large Carbonite footprint the buffer pool cannot see: published shuffle output.

A mapper's output bucket is never *reserved*. It is produced, handed to the local Flight
store, and held until a reducer fetches it, so no reservation covers it and the pool's
`used` reads zero while the process holds the node's whole share of the shuffle. That is
why `PressureMonitor` falls back to process RSS rather than trusting the pool on a
shuffle-heavy worker.

`shuffle_store_cap` is Carbonite deciding a bound for it, which the transport enforces by
spilling its largest buckets to local disk and reading them back on fetch. These tests pin
the *decision*: that it scales with the envelope, that it never exceeds what the query as a
whole may use, and that an unknown envelope stays unbounded rather than guessing a number
that would spill a query which fits.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.policies import shuffle_store_cap
from batcher.config import Config, ExecutionConfig, MemoryConfig

pytestmark = pytest.mark.unit


def _cfg(hard_limit: float = 0.8, morsel_bytes: int = 1 << 20) -> Config:
    return Config().replace(
        memory=MemoryConfig(hard_limit=hard_limit),
        execution=ExecutionConfig(morsel_bytes=morsel_bytes),
    )


def test_the_cap_scales_with_the_envelope() -> None:
    """Twice the memory is twice the shuffle a worker may hold before it spills."""
    small = shuffle_store_cap(_cfg(), 4 << 30)
    large = shuffle_store_cap(_cfg(), 8 << 30)
    assert large == small * 2


def test_the_cap_is_a_minority_of_the_envelope() -> None:
    """The shuffle store shares the machine with the operators it feeds."""
    envelope = 8 << 30
    assert shuffle_store_cap(_cfg(), envelope) < envelope // 2


def test_the_hard_limit_binds_when_it_is_tighter_than_the_fraction() -> None:
    """The store is not entitled to memory the whole query may not use.

    A deployment that caps the query at 10% of the machine must not then let one
    un-reserved shuffle buffer take 25% of it.
    """
    envelope = 8 << 30
    tight = shuffle_store_cap(_cfg(hard_limit=0.10), envelope)
    assert tight == int(envelope * 0.10)
    assert tight < shuffle_store_cap(_cfg(hard_limit=0.9), envelope)


def test_an_unknown_envelope_stays_unbounded() -> None:
    """`0` means "no cap" downstream, which is the historical behaviour.

    Guessing a number here would spill a query that fits, and a spill that buys nothing is
    strictly a regression. Not knowing the envelope is a reason to keep the old behaviour,
    not to invent a bound.
    """
    assert shuffle_store_cap(_cfg(), 0) == 0
    assert shuffle_store_cap(_cfg(), -1) == 0


def test_a_tiny_envelope_still_admits_one_morsel() -> None:
    """A cap below one batch would spill every bucket the instant it is published.

    The store's unit of work is a `RecordBatch`; a cap under that size means the very first
    publish exceeds it, so every fetch pays a disk round trip and nothing is ever served
    from memory. The floor makes the bound useless rather than pathological.
    """
    morsel = 4 << 20
    assert shuffle_store_cap(_cfg(morsel_bytes=morsel), 1 << 20) >= morsel


def test_the_default_envelope_is_the_machine() -> None:
    """Called with no envelope the policy reads the machine, and answers something real."""
    assert shuffle_store_cap(_cfg()) > 0


def test_the_engine_accepts_the_cap_the_policy_produces() -> None:
    """The decision must survive the FFI boundary it exists to cross.

    Carbonite deciding a bound the transport cannot be told is not a bound. This is the
    one assertion that fails if the parameter is dropped from either side of the signature.
    """
    from batcher._internal.native import engine

    nat = engine()
    cap = shuffle_store_cap(_cfg(), 2 << 30)
    nat.set_flight_transport_config(0, 0, 0, None, cap)
    # And the historical 4-argument call must keep working, since a worker built against
    # an older driver still makes it.
    nat.set_flight_transport_config(0)
