"""The N-stage overlap loop's scheduling algebra, driven deterministically without Ray.

`tests/integration/test_stream_inference.py` runs this pipeline for real, and cannot run in
CI: there is no Ray there, and on a loaded box a raylet takes minutes to come up or does not.
So the properties that a distributed scheduler gets *silently* wrong had no automated cover at
all — and they are the worst class of bug in this repo, because a lost row and a duplicated
row both come back as a plausible number rather than an error.

This drives `run_streamed` against stand-in actors and a stand-in `ray`, so every property
below is checked in milliseconds with no cluster:

* every input row reaches the last stage **exactly once** — through a three-stage chain, and
  through a stage that fans one morsel out into several;
* the same holds after a **middle stage is preempted**, which is the case the "hold a morsel
  until its subtree is done" rule exists for: its input is still on the stage below, so the
  replay is local. A replay that minted *new* morsel names would leave the results it had
  already recorded behind as duplicates, which is exactly the failure no exception reports;
* the same holds after a **producer is preempted**, where the whole partition replays;
* an actor never holds more published-but-unreleased morsels than the credit window allows;
* a stage that loses every actor and cannot replace one **raises** rather than returning a
  partial answer that looks complete.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.transfer import ShuffleTicket

pytestmark = pytest.mark.unit


# --- a stand-in for Ray ----------------------------------------------------------------


class _Ref:
    """One completed call. `get` returns its value or re-raises its error.

    The call runs when it is *issued*, not when it is waited on, because that is what
    `.remote()` does: it starts the work. The loop also issues `release` fire-and-forget and
    never waits on it, so a ref that only ran on `wait` would leave every morsel published
    forever — and the credit-window bound below would read as broken when it is not.
    """

    __slots__ = ("error", "value")

    def __init__(self, bound, args) -> None:
        self.value = None
        self.error: BaseException | None = None
        try:
            self.value = bound(*args)
        except BaseException as exc:
            self.error = exc


class _Method:
    def __init__(self, bound) -> None:
        self._bound = bound

    def remote(self, *args):
        return _Ref(self._bound, args)


class _Handle:
    """An actor handle: `handle.method.remote(...)` returns a `_Ref` the loop later waits on."""

    def __init__(self, impl) -> None:
        object.__setattr__(self, "_impl", impl)

    def __getattr__(self, name: str) -> _Method:
        return _Method(getattr(object.__getattribute__(self, "_impl"), name))

    @property
    def impl(self):
        return object.__getattribute__(self, "_impl")


class _FakeRay:
    """`wait` completes the *oldest* outstanding call, so a run is reproducible."""

    class exceptions:
        class RayActorError(Exception):
            pass

        class RayTaskError(Exception):
            pass

    @staticmethod
    def wait(refs, num_returns=1):
        return [refs[0]], refs[1:]

    @staticmethod
    def get(ref):
        if isinstance(ref, list):
            return [_FakeRay.get(r) for r in ref]
        if ref.error is not None:
            raise ref.error
        return ref.value


class _Preempted(_FakeRay.exceptions.RayActorError):
    """What a lost actor raises here — a `RayActorError`, which is what the loop recovers from."""


# --- stand-in stages -------------------------------------------------------------------


class _Flight:
    """The published morsels of every actor, keyed by `(addr, ticket)` — a stand-in wire."""

    def __init__(self) -> None:
        self.store: dict[tuple, list] = {}
        self.peak: dict[str, int] = {}

    def publish(self, addr: str, ticket, rows: list) -> None:
        self.store[(addr, str(ticket))] = rows
        held = sum(1 for a, _t in self.store if a == addr)
        self.peak[addr] = max(self.peak.get(addr, 0), held)

    def fetch(self, addr: str, ticket) -> list:
        return self.store[(addr, str(ticket))]

    def release(self, addr: str, ticket) -> None:
        self.store.pop((addr, str(ticket)), None)


class _Producer:
    """Publishes one morsel per row of its partition, so a small run has many morsels."""

    def __init__(self, flight: _Flight, addr: str) -> None:
        self._flight = flight
        self._addr = addr
        self._rows: list = []

    def open(self, partition) -> str:
        self._rows = list(partition["rows"])
        return self._addr

    def publish_next(self, ticket) -> bool:
        if not self._rows:
            return False
        self._flight.publish(self._addr, ticket, [self._rows.pop(0)])
        return True

    def release(self, ticket) -> None:
        self._flight.release(self._addr, ticket)


class _Relay:
    """Maps a morsel and republishes it, optionally fanning it out into `fanout` morsels."""

    def __init__(self, flight: _Flight, addr: str, *, fanout: int = 1, fail_first: int = 0) -> None:
        self._flight = flight
        self._addr = addr
        self._fanout = fanout
        self._fail_first = fail_first

    def addr(self) -> str:
        return self._addr

    def node_host(self) -> str:
        return self._addr

    def gpu_stats(self):
        return None

    def consume(self, up_addr, up_ticket, plan_id, stage_id, morsel) -> int:
        if self._fail_first > 0:
            self._fail_first -= 1
            raise _Preempted("simulated relay preemption")
        rows = self._flight.fetch(up_addr, up_ticket)
        published = 0
        for i in range(self._fanout):
            # A fan-out marks each piece, so a duplicated morsel is distinguishable from a
            # legitimate second piece of the same input.
            payload = [(value, i) for value in rows]
            self._flight.publish(self._addr, ShuffleTicket(plan_id, stage_id, morsel, i), payload)
            published += 1
        return published

    def release(self, ticket) -> None:
        self._flight.release(self._addr, ticket)


class _Consumer:
    """The last stage: returns the morsel's rows to the driver."""

    def __init__(self, flight: _Flight, *, fail_first: int = 0) -> None:
        self._flight = flight
        self._fail_first = fail_first

    def node_host(self) -> str:
        return "consumer"

    def gpu_stats(self):
        return None

    def run_split(self, addr, ticket) -> list:
        if self._fail_first > 0:
            self._fail_first -= 1
            raise _Preempted("simulated consumer preemption")
        return list(self._flight.fetch(addr, ticket))


class _DeadProducer:
    """A producer whose partition open always fails — a preempted node."""

    def open(self, partition):
        raise _Preempted("simulated producer preemption")

    def publish_next(self, ticket):
        return False

    def release(self, ticket):
        return None


@pytest.fixture
def loop(monkeypatch):
    """`run_streamed` with `ray` and the host probe replaced by stand-ins."""
    from batcher.dist.streaming.pipeline import schedule

    monkeypatch.setattr(schedule, "ray", _FakeRay)
    # The loop asks the real `ray` module which exception types mean "lost actor". Here the
    # losses are simulated, so the answer has to be the simulated type — otherwise a
    # preemption escapes as an ordinary error and the recovery paths are never entered.
    monkeypatch.setattr(schedule, "_worker_loss_errors", lambda: (_Preempted,))
    monkeypatch.setattr(schedule, "_probe", lambda pool: {})
    monkeypatch.setattr(
        schedule,
        "_probe_addrs",
        lambda pool: {a: a.impl.addr() for a in pool if hasattr(a.impl, "addr")},
    )
    return schedule.run_streamed


def _partitions(counts: list[int]) -> list[dict]:
    """One partition per entry, holding `count` distinct row values."""
    out, start = [], 0
    for count in counts:
        out.append({"rows": list(range(start, start + count))})
        start += count
    return out


def _values(results: dict) -> list:
    return sorted(v for out in results.values() if out for v in out)


# --- the properties ---------------------------------------------------------------------


def test_every_row_reaches_the_last_stage_exactly_once(loop):
    flight = _Flight()
    producers = [_Handle(_Producer(flight, f"p{i}")) for i in range(2)]
    relays = [_Handle(_Relay(flight, f"r{i}")) for i in range(2)]
    consumers = [_Handle(_Consumer(flight)) for _ in range(2)]
    results = loop([producers, relays, consumers], _partitions([4, 3]), 1, 2)
    assert _values(results) == [(v, 0) for v in range(7)]


def test_a_stage_that_fans_a_morsel_out_keeps_every_piece(loop):
    flight = _Flight()
    producers = [_Handle(_Producer(flight, "p0"))]
    relays = [_Handle(_Relay(flight, "r0", fanout=3))]
    consumers = [_Handle(_Consumer(flight))]
    results = loop([producers, relays, consumers], _partitions([3]), 1, 2)
    assert _values(results) == sorted((v, i) for v in range(3) for i in range(3))


def test_a_preempted_relay_replays_from_the_stage_below_without_duplicating(loop):
    """The property the hold-until-settled rule buys, and the one a wrong fix would break.

    Half the morsels fail on their first pass, so the run mixes recovered and un-recovered
    work — a replay that minted fresh morsel names would leave the already-recorded results
    behind as duplicates, and the count below would be too high rather than an error.
    """
    flight = _Flight()
    producers = [_Handle(_Producer(flight, "p0"))]
    relays = [_Handle(_Relay(flight, "r0", fail_first=3))]
    consumers = [_Handle(_Consumer(flight))]
    spawned: list = []

    def spawn_relay():
        handle = _Handle(_Relay(flight, f"r{len(spawned) + 1}"))
        spawned.append(handle)
        return handle

    results = loop(
        [producers, relays, consumers],
        _partitions([6]),
        1,
        2,
        spawn=[None, spawn_relay, None],
    )
    assert _values(results) == [(v, 0) for v in range(6)]
    assert spawned, "a lost relay must be replaced, not merely dropped"


def test_a_preempted_consumer_redispatches_its_morsel(loop):
    flight = _Flight()
    producers = [_Handle(_Producer(flight, "p0"))]
    relays = [_Handle(_Relay(flight, "r0"))]
    consumers = [_Handle(_Consumer(flight, fail_first=2))]
    results = loop(
        [producers, relays, consumers],
        _partitions([5]),
        1,
        2,
        spawn=[None, None, lambda: _Handle(_Consumer(flight))],
    )
    assert _values(results) == [(v, 0) for v in range(5)]


def test_a_preempted_producer_replays_its_whole_partition(loop):
    flight = _Flight()
    dead = _Handle(_DeadProducer())
    relays = [_Handle(_Relay(flight, "r0"))]
    consumers = [_Handle(_Consumer(flight))]
    results = loop(
        [[dead], relays, consumers],
        _partitions([4]),
        1,
        2,
        spawn=[lambda: _Handle(_Producer(flight, "p1")), None, None],
    )
    assert _values(results) == [(v, 0) for v in range(4)]


def test_no_actor_ever_holds_more_than_the_credit_window(loop):
    """The bound the pipeline advertises, checked at the relay hop and not only at the first.

    A fan-out of 2 with a window of 2 is the shape that catches a relay left unwindowed: it
    publishes twice per call, so an unbounded relay races ahead of its consumer immediately.
    """
    flight = _Flight()
    credits = 2
    producers = [_Handle(_Producer(flight, "p0"))]
    relays = [_Handle(_Relay(flight, "r0", fanout=2))]
    consumers = [_Handle(_Consumer(flight))]
    loop([producers, relays, consumers], _partitions([12]), 1, credits)
    assert flight.peak["p0"] <= credits
    # A single `consume` publishes its whole fan-out, so the relay's peak is bounded by the
    # window plus what one call adds — never by the partition size, which is the point.
    assert flight.peak["r0"] <= credits + 2


def test_a_stage_that_loses_every_actor_raises_instead_of_returning_partial_rows(loop):
    """Returning would hand back an answer that looks complete and is missing rows."""
    from batcher._internal.errors import ResourceError

    flight = _Flight()
    producers = [_Handle(_Producer(flight, "p0"))]
    relays = [_Handle(_Relay(flight, "r0"))]
    consumers = [_Handle(_Consumer(flight, fail_first=99))]
    with pytest.raises((ResourceError, _Preempted)):
        loop([producers, relays, consumers], _partitions([4]), 1, 2)


def test_two_stages_still_work_as_the_degenerate_case(loop):
    """The N-stage loop is the only loop, so the pipeline it replaced has to be an N of 2."""
    flight = _Flight()
    producers = [_Handle(_Producer(flight, "p0"))]
    consumers = [_Handle(_Consumer(flight))]
    results = loop([producers, consumers], _partitions([5]), 1, 2)
    assert _values(results) == list(range(5))


# --- per-stage autoscaling ---------------------------------------------------------------


def test_a_backlogged_stage_grows_toward_its_ceiling(loop):
    """`concurrency=(min, max)` is documented as autoscaling, so a starved stage must widen.

    The pipeline's slowest stage decides the whole query: a consumer pool that cannot keep
    up leaves morsels queued while the cluster has room for more actors. Pool sizes were
    fixed at construction, so a stage that fell behind stayed behind for the entire run.
    """
    flight = _Flight()
    spawned: list = []

    def _spawn_consumer():
        handle = _Handle(_Consumer(flight))
        spawned.append(handle)
        return handle

    producers = [_Handle(_Producer(flight, f"p{i}")) for i in range(3)]
    consumers = [_Handle(_Consumer(flight))]
    pools = [producers, consumers]
    results = loop(
        pools,
        _partitions([4, 4, 4]),
        1,
        3,
        spawn=[None, _spawn_consumer],
        ceilings=[3, 4],
    )
    assert _values(results) == list(range(12))
    assert spawned, "a stage with queued morsels and headroom should have grown"
    assert len(pools[1]) <= 4, "growth must stop at the ceiling"


def test_growth_never_exceeds_the_ceiling(loop):
    """The ceiling is the `max` the user wrote, and nothing may push past it."""
    flight = _Flight()
    producers = [_Handle(_Producer(flight, f"p{i}")) for i in range(4)]
    consumers = [_Handle(_Consumer(flight))]
    pools = [producers, consumers]
    loop(
        pools,
        _partitions([6, 6, 6, 6]),
        1,
        4,
        spawn=[None, lambda: _Handle(_Consumer(flight))],
        ceilings=[4, 2],
    )
    assert len(pools[1]) == 2


def test_a_fixed_size_stage_is_left_alone(loop):
    """A plain-int or absent `concurrency` must schedule exactly as it did before.

    `ceilings` equal to the pool it was given is how a stage says "do not scale"; if that
    read as headroom, every existing pipeline would silently start spawning actors.
    """
    flight = _Flight()
    spawned: list = []

    def _spawn_consumer():
        handle = _Handle(_Consumer(flight))
        spawned.append(handle)
        return handle

    producers = [_Handle(_Producer(flight, f"p{i}")) for i in range(2)]
    consumers = [_Handle(_Consumer(flight))]
    results = loop(
        [producers, consumers],
        _partitions([5, 5]),
        1,
        2,
        spawn=[None, _spawn_consumer],
        ceilings=[2, 1],
    )
    assert _values(results) == list(range(10))
    assert spawned == []


def test_omitting_ceilings_keeps_every_stage_fixed(loop):
    """The default is today's behaviour: no `ceilings` argument, no growth anywhere."""
    flight = _Flight()
    spawned: list = []

    def _spawn_relay():
        handle = _Handle(_Relay(flight, f"extra{len(spawned)}"))
        spawned.append(handle)
        return handle

    producers = [_Handle(_Producer(flight, "p0"))]
    relays = [_Handle(_Relay(flight, "r0"))]
    consumers = [_Handle(_Consumer(flight))]
    results = loop(
        [producers, relays, consumers],
        _partitions([6]),
        1,
        2,
        spawn=[None, _spawn_relay, None],
    )
    assert _values(results) == [(v, 0) for v in range(6)]
    assert spawned == []


def test_a_grown_stage_still_delivers_every_row_exactly_once(loop):
    """Growth is a scheduling change, so the multiset of rows must be untouched by it.

    Run the same input through a fixed pool and a growing one and compare: a new actor that
    replayed or dropped a morsel would show up here and nowhere else.
    """
    flight_fixed = _Flight()
    fixed = loop(
        [
            [_Handle(_Producer(flight_fixed, "p0"))],
            [_Handle(_Relay(flight_fixed, "r0"))],
            [_Handle(_Consumer(flight_fixed))],
        ],
        _partitions([9]),
        1,
        2,
    )
    flight_grown = _Flight()
    counter = [0]

    def _spawn_consumer():
        counter[0] += 1
        return _Handle(_Consumer(flight_grown))

    grown = loop(
        [
            [_Handle(_Producer(flight_grown, "p0"))],
            [_Handle(_Relay(flight_grown, "r0"))],
            [_Handle(_Consumer(flight_grown))],
        ],
        _partitions([9]),
        1,
        2,
        spawn=[None, None, _spawn_consumer],
        ceilings=[1, 1, 4],
    )
    assert _values(grown) == _values(fixed)
