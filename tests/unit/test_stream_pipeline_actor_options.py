"""A streaming stage's actor options must be applied in exactly one `.options(...)` call.

Ray's `.options(...)` does not return the actor class. It returns a small wrapper exposing
only `remote` and `bind`, so calling `.options(...)` on the result raises `AttributeError`.
The pipeline builder applied two fragments — the accelerator request when it built the pool,
the shipped `runtime_env` when it spawned an actor — and the second call landed on the
wrapper the first had produced.

It needed *both* fragments to be non-empty to fire, which is why nothing caught it: the
`runtime_env` fragment is empty whenever Batcher started Ray itself (the whole test suite and
the default path), and the accelerator fragment is empty for a CPU-only chain. The
intersection is a GPU stage on a cluster the user attached to themselves — that is, every
stage-overlapped CPU->GPU inference pipeline on a real cluster, which is the AI moat's own
shape. It failed loudly there and nowhere else.

The stubs below mimic Ray's contract exactly on the one point that matters: `options()`
returns something that has no `options()`.
"""

from __future__ import annotations

import pytest

from batcher.dist.streaming.pipeline import driver


class _Bound:
    """What Ray's `.options(...)` returns: `remote` and nothing else."""

    def __init__(self, cls: _StubActorClass, options: dict) -> None:
        self._cls = cls
        self._options = options

    def remote(self, *args):
        self._cls.spawned.append((self._options, args))
        return f"actor{len(self._cls.spawned)}"


class _StubActorClass:
    """A Ray-remote actor class, recording every `.options(...)` fragment it is given."""

    def __init__(self) -> None:
        self.option_calls: list[dict] = []
        self.spawned: list[tuple[dict, tuple]] = []

    def options(self, **kwargs) -> _Bound:
        self.option_calls.append(kwargs)
        return _Bound(self, kwargs)

    def remote(self, *args):
        self.spawned.append(({}, args))
        return f"actor{len(self.spawned)}"


class _Stage:
    """The fields `_build_pools` reads off a resource stage."""

    def __init__(self, num_gpus: float) -> None:
        self.num_gpus = num_gpus
        self.accelerator_type = None
        self.concurrency = None
        self.sub_plan = f"plan-gpu{num_gpus}"


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Point the builder at stub actor classes and a non-empty shipped `runtime_env`.

    Returns the `(producer, relay, terminal)` stubs so a test can read what each was asked
    for. `consumer_batch_rows` is stubbed too: it inspects a real sub-plan, and these are
    strings.
    """
    from batcher.dist.executors import map as dist_map
    from batcher.dist.streaming import relay as relay_mod

    producer, relay, terminal = _StubActorClass(), _StubActorClass(), _StubActorClass()
    monkeypatch.setattr(dist_map, "_MapActor", terminal, raising=False)
    monkeypatch.setattr(relay_mod, "RelayActor", relay, raising=False)
    monkeypatch.setattr(driver, "ProducerActor", producer, raising=False)
    monkeypatch.setattr(driver, "consumer_batch_rows", lambda _plan: 4096)
    monkeypatch.setattr(driver, "_shipping_options", lambda: {"runtime_env": {"env_vars": {}}})
    return producer, relay, terminal


def test_a_gpu_stage_with_a_shipped_runtime_env_spawns(stub_pipeline):
    """The regression: two fragments, one `.options(...)` call, an actor at the end of it."""
    _producer, _relay, terminal = stub_pipeline
    stages = [_Stage(0.0), _Stage(1.0)]
    pools, spawns, ceilings = driver._build_pools(stages, [(1, 1), (2, 2)], credits=4)

    assert [len(p) for p in pools] == [1, 2]
    assert ceilings == [1, 2]
    assert len(spawns) == 2
    # One call per spawned actor, never one per actor plus one for the pool.
    assert len(terminal.option_calls) == 2
    for opts in terminal.option_calls:
        assert opts["num_gpus"] == 1.0
        assert opts["runtime_env"] == {"env_vars": {}}


def test_a_host_stage_still_ships_its_runtime_env(stub_pipeline):
    """A stage that asks for no accelerator must keep the `runtime_env` fragment."""
    _producer, relay, _terminal = stub_pipeline
    stages = [_Stage(0.0), _Stage(0.0), _Stage(1.0)]
    driver._build_pools(stages, [(1, 1), (1, 1), (1, 1)], credits=4)

    assert len(relay.option_calls) == 1
    assert relay.option_calls[0] == {"runtime_env": {"env_vars": {}}}
    assert "num_gpus" not in relay.option_calls[0]


def test_no_options_call_when_there_is_nothing_to_apply(monkeypatch, stub_pipeline):
    """With no accelerator and no shipped env, the raw class is used — no wrapper at all."""
    _producer, relay, _terminal = stub_pipeline
    monkeypatch.setattr(driver, "_shipping_options", dict)
    driver._build_pools([_Stage(0.0), _Stage(0.0), _Stage(1.0)], [(1, 1)] * 3, credits=4)

    assert relay.option_calls == []
    assert len(relay.spawned) == 1


# --------------------------------------------------------------------------- #
# Stage 0's `concurrency` reaches the partitioning, not just the pool
# --------------------------------------------------------------------------- #
def test_the_host_stage_honours_an_explicit_concurrency():
    """`map_batches(..., concurrency=N)` on the host stage was silently discarded.

    Stage 0 reads partitions, so its pool size and the partition count are one number, and
    that number came from `workers` -- the relational fleet width. On a 192-CPU cluster
    feeding eight devices, a caller asking for 48 decode actors got 16, and the argument the
    public API documents as controlling pool size did nothing on this path while doing exactly
    what it says on the non-streamed one.
    """
    assert driver.producer_fanout(_Stage(0.0), workers=16) == 16, "no spec: the fleet width"

    explicit = _Stage(0.0)
    explicit.concurrency = 48
    assert driver.producer_fanout(explicit, workers=16) == 48

    smaller = _Stage(0.0)
    smaller.concurrency = 4
    assert driver.producer_fanout(smaller, workers=16) == 4, "a smaller ask is also an ask"

    ranged = _Stage(0.0)
    ranged.concurrency = (2, 32)
    assert driver.producer_fanout(ranged, workers=16) == 32, "a range needs room to grow into"


def test_an_explicit_host_concurrency_reaches_the_pool_bounds(stub_pipeline):
    """The number has to survive into `_pool_bounds`, which is what sizes the pool."""
    stages = [_Stage(0.0), _Stage(1.0)]
    stages[0].concurrency = 48
    # 48 partitions available, so nothing clamps the ask back down.
    bounds = driver._pool_bounds(stages, workers=16, num_partitions=48)
    assert bounds[0] == (48, 48)


def test_the_partition_count_still_caps_the_host_pool(stub_pipeline):
    """An ask larger than the input can supply is still bounded by the partitions."""
    stages = [_Stage(0.0), _Stage(1.0)]
    stages[0].concurrency = 48
    assert driver._pool_bounds(stages, workers=16, num_partitions=6)[0] == (6, 6)
