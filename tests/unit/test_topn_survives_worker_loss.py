"""A lost worker must not take a distributed top-N with it.

`execute_topn_flight` gathered its per-worker results with a bare `ray.get`, so a worker that
died mid-fold raised `ActorDiedError` and lost the query. Measured on a 72M-row
`ORDER BY ... LIMIT` against a real cluster: killing a `_FlightWorker` **mid-method** lost the
query 5 times out of 5, while the same fault against an *idle* fleet actor was survived 9 out
of 9. The fleet reforms fine; the in-flight partition had nowhere to go.

These tests drive `topn_partition` directly with fake actors rather than through a cluster,
because the behaviour under test is the retry policy and CI has no Ray. That is a real limit:
they prove the policy recomputes, re-raises and gives up correctly, and they do NOT prove the
distributed path is wired to it -- only the recorded cluster run in
`benchmarks/BENCHMARK_RESULTS.md` does that (`.claude/rules/testing.md`, and CI installs no
Ray).

The positive control matters as much as the recovery cases: `test_a_clean_gather_never_retries`
fails if the helper degenerates into "always recompute", which would pass every recovery test
here while quietly doubling the work of every healthy query.
"""

from __future__ import annotations

import sys
import types

import pytest

from batcher._internal.errors import ResourceError, RetryableShuffleError


class _Batch:
    """The one thing the fold asks of a batch: how many rows it has."""

    def __init__(self, rows: int) -> None:
        self.num_rows = rows


class _Ref:
    """A stand-in Ray ObjectRef: either batches to hand back, or an exception to raise."""

    def __init__(self, payload) -> None:
        self.payload = payload


class _Method:
    def __init__(self, actor: _FakeActor) -> None:
        self.actor = actor

    def remote(self, _local_ir, _part):
        self.actor.calls += 1
        return _Ref(self.actor.payload)


class _FakeActor:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls = 0

    @property
    def local_topn(self) -> _Method:
        return _Method(self)


@pytest.fixture
def topn(monkeypatch):
    """`topn_partition` with a fake `ray` whose `get` replays whatever the ref carries."""
    fake = types.ModuleType("ray")

    def _get(ref):
        if isinstance(ref.payload, BaseException):
            raise ref.payload
        return ref.payload

    fake.get = _get
    monkeypatch.setitem(sys.modules, "ray", fake)

    from batcher.dist.executors.ray_runtime import topn_partition

    return topn_partition


def _loss() -> BaseException:
    """A failure the classifier reads as a lost worker rather than a deterministic bug."""
    return RetryableShuffleError("peer unreachable")


def test_a_clean_gather_never_retries(topn):
    """The positive control: a healthy fold must not recompute anything.

    Without this, a helper that always recomputed would pass every other test in this file
    while doubling the work of every query that never lost a worker.
    """
    actors = [_FakeActor([_Batch(3)]) for _ in range(4)]
    refs = [_Ref([_Batch(3)]) for _ in range(4)]
    dead: set[int] = set()

    got = topn(refs, 1, actors, "ir", [None] * 4, dead)

    assert [b.num_rows for b in got] == [3]
    assert dead == set()
    assert [a.calls for a in actors] == [0, 0, 0, 0], "a healthy gather resubmitted work"


def test_empty_batches_are_dropped(topn):
    """The fold concatenates, so zero-row batches are cost without content."""
    refs = [_Ref([_Batch(0), _Batch(5), _Batch(0)])]
    got = topn(refs, 0, [_FakeActor([])], "ir", [None], set())
    assert [b.num_rows for b in got] == [5]


def test_a_lost_worker_is_recomputed_on_a_survivor(topn):
    """The defect this file exists for: the query survives, with the partition's rows."""
    actors = [_FakeActor(_loss()), _FakeActor([_Batch(7)]), _FakeActor([_Batch(7)])]
    refs = [_Ref(_loss()), _Ref([_Batch(1)]), _Ref([_Batch(1)])]
    dead: set[int] = set()

    got = topn(refs, 0, actors, "ir", [None] * 3, dead)

    assert [b.num_rows for b in got] == [7], "partition 0 did not come back"
    assert 0 in dead, "the worker that died was not recorded as dead"
    assert sum(a.calls for a in actors) == 1, "recompute should stop at the first survivor"


def test_a_deterministic_bug_is_raised_not_retried(topn):
    """A UDF bug fails identically everywhere, so retrying it buries the real traceback.

    This is the failure mode `_faults` documents: absorb a deterministic error and every
    round blames another host until the query reports a resource error for a Python bug.
    """
    actors = [_FakeActor([_Batch(1)]) for _ in range(3)]
    refs = [_Ref(ZeroDivisionError("bug in a projection")), _Ref([]), _Ref([])]

    with pytest.raises(ZeroDivisionError):
        topn(refs, 0, actors, "ir", [None] * 3, set())

    assert sum(a.calls for a in actors) == 0, "a deterministic bug was retried"


def test_it_skips_workers_already_known_dead(topn):
    """One fold must not re-try a worker an earlier partition already buried."""
    dying, survivor = _FakeActor(_loss()), _FakeActor([_Batch(2)])
    actors = [_FakeActor(_loss()), dying, survivor]
    refs = [_Ref(_loss()), _Ref([]), _Ref([])]
    dead = {1}  # worker 1 was lost collecting an earlier partition

    got = topn(refs, 0, actors, "ir", [None] * 3, dead)

    assert [b.num_rows for b in got] == [2]
    assert dying.calls == 0, "a worker already known dead was asked to recompute"


def test_losing_every_worker_reports_a_resource_error(topn, monkeypatch):
    """When nothing survives, the error names the fleet-wide loss rather than the partition."""
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.policies._topn.recovery_policy",
        lambda: types.SimpleNamespace(max_attempts=2, backoff_base_s=0.0),
    )
    actors = [_FakeActor(_loss()) for _ in range(3)]
    refs = [_Ref(_loss()) for _ in range(3)]
    dead: set[int] = set()

    with pytest.raises(ResourceError, match="could not be recomputed"):
        topn(refs, 0, actors, "ir", [None] * 3, dead)

    assert dead == {0, 1, 2}, "every worker should have been tried and recorded"


def test_the_recomputed_partition_reads_its_own_split(topn):
    """A recompute must re-run the LOST partition, not the survivor's own.

    Folding a survivor's partition twice would return a plausible top-N that is missing the
    dead worker's rows entirely -- a wrong answer that no row count would reveal.
    """
    seen: list = []

    class _Recording(_FakeActor):
        @property
        def local_topn(self):
            actor = self

            class _M:
                def remote(self, local_ir, part):
                    seen.append(part)
                    actor.calls += 1
                    return _Ref(actor.payload)

            return _M()

    actors = [_FakeActor(_loss()), _Recording([_Batch(4)])]
    parts = ["split-for-0", "split-for-1"]

    topn([_Ref(_loss()), _Ref([])], 0, actors, "ir", parts, set())

    assert seen == ["split-for-0"], f"recomputed the wrong split: {seen}"
