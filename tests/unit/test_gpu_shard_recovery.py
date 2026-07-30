"""Recovering a GPU shard that did not fit, and asking the autoscaler for what the plan wants.

Two failures decide whether a GPU fan-out scales in practice, and neither is about the answer:

* **a shard that does not fit.** The shard count is fixed before the query runs, from an
  estimate, and estimates are wrong exactly where it matters — a skewed key, a wider row than
  the footer promised, a neighbouring tenant on the device. Handing that shard to the host is
  correct and hands the largest piece of the work to the slowest executor. Subdividing is exact
  because the stage is mergeable, which is the property these cases pin.
* **asking for the devices the cluster already has.** That pins the autoscaler's floor against
  reclamation and can never grow it, so a query that could use thirty-two devices runs on the
  four it happened to find.

No Ray and no GPU here: the subdivision ladder is a pure function of a descriptor and a
callable, and the sizing is a pure function of the plan.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.dist.gpu.shards import is_memory_failure, run_subdivided, split_descriptor

pytestmark = pytest.mark.unit


def _batch_descriptor(n: int) -> dict:
    table = pa.table({"v": list(range(n))})
    return {"batches": [table.slice(i, 1).to_batches()[0] for i in range(n)]}


# --- classifying the failure ---------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        MemoryError("device"),
        RuntimeError("std::bad_alloc"),
        RuntimeError("RMM failure: out_of_memory"),
        RuntimeError("CUDA error: cudaErrorMemoryAllocation"),
        RuntimeError("parallel_for failed: out of memory"),
    ],
)
def test_allocation_failures_are_worth_subdividing(exc):
    assert is_memory_failure(exc)


@pytest.mark.parametrize(
    "exc",
    [
        KeyError("no such column"),
        ValueError("unsupported expression"),
        RuntimeError("worker died unexpectedly"),
    ],
)
def test_deterministic_failures_are_not(exc):
    """A deterministic error fails identically on a smaller shard.

    Retrying it in pieces would pay the split factor to reach the same conclusion, so it takes
    the other rung of the ladder — the CPU engine — immediately.
    """
    assert not is_memory_failure(exc)


def test_a_wrapped_allocation_failure_is_still_recognized():
    """The error arrives through Ray, which re-raises it as its own type."""
    try:
        try:
            raise RuntimeError("rmm::bad_alloc: out_of_memory")
        except RuntimeError as inner:
            raise RuntimeError("task failed") from inner
    except RuntimeError as wrapped:
        assert is_memory_failure(wrapped)


# --- subdividing the shard -----------------------------------------------------------


def test_a_descriptor_splits_into_smaller_reads():
    pieces = split_descriptor(_batch_descriptor(8), parts=4)
    assert len(pieces) == 4
    assert sum(len(p["batches"]) for p in pieces) == 8


def test_an_indivisible_descriptor_reports_itself():
    """One split is already the smallest thing a worker can be asked to read.

    Returning it unchanged is what lets the caller tell "divide again" from "cannot", instead
    of looping on a descriptor that will never get smaller.
    """
    single = {"splits": ["only-one"], "projection": None, "predicate": None}
    assert split_descriptor(single, parts=4) == [single]


def test_split_manifests_keep_their_pushdown():
    """Dividing the manifest must not drop the projection and predicate pushed into it —
    a piece that reads every column would reintroduce the I/O the pushdown removed."""
    desc = {"splits": list(range(6)), "projection": ["a"], "predicate": {"e": "col"}}
    for piece in split_descriptor(desc, parts=3):
        assert piece["projection"] == ["a"]
        assert piece["predicate"] == {"e": "col"}


def test_subdividing_recovers_a_shard_that_did_not_fit():
    """The whole shard fails; its pieces succeed; the concatenation is the shard's own value."""
    calls: list[int] = []

    def run(desc):
        n = len(desc["batches"])
        calls.append(n)
        if n > 2:
            raise RuntimeError("out of memory")
        return pa.table({"v": [b.column(0)[0].as_py() for b in desc["batches"]]})

    out = run_subdivided(_batch_descriptor(8), run, parts=2, rounds=3)
    assert sorted(out.column("v").to_pylist()) == list(range(8))
    assert max(calls) <= 4  # it never re-ran the whole shard


def test_subdividing_gives_up_rather_than_looping_forever():
    """A shard that fails however small it gets must raise, so the CPU rung is reached."""

    def always_oom(_desc):
        raise RuntimeError("out of memory")

    with pytest.raises(Exception, match=r"(?i)memory"):
        run_subdivided(_batch_descriptor(8), always_oom, parts=2, rounds=2)


def test_subdividing_reraises_a_deterministic_error_immediately():
    """A non-memory failure inside a piece stops the ladder rather than splitting further."""

    def bad_expression(_desc):
        raise ValueError("unsupported expression")

    with pytest.raises(ValueError, match="unsupported"):
        run_subdivided(_batch_descriptor(8), bad_expression, parts=2, rounds=3)


def test_empty_pieces_do_not_become_empty_tables():
    """A piece with no surviving rows contributes nothing, rather than a schema to reconcile."""

    def empty(_desc):
        return None

    assert run_subdivided(_batch_descriptor(4), empty, parts=2, rounds=1) is None


# --- asking for the right number of devices ------------------------------------------


def test_the_decision_asks_for_the_devices_the_plan_wants():
    """Sized from the working set, not from the cluster — otherwise it can never scale up."""
    from batcher.kyber.gpu.policy import decide_gpu_backend

    rows = 200_000
    q = bt.from_pydict({"k": list(range(rows)), "v": list(range(rows))})
    q = q.group_by("k").agg(s=bt.col("v").sum())
    # A device far smaller than the working set, but large enough that a shard of it fits:
    # the plan wants many devices, and the cluster has two.
    decision = decide_gpu_backend(
        q._plan, q._sources, gpu_count=2, force=True, gpu_memory_gb=0.0005
    )
    assert decision.use_gpu and decision.distributed
    assert decision.desired_gpus > 2


def test_the_device_request_is_capped():
    """A badly-estimated query must not be able to ask a cluster to grow without bound."""
    import dataclasses

    from batcher.config import active_config, config_context
    from batcher.kyber.gpu.policy import decide_gpu_backend

    cfg = active_config()
    scoped = cfg.replace(
        distributed=dataclasses.replace(cfg.distributed, gpu_max_autoscale_devices=3)
    )
    q = bt.from_pydict({"k": list(range(200_000)), "v": list(range(200_000))})
    q = q.group_by("k").agg(s=bt.col("v").sum())
    with config_context(scoped):
        decision = decide_gpu_backend(
            q._plan, q._sources, gpu_count=2, force=True, gpu_memory_gb=0.0005
        )
    assert decision.desired_gpus == 3


def test_a_single_device_plan_asks_for_one():
    from batcher.kyber.gpu.policy import decide_gpu_backend

    q = bt.from_pydict({"k": [1, 2], "v": [1.0, 2.0]}).group_by("k").agg(s=bt.col("v").sum())
    decision = decide_gpu_backend(q._plan, q._sources, gpu_count=8, force=True, gpu_memory_gb=80.0)
    assert decision.use_gpu and not decision.distributed
    assert decision.desired_gpus == 1


# --- recovering several shards at once ------------------------------------------------


def test_recoveries_are_submitted_before_any_is_awaited():
    """A spot reclamation takes several nodes at once, which is the case that matters.

    Awaiting each recovery where its failure was noticed would run them one after another,
    turning the one event a fan-out most needs to absorb into the slowest possible response.
    The barrier's failure hook returns a handle instead, so every recovery is already in flight
    by the time the first is awaited.
    """
    from batcher.dist.gpu.aggregate import _await_recoveries, _Recovering

    class _Ref:
        def __init__(self, value):
            self.value = value

    awaited: list = []

    class _FakeRay:
        @staticmethod
        def get(refs):
            awaited.append(list(refs))
            return [r.value for r in refs]

    import sys

    saved = sys.modules.get("ray")
    sys.modules["ray"] = _FakeRay
    try:
        results = ["a", _Recovering(_Ref("b")), "c", _Recovering(_Ref("d"))]
        assert _await_recoveries(results) == ["a", "b", "c", "d"]
    finally:
        if saved is None:
            del sys.modules["ray"]
        else:
            sys.modules["ray"] = saved
    # one wait covering both, not one wait each
    assert len(awaited) == 1
    assert len(awaited[0]) == 2


def test_a_run_with_no_recoveries_is_untouched():
    """The common case must not pay for the recovery machinery, or reorder anything."""
    from batcher.dist.gpu.aggregate import _await_recoveries

    results = ["a", "b", None, "c"]
    assert _await_recoveries(results) is results


# --- saying so when a run degraded ----------------------------------------------------


def _captured_events():
    """Collect published events for the duration of a `with` block."""
    import contextlib

    from batcher._internal import events

    @contextlib.contextmanager
    def _capture():
        seen: list = []
        unsubscribe = events.subscribe(seen.append)
        try:
            yield seen
        finally:
            if callable(unsubscribe):
                unsubscribe()

    return _capture


def test_a_clean_fan_out_says_nothing():
    """The ordinary case must add no noise, or the signal is worthless."""
    from batcher.dist.gpu.shards import ShardReport

    with _captured_events()() as seen:
        ShardReport("gpu-chain", 8).publish()
    assert [e for e in seen if e.fields.get("event") == "shard_degraded"] == []


def test_a_degraded_fan_out_reports_what_happened():
    """A run where a third of the shards fell back is a different run, and looks identical.

    Same rows, same schema, just slower — so without this event an operator has no way to tell
    a healthy cluster from one whose devices are too small for the shard size, or whose spot
    pool is being reclaimed faster than the work finishes.
    """
    from batcher.dist.gpu.shards import ShardReport

    report = ShardReport("gpu-chain", 9)
    report.note_subdivided()
    report.note_recovered()
    report.note_recovered()
    with _captured_events()() as seen:
        report.publish()
    degraded = [e for e in seen if e.fields.get("event") == "shard_degraded"]
    assert len(degraded) == 1
    assert degraded[0].fields["shards"] == 9
    assert degraded[0].fields["subdivided"] == 1
    assert degraded[0].fields["recovered_on_cpu"] == 2
    assert degraded[0].name == "gpu-chain"
