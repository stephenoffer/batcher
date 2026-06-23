"""Credit-backpressured multi-stage pipeline (A) — correctness + bounded memory.

The competitive moat is keeping a downstream (GPU) stage fed via overlap while a slow
consumer can't blow memory: stages run concurrently, bounded by credits. Verified on
CPU (the multi-node GPU placement layers on the same shape, tested on hardware).
"""

from __future__ import annotations

import time

import pyarrow as pa
import pytest

from batcher.ml.pipeline import Stage, run_pipeline


def _batch(v: int) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays([pa.array([v], type=pa.int64())], names=["x"])


def _add(k: int):
    import pyarrow.compute as pc

    def factory():
        return lambda b: pa.RecordBatch.from_arrays([pc.add(b.column("x"), k)], names=["x"])

    return factory


def _vals(batches) -> list[int]:
    out = []
    for b in batches:
        out.extend(b.column("x").to_pylist())
    return out


def test_pipeline_equals_sequential_composition_in_order():
    stages = [Stage(_add(1)), Stage(_add(10))]  # x -> x+1 -> x+11
    out = list(run_pipeline((_batch(i) for i in range(20)), stages))
    assert _vals(out) == [i + 11 for i in range(20)]  # order preserved


def test_empty_stage_list_is_identity():
    out = list(run_pipeline((_batch(i) for i in range(5)), []))
    assert _vals(out) == list(range(5))


def test_factory_built_once_per_stage():
    builds = []

    def factory():
        builds.append(1)
        return lambda b: b

    stages = [Stage(factory), Stage(factory)]
    list(run_pipeline((_batch(i) for i in range(30)), stages))
    assert len(builds) == 2  # once per stage thread, never per batch


def test_backpressure_keeps_producer_lead_bounded():
    # A slow consumer must NOT cause the source to be drained ahead unboundedly:
    # with credits, the producer leads by at most a constant (bounded memory), not by
    # the whole stream.
    produced = [0]

    def source():
        for i in range(200):
            produced[0] += 1
            yield _batch(i)

    stages = [Stage(_add(0), credits=2), Stage(_add(0), credits=2)]
    max_lead = 0
    consumed = 0
    for _ in run_pipeline(source(), stages):
        consumed += 1
        time.sleep(0.0005)  # slow consumer
        max_lead = max(max_lead, produced[0] - consumed)
    assert consumed == 200
    # Total buffering ≈ source_q(2) + 2 stage queues(2+2) + in-flight workers — a
    # small constant, independent of the 200-row stream. Generous ceiling:
    assert max_lead <= 16, max_lead


def test_stage_error_propagates_to_consumer():
    def boom_factory():
        def worker(_b):
            raise RuntimeError("stage exploded")

        return worker

    import pytest

    with pytest.raises(RuntimeError, match="stage exploded"):
        list(run_pipeline((_batch(i) for i in range(10)), [Stage(_add(1)), Stage(boom_factory)]))


def test_error_does_not_leak_threads():
    # On a mid-pipeline error, upstream stages blocked on a full output queue must
    # still exit (stop-aware puts), not hang — else each failed pipeline leaks threads.
    import threading
    import time

    def slow_source():
        for i in range(1000):
            yield _batch(i)

    def boom_factory():
        def worker(_b):
            raise RuntimeError("kaboom")

        return worker

    before = threading.active_count()
    # A fast source + tiny credits keeps the upstream stage blocked on put when the
    # downstream stage explodes — the exact leak condition.
    with pytest.raises(RuntimeError, match="kaboom"):
        list(
            run_pipeline(slow_source(), [Stage(_add(0), credits=1), Stage(boom_factory, credits=1)])
        )
    # Give the stop-aware threads a moment to wind down, then assert no net leak.
    deadline = time.time() + 3.0
    while threading.active_count() > before and time.time() < deadline:
        time.sleep(0.05)
    assert threading.active_count() <= before, (
        f"leaked threads: {threading.active_count()} > {before}"
    )


def test_error_with_paused_consumer_does_not_deadlock():
    # The nasty case: the consumer pauses (final queue fills), then an UPSTREAM stage
    # errors. The last stage abandons its blocked put without emitting _DONE, so a
    # plain consumer get would hang forever. The stop-aware consumer must still
    # terminate and raise. Run under a watchdog so a hang fails instead of stalling.
    import threading
    import time

    def source():
        for i in range(1000):
            yield _batch(i)

    def boom_on_third():
        state = {"n": 0}

        def worker(b):
            state["n"] += 1
            if state["n"] >= 3:
                raise RuntimeError("upstream boom")
            return b

        return worker

    result = {}

    def consume():
        gen = run_pipeline(source(), [Stage(boom_on_third, credits=1), Stage(_add(0), credits=1)])
        try:
            next(gen)  # pull one, then pause so the final queue fills
            time.sleep(0.2)
            list(gen)  # resume — must terminate, not hang
            result["outcome"] = "no-error"
        except RuntimeError as e:
            result["outcome"] = str(e)
        except StopIteration:
            result["outcome"] = "stopped"

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "pipeline deadlocked on error with a paused consumer"
    assert "boom" in result.get("outcome", "") or result.get("outcome") == "stopped"
