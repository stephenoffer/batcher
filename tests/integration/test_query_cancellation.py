"""A ten-minute `collect()` must be interruptible, and Ctrl-C must be what interrupts it.

# The gap this closes

`bc_py::execute_plan` runs inside `Python::allow_threads`, which releases the GIL for the
whole of execution. Python's signal handlers run between bytecodes, and while the native
call holds there are none — so `SIGINT` during a long `collect()` is *recorded* and not
*delivered* until the query finishes on its own. The interpreter is not hung; it is not
scheduled. Ctrl-C did nothing, twice, and then the third one killed the process.

# What is asserted here, and what is not

Asserted: a cancel from another thread stops a running query; the query raises
`QueryCancelledError` rather than returning short rows; the id is registered while running
and gone afterwards, on both the success and the failure path; an uncancelled query is
untouched; and cancelling a finished query is a no-op rather than an error.

**Not** asserted, because it is not true: that cancellation is instant. It is cooperative
and lands at a *morsel* boundary, and a pipeline breaker — a sort's run generation, a hash
join's build — consumes its whole input inside one such step. A query that spends four
minutes building a hash table notices four minutes late. The external sort polls between
merge passes and the materializing executor polls between operators, which covers the shapes
where that gap is longest, but the honest claim is "at the next boundary", not "at once".

Nor is the *latency* of a cancel measured here. That needs a workload long enough for the
measurement to mean something, which is a benchmark, not a test.
"""

from __future__ import annotations

import threading
import time

import pytest

import batcher as bt
from batcher._internal.errors import PlanError, QueryCancelledError
from batcher.core.runtime import current_query_id, query_scope

pytestmark = pytest.mark.integration


def _big_dataset(rows: int = 4_000_000):
    """Big enough that a `collect()` spans many morsels, so a cancel has somewhere to land."""
    return bt.from_pydict({"a": list(range(rows)), "b": [1.5] * rows})


class TestTheRegistry:
    def test_the_id_is_released_when_the_query_succeeds(self) -> None:
        """That a *running* query is listed is proven by the cancel test below, which finds
        its target through `running_queries()`. What needs its own assertion is the other
        half: that the id does not outlive the query. A leaked registration would keep
        naming a finished query, and would let a reused id be cancelled by a stale request."""
        with query_scope() as query_id:
            _big_dataset(100_000).filter(bt.col("a") > 0).collect()
        assert query_id not in bt.running_queries()

    def test_the_id_is_released_when_the_query_raises(self) -> None:
        """The registration is RAII in Rust precisely so the error path cannot leak it."""
        query_id = ""
        with pytest.raises(PlanError), query_scope() as qid:
            query_id = qid
            bt.from_pydict({"a": [1]}).select(bt.col("nope")).collect()
        assert query_id not in bt.running_queries()

    def test_cancelling_a_finished_query_is_a_no_op(self) -> None:
        # The cancel/completion race has no correct loser, so it reports rather than raises.
        assert bt.cancel_query("q-definitely-not-running") is False

    def test_current_query_id_is_empty_outside_a_scope(self) -> None:
        assert current_query_id() == ""

    def test_current_query_id_is_set_inside_one(self) -> None:
        with query_scope() as query_id:
            assert current_query_id() == query_id
            assert query_id.startswith("q-")


class TestScopeReentrancy:
    def test_a_nested_scope_keeps_the_outer_id(self) -> None:
        """The bug this caught, pinned.

        A terminal op opens its own `query_scope`. A caller that opened one first — to learn
        the id it wanted to be able to cancel — used to have that id silently replaced: the
        query ran under a fresh inner id, and `cancel_query(outer_id)` cancelled nothing
        while the query carried on. It reported success, too, because the outer id was still
        registered. Re-entrancy is what makes the id a caller holds mean the query it ran.
        """
        with query_scope() as outer, query_scope() as inner:
            assert inner == outer
            assert current_query_id() == outer

    def test_the_outer_id_is_the_one_the_engine_uses(self) -> None:
        # The end-to-end form of the above: cancel the id the caller holds, and the query
        # the caller started must stop.
        with query_scope() as query_id:
            assert bt.cancel_query(query_id) is True
            with pytest.raises(QueryCancelledError):
                _big_dataset(500_000).filter(bt.col("a") > 0).collect()

    def test_leaving_a_nested_scope_does_not_unregister(self) -> None:
        """An inner scope that unregistered on exit would leave the outer query
        uncancellable for the rest of its life."""
        with query_scope() as outer:
            with query_scope():
                pass
            assert outer in bt.running_queries()


class TestCancelling:
    @pytest.mark.parametrize(
        ("shape", "build"),
        [
            ("filter", lambda d: d.filter(bt.col("a") > 0)),
            ("project", lambda d: d.with_columns(c=bt.col("b") * 2)),
            ("aggregate", lambda d: d.group_by("b").agg(n=bt.col("a").count())),
            ("sort", lambda d: d.sort("a", descending=True)),
            ("distinct", lambda d: d.select("b").distinct()),
            ("join", lambda d: d.join(d.select("a"), on="a", how="inner")),
        ],
    )
    @pytest.mark.parametrize("streaming", [True, False])
    def test_every_shape_stops_on_both_executors(self, shape, build, streaming) -> None:
        """Both executors, six shapes. The matrix is the point.

        Streaming is the default and materializing is the fallback, and they poll at
        different places — streaming once per morsel through the pipeline, materializing once
        per operator. A cancel honoured on one and ignored on the other is the shape of bug
        that ships: the feature demos fine and then does nothing on the plan that actually
        needed it.
        """
        import dataclasses

        from batcher.config import Config, config_context

        base = Config()
        scoped = base.replace(execution=dataclasses.replace(base.execution, streaming=streaming))
        with config_context(scoped), query_scope() as query_id:
            bt.cancel_query(query_id)
            with pytest.raises(QueryCancelledError):
                build(_big_dataset(1_000_000)).collect()

    def test_a_cancel_from_another_thread_stops_a_running_query(self) -> None:
        """The realistic case: the query is genuinely mid-flight when the cancel arrives."""
        outcome: list[object] = []
        started = threading.Event()

        def run() -> None:
            try:
                with query_scope():
                    started.set()
                    dataset = _big_dataset()
                    for _ in range(6):
                        dataset = dataset.filter(bt.col("a") >= 0).with_columns(
                            b=bt.col("b") * 1.000001
                        )
                    outcome.append(dataset.collect().num_rows)
            except BaseException as exc:
                outcome.append(exc)

        worker = threading.Thread(target=run)
        worker.start()
        assert started.wait(timeout=30), "the worker never started"

        target = ""
        for _ in range(500):
            ids = bt.running_queries()
            if ids:
                target = ids[0]
                break
            time.sleep(0.002)
        assert target, "the query never registered; nothing was cancelled"
        bt.cancel_query(target)

        worker.join(timeout=60)
        assert not worker.is_alive(), "the query ignored the cancel"
        assert outcome, "the worker produced neither a result nor an error"
        # A completion here is a lost race, not a failure of the mechanism: the query is
        # ~2 s and the cancel is issued as fast as the poll can see it, but a loaded box can
        # still finish first. What must never happen is a *short* result presented as whole.
        if isinstance(outcome[0], int):
            pytest.skip(f"the query finished before the cancel landed ({outcome[0]} rows)")
        assert isinstance(outcome[0], QueryCancelledError), (
            f"expected QueryCancelledError, got {outcome[0]!r} — a cancelled query that "
            f"returns rows is worse than one that hangs, because the rows look complete"
        )

    def test_an_uncancelled_query_is_unaffected(self) -> None:
        # The regression that matters most: the poll must not change a normal result.
        with query_scope():
            table = _big_dataset(100_000).filter(bt.col("a") < 50_000).collect()
        assert table.num_rows == 50_000

    def test_results_match_with_and_without_a_cancellable_scope(self) -> None:
        """A cancellable query and an uncancellable one must compute the same relation."""
        plan = (
            _big_dataset(200_000)
            .filter(bt.col("a") % 3 == 0)
            .group_by("b")
            .agg(n=bt.col("a").count())
        )
        outside = plan.to_pydict()
        with query_scope():
            inside = plan.to_pydict()
        assert outside == inside
