"""`execution.query_timeout_s`: a terminal operation that runs too long is cancelled.

The timeout is a cancellation that knows why it happened. It arms a timer when the query's
scope opens and, at the deadline, calls the same `cancel_query` a user would, so the
engine stops at its next morsel boundary. What the engine's flag cannot reach -- the Python
loop that runs a `map_batches` function per batch, a process pool waiting on its children,
a consumer pulling from `iter_batches` -- is bounded separately, and each is tested here.

Asserted: each terminal raises `QueryCancelledError` naming the limit and the phase, well
before the work would have finished; the query's id is released; with no limit nothing
changes; and an abandoned or timed-out stream closes its source. **Not** asserted: a tight
latency. Cancellation is cooperative, so the bounds below are generous multiples of the
limit and far below the uncancelled run time.
"""

from __future__ import annotations

import time

import pytest

import batcher as bt
from batcher._internal.errors import ConfigError, QueryCancelledError
from batcher.config import Config, ExecutionConfig, option_context
from batcher.core.runtime import bounded_iteration

pytestmark = pytest.mark.integration

LIMIT_S = 0.3
SLEEP_S = 0.2
BATCHES = 200  # 200 x 200 ms = 40 s uncancelled single-threaded, 20 s on two workers
#: Far below any uncancelled run, and generous above the limit: cancellation is cooperative,
#: a process pool takes a moment to tear down, and the shared box this runs on is loaded.
BOUND_S = 10.0


def _slow(batch):
    time.sleep(SLEEP_S)
    return batch


def _slow_pipeline(**map_kwargs):
    rows = BATCHES * 100
    return bt.from_pydict({"a": list(range(rows))}).map_batches(_slow, batch_size=100, **map_kwargs)


def _assert_times_out(run, *, stage: str) -> None:
    started = time.monotonic()
    with (
        option_context("execution.query_timeout_s", LIMIT_S),
        pytest.raises(QueryCancelledError, match="query_timeout_s") as info,
    ):
        run()
    elapsed = time.monotonic() - started
    assert stage in str(info.value)
    assert elapsed < BOUND_S, f"took {elapsed:.2f}s to honour a {LIMIT_S}s limit"
    assert bt.running_queries() == []


@pytest.fixture(autouse=True, scope="module")
def _warm():
    """Pay the first-use imports once, so they are not charged to the first timeout."""
    bt.from_pydict({"a": [1, 2]}).map_batches(lambda b: b, num_workers=2).collect()


class TestCollect:
    def test_a_thread_pool_udf_is_cancelled_between_batches(self) -> None:
        _assert_times_out(lambda: _slow_pipeline(num_workers=2).collect(), stage="map_batches")

    def test_a_default_udf_pipeline_is_cancelled(self) -> None:
        """Whichever strategy the planner picks (process or thread pool), the limit holds."""
        _assert_times_out(lambda: _slow_pipeline().collect(), stage="map_batches")

    def test_a_native_relational_query_is_cancelled_in_the_engine(self) -> None:
        # The query must outlast LIMIT_S on any box the gate runs on. The previous shape, a
        # self-join grouped by its own key, finished in 60 ms on a 16-core node, so it never
        # timed out. Here the aggregate reads both sides of every joined pair, 1.2 billion of
        # them (200 rows on each side of each of 30,000 keys): 3.0 s uncancelled on 16 cores
        # at a 0.7 GB peak. Do not raise the per-key fan-out to make it slower: at 400 rows a
        # side the same join peaked at 49 GB, and at 800 it was OOM-killed before the
        # cancellation landed.
        rows = 6_000_000
        ds = bt.from_pydict(
            {"a": [i % 30_000 for i in range(rows)], "b": [i * 0.5 for i in range(rows)]}
        )
        query = ds.join(ds, on="a").agg(m=(bt.col("b") - bt.col("b_right")).abs().max())
        _assert_times_out(query.collect, stage="core.execute")

    def test_a_high_fan_out_join_under_an_aggregate_is_cancelled_too(self) -> None:
        """The shape the comment above warns off, now that it no longer OOMs.

        800 rows a side per key was routed out of core by the spill gate's widest-intermediate
        term -- the 3.2 billion joined rows -- and the out-of-core path materializes exactly
        those, so the process died before the timer's cancellation landed. The aggregate folds
        them on the streaming executor (0.6 GB at 4M rows, measured), which polls the
        cancellation per morsel. 800 million pairs here: ~1.8 s uncancelled on 16 cores.
        """
        rows = 1_000_000
        ds = bt.from_pydict(
            {"a": [i % 1_250 for i in range(rows)], "b": [i * 0.5 for i in range(rows)]}
        )
        query = ds.join(ds, on="a").agg(m=(bt.col("b") - bt.col("b_right")).abs().max())
        _assert_times_out(query.collect, stage="core.execute")


def test_a_cancelled_process_stage_does_not_disable_the_process_pool(monkeypatch) -> None:
    """A timeout is not a broken pool: no thread fallback, and processes stay enabled.

    The process route falls back to threads, and turns processes off for the session, on
    any exception from the pool. A cancellation reaching it through that handler re-ran the
    stage on threads and quietly cost every later `map_batches` its process pool.
    """
    from batcher.core.udf import processes, strategy

    def cancelled(*_args, **_kwargs):
        raise QueryCancelledError("query exceeded execution.query_timeout_s during map_batches")

    monkeypatch.setattr(strategy, "_processes_disabled", False)
    monkeypatch.setattr(strategy, "wants_processes", lambda *_a: True)
    monkeypatch.setattr(processes, "run_map_processes", cancelled)
    with pytest.raises(QueryCancelledError):
        _slow_pipeline(num_workers=2).collect()
    assert strategy._processes_disabled is False


class TestWrite:
    def test_a_write_is_cancelled(self, tmp_path) -> None:
        target = str(tmp_path / "out.parquet")
        _assert_times_out(lambda: _slow_pipeline().write.parquet(target), stage="map_batches")


class TestIteration:
    def test_iter_batches_is_cancelled(self) -> None:
        def drain():
            for _ in _slow_pipeline().iter_batches():
                pass

        _assert_times_out(drain, stage="iter_batches")

    def test_consumer_time_is_not_charged(self) -> None:
        """A slow consumer is not cut off: only producing the batches counts."""

        def produce():
            yield from range(4)

        with option_context("execution.query_timeout_s", 0.1):
            seen = []
            for item in bounded_iteration(produce()):
                time.sleep(0.06)  # 4 x 60 ms of consumer time, past the 100 ms limit
                seen.append(item)
        assert seen == [0, 1, 2, 3]

    def test_a_timed_out_stream_closes_its_source(self) -> None:
        closed = []

        def produce():
            try:
                while True:
                    time.sleep(0.05)
                    yield 1
            finally:
                closed.append(True)

        with (
            option_context("execution.query_timeout_s", 0.12),
            pytest.raises(QueryCancelledError, match="iter_batches"),
        ):
            for _ in bounded_iteration(produce()):
                pass
        assert closed == [True]

    def test_an_abandoned_stream_closes_its_source(self) -> None:
        closed = []

        def produce():
            try:
                yield from range(10)
            finally:
                closed.append(True)

        with option_context("execution.query_timeout_s", 30.0):
            stream = bounded_iteration(produce())
            next(stream)
            stream.close()
        assert closed == [True]


class TestNoLimit:
    def test_the_default_is_no_limit(self) -> None:
        assert ExecutionConfig().query_timeout_s is None

    def test_without_a_limit_results_are_unchanged(self) -> None:
        ds = bt.from_pydict({"a": [3, 1, 2]}).map_batches(lambda b: b)
        assert sorted(ds.collect().column("a").to_pylist()) == [1, 2, 3]
        assert bt.running_queries() == []

    def test_a_query_inside_the_limit_returns_normally(self) -> None:
        with option_context("execution.query_timeout_s", 30.0):
            out = bt.from_pydict({"a": [1, 2, 3]}).agg(s=bt.col("a").sum()).collect()
        assert out.to_pydict() == {"s": [6]}

    @pytest.mark.parametrize("bad", [0, -1.0, True])
    def test_a_non_positive_limit_is_rejected(self, bad) -> None:
        with pytest.raises(ConfigError, match="query_timeout_s"):
            Config().replace(execution=ExecutionConfig(query_timeout_s=bad)).validate()
