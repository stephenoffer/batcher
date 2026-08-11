"""Bounded-state streaming drivers: source pushdown, window eviction, and stop latency.

These are cost contracts rather than result contracts, so each one is asserted by observing
what the driver *asked the source for* or *asked the engine to run* — the results themselves
are already pinned by the differential suite.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher.api.dataset.frame import Dataset
from batcher.io.formats.streaming.broker import BrokerSource
from batcher.plan.logical import Scan
from batcher.plan.schema import SchemaRef


class _WideStream:
    """An unbounded-shaped source that records the projection every read asks for."""

    bounded = True  # bounded so a test can drain it; the projection contract is the point

    def __init__(self, rows: int = 8, columns: int = 6) -> None:
        self.projections: list[list[str] | None] = []
        self._data = {f"c{i}": [i * 100 + r for r in range(rows)] for i in range(columns)}
        self._data["k"] = ["a", "b"] * (rows // 2)

    def schema(self) -> pa.Schema:
        return pa.table(self._data).schema

    def row_count(self) -> int | None:
        return len(self._data["k"])

    def read(self, projection=None):
        return list(self.iter_batches(projection))

    def iter_batches(self, projection=None):
        self.projections.append(list(projection) if projection is not None else None)
        table = pa.table(self._data)
        if projection is not None:
            table = table.select(projection)
        yield from table.to_batches()

    def identity(self) -> str:
        return "wide-stream"

    def splits(self, target_size=None):
        return []


def _ds(source: _WideStream) -> Dataset:
    return Dataset(Scan(0, SchemaRef.from_arrow(source.schema())), [source])


def _read_columns(source: _WideStream) -> set[str]:
    assert source.projections, "the driver never read the source"
    narrowed = [p for p in source.projections if p is not None]
    assert narrowed, "the driver read every column"
    return set(narrowed[0])


def test_a_streaming_aggregate_reads_only_the_columns_it_groups_and_sums():
    """The bounded-state drivers read the source directly and read it whole: a two-column
    aggregate over a wide event decoded every other column per micro-batch and threw it away."""
    source = _WideStream()
    ds = _ds(source).group_by("k").agg(total=bt.col("c0").sum())
    rows = list(ds.iter_batches())

    assert sum(b.num_rows for b in rows) == 2
    assert _read_columns(source) == {"k", "c0"}


def test_a_streaming_distinct_reads_only_its_key_columns():
    source = _WideStream()
    ds = _ds(source).select("k", "c1").distinct()
    assert sum(b.num_rows for b in ds.iter_batches()) == 8  # c1 is unique per row
    assert _read_columns(source) == {"k", "c1"}


def test_a_streaming_top_n_reads_only_its_sort_and_output_columns():
    source = _WideStream()
    ds = _ds(source).select("k", "c2").sort("c2", descending=True).head(3)
    assert sum(b.num_rows for b in ds.iter_batches()) == 3
    assert _read_columns(source) == {"k", "c2"}


def test_a_streaming_limit_reads_only_its_output_columns():
    source = _WideStream()
    ds = _ds(source).select("c3").limit(2)
    assert sum(b.num_rows for b in ds.iter_batches()) == 2
    assert _read_columns(source) == {"c3"}


def test_the_pushdown_never_changes_the_answer():
    """The projection is a cost decision, so the result must be identical either way."""
    narrowed = _ds(_WideStream()).group_by("k").agg(t=bt.col("c0").sum())
    whole = _ds(_WideStream()).group_by("k").agg(t=bt.col("c0").sum())
    streamed = pa.Table.from_batches(list(narrowed.iter_batches())).sort_by("k").to_pydict()
    collected = whole.collect().sort_by("k").to_pydict()
    assert streamed == collected


# --------------------------------------------------------------------------
# The windowed fold must not re-sweep state on every micro-batch.
# --------------------------------------------------------------------------
def test_window_eviction_is_skipped_when_no_window_can_have_closed():
    """A window closes when `window_start <= watermark - width`. With a wide window and a
    fast trigger, almost every micro-batch closes nothing — and each sweep costs two full
    relational passes over the whole open-window state plus a re-combine."""
    import datetime as dt

    from batcher.core.streaming import _window_key, _WindowedAggFold

    base = dt.datetime(2024, 1, 1)
    plan = (
        bt.from_pydict({"t": [base], "v": [1]})
        .with_watermark("t", "5m")
        .group_by(w=bt.window(bt.col("t"), "10m"))
        .agg(s=bt.col("v").sum())
        ._plan
    )
    fold = _WindowedAggFold(plan, _window_key(plan))

    # `_evicted_through` advances exactly when a sweep runs, so counting its changes counts
    # the sweeps without patching a `__slots__` object.
    sweeps = 0
    previous = fold._evicted_through
    for i in range(12):
        fold.push(pa.record_batch({"t": [base + dt.timedelta(seconds=i)], "v": [1]}))
        if fold._evicted_through != previous:
            sweeps += 1
            previous = fold._evicted_through
    # Twelve micro-batches inside one ten-minute window: the close threshold never moves
    # past the first sweep, so only that first one may do the work.
    assert sweeps <= 1

    # Crossing into the next window moves the threshold and must sweep again.
    fold.push(pa.record_batch({"t": [base + dt.timedelta(minutes=25)], "v": [1]}))
    assert fold._evicted_through != previous


def test_a_restored_fold_sweeps_on_its_next_push():
    """A restored fold has swept nothing yet; assuming the dead run emitted its closed
    windows would strand them forever."""
    import datetime as dt

    from batcher.core.streaming import _window_key, _WindowedAggFold

    base = dt.datetime(2024, 1, 1)
    plan = (
        bt.from_pydict({"t": [base], "v": [1]})
        .with_watermark("t", "5m")
        .group_by(w=bt.window(bt.col("t"), "1m"))
        .agg(s=bt.col("v").sum())
        ._plan
    )
    key = _window_key(plan)
    fold = _WindowedAggFold(plan, key)
    fold.push(pa.record_batch({"t": [base], "v": [1]}))
    fold.push(pa.record_batch({"t": [base + dt.timedelta(minutes=5)], "v": [1]}))
    state = fold.state()

    revived = _WindowedAggFold(plan, key)
    revived.restore(state)
    assert revived._evicted_through is None


# --------------------------------------------------------------------------
# An idle unbounded source must not make `stop()` block forever.
# --------------------------------------------------------------------------
class _NeverPublishes(BrokerSource):
    """A broker that is reachable, healthy, and simply has nothing to say."""

    format_name = "never_publishes"
    __slots__ = ("polls",)

    def __init__(self) -> None:
        super().__init__("idle", poll_size=4)
        self.polls = 0

    def _discover_partitions(self):
        return [0]

    def _poll(self):
        self.polls += 1
        return []


def test_an_idle_broker_ends_its_poll_loop_when_the_stop_signal_fires():
    """Stopping a query parked on an idle topic did not merely take a while: `stage()` sat
    inside `next()` waiting for data that might never come, and `stop()` blocked joining it."""
    source = _NeverPublishes()
    stopped = {"now": False}
    source.set_stop_signal(lambda: stopped["now"])

    stream = source.iter_batches()
    stopped["now"] = True
    assert list(stream) == []  # the loop ends itself rather than polling forever
    assert source.polls == 0


def test_a_broker_carries_no_stop_signal_until_a_driver_attaches_one():
    """An unattached source keeps its original poll-forever behavior — the signal is opt-in,
    so a plain `iter_batches()` consumer is unaffected."""
    source = _NeverPublishes()
    assert source._should_stop is None
    predicate = lambda: True  # noqa: E731
    source.set_stop_signal(predicate)
    assert source._should_stop is predicate
    source.set_stop_signal(None)
    assert source._should_stop is None
