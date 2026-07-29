"""The distributed micro-batch driver must not re-enumerate splits five times a second.

`splits()` costs a round trip for most unbounded sources, and the idle loop asks on every
pass — but for the one source whose `splits()` *is* the discovery, asking is mandatory.
Both halves are pinned here, because getting the second one wrong strands data silently.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.dist.streaming.microbatch import _SPLIT_REFRESH_SECONDS, DistributedRunner


class _CountingBroker:
    """An unbounded source whose `splits()` is a pure question that costs a round trip."""

    bounded = False
    partitionable = True

    def __init__(self) -> None:
        self.split_calls = 0

    def schema(self) -> pa.Schema:
        return pa.schema([("a", pa.int64())])

    def row_count(self) -> int | None:
        return None

    def splits(self, target_size: int | None = None):
        self.split_calls += 1
        return ["p0", "p1"]


class _ClaimingSource(_CountingBroker):
    """The incremental-file shape: `splits()` hands out work and withholds it thereafter."""

    def __init__(self) -> None:
        super().__init__()
        self.handed = 0

    def splits(self, target_size: int | None = None):
        self.split_calls += 1
        self.handed += 1
        return [f"file{self.handed}"] if self.handed <= 3 else []

    def confirm(self) -> None:
        """The marker for a source whose `splits()` claims what it returns."""


def _runner(source) -> DistributedRunner:
    return DistributedRunner(
        plan_ir="{}",
        projection=None,
        source=source,
        path="/tmp/does-not-matter",
        fmt="parquet",
        sink_kwargs={},
        query_name="q",
        num_workers=1,
        drain=True,
        should_stop=lambda: False,
    )


def test_a_broker_is_enumerated_once_inside_the_refresh_window(monkeypatch):
    source = _CountingBroker()
    runner = _runner(source)
    for _ in range(10):
        assert runner._splits() == ["p0", "p1"]
    assert source.split_calls == 1


def test_the_cache_expires_so_a_rescaled_topic_is_noticed(monkeypatch):
    import time

    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    source = _CountingBroker()
    runner = _runner(source)

    runner._splits()
    clock["now"] += _SPLIT_REFRESH_SECONDS / 2
    runner._splits()
    assert source.split_calls == 1

    clock["now"] += _SPLIT_REFRESH_SECONDS
    runner._splits()
    assert source.split_calls == 2


def test_a_claiming_source_is_asked_every_pass():
    """For the incremental file source, asking *is* the discovery — caching the answer would
    replay the same files forever and never see a new one."""
    source = _ClaimingSource()
    runner = _runner(source)
    assert runner._splits() == ["file1"]
    assert runner._splits() == ["file2"]
    assert runner._splits() == ["file3"]
    assert runner._splits() == []
    assert source.split_calls == 4
