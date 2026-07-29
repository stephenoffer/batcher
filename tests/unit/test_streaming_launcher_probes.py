"""The streaming launchers must not consume work while sizing, or leak a store on failure.

Both contracts are about what a launcher does *before* the query runs, so neither shows up
in a result: one silently strands the first batch of discovered files, the other leaks two
SQLite connections per failed start.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.api.streaming._distributed import _drain_workers


class _PlainSplittable:
    """A source whose `splits()` is a pure question, as the protocol intends."""

    bounded = True

    def __init__(self, n: int) -> None:
        self.split_calls = 0
        self._n = n

    def schema(self) -> pa.Schema:
        return pa.schema([("a", pa.int64())])

    def splits(self, target_size: int | None = None):
        self.split_calls += 1
        return list(range(self._n))


class _ClaimingSource(_PlainSplittable):
    """A source whose `splits()` *claims* the work it returns, the Auto Loader shape.

    Files handed out become pending and are withheld from every later pass until the epoch
    that read them is published, so asking twice loses a pass.
    """

    bounded = False

    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.claimed: list[int] = []

    def splits(self, target_size: int | None = None):
        self.split_calls += 1
        fresh = [i for i in range(self._n) if i not in self.claimed]
        self.claimed.extend(fresh)
        return fresh

    def confirm(self) -> None:
        """The other half of the discover/confirm protocol — the marker for a claiming source."""


def test_a_pure_splittable_source_is_sized_from_its_partitions():
    source = _PlainSplittable(3)
    assert _drain_workers(source) >= 1
    assert source.split_calls == 1


def test_a_claiming_source_is_never_probed():
    """Counting partitions by calling `splits()` consumed a whole discovery pass whose files
    no epoch ever read — silently stranding the query's first batch of new files, for the
    life of a continuous run."""
    source = _ClaimingSource(5)
    workers = _drain_workers(source)
    assert workers >= 1
    assert source.split_calls == 0
    assert source.claimed == []  # nothing was taken out of circulation


def test_a_source_that_cannot_split_falls_back_to_one_worker():
    class _Angry(_PlainSplittable):
        def splits(self, target_size: int | None = None):
            raise RuntimeError("cannot enumerate")

    assert _drain_workers(_Angry(0)) == 1


def test_a_failed_distributed_start_closes_its_checkpoint_store(tmp_path, monkeypatch):
    """`start()` opens the checkpoint and recovers before the loop thread exists, so a failure
    there leaves a store nothing will ever close — two SQLite connections per failed start."""
    from batcher.api.streaming import _distributed
    from batcher.io.formats.streaming.checkpoint import CheckpointStore

    closed: list[CheckpointStore] = []
    real_close = CheckpointStore.close

    def spy_close(self):
        closed.append(self)
        real_close(self)

    monkeypatch.setattr(CheckpointStore, "close", spy_close)

    class _Boom:
        def start(self):
            raise RuntimeError("sink refused to open")

    import batcher as bt
    from batcher import core

    monkeypatch.setattr(core, "StreamingQueryEngine", lambda **_: _Boom())
    # The plan is never optimized because `start()` raises first, so a bare scan suffices.
    monkeypatch.setattr(_distributed, "_drain_workers", lambda _source: 1, raising=False)

    plan = bt.from_pydict({"a": [1]})._plan
    with pytest.raises(RuntimeError, match="sink refused"):
        _distributed.start_distributed_stream(
            plan=plan,
            sources=[_PlainSplittable(1)],
            path=str(tmp_path / "out"),
            fmt="parquet",
            sink_kwargs={},
            trigger=bt.Trigger.processing_time(1),
            checkpoint=str(tmp_path / "ck"),
        )
    assert len(closed) == 1

    # And the query did not linger in the registry as a phantom active stream.
    from batcher.api.streaming._query import active_streams

    assert active_streams() == []
