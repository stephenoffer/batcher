"""Exactly-once across every crash window of a micro-batch, for each exactly-once sink.

One micro-batch is: record the source position (write-ahead), hand the rows to the sink,
snapshot state, then write the commit log. A process can die between any two of those,
and the claim is that a restart on the same checkpoint produces exactly the table an
uninterrupted run would have. The narrative argues it; this file checks it, by killing the
query at each boundary during its second micro-batch, restarting it, and comparing the
destination against a run that never crashed.

The kill is a non-restartable exception, so the engine does not self-heal in-process and
the second run is a genuine restart from the checkpoint on disk.
"""

from __future__ import annotations

import glob
import os

import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.io.formats.streaming.checkpoint.store import CheckpointStore
from batcher.io.formats.streaming.sinks import FileStreamSink, TransactionalStreamSink

pytestmark = pytest.mark.integration

#: The micro-batch each query dies in. The stateless query writes from its first batch;
#: the windowed one emits nothing until the watermark closes its first window in batch 3,
#: so a crash any earlier would never reach the sink and would test nothing there.
_CRASH_BATCH = {False: 1, True: 3}


class _Killed(RuntimeError):
    """Stands in for the process dying: not a transient fault, so nothing retries it."""


def _query(sink: str, dest: str, ck: str, *, stateful: bool):
    ds = bt.read.rate(5, num_rows=20, pace=False)
    if stateful:
        # Two-second windows over one-second micro-batches: every window's running sum is
        # carried in checkpointed state across a batch boundary before it closes.
        ds = (
            ds.with_watermark("timestamp", "1s")
            .group_by(w=bt.window(bt.col("timestamp"), "2s"))
            .agg(total=bt.col("value").sum(), n=bt.col("value").count())
        )
    else:
        ds = ds.select("value")
    kw = {"trigger": bt.Trigger.available_now(), "checkpoint": ck}
    if sink == "delta":
        return ds.write.delta(dest, **kw)
    return ds.write(dest, format="parquet", **kw)


def _rows(sink: str, dest: str) -> list[tuple]:
    if not os.path.exists(dest):
        return []
    if sink == "delta":
        import deltalake

        # A crash between the data file and the first commit leaves a directory holding an
        # orphan file and no log: not a table yet, and no rows.
        if not deltalake.DeltaTable.is_deltatable(dest):
            return []
        table = bt.read.delta(dest).collect()
    else:
        files = sorted(glob.glob(os.path.join(dest, "*.parquet")))
        table = pq.read_table(files) if files else None
    if table is None:
        return []
    cols = table.column_names
    return sorted(zip(*(table.column(c).to_pylist() for c in cols), strict=True))


def _arm(monkeypatch, window: str, sink: str, crash_batch: int) -> None:
    """Make the query die at `window` during micro-batch `crash_batch`, once."""
    fired = {"done": False}

    def once(batch_id: int) -> None:
        if batch_id == crash_batch and not fired["done"]:
            fired["done"] = True
            raise _Killed(window)

    if window == "before_sink_write":
        cls = TransactionalStreamSink if sink == "delta" else FileStreamSink
        real_write = cls.write_batch

        def write_batch(self, batch_id, table):
            once(batch_id)
            return real_write(self, batch_id, table)

        monkeypatch.setattr(cls, "write_batch", write_batch)
    elif window == "after_sink_before_commit_log":
        real_commit = CheckpointStore.commit

        def commit(self, batch_id, sink_token=None):
            once(batch_id)
            return real_commit(self, batch_id, sink_token)

        monkeypatch.setattr(CheckpointStore, "commit", commit)
    elif window == "data_file_written_txn_not_committed":
        from batcher.io.formats import SINKS

        delta_cls = SINKS.get("delta")
        real_commit = delta_cls.commit

        def delta_commit(self, manifest, path):
            txn = self._app_txn
            if txn is not None:
                once(txn[1])
            return real_commit(self, manifest, path)

        monkeypatch.setattr(delta_cls, "commit", delta_commit)
    else:  # pragma: no cover - the parametrization names every window
        raise AssertionError(window)


_WINDOWS = [
    ("delta", "before_sink_write"),
    ("delta", "data_file_written_txn_not_committed"),
    ("delta", "after_sink_before_commit_log"),
    ("parquet", "before_sink_write"),
    ("parquet", "after_sink_before_commit_log"),
]


@pytest.mark.parametrize("stateful", [False, True], ids=["stateless", "stateful"])
@pytest.mark.parametrize(("sink", "window"), _WINDOWS)
def test_a_crash_at_each_boundary_restarts_to_the_uninterrupted_result(
    tmp_path, monkeypatch, sink, window, stateful
):
    if sink == "delta":
        pytest.importorskip("deltalake")
    ref_dest, ref_ck = str(tmp_path / "ref"), str(tmp_path / "ref_ck")
    _query(sink, ref_dest, ref_ck, stateful=stateful).await_termination()
    expected = _rows(sink, ref_dest)
    assert expected, "the reference run wrote nothing, so the comparison would be vacuous"

    dest, ck = str(tmp_path / "out"), str(tmp_path / "ck")
    with monkeypatch.context() as patch:
        _arm(patch, window, sink, _CRASH_BATCH[stateful])
        with pytest.raises(_Killed):
            _query(sink, dest, ck, stateful=stateful).await_termination()
    # The crash really interrupted the stream: the destination is short of the answer.
    assert _rows(sink, dest) != expected

    _query(sink, dest, ck, stateful=stateful).await_termination()
    assert _rows(sink, dest) == expected
