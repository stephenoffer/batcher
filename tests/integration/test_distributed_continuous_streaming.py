"""Distributed continuous streaming — the epoch fans out, the commit does not.

A `distributed=True` streaming write with a continuous / processing-time trigger runs each
micro-batch as a cluster-wide epoch: the workers read their share, run the plan, and stage
data files *without committing*; the driver then publishes the whole epoch as **one**
transaction. What these tests hold the implementation to:

* the log records **one transaction per micro-batch** — not one per worker, and not one per
  file (the thing that makes a Delta log stop describing the stream that wrote it);
* a **replayed** epoch adds no rows and no transaction (exactly-once), because it finds its
  own ``txn`` already committed;
* an **interrupted** epoch loses nothing: its files were never confirmed, so the next run
  picks them up again;
* the distributed result is **identical** to the single-node one — including for a
  streaming aggregation, which fans out as `partial` and merges with `combine`.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.integration

deltalake = pytest.importorskip("deltalake")


def _land(directory, batch: int, files: int = 3, per_file: int = 10) -> None:
    """Drop `files` new parquet files into the watched directory (an Auto Loader arrival)."""
    for i in range(files):
        start = batch * 1000 + i * per_file
        pq.write_table(
            pa.table({"id": list(range(start, start + per_file))}),
            f"{directory}/b{batch:03d}f{i:03d}.parquet",
        )


def _stream(src, state, out, ckpt, *, trigger=None, workers=4, plan=lambda ds: ds):
    ds = bt.read.files_incremental(str(src), "parquet", state_dir=str(state))
    query = plan(ds).write(
        str(out),
        format="delta",
        trigger=trigger or bt.Trigger.available_now(),
        checkpoint=str(ckpt),
        distributed=True,
        num_workers=workers,
    )
    query.await_termination()
    return query


def _ids(path) -> list[int]:
    return sorted(bt.read(str(path), format="delta").to_pydict()["id"])


def _commits(path) -> int:
    return len(deltalake.DeltaTable(str(path)).history())


def test_each_micro_batch_is_exactly_one_transaction(tmp_path):
    src, state, out, ckpt = (tmp_path / n for n in ("src", "state", "tbl", "ckpt"))
    src.mkdir()

    for batch in range(3):
        _land(src, batch)
        query = _stream(src, state, out, ckpt)
        assert [p.batch_id for p in query.recent_progress] == [batch]

    table = deltalake.DeltaTable(str(out))
    # Three arrivals, three micro-batches, three commits — even though several workers
    # each wrote their own data file within every one of them.
    assert _commits(out) == 3
    assert len(table.file_uris()) > 3, "the epochs should have been written in parallel"
    assert _ids(out) == sorted(i for b in range(3) for i in range(b * 1000, b * 1000 + 30))


def test_a_replayed_micro_batch_adds_no_rows_and_no_transaction(tmp_path):
    src, state, out, ckpt = (tmp_path / n for n in ("src", "state", "tbl", "ckpt"))
    src.mkdir()
    _land(src, 0)
    _stream(src, state, out, ckpt)
    before_rows, before_commits = _ids(out), _commits(out)

    # Re-run the same query against the same checkpoint with nothing new to read.
    _stream(src, state, out, ckpt)

    assert _ids(out) == before_rows
    assert _commits(out) == before_commits


def test_an_interrupted_epoch_is_replayed_not_lost(tmp_path):
    """A crash between discovery and the commit must not swallow the files.

    Discovery used to mark a file seen the moment it was *listed*, so a query that died
    mid-epoch came back to a store that already claimed those files and skipped them
    forever. The durable store now only records what has been published.
    """
    from batcher.io.formats.streaming.autoloader import IncrementalFileSource

    src, state = tmp_path / "src", tmp_path / "state"
    src.mkdir()
    _land(src, 0)

    # A source that discovers the pass and then dies before anything is published.
    doomed = IncrementalFileSource(str(src), "parquet", state_dir=str(state))
    assert len(doomed.discover()) == 3
    del doomed

    # The restart re-offers every file: nothing was recorded, because nothing was published.
    survivor = IncrementalFileSource(str(src), "parquet", state_dir=str(state))
    assert sum(b.num_rows for b in survivor.iter_batches()) == 30

    # And now that the pass *has* been consumed, it is durable — no second helping.
    assert IncrementalFileSource(str(src), "parquet", state_dir=str(state)).discover() == []


def test_distributed_result_matches_single_node(tmp_path):
    src, state_d, state_s = tmp_path / "src", tmp_path / "sd", tmp_path / "ss"
    dist_out, single_out = tmp_path / "dist", tmp_path / "single"
    ck_d, ck_s = tmp_path / "ckd", tmp_path / "cks"
    src.mkdir()
    for batch in range(2):
        _land(src, batch, files=4)

    keep = lambda ds: ds.filter(bt.col("id") % 3 == 0)  # noqa: E731
    _stream(src, state_d, dist_out, ck_d, workers=4, plan=keep)
    bt.read.files_incremental(str(src), "parquet", state_dir=str(state_s)).pipe(keep).write(
        str(single_out), format="delta", trigger=bt.Trigger.available_now(), checkpoint=str(ck_s)
    ).await_termination()

    assert _ids(dist_out) == _ids(single_out)


def test_distributed_streaming_aggregate_merges_worker_partials(tmp_path):
    """A streaming aggregation fans out as `partial` and merges with `combine`.

    Each worker aggregates only its share of the epoch and returns a partial state — small,
    bounded by the group count — which the driver combines into the running result. The
    answer must be the one a single node computes, because it is the same mergeable operator
    (`partial → combine → finalize`), not a second implementation of it.
    """
    src, state, out, ckpt = (tmp_path / n for n in ("src", "state", "tbl", "ckpt"))
    src.mkdir()
    _land(src, 0, files=4)

    agg = lambda ds: (  # noqa: E731
        ds.with_columns(bucket=bt.col("id") % 4).group_by("bucket").agg(n=bt.col("id").count())
    )
    _stream(src, state, out, ckpt, workers=4, plan=agg, trigger=bt.Trigger.available_now())

    got = bt.read(str(out), format="delta").to_pydict()
    truth = agg(bt.read(str(src), format="parquet")).to_pydict()
    assert sorted(zip(got["bucket"], got["n"], strict=True)) == sorted(
        zip(truth["bucket"], truth["n"], strict=True)
    )


def test_a_continuous_stream_spans_arrivals_and_stops_on_demand(tmp_path):
    """One long-running query, several epochs — an idle moment is not the end of a stream.

    This is what makes it *continuous* rather than a drain re-run by hand: the query stays
    up across a lull, picks up the next arrival as its own micro-batch, and commits it as
    its own transaction.
    """

    src, state, out, ckpt = (tmp_path / n for n in ("src", "state", "tbl", "ckpt"))
    src.mkdir()
    _land(src, 0, files=2)

    ds = bt.read.files_incremental(str(src), "parquet", state_dir=str(state))
    query = ds.write(
        str(out),
        format="delta",
        trigger=bt.Trigger.processing_time(0.05),
        checkpoint=str(ckpt),
        distributed=True,
        num_workers=2,
    )
    try:
        _wait_for(lambda: len(query.recent_progress) >= 1)
        _land(src, 1, files=2)  # a second arrival, while the same query is still running
        _wait_for(lambda: len(query.recent_progress) >= 2)
    finally:
        query.stop()

    assert not query.is_active
    assert _commits(out) == 2, "each arrival should be its own transaction"
    assert _ids(out) == sorted(i for b in range(2) for i in range(b * 1000, b * 1000 + 20))


def _wait_for(predicate, timeout: float = 30.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("timed out waiting for the stream to make progress")


def test_distributed_streaming_to_iceberg_is_refused_not_downgraded(tmp_path):
    """Iceberg has no transaction-id check, so a replay there would duplicate rows."""
    src, state = tmp_path / "src", tmp_path / "state"
    src.mkdir()
    _land(src, 0, files=1)

    ds = bt.read.files_incremental(str(src), "parquet", state_dir=str(state))
    with pytest.raises(PlanError, match="transaction-id check"):
        ds.write(
            str(tmp_path / "ice"),
            format="iceberg",
            trigger=bt.Trigger.processing_time(0),
            distributed=True,
        )
