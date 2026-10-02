"""A checkpoint's identity: its stream id, the plan that wrote it, and its one owner.

Three hazards `checkpoint.identity` closes, each of which lost or corrupted data while every
log entry looked correct:

* the Delta sink's exactly-once check keyed on an app id derived from the destination, so
  a second unnamed stream into the same table -- or a rerun without a checkpoint, whose
  batch counter restarts at 0 -- found its batches "already committed" and wrote nothing;
* a stateful restart under a different plan restored state another computation produced;
* two drivers on one checkpoint raced its offsets and commits.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import CommitError, PlanError
from batcher.io.formats.streaming.checkpoint import CheckpointStore, identity
from batcher.io.formats.streaming.checkpoint.identity import CheckpointOwner, stream_app_id

pytestmark = pytest.mark.integration


def _append(values: list[int], table: str, **kw) -> None:
    ds = bt.from_pydict({"v": values})
    ds.write.delta(table, trigger=bt.Trigger.available_now(), **kw).await_termination()


def test_two_unnamed_uncheckpointed_runs_both_land(tmp_path):
    pytest.importorskip("deltalake")
    table = str(tmp_path / "t")
    _append([0, 1, 2], table)
    _append([10, 11, 12], table)
    assert sorted(bt.read.delta(table).to_pydict()["v"]) == [0, 1, 2, 10, 11, 12]


def test_two_unnamed_checkpointed_streams_into_one_table_both_land(tmp_path):
    pytest.importorskip("deltalake")
    table = str(tmp_path / "t")
    _append([0, 1], table, checkpoint=str(tmp_path / "ck_a"))
    _append([5, 6], table, checkpoint=str(tmp_path / "ck_b"))
    assert sorted(bt.read.delta(table).to_pydict()["v"]) == [0, 1, 5, 6]


def test_a_checkpointed_stream_keeps_its_id_across_restarts(tmp_path):
    ck = str(tmp_path / "ck")
    first = stream_app_id(None, ck, "lake/t")
    assert stream_app_id(None, ck, "lake/t") == first
    assert stream_app_id(None, str(tmp_path / "other"), "lake/t") != first
    assert stream_app_id("nightly", ck, "lake/t") == "nightly"


def test_without_a_checkpoint_every_run_gets_its_own_id():
    assert stream_app_id("q", None, "lake/t") != stream_app_id("q", None, "lake/t")


def test_a_checkpoint_from_before_stream_ids_keeps_its_old_id(tmp_path):
    """Its committed batches carry the destination-derived id; a new one would re-append
    the in-flight batch on the first restart after the upgrade."""
    ck = str(tmp_path / "ck")
    store = CheckpointStore(ck)
    store.record_offsets(0, {0: {"offset": 1}})
    store.close()
    assert stream_app_id(None, ck, "lake/t/") == "batcher-stream:lake/t"


def test_a_second_driver_on_a_local_checkpoint_is_refused(tmp_path):
    ck = str(tmp_path / "ck")
    first = CheckpointStore(ck)
    first.claim("fp", stateful=False)
    second = CheckpointStore(ck)
    with pytest.raises(CommitError, match="in use by another running streaming query"):
        second.claim("fp", stateful=False)
    second.close()
    first.close()
    third = CheckpointStore(ck)
    third.claim("fp", stateful=False)  # released with the first store
    third.close()


def test_a_running_query_holds_its_checkpoint(tmp_path):
    ck = str(tmp_path / "ck")
    q = (
        bt.read.rate(5, pace=True)
        .select("value")
        .write.memory("ck_owner", trigger=bt.Trigger.processing_time("1 second"), checkpoint=ck)
    )
    try:
        with pytest.raises(CommitError, match="in use"):
            bt.read.rate(5, pace=True).select("value").write.memory(
                "ck_owner_2", trigger=bt.Trigger.processing_time("1 second"), checkpoint=ck
            )
    finally:
        q.stop()


def test_on_a_remote_store_the_newest_driver_fences_the_older(tmp_path, monkeypatch):
    # An object store has no lock; the owner document is the fence. A local directory
    # stands in for the remote store by taking the non-local branch.
    monkeypatch.setattr(identity, "is_local_location", lambda _loc: False)
    ck = str(tmp_path / "ck")
    old = CheckpointOwner(ck)
    new = CheckpointOwner(ck)
    new.verify()
    with pytest.raises(CommitError, match="no longer owns its checkpoint"):
        old.verify()


def _agg_query(ck: str, name: str, *, keyed_by: int):
    return (
        bt.read.rate(5, num_rows=20, pace=False)
        .group_by(k=bt.col("value") % keyed_by)
        .agg(n=bt.col("value").count())
        .write.memory(
            name, trigger=bt.Trigger.available_now(), output_mode="complete", checkpoint=ck
        )
    )


def test_a_stateful_restart_under_a_different_plan_is_refused(tmp_path):
    ck = str(tmp_path / "ck")
    _agg_query(ck, "plan_a", keyed_by=3).await_termination()
    _agg_query(ck, "plan_a", keyed_by=3).await_termination()  # the same plan resumes
    with pytest.raises(PlanError, match="different streaming plan"):
        _agg_query(ck, "plan_b", keyed_by=4)


def test_a_stateless_plan_may_change_between_runs(tmp_path):
    ck = str(tmp_path / "ck")
    src = bt.read.rate(5, num_rows=10, pace=False)
    src.select("value").write.memory(
        "sl_a", trigger=bt.Trigger.available_now(), checkpoint=ck
    ).await_termination()
    src.filter(bt.col("value") > 2).select("value").write.memory(
        "sl_b", trigger=bt.Trigger.available_now(), checkpoint=ck
    ).await_termination()
