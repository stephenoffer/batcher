"""Workstream D — the checkpoint offset/commit logs and recovery decision (unit)."""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.io.formats.streaming.checkpoint import CheckpointStore, recover

pytestmark = pytest.mark.unit


def test_offset_and_commit_logs_round_trip(tmp_path):
    store = CheckpointStore(str(tmp_path / "ckpt"))
    store.record_offsets(0, {0: {"value": 5}})
    store.record_offsets(1, {0: {"value": 10}})
    store.commit(0, "tok0")
    assert store.offsets.latest_batch() == 1
    assert store.commits.last_committed() == 0
    assert store.commits.is_committed(0) and not store.commits.is_committed(1)
    assert store.offsets.position_at(1) == {0: {"value": 10}}


def test_recover_fresh_query(tmp_path):
    store = CheckpointStore(str(tmp_path / "ckpt"))
    plan = recover(store)
    assert plan.start_batch == 0 and plan.seek == {} and plan.state is None


def test_recover_resumes_after_last_commit(tmp_path):
    store = CheckpointStore(str(tmp_path / "ckpt"))
    for b in range(3):
        store.record_offsets(b, {0: {"value": (b + 1) * 5}})
        store.commit(b)
    # An in-flight (recorded but uncommitted) batch 3.
    store.record_offsets(3, {0: {"value": 20}})
    plan = recover(store)
    assert plan.start_batch == 3  # first uncommitted
    assert plan.seek == {0: {"value": 15}}  # position recorded at the last commit (b=2)


def test_state_snapshot_round_trip(tmp_path):
    store = CheckpointStore(str(tmp_path / "ckpt"))
    state = pa.record_batch({"k": [1, 2], "s": [10, 20]})
    store.record_offsets(4, {0: {"value": 99}})
    store.snapshot_state(4, state)
    store.commit(4)
    restored = store.state.restore(4)
    assert restored.to_pydict() == {"k": [1, 2], "s": [10, 20]}
    # recover() restores the snapshot taken at the last committed batch.
    assert recover(store).state.to_pydict() == {"k": [1, 2], "s": [10, 20]}
