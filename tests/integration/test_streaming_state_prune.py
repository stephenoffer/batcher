"""Bounded checkpoint state — a long stateful stream keeps one live snapshot.

A stateful streaming query (grouped aggregation, complete/update output) snapshots its
running state to the checkpoint every micro-batch. Recovery only ever restores the
*latest committed* snapshot, so the engine prunes the older ones after each commit —
the ``state/`` directory stays bounded instead of growing one file per batch forever,
while exactly-once resume is preserved.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.integration


def _state_snapshots(ckpt: str) -> list[str]:
    state_dir = os.path.join(ckpt, "state")
    if not os.path.isdir(state_dir):
        return []
    return [f for f in os.listdir(state_dir) if f.endswith(".arrow")]


def test_state_dir_bounded_across_many_microbatches(tmp_path):
    ckpt = str(tmp_path / "ckpt")

    # 40 rows at 5/batch = 8 micro-batches, each snapshotting the running aggregate.
    q = (
        bt.read.rate(5, num_rows=40, pace=False)
        .group_by(k=bt.col("value") % 3)
        .agg(n=bt.col("value").count())
        .write.memory(
            "prune_agg",
            trigger=bt.Trigger.available_now(),
            output_mode="complete",
            checkpoint=ckpt,
        )
    )
    q.await_termination()

    # Without pruning this would be 8 snapshots; pruning keeps only the live one.
    assert len(_state_snapshots(ckpt)) <= 1

    # The aggregate is still correct: values 0..39 bucketed by v % 3.
    result = {r["k"]: r["n"] for r in bt.read_memory("prune_agg").collect().to_pylist()}
    assert result == {0: 14, 1: 13, 2: 13}


def test_stateful_resume_after_prune_is_exactly_once(tmp_path):
    ckpt = str(tmp_path / "ckpt")

    # Run A: fold 20 rows, pruning as it commits.
    (
        bt.read.rate(5, num_rows=20, pace=False)
        .group_by(k=bt.col("value") % 3)
        .agg(n=bt.col("value").count())
        .write.memory(
            "resume_agg",
            trigger=bt.Trigger.available_now(),
            output_mode="complete",
            checkpoint=ckpt,
        )
    ).await_termination()

    # Run B: the same query over 40 rows resumes from the pruned-but-live snapshot and
    # continues the fold — the final counts cover 0..39 exactly once (no double count).
    (
        bt.read.rate(5, num_rows=40, pace=False)
        .group_by(k=bt.col("value") % 3)
        .agg(n=bt.col("value").count())
        .write.memory(
            "resume_agg",
            trigger=bt.Trigger.available_now(),
            output_mode="complete",
            checkpoint=ckpt,
        )
    ).await_termination()

    result = {r["k"]: r["n"] for r in bt.read_memory("resume_agg").collect().to_pylist()}
    assert result == {0: 14, 1: 13, 2: 13}
    assert len(_state_snapshots(ckpt)) <= 1
