"""Incremental (changelog) state checkpointing, and the invariant that makes it sound.

Rewriting the whole running state every micro-batch makes a checkpoint's cost grow with the
state it protects: an aggregate with no watermark never evicts, so a query holding ten
million groups wrote ten million rows on every trigger, forever, with the fsyncs on the
critical path of every epoch.

A delta records the *partial* the micro-batch folded in instead — bounded by the batch's
distinct group count, not the query's — and recovery combines the newest snapshot with the
deltas after it. That is sound for exactly one reason: `combine` is associative and
commutative (invariant #7), so combining a base with every partial recorded after it is the
same state the full snapshot would have held.

The load-bearing test here is `test_a_delta_chain_recovers_the_same_state_a_snapshot_would`.
Everything else guards an edge where the chain could silently lose or double-count rows —
and "silently" is the word: a chain with a hole in it produces short totals, never an error.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import active_config, option_context
from batcher.io.formats.streaming.checkpoint.store import CheckpointStore

pytestmark = pytest.mark.integration


@pytest.fixture
def always_delta():
    """Force every eligible epoch to record a delta, so a chain actually forms."""
    with option_context("streaming.checkpoint_delta_interval", 1000):
        yield


@pytest.fixture
def never_delta():
    """Disable incremental checkpointing — the whole-snapshot behaviour, for comparison."""
    with option_context("streaming.checkpoint_delta_interval", 0):
        yield


def _run(
    ckpt: str, rows: int, *, per_batch: int = 4, buckets: int | None = 3
) -> list[tuple[int, int, int]]:
    """Fold `rows` of the rate source into a keyed running aggregate against `ckpt`.

    An unwatermarked aggregate is exactly the shape incremental checkpointing is for:
    nothing ever closes a group, so the state only grows and a whole-snapshot checkpoint gets
    more expensive every epoch. `append` is refused for it — no watermark can make a row
    final — so it runs in `update` mode into `for_each_batch`, which is the sink shape a
    running aggregate has.

    Args:
        ckpt: The checkpoint location to resume from and record into.
        rows: How many rows the (bounded, replayable) rate source offers this run.
        per_batch: Rows per micro-batch, which sets how many epochs the run takes.
        buckets: How many distinct group keys the stream has, or None for one per row —
            the high-cardinality shape where a delta is genuinely smaller than the state,
            and so the only shape the size rule lets a delta be written for.

    Returns:
        The last emitted value per bucket, as sorted ``(bucket, total, n)`` triples.
    """
    latest: dict[int, tuple[int, int]] = {}

    def collect(table: pa.Table, _batch_id: int) -> None:
        emitted = table.to_pydict()
        for bucket, total, n in zip(emitted["bucket"], emitted["total"], emitted["n"], strict=True):
            latest[bucket] = (total, n)

    query = (
        bt.read.rate(per_batch, num_rows=rows, pace=False)
        .with_columns(bucket=bt.col("value") if buckets is None else bt.col("value") % buckets)
        .group_by("bucket")
        .agg(total=bt.col("value").sum(), n=bt.col("value").count())
        .write.for_each_batch(
            collect,
            trigger=bt.Trigger.available_now(),
            checkpoint=ckpt,
            output_mode="update",
        )
    )
    query.await_termination()
    return sorted((bucket, total, n) for bucket, (total, n) in latest.items())


# --- the invariant ---------------------------------------------------------


def test_a_delta_chain_recovers_the_same_state_a_snapshot_would(tmp_path, always_delta):
    """The whole feature in one assertion: restarting off a chain must land on the same
    aggregate as restarting off a whole snapshot. A chain that dropped or double-counted a
    partial shows up here as a wrong total and nowhere else."""
    inc_ckpt = str(tmp_path / "inc-ckpt")
    _run(inc_ckpt, rows=12)
    incremental = _run(inc_ckpt, rows=24)

    whole_ckpt = str(tmp_path / "whole-ckpt")
    with option_context("streaming.checkpoint_delta_interval", 0):
        _run(whole_ckpt, rows=12)
        whole = _run(whole_ckpt, rows=24)

    assert incremental == whole


def test_the_recovered_aggregate_matches_a_single_uninterrupted_run(tmp_path, always_delta):
    """The oracle a restart is actually held to: the same numbers as never having stopped."""
    restart_ckpt = str(tmp_path / "r-ckpt")
    _run(restart_ckpt, rows=12)
    restarted = _run(restart_ckpt, rows=24)

    straight = _run(str(tmp_path / "s-ckpt"), rows=24)

    assert restarted == straight


def test_a_restart_across_several_deltas_still_lands_on_the_right_total(tmp_path, always_delta):
    """Three runs, so the second restart resumes from a chain the first restart wrote."""
    ckpt = str(tmp_path / "ckpt")
    _run(ckpt, rows=8)
    _run(ckpt, rows=16)
    stepped = _run(ckpt, rows=24)

    assert stepped == _run(str(tmp_path / "straight"), rows=24)


def test_incremental_checkpointing_is_what_the_default_config_does():
    """Off by default would make this a feature nobody gets. The size rule is what makes
    that safe — see `state_policy.worth_a_delta`."""
    assert active_config().streaming.checkpoint_delta_interval > 0


# --- what actually lands on disk -------------------------------------------


def _state_files(ckpt: str) -> tuple[list[str], list[str]]:
    """``(snapshots, deltas)`` currently in a checkpoint's state directory."""
    root = os.path.join(ckpt, "state")
    if not os.path.isdir(root):
        return [], []
    names = sorted(n for n in os.listdir(root) if n.endswith(".arrow"))
    return (
        [n for n in names if not n.endswith(".delta.arrow")],
        [n for n in names if n.endswith(".delta.arrow")],
    )


def test_a_high_cardinality_aggregate_records_deltas_not_whole_snapshots(tmp_path, always_delta):
    """The shape the feature exists for: state grows with every batch while each batch
    touches only its own handful of keys, so the delta is a small fraction of the state.

    Observed *during* the run, not after. A draining trigger ends with a checkpoint marker
    that writes a whole snapshot and prunes the chain under it — correct, and it means the
    final directory of a finished `available_now` run never shows a delta whether or not one
    was ever written.
    """
    ckpt = str(tmp_path / "ckpt")
    seen: list[int] = []

    def watch(_table: pa.Table, _batch_id: int) -> None:
        seen.append(len(_state_files(ckpt)[1]))

    query = (
        bt.read.rate(4, num_rows=200, pace=False)
        .with_columns(bucket=bt.col("value"))
        .group_by("bucket")
        .agg(total=bt.col("value").sum(), n=bt.col("value").count())
        .write.for_each_batch(
            watch,
            trigger=bt.Trigger.available_now(),
            checkpoint=ckpt,
            output_mode="update",
        )
    )
    query.await_termination()

    assert max(seen) > 1, (
        f"no chain ever formed (deltas per epoch: {seen}), so the epoch cost still scales "
        "with the state"
    )


def test_a_low_cardinality_aggregate_keeps_whole_snapshots(tmp_path, always_delta):
    """The size rule refusing a delta is not a missed optimization, it is the guard that
    keeps this change from ever costing more than it saves: with three groups the delta
    *is* the state, so a chain would write it repeatedly and then a snapshot as well."""
    ckpt = str(tmp_path / "ckpt")
    _run(ckpt, rows=40, per_batch=4, buckets=3)
    snapshots, deltas = _state_files(ckpt)
    assert snapshots and not deltas


def test_the_high_cardinality_chain_also_recovers_to_the_uninterrupted_answer(
    tmp_path, always_delta
):
    """Deltas are only actually exercised at high cardinality, so the recovery invariant
    has to be checked there too and not only where the size rule declines them."""
    ckpt = str(tmp_path / "ckpt")
    _run(ckpt, rows=100, per_batch=4, buckets=None)
    restarted = _run(ckpt, rows=200, per_batch=4, buckets=None)

    assert restarted == _run(str(tmp_path / "straight"), rows=200, per_batch=4, buckets=None)


def test_disabling_the_interval_restores_whole_snapshots(tmp_path, never_delta):
    ckpt = str(tmp_path / "ckpt")
    _run(ckpt, rows=40, per_batch=4)
    snapshots, deltas = _state_files(ckpt)
    assert snapshots and not deltas


def test_the_state_directory_stays_bounded_across_many_micro_batches(tmp_path):
    """A chain that is never collapsed is a directory that grows one file per trigger, so
    the interval has to actually force a snapshot and pruning has to actually run."""
    ckpt = str(tmp_path / "ckpt")
    with option_context("streaming.checkpoint_delta_interval", 5):
        _run(ckpt, rows=200, per_batch=2, buckets=None)
    snapshots, deltas = _state_files(ckpt)
    assert len(snapshots) <= 2, snapshots
    assert len(deltas) <= 6, deltas


# --- the store's own contracts ---------------------------------------------


def _batch(value: int) -> pa.RecordBatch:
    return pa.record_batch({"k": pa.array([value], type=pa.int64())})


def _chain(store: CheckpointStore, batch_id: int) -> list[int]:
    return [b.column("k")[0].as_py() for b in store.state.restore_chain(batch_id)]


def test_pruning_never_deletes_the_base_a_delta_still_depends_on(tmp_path):
    """Deleting the base and keeping the deltas leaves recovery combining partials with
    nothing under them — a fraction of the aggregate, silently."""
    store = CheckpointStore(str(tmp_path / "ckpt"))
    store.state.snapshot(5, _batch(5))
    store.state.snapshot_delta(6, _batch(6))
    store.state.snapshot_delta(7, _batch(7))
    store.state.prune(7)

    assert _chain(store, 7) == [5, 6, 7]


def test_pruning_drops_a_base_superseded_by_a_newer_snapshot(tmp_path):
    store = CheckpointStore(str(tmp_path / "ckpt"))
    store.state.snapshot(1, _batch(1))
    store.state.snapshot_delta(2, _batch(2))
    store.state.snapshot(3, _batch(3))
    store.state.prune(3)

    assert _chain(store, 3) == [3]
    assert _state_files(str(tmp_path / "ckpt")) == (["batch-00000003.arrow"], [])


def test_a_chain_excludes_deltas_from_an_uncommitted_epoch(tmp_path):
    """A recorded-but-uncommitted epoch is about to be re-read, so replaying its delta
    would fold the same micro-batch in twice."""
    store = CheckpointStore(str(tmp_path / "ckpt"))
    store.state.snapshot(1, _batch(1))
    store.state.snapshot_delta(2, _batch(2))
    store.state.snapshot_delta(3, _batch(3))

    assert _chain(store, 2) == [1, 2]


def test_a_chain_resolves_to_the_newest_snapshot_at_or_before_the_batch(tmp_path):
    """A delta-checkpointed epoch writes no snapshot of its own, so an exact-name lookup
    would find nothing and resume a stateful query with empty state."""
    store = CheckpointStore(str(tmp_path / "ckpt"))
    store.state.snapshot(4, _batch(4))
    store.state.snapshot_delta(5, _batch(5))

    assert _chain(store, 5) == [4, 5]


def test_an_idle_epoch_that_wrote_nothing_still_recovers_the_last_state(tmp_path):
    store = CheckpointStore(str(tmp_path / "ckpt"))
    store.state.snapshot(2, _batch(2))

    assert _chain(store, 9) == [2]


def test_a_checkpoint_with_no_state_recovers_nothing_rather_than_raising(tmp_path):
    store = CheckpointStore(str(tmp_path / "ckpt"))
    assert store.state.restore_chain(3) == []
    assert store.state.restore(3) is None


def test_a_delta_is_refused_when_it_is_not_actually_smaller_than_the_state(tmp_path):
    """Without this rule a stream whose every batch touches every group writes an
    interval's worth of state-sized deltas *and* a snapshot — worse than before."""
    from batcher.core.streaming_query.state_policy import worth_a_delta

    small = pa.record_batch({"k": pa.array(range(10), type=pa.int64())})
    big = pa.record_batch({"k": pa.array(range(1000), type=pa.int64())})
    assert worth_a_delta(small, big)
    assert not worth_a_delta(big, big)
    assert not worth_a_delta(small, None)


# --- the safety property: only a monotone fold may use a chain ---------------


def test_a_processor_offers_a_delta_only_if_its_removals_are_expressible():
    """A chain records what was folded *in*. An operator may use one only if what it takes
    *out* can be replayed too, and there are exactly two ways for that to be true:

    * it never removes anything — the unwatermarked running aggregate; or
    * every removal is a **prefix** of a totally ordered axis, so the whole tombstone is one
      bound. That is the windowed aggregate: eviction drops every window at or below a
      threshold, and the threshold rides in each entry's metadata.

    Keyed state is neither. Its TTL expires arbitrary keys at arbitrary times, so there is no
    bound that describes what went — it would need a real tombstone per key, which this
    changelog has no way to carry. If it ever gains `snapshot_delta` without gaining that,
    nothing else in the suite fails: the symptom is state that comes back from the dead and
    is emitted a second time.
    """
    from batcher.core.streaming_query.processors import (
        AggregateProcessor,
        KeyedStateProcessor,
        StatelessProcessor,
        WindowedAggregateProcessor,
    )

    for expressible in (AggregateProcessor, WindowedAggregateProcessor):
        assert hasattr(expressible, "snapshot_delta"), (
            f"{expressible.__name__} can express its removals and should not be paying for a "
            "whole snapshot every epoch"
        )
    for inexpressible in (KeyedStateProcessor, StatelessProcessor):
        assert not hasattr(inexpressible, "snapshot_delta"), (
            f"{inexpressible.__name__} removes state a chain cannot describe, so replaying "
            "one would resurrect it"
        )


def test_offering_a_delta_and_being_able_to_replay_one_go_together():
    """Half of the pair is worse than neither: a processor that writes deltas but cannot
    combine them recovers from the base alone and silently drops every epoch after it."""
    from batcher.core.streaming_query.processors import AggregateProcessor

    assert hasattr(AggregateProcessor, "restore_state_chain")


def test_a_windowed_query_still_recovers_across_a_restart(tmp_path):
    """The fold that must *not* use deltas still has to survive a restart unchanged."""
    ckpt = str(tmp_path / "ckpt")
    seen: list[tuple] = []

    def run(rows: int) -> None:
        query = (
            bt.read.rate_micro_batch(4, num_rows=rows, pace=False, advance_ms_per_batch=1000)
            .with_watermark("timestamp", "1 second")
            .group_by(w=bt.window(bt.col("timestamp"), "2 seconds"))
            .agg(total=bt.col("value").sum())
            .write.for_each_batch(
                lambda table, _b: seen.extend(zip(*table.to_pydict().values(), strict=True)),
                trigger=bt.Trigger.available_now(),
                checkpoint=ckpt,
            )
        )
        query.await_termination()

    run(12)
    run(24)
    assert len(seen) == len(set(seen)), f"a window was emitted twice: {seen}"


def test_every_runner_that_can_hold_state_can_read_a_chain():
    """A checkpoint is portable across runners, so a single-node query restarted with
    `distributed=True` resumes from the chain the single-node run left behind.

    Writing deltas is opt-in; *reading* them cannot be, or that restart silently resumes
    with a fraction of its aggregate — short totals, no error, and only on the restart that
    changed configuration.
    """
    from batcher.core.streaming_runner import LocalRunner
    from batcher.dist.streaming.microbatch import DistributedRunner

    for runner in (LocalRunner, DistributedRunner):
        assert hasattr(runner, "restore_state_chain"), (
            f"{runner.__name__} cannot read a checkpoint another runner may have written"
        )


def test_a_batch_id_never_holds_both_a_snapshot_and_a_delta(tmp_path):
    """A crash can leave the other kind of file behind for the same id: run 1 recorded a
    delta for batch 58 and died before committing, run 2 reprocessed 58 and snapshotted.
    Which one a chain picked would then depend on the file naming rather than on anything
    meaningful, so the write that lands clears its counterpart."""
    store = CheckpointStore(str(tmp_path / "ckpt"))
    store.state.snapshot_delta(58, _batch(1))
    store.state.snapshot(58, _batch(2))
    assert _state_files(str(tmp_path / "ckpt")) == (["batch-00000058.arrow"], [])
    assert _chain(store, 58) == [2]

    store.state.snapshot_delta(58, _batch(3))
    assert _state_files(str(tmp_path / "ckpt")) == ([], ["batch-00000058.delta.arrow"])


def test_the_chain_length_survives_a_restart(tmp_path):
    """A restart that reset the delta count would write another full interval on top of the
    chain it just replayed, so the chain grows by an interval per restart — and a query
    restarted often enough replays an unbounded one."""
    ckpt = str(tmp_path / "ckpt")
    with option_context("streaming.checkpoint_delta_interval", 4):
        for _ in range(6):
            _run(ckpt, rows=len(_state_files(ckpt)[1]) * 8 + 40, per_batch=2, buckets=None)
    _snapshots, deltas = _state_files(ckpt)
    assert len(deltas) <= 4, deltas


# --- the exactly-once boundary ---------------------------------------------


def test_a_crash_between_the_delta_and_the_commit_does_not_double_count(tmp_path):
    """The ordering the durability rules exist for, exercised on the chain.

    The engine records state and *then* commits, so a crash in between leaves a delta on
    disk for an epoch the commit log never acknowledged. Recovery must resume *before* that
    epoch and re-read it — and must not also replay its delta, or the micro-batch is folded
    in twice and every group it touched is silently doubled.
    """
    from batcher.io.formats.streaming.checkpoint.recovery import recover

    ckpt = str(tmp_path / "ckpt")
    store = CheckpointStore(ckpt)
    # Two epochs land cleanly...
    store.record_offsets(0, {0: {"value": 8}})
    store.snapshot_state(0, _batch(10))
    store.commit(0)
    store.record_offsets(1, {0: {"value": 16}})
    store.snapshot_state_delta(1, _batch(20))
    store.commit(1)
    # ...and the third is recorded but the process dies before the commit.
    store.record_offsets(2, {0: {"value": 24}})
    store.snapshot_state_delta(2, _batch(40))

    plan = recover(store)

    assert plan.start_batch == 2, "the uncommitted epoch is re-read"
    assert plan.seek == {0: {"value": 16}}, "from where the last committed epoch stopped"
    chain = [plan.state, *plan.state_deltas]
    assert [b.column("k")[0].as_py() for b in chain] == [10, 20], (
        "the uncommitted epoch's delta must not be replayed — it is about to be re-read"
    )


def test_recovery_reports_a_chain_the_engine_can_replay_in_order(tmp_path):
    """`ResumePlan` splits base from deltas rather than pre-combining them, because
    combining partials is the aggregate algebra and this layer is neutral."""
    from batcher.io.formats.streaming.checkpoint.recovery import recover

    store = CheckpointStore(str(tmp_path / "ckpt"))
    store.record_offsets(0, {0: {"value": 1}})
    store.snapshot_state(0, _batch(1))
    store.commit(0)
    for batch_id in (1, 2, 3):
        store.record_offsets(batch_id, {0: {"value": batch_id}})
        store.snapshot_state_delta(batch_id, _batch(batch_id * 10))
        store.commit(batch_id)

    plan = recover(store)

    assert plan.state.column("k")[0].as_py() == 1
    assert [b.column("k")[0].as_py() for b in plan.state_deltas] == [10, 20, 30]
