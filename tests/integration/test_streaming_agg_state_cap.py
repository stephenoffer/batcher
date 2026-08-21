"""A streaming aggregate with no watermark must fail loudly, not leak for days.

Kyber can already name the shape (`kyber.streaming.retains_unbounded_state`): a *grouped*
aggregate over a stream with no watermark holds one entry per group for the life of the
query. The windowed fold has been capped since it was written; the plain one had no cap at
all, so the query that actually leaks was the one nothing was watching.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import ResourceError
from batcher.config import Config, MemoryConfig, config_context
from batcher.core.streaming import check_agg_state_bounded, streaming_state_budget
from batcher.core.streaming_query import AggregateProcessor
from batcher.plan.logical import Aggregate


def _plan(*, keyed: bool) -> Aggregate:
    ds = bt.from_pydict({"k": [1, 2, 3], "v": [10, 20, 30]})
    agg = ds.group_by("k").agg(n=bt.col("v").sum()) if keyed else ds.agg(n=bt.col("v").sum())
    return agg._plan


def _tiny_budget():
    base = Config()
    return base.replace(memory=dataclasses.replace(base.memory, streaming_state_max_bytes=1))


def test_a_keyed_streaming_aggregate_refuses_to_grow_past_its_envelope():
    with config_context(_tiny_budget()):
        processor = AggregateProcessor(_plan(keyed=True))
        with pytest.raises(ResourceError, match="no watermark"):
            processor.process(pa.record_batch({"k": [1, 2], "v": [1, 2]}))


def test_the_message_names_the_fix_rather_than_just_the_number():
    with config_context(_tiny_budget()):
        processor = AggregateProcessor(_plan(keyed=True))
        with pytest.raises(ResourceError) as excinfo:
            processor.process(pa.record_batch({"k": [1], "v": [1]}))
    message = str(excinfo.value)
    assert "with_watermark" in message
    assert "narrow the group keys" in message
    assert "memory.streaming_state_max_bytes" in message


def test_a_keyless_aggregate_is_never_capped():
    """One row of state, no matter how long the stream runs — there is nothing to bound."""
    with config_context(_tiny_budget()):
        processor = AggregateProcessor(_plan(keyed=False))
        for _ in range(5):
            out = processor.process(pa.record_batch({"k": [1], "v": [1]}))
        assert out and out[0].num_rows == 1


def test_an_ordinary_budget_lets_a_keyed_aggregate_run():
    processor = AggregateProcessor(_plan(keyed=True))
    out = processor.process(pa.record_batch({"k": [1, 2, 1], "v": [1, 2, 3]}))
    assert out and out[0].num_rows == 2


def test_the_shared_check_is_silent_inside_its_budget():
    class _Fold:
        def nbytes(self):
            return 10

    check_agg_state_bounded(_Fold(), 100, "because", label="test")
    with pytest.raises(ResourceError, match="test state reached 10 bytes"):
        check_agg_state_bounded(_Fold(), 5, "because", label="test")


def test_the_budget_helper_reads_the_active_config():
    assert streaming_state_budget() > 0
    with config_context(_tiny_budget()):
        assert streaming_state_budget() == 1


def test_the_windowed_fold_still_names_its_own_cause():
    """The two folds release state differently — one by an advancing watermark, one not at
    all — so they must diagnose differently even though they share the check."""
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
    with config_context(_tiny_budget()):
        fold = _WindowedAggFold(plan, key)
        with pytest.raises(ResourceError, match="watermark on 't' is not advancing"):
            fold.push(pa.record_batch({"t": [base], "v": [1]}))


def test_the_memory_config_knob_is_the_same_one_for_both():
    assert isinstance(MemoryConfig().streaming_state_budget_bytes(), int)


# --- what `update` mode retains beside the fold ----------------------------


def _growing(epochs: int = 8, per_batch: int = 5_000):
    """A stream introducing `per_batch` brand-new group keys every epoch."""
    schema = pa.schema([("k", pa.string()), ("v", pa.int64())])

    def batches():
        for epoch in range(epochs):
            start = epoch * per_batch
            yield pa.record_batch(
                {
                    "k": pa.array([f"key-{start + i:08d}" for i in range(per_batch)]),
                    "v": pa.array([1] * per_batch, type=pa.int64()),
                },
                schema=schema,
            )

    return bt.from_batches(batches, schema, bounded=False)


def _run_update(cap: int | None = None):
    """Run a keyed aggregate in `update` mode, returning its last progress record."""
    import contextlib

    from batcher.config import active_config, config_context

    scope = contextlib.nullcontext()
    if cap is not None:
        config = active_config()
        scope = config_context(
            config.replace(memory=dataclasses.replace(config.memory, streaming_state_max_bytes=cap))
        )
    with scope:
        query = (
            _growing()
            .group_by("k")
            .agg(total=bt.col("v").sum())
            .write.for_each_batch(
                lambda _t, _b: None,
                trigger=bt.Trigger.available_now(),
                output_mode="update",
            )
        )
        query.await_termination()
        if query.exception() is not None:
            raise query.exception()
        return query.recent_progress[-1]


def test_update_mode_counts_the_copy_it_diffs_against():
    """`update` keeps a full copy of the last emitted result to diff against — exactly as
    large as the fold's own state — and nothing counted it. A query configured with a
    one-gigabyte cap therefore held two, and `memory_used_bytes` reported half of what the
    operator was really using."""
    from batcher.core.streaming.folds import streaming_state_budget  # noqa: F401

    progress = _run_update()
    operator = progress.state_operators[0]
    rows = operator.num_rows_total
    assert rows > 0
    # The fold alone is a little over 24 bytes/row for this schema; the reported figure has
    # to be materially above that, because the previous-result copy is the same size again.
    assert operator.memory_used_bytes > rows * 40, (
        f"{operator.memory_used_bytes} bytes for {rows} rows looks like the fold alone, so "
        "the update-mode copy is still uncounted"
    )


def test_the_cap_now_refuses_usage_that_used_to_be_invisible():
    """A budget between the fold's size and the true retained total. Before, this ran to
    completion while using twice the configured envelope; the failure mode was an OOM at
    2x the budget rather than the error the budget exists to produce."""
    unbounded = _run_update().state_operators[0]
    fold_only = unbounded.memory_used_bytes // 2
    with pytest.raises(ResourceError, match="streaming aggregate state"):
        _run_update(cap=int(fold_only * 1.4))


def test_a_complete_mode_aggregate_keeps_no_such_copy():
    """Only `update` diffs, so only `update` pays. Pinned so the accounting is not quietly
    charged to a mode that does not hold the copy."""
    query = (
        _growing(epochs=3)
        .group_by("k")
        .agg(total=bt.col("v").sum())
        .write.memory(
            "cap_probe_complete", output_mode="complete", trigger=bt.Trigger.available_now()
        )
    )
    query.await_termination()
    operator = query.recent_progress[-1].state_operators[0]
    assert operator.memory_used_bytes <= operator.num_rows_total * 40
