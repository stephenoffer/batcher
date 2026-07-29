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
    alias, width = _window_key(plan)
    with config_context(_tiny_budget()):
        fold = _WindowedAggFold(plan, alias, width)
        with pytest.raises(ResourceError, match="watermark on 't' is not advancing"):
            fold.push(pa.record_batch({"t": [base], "v": [1]}))


def test_the_memory_config_knob_is_the_same_one_for_both():
    assert isinstance(MemoryConfig().streaming_state_budget_bytes(), int)
