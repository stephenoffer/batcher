"""A stream-stream join can be written to a sink, not only iterated.

Until this worked, `joined.write.delta(...)` raised and the *only* way to consume an
interval join was `iter_batches()` — which the streaming cookbook called the sharpest
edge in Batcher's streaming story, because it meant every restart story for a joined
stream was the user's own code.

The rows written must be the rows `iter_batches()` yields: the driver is shared, so this
checks that the seam did not introduce a second definition of what the join means.
"""

from __future__ import annotations

import datetime
import os

import pyarrow as pa
import pytest

import batcher as bt

_LEFT_SCHEMA = pa.schema([("k", pa.int64()), ("lt", pa.timestamp("us")), ("lv", pa.string())])
_RIGHT_SCHEMA = pa.schema([("k", pa.int64()), ("rt", pa.timestamp("us")), ("rv", pa.string())])
_BASE = datetime.datetime(2024, 1, 1)


def _left_feed():
    yield pa.record_batch(
        {"k": [1, 2], "lt": [_BASE, _BASE], "lv": ["a", "b"]}, schema=_LEFT_SCHEMA
    )


def _right_feed():
    yield pa.record_batch({"k": [1], "rt": [_BASE], "rv": ["A"]}, schema=_RIGHT_SCHEMA)


def _joined(how: str = "inner"):
    left = bt.from_batches(_left_feed, _LEFT_SCHEMA, bounded=False)
    right = bt.from_batches(_right_feed, _RIGHT_SCHEMA, bounded=False)
    return left.join_stream(right, on="k", left_time="lt", right_time="rt", within="10m", how=how)


def _pairs(table) -> list[tuple[str, str]]:
    data = table.to_pydict()
    return sorted((str(a), str(b)) for a, b in zip(data["lv"], data["rv"], strict=True))


@pytest.mark.integration
def test_an_inner_stream_join_writes_its_matched_pairs():
    query = _joined().write.memory("sj_inner", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    assert query.exception() is None
    assert _pairs(bt.read_memory("sj_inner")) == [("a", "A")]


@pytest.mark.integration
def test_an_outer_stream_join_writes_its_null_padded_rows_too():
    query = _joined("left").write.memory("sj_left", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    assert _pairs(bt.read_memory("sj_left")) == [("a", "A"), ("b", "None")]


@pytest.mark.integration
def test_the_sink_receives_exactly_what_iter_batches_yields():
    """One driver, two consumers — not a second definition of the join."""
    iterated: list[dict] = []
    for batch in _joined("left").iter_batches():
        iterated.extend(batch.to_pylist())

    query = _joined("left").write.memory("sj_parity", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True

    written = bt.read_memory("sj_parity").to_pydict()
    assert sorted((str(r["lv"]), str(r["rv"])) for r in iterated) == sorted(
        (str(a), str(b)) for a, b in zip(written["lv"], written["rv"], strict=True)
    )


@pytest.mark.integration
def test_it_writes_files_too_not_only_the_memory_sink(tmp_path):
    out = str(tmp_path / "joined")
    query = _joined().write(out, format="parquet", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    assert query.exception() is None
    assert os.listdir(out), "the file sink wrote nothing"
    assert _pairs(bt.read(out, format="parquet").collect()) == [("a", "A")]


@pytest.mark.integration
def test_progress_is_recorded_per_micro_batch():
    query = _joined("left").write.memory("sj_progress", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    progress = query.recent_progress
    assert progress, "a driver-produced stream recorded no progress at all"
    assert sum(p.num_output_rows for p in progress) == 2


@pytest.mark.integration
def test_a_checkpoint_is_refused_rather_than_silently_useless(tmp_path):
    """The join's state is two buffers and two watermarks, none of it offset-addressable.
    Accepting a checkpoint would restart from an empty join while looking like recovery."""
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="no checkpointable position"):
        _joined().write.memory(
            "sj_ckpt", trigger=bt.Trigger.available_now(), checkpoint=str(tmp_path / "ckpt")
        )


@pytest.mark.integration
def test_an_aggregating_output_mode_is_refused_by_name():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="needs an aggregation"):
        _joined().write.memory(
            "sj_mode", trigger=bt.Trigger.available_now(), output_mode="complete"
        )
