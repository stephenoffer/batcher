"""The streaming API is consolidated with the batch API.

Every streaming construct is a method on the same `Dataset` (or the same `bt.read` /
`ds.write` namespace) and works on a *bounded* source with batch semantics — so the
identical pipeline runs as a one-off job or a continuous query with no rewrite. These
tests pin both the public surface and that batch/streaming parity.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.integration

_BASE = dt.datetime(2024, 1, 1)


def _ds():
    return bt.from_pydict(
        {
            "k": ["a", "b", "a"],
            "ts": [_BASE, _BASE + dt.timedelta(minutes=30), _BASE + dt.timedelta(minutes=90)],
            "v": [1, 2, 3],
        }
    )


def test_public_surface_is_present():
    for name in ("Trigger", "OutputMode", "window", "streams", "read_memory"):
        assert hasattr(bt, name), f"bt.{name} missing"
    ds = _ds()
    for m in (
        "with_watermark",
        "session_window",
        "drop_duplicates_within_watermark",
        "join_stream",
        "iter_batches",
        "is_streaming",
    ):
        assert hasattr(ds, m), f"Dataset.{m} missing"
    for s in ("console", "memory", "for_each_batch", "for_each", "delta", "parquet"):
        assert hasattr(ds.write, s), f"ds.write.{s} missing"
    for r in ("kafka", "kinesis", "files_incremental", "rate", "socket", "read_change_feed"):
        assert hasattr(bt.read, r), f"bt.read.{r} missing"


def test_window_runs_on_bounded_source():
    out = _ds().group_by(w=bt.window(col("ts"), "1h")).agg(t=col("v").sum()).to_pydict()
    assert dict(zip(out["w"], out["t"], strict=True)) == {
        _BASE: 3,
        _BASE + dt.timedelta(hours=1): 3,
    }


def test_watermark_window_runs_on_bounded_source():
    # `.with_watermark` is a no-op-friendly annotation on a bounded source.
    out = (
        _ds()
        .with_watermark("ts", "10m")
        .group_by(w=bt.window(col("ts"), "1h"))
        .agg(t=col("v").sum())
        .to_pydict()
    )
    assert sorted(out["t"]) == [3, 3]


def test_session_window_runs_on_bounded_source():
    out = _ds().session_window("ts", "1h", t=col("v").sum()).to_pydict()
    assert out["t"] == [6]  # all within 1h gaps → one session


def test_dedup_runs_on_bounded_source():
    out = _ds().drop_duplicates_within_watermark(["k"], event_time="ts", lateness="1h").to_pydict()
    assert sorted(out["k"]) == ["a", "b"]


def test_join_stream_runs_on_bounded_source():
    right = bt.from_pydict({"k": ["a"], "ts2": [_BASE + dt.timedelta(minutes=20)], "rv": [9]})
    out = (
        _ds().join_stream(right, on="k", left_time="ts", right_time="ts2", within="1h").to_pydict()
    )
    assert out["rv"] == [9]  # the a@0 row joins a@20 within 1h


def test_write_returns_manifest_for_bounded_no_trigger(tmp_path):
    from batcher.io.manifest import WriteManifest

    manifest = _ds().write(str(tmp_path / "out.parquet"))
    assert isinstance(manifest, WriteManifest)


def test_write_returns_streaming_query_for_trigger(tmp_path):
    from batcher.api.streaming import StreamingQuery

    q = _ds().write(str(tmp_path / "out"), format="parquet", trigger=bt.Trigger.available_now())
    q.await_termination()
    assert isinstance(q, StreamingQuery)
