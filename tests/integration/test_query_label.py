"""`observability.query_label`: a caller-chosen name every query in a scope carries.

A batch query had no user-facing name, so "which run was the nightly ETL?" had no answer in
the history, the progress events or a write's manifest. The label is plain config, set in a
scope, and each of those surfaces retains it. Without one, every surface is unchanged.
"""

from __future__ import annotations

import dataclasses

import pytest

import batcher as bt
from batcher._internal import events
from batcher.api.history import query_history
from batcher.api.terminal.event_log import flush_event_log
from batcher.config import active_config, config_context, option_context

pytestmark = pytest.mark.integration

LABEL = "nightly-orders-etl"


def _grouped() -> bt.Dataset:
    """A grouped aggregate: a shape that reaches the executor rather than a metadata answer."""
    return (
        bt.from_pydict({"k": [1, 1, 2], "v": [1.0, 2.0, 3.0]})
        .group_by("k")
        .agg(s=bt.col("v").sum())
    )


@pytest.fixture
def logged(tmp_path, monkeypatch):
    """The event log on and pointed at a fresh directory; yields that directory."""
    monkeypatch.setenv("BATCHER_HOME", str(tmp_path))
    log_dir = tmp_path / "logs"
    base = active_config()
    cfg = base.replace(
        observability=dataclasses.replace(
            base.observability, event_log=True, event_log_dir=str(log_dir)
        )
    )
    with config_context(cfg):
        yield str(log_dir)


def _history_labels(log_dir: str) -> list[object]:
    assert flush_event_log(10.0)
    return query_history(log_dir).collect().column("query_label").to_pylist()


def test_query_history_records_the_label(logged):
    with option_context("observability.query_label", LABEL):
        _grouped().collect()
    _grouped().collect()  # unlabelled
    labels = _history_labels(logged)
    assert len(labels) == 2
    assert sorted(labels, key=str) == sorted([LABEL, None], key=str)


def test_the_column_is_typed_on_an_empty_history(tmp_path):
    history = query_history(str(tmp_path))
    assert str(history.schema.field("query_label").type) == "string"
    assert history.count() == 0


def test_the_query_start_event_carries_the_label():
    seen: list[events.Event] = []
    detach = events.subscribe(seen.append)
    try:
        with option_context("observability.query_label", LABEL):
            _grouped().collect()
        _grouped().collect()
    finally:
        detach()
    starts = [e for e in seen if e.kind == events.QUERY_START]
    assert len(starts) == 2
    assert [e.fields.get("query_label") for e in starts] == [LABEL, ""]


def test_a_streamed_query_announces_the_label():
    seen: list[events.Event] = []
    detach = events.subscribe(seen.append)
    try:
        with option_context("observability.query_label", LABEL):
            list(bt.from_pydict({"a": [1, 2, 3]}).iter_batches())
    finally:
        detach()
    starts = [e for e in seen if e.kind == events.QUERY_START]
    assert starts
    assert all(e.fields.get("query_label") == LABEL for e in starts)


def _fails(batch):
    raise ValueError("bad batch")


def _failing() -> bt.Dataset:
    """A query that fails while executing, not while being built."""
    return bt.from_pydict({"a": [1, 2]}).map_batches(_fails)


def test_a_failing_query_names_its_label():
    with (
        option_context("observability.query_label", LABEL),
        pytest.raises(Exception, match="bad batch") as info,
    ):
        _failing().collect()
    assert f"query_label: {LABEL}" in getattr(info.value, "__notes__", [])


def test_an_unlabelled_failure_gains_no_note():
    with pytest.raises(Exception, match="bad batch") as info:
        _failing().collect()
    assert not any("query_label" in n for n in getattr(info.value, "__notes__", []))


def test_a_write_manifest_carries_the_label(tmp_path):
    with option_context("observability.query_label", LABEL):
        manifest = _grouped().write.parquet(str(tmp_path / "labelled.parquet"))
    assert manifest.query_label == LABEL
    plain = _grouped().write.parquet(str(tmp_path / "plain.parquet"))
    assert plain.query_label == ""
