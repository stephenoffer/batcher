"""Edges of the observability surfaces: query history, OpenLineage run identity, metadata.

Each class here pins a defect that passed every existing test while being wrong:

* `query_history()` was implemented, documented in its own docstring, and unreachable —
  `bt.query_history` raised `AttributeError`, so its doctest failed.
* A failed query's FAIL event was emitted under the job name `batcher.query` while its
  START used `batcher.query.<signature>`, so a lineage backend recorded one open run on the
  real job and one orphan failure on a job nothing else ever ran.
* `explain(analyze=True)` and `stats()` posted a COMPLETE with no START before it.
* Every in-memory input was named `<source 0>`, so two unrelated in-memory relations were
  the same dataset in a lineage backend.
* `MetadataConfig(backend="layered")` and `"rocksdb"` were rejected by config validation
  while the backend factory built both, so two documented backends were unreachable.

The lineage events are captured by replacing `_emit`, the one hand-off point to the drain
thread. `tests/integration/test_openlineage.py` covers the HTTP transport itself against a
real receiver; what is under test here is the *content* of the events, which a synchronous
capture checks without a race against the background thread.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from typing import Any

import pytest

import batcher as bt
from batcher.api.terminal import lineage
from batcher.config import Config, MetadataConfig, active_config, config_context

pytestmark = pytest.mark.unit


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A temporary `$BATCHER_HOME`, so nothing a test runs writes under `~/.batcher`."""
    monkeypatch.setenv("BATCHER_HOME", str(tmp_path))
    return tmp_path


def _observed(tmp_path, **fields: Any) -> Config:
    """The active config with the event log on, pointed at `tmp_path`, and `fields` set."""
    base = active_config()
    return base.replace(
        observability=dataclasses.replace(
            base.observability,
            event_log=True,
            event_log_dir=str(tmp_path / "logs"),
            **fields,
        )
    )


@pytest.fixture()
def captured(home, monkeypatch):
    """OpenLineage switched on, with every event captured synchronously instead of posted."""
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(lineage, "_emit", events.append)
    cfg = _observed(home, openlineage=True, openlineage_url="http://127.0.0.1:9")
    with config_context(cfg):
        yield events


def _executing(ds: bt.Dataset) -> bt.Dataset:
    """A grouped aggregate over `ds`: a shape that reaches the executor.

    A keyless aggregate over in-memory data is answered from metadata and never executes,
    so it opens no lineage run and writes no event-log document.
    """
    return ds.group_by("k").agg(s=bt.col("v").sum())


def _by_type(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [e for e in events if e["eventType"] == kind]


class TestQueryHistoryIsPublic:
    def test_it_is_reachable_from_the_package(self):
        from batcher.api.history import query_history

        assert bt.query_history is query_history
        assert "query_history" in bt.__all__

    def test_it_reads_what_the_event_log_wrote(self, home):
        with config_context(_observed(home)):
            _executing(bt.from_pydict({"k": [1, 1, 2], "v": [1, 2, 3]})).collect()
            history = bt.query_history()
        assert history.count() == 1
        [path] = history.to_pydict()["profile_path"]
        assert path.startswith(str(home / "logs"))


class TestLineageRunIdentity:
    def test_a_failed_query_fails_the_job_it_started(self, captured):
        def _boom(batch):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            bt.from_pydict({"x": [1, 2]}).map_batches(_boom).collect()

        [start] = _by_type(captured, "START")
        [fail] = _by_type(captured, "FAIL")
        assert start["run"]["runId"] == fail["run"]["runId"]
        # The positive control: START names the plan signature, so the equality below
        # is not two fallbacks agreeing.
        assert start["job"]["name"].startswith("batcher.query.")
        assert fail["job"]["name"] == start["job"]["name"]

    def test_a_completed_query_completes_the_job_it_started(self, captured):
        _executing(bt.from_pydict({"k": [1, 1, 2], "v": [1, 2, 3]})).collect()

        [start] = _by_type(captured, "START")
        [complete] = _by_type(captured, "COMPLETE")
        assert start["run"]["runId"] == complete["run"]["runId"]
        assert start["job"]["name"] == complete["job"]["name"]

    @pytest.mark.parametrize("terminal", ["explain", "stats"])
    def test_a_profiled_run_opens_the_run_it_completes(self, captured, terminal):
        ds = _executing(bt.from_pydict({"k": [1, 1, 2], "v": [1, 2, 3]}))
        if terminal == "explain":
            ds.explain(analyze=True)
        else:
            ds.stats()

        [start] = _by_type(captured, "START")
        [complete] = _by_type(captured, "COMPLETE")
        assert start["run"]["runId"] == complete["run"]["runId"]
        assert start["job"]["name"] == complete["job"]["name"]

    def test_the_remembered_job_names_are_bounded(self, captured, monkeypatch):
        monkeypatch.setattr(lineage, "_JOB_NAMES_KEPT", 4)
        monkeypatch.setattr(lineage, "_job_names", OrderedDict())
        ds = bt.from_pydict({"x": [1]})
        for i in range(10):
            lineage.emit_run_start(f"q{i}", ds._plan, ds._sources)
        assert len(lineage._job_names) == 4


class TestInMemoryDatasetNames:
    def test_two_queries_over_different_data_read_two_datasets(self, captured):
        # One input each, so the positional index cannot tell them apart: that is how every
        # in-memory input in the backend used to be the one dataset `<source 0>`.
        _executing(bt.from_pydict({"k": [1, 2], "v": [1, 2]})).collect()
        _executing(bt.from_pydict({"k": [7, 8], "v": [5, 6]})).collect()

        starts = _by_type(captured, "START")
        names = [i["name"] for s in starts for i in s["inputs"]]
        assert len(names) == 2
        assert len(set(names)) == 2, names

    def test_the_same_in_memory_input_keeps_its_name(self):
        ds = bt.from_pydict({"x": [1]})
        other = bt.from_pydict({"x": [1]})
        first = lineage._source_names(ds._sources)
        assert lineage._source_names(ds._sources) == first
        # Same schema and same rows is still a different relation.
        assert lineage._source_names(other._sources) != first

    def test_a_file_backed_input_is_named_by_its_path(self, tmp_path):
        path = str(tmp_path / "t.parquet")
        bt.from_pydict({"x": [1]}).write(path, format="parquet")
        assert lineage._source_names(bt.read.parquet(path)._sources) == [path]


class TestEveryFactoryBackendIsConfigurable:
    def test_validation_and_factory_name_the_same_backends(self):
        from batcher.config.validation.sections import METADATA_BACKENDS
        from batcher.metadata.backends import BACKEND_NAMES

        assert set(METADATA_BACKENDS) == set(BACKEND_NAMES)

    def test_layered_is_accepted_and_is_what_the_hub_uses(self, tmp_path):
        from batcher.core import default_hub
        from batcher.metadata.backends.layered import LayeredBackend

        meta = MetadataConfig(backend="layered", uri=str(tmp_path / "store"))
        with config_context(active_config().replace(metadata=meta)):
            backend = default_hub()._backend
            assert isinstance(backend, LayeredBackend)
            backend.put("t", ("k",), b"v")
        # Durable through the shared store, not just the local cache in front of it.
        assert LayeredBackend.from_uri(str(tmp_path / "store")).get("t", ("k",)) == b"v"

    def test_rocksdb_is_accepted_and_is_what_the_hub_uses(self, tmp_path):
        pytest.importorskip("rocksdict")
        from batcher.core import default_hub
        from batcher.metadata.backends.rocksdb import RocksDBBackend

        meta = MetadataConfig(backend="rocksdb", uri=str(tmp_path / "stats.rocksdb"))
        with config_context(active_config().replace(metadata=meta)):
            assert isinstance(default_hub()._backend, RocksDBBackend)

    def test_an_unknown_backend_is_still_refused(self):
        from batcher._internal.errors import ConfigError

        bad = active_config().replace(metadata=MetadataConfig(backend="casandra"))
        with pytest.raises(ConfigError, match=r"metadata\.backend"), config_context(bad):
            pass
