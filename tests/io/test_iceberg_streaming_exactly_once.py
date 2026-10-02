"""A streaming write to Iceberg is exactly-once across a restart, as Delta's is.

The sink used to have no per-batch marker, so a micro-batch replayed after a crash between
the snapshot commit and the checkpoint commit appended its rows a second time (and the sink
warned that it was at-least-once). Each micro-batch's ``(app_id, batch_id)`` now goes in the
snapshot summary, and a replay that finds it commits nothing.
"""

from __future__ import annotations

import warnings

import pyarrow as pa
import pytest

import batcher as bt
from batcher.io.formats.streaming.checkpoint.store import CheckpointStore

pytest.importorskip("pyiceberg", reason="pyiceberg not installed")

pytestmark = pytest.mark.integration


class _Killed(RuntimeError):
    pass


@pytest.fixture
def spec(tmp_path) -> dict:
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "wh"
    warehouse.mkdir()
    spec = {
        "type": "sql",
        "uri": f"sqlite:///{warehouse}/catalog.db",
        "warehouse": f"file://{warehouse}",
    }
    catalog = SqlCatalog("default", uri=spec["uri"], warehouse=spec["warehouse"])
    catalog.create_namespace("db")
    catalog.create_table("db.t", schema=pa.schema([pa.field("value", pa.int64())]))
    return spec


def _run(spec: dict, ck: str) -> None:
    bt.read.rate(5, num_rows=20, pace=False).select("value").write(
        "db.t",
        format="iceberg",
        catalog=spec,
        trigger=bt.Trigger.available_now(),
        checkpoint=ck,
    ).await_termination()


def _values(spec: dict) -> list[int]:
    return sorted(bt.read.iceberg("db.t", catalog=spec).to_pydict()["value"])


def test_a_replay_after_the_snapshot_commit_adds_no_rows(spec, tmp_path, monkeypatch):
    ck = str(tmp_path / "ck")
    real_commit = CheckpointStore.commit
    fired = {"done": False}

    def commit(self, batch_id, sink_token=None):
        if batch_id == 1 and not fired["done"]:
            fired["done"] = True
            raise _Killed("after the Iceberg snapshot, before the commit log")
        return real_commit(self, batch_id, sink_token)

    with monkeypatch.context() as patch:
        patch.setattr(CheckpointStore, "commit", commit)
        with pytest.raises(_Killed):
            _run(spec, ck)
    assert _values(spec) == list(range(10)), "the crash did not land after batch 1's commit"

    _run(spec, ck)
    assert _values(spec) == list(range(20))


def test_the_iceberg_stream_sink_no_longer_warns_at_least_once(spec, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _run(spec, str(tmp_path / "ck"))
    assert _values(spec) == list(range(20))
