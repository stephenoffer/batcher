"""Config immutability/derivation and MetadataHub backends."""

from __future__ import annotations

import dataclasses

from batcher.config import Config, config_context
from batcher.config.config import MemoryConfig
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend, SQLiteBackend
from batcher.plan.feedback import OperatorFeedback
from batcher.plan.ids import OpId


def test_config_is_frozen_and_derived():
    cfg = Config()
    assert dataclasses.is_dataclass(cfg)
    derived = cfg.replace(memory=MemoryConfig(soft_limit=0.5))
    assert derived.memory.soft_limit == 0.5
    assert cfg.memory.soft_limit == 0.85  # original untouched


def test_config_from_env():
    cfg = Config.from_env({"BATCHER_EXECUTION_MORSEL_ROWS": "4096"})
    assert cfg.execution.morsel_rows == 4096


def test_config_context_scoped():
    from batcher.config import active_config
    from batcher.config.config import ExecutionConfig

    base = active_config().execution.morsel_rows
    scoped = Config().replace(execution=ExecutionConfig(morsel_rows=999))
    with config_context(scoped):
        assert active_config().execution.morsel_rows == 999
    # restored on exit
    assert active_config().execution.morsel_rows == base


def _fb(op: int, n: int) -> OperatorFeedback:
    return OperatorFeedback(
        op_id=OpId(op),
        kind="pipeline",
        n_actual=n,
        t_op_ms=1.0,
        m_peak_bytes=0,
        selectivity=1.0,
        batch_size=0,
    )


def test_inprocess_hub_records_and_reads():
    hub = MetadataHub(InProcessBackend())
    hub.record(_fb(0, 5))
    hub.record(_fb(0, 7))
    hist = hub.operator_history(0)
    assert [h["n_actual"] for h in hist] == [5, 7]


def test_sqlite_hub_params_roundtrip():
    hub = MetadataHub(SQLiteBackend(":memory:"))
    hub.save_params("kyber.cardinality", {"f_p": 1.3})
    assert hub.load_params("kyber.cardinality") == {"f_p": 1.3}
    assert hub.load_params("missing") == {}


def test_keyed_params_isolate_writes_and_reassemble():
    # Per-key learned-stats writes: each entry is its own backend key, so writing
    # one signature never clobbers another (the lost-update race the whole-blob
    # read-modify-write had), and `load_keyed_params` reassembles the same dict.
    for backend in (InProcessBackend(), SQLiteBackend(":memory:")):
        hub = MetadataHub(backend)
        hub.put_keyed_param("kyber.stats", "sigA", {"rows": 10.0})
        hub.put_keyed_param("kyber.stats", "sigB", {"selectivity": 0.5})
        hub.put_keyed_param("kyber.stats", "sigA", {"rows": 20.0})  # update A only
        got = hub.load_keyed_params("kyber.stats")
        assert got == {"sigA": {"rows": 20.0}, "sigB": {"selectivity": 0.5}}
        assert hub.get_keyed_param("kyber.stats", "sigB") == {"selectivity": 0.5}
        assert hub.get_keyed_param("kyber.stats", "missing") is None


def test_keyed_params_merge_legacy_blob():
    # A store written by the old whole-blob path still reads back: load_keyed_params
    # merges the legacy `(namespace,)` blob underneath the per-key entries, which win.
    hub = MetadataHub(SQLiteBackend(":memory:"))
    hub.save_params("kyber.stats", {"old": {"rows": 1.0}, "shared": {"rows": 2.0}})
    hub.put_keyed_param("kyber.stats", "shared", {"rows": 99.0})  # per-key supersedes
    hub.put_keyed_param("kyber.stats", "new", {"rows": 3.0})
    got = hub.load_keyed_params("kyber.stats")
    assert got == {
        "old": {"rows": 1.0},  # legacy-only entry preserved
        "shared": {"rows": 99.0},  # per-key wins over the legacy blob
        "new": {"rows": 3.0},
    }
    # A miss falls back to the legacy blob's entry during migration.
    assert hub.get_keyed_param("kyber.stats", "old") == {"rows": 1.0}
