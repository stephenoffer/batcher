"""Cluster-shared metadata backends: object storage, redis, and the layered cache.

These let learned statistics (cardinality, cost calibration) compound across drivers
on a cluster instead of resetting per process — the same `MetadataBackend` protocol
the Hub already speaks, so nothing above changes.
"""

from __future__ import annotations

import pytest

from batcher.metadata import MetadataHub
from batcher.metadata.backends import make_backend
from batcher.metadata.backends.layered import LayeredBackend
from batcher.metadata.backends.object_storage import ObjectStorageBackend


def _object_backend(tmp_path) -> ObjectStorageBackend:
    return ObjectStorageBackend(f"file://{tmp_path}")


def test_object_storage_protocol_roundtrip(tmp_path):
    be = _object_backend(tmp_path)
    assert be.get("t", ("a",)) is None
    be.put("t", ("a",), b"1")
    be.put("t", ("b", 2), b"2")
    assert be.get("t", ("a",)) == b"1"
    assert be.get("t", ("b", 2)) == b"2"
    assert dict(be.scan("t")) == {("a",): b"1", ("b", 2): b"2"}
    # Re-writing a key replaces it (per-key granularity, no clobber of the other).
    be.put("t", ("a",), b"9")
    assert be.get("t", ("a",)) == b"9"
    assert be.get("t", ("b", 2)) == b"2"


def test_object_storage_scan_prefix_and_batch(tmp_path):
    be = _object_backend(tmp_path)
    be.batch_put("s", [(("k", 1), b"x"), (("k", 2), b"y"), (("j", 1), b"z")])
    assert dict(be.scan("s", ("k",))) == {("k", 1): b"x", ("k", 2): b"y"}
    assert dict(be.scan("s", ("j",))) == {("j", 1): b"z"}


def test_object_storage_shares_across_drivers(tmp_path):
    # The W5 win: a second driver pointed at the same root reads what the first wrote.
    driver_a = _object_backend(tmp_path)
    driver_a.put("kyber.stats", ("sigA",), b"learned")
    driver_b = _object_backend(tmp_path)
    assert driver_b.get("kyber.stats", ("sigA",)) == b"learned"


def test_object_storage_hub_keyed_params_roundtrip(tmp_path):
    hub = MetadataHub(_object_backend(tmp_path))
    hub.put_keyed_param("kyber.stats", "sigA", {"rows": 10.0})
    hub.put_keyed_param("kyber.stats", "sigB", {"selectivity": 0.5})
    hub.put_keyed_param("kyber.stats", "sigA", {"rows": 20.0})  # update A only
    assert hub.load_keyed_params("kyber.stats") == {
        "sigA": {"rows": 20.0},
        "sigB": {"selectivity": 0.5},
    }


def test_layered_caches_reads_and_writes_through(tmp_path):
    shared = _object_backend(tmp_path)
    layered = LayeredBackend(shared)
    layered.put("t", ("a",), b"1")
    # Write went through to the durable store (another driver could read it).
    assert _object_backend(tmp_path).get("t", ("a",)) == b"1"
    # A read is served (and the cache warmed) — drop the shared store and the cached
    # value still returns.
    assert layered.get("t", ("a",)) == b"1"
    layered._shared = _object_backend(tmp_path.with_name("empty"))  # no such data
    assert layered.get("t", ("a",)) == b"1"  # served from cache


def test_layered_refresh_picks_up_another_drivers_write(tmp_path):
    shared = _object_backend(tmp_path)
    layered = LayeredBackend(shared)
    assert layered.get("kyber.stats", ("sigA",)) is None  # caches the miss-through
    # Another driver writes the same shared store.
    _object_backend(tmp_path).put("kyber.stats", ("sigA",), b"fresh")
    # refresh() drops the cache so the next read re-pulls the shared store.
    layered.refresh()
    assert layered.get("kyber.stats", ("sigA",)) == b"fresh"


def test_make_backend_constructs_each_kind(tmp_path):
    assert isinstance(make_backend("object_storage", f"file://{tmp_path}"), ObjectStorageBackend)
    assert isinstance(make_backend("layered", f"file://{tmp_path}"), LayeredBackend)
    with pytest.raises(ValueError, match="unknown metadata backend"):
        make_backend("nope")
    with pytest.raises(ValueError, match="requires a uri"):
        make_backend("object_storage", None)


def test_sqlite_no_uri_persists_to_default_path(tmp_path, monkeypatch):
    # `backend="sqlite"` with no URI must persist to an on-disk file (cross-restart
    # learning), not a silent ephemeral `:memory:` store. Honors $BATCHER_HOME.
    from batcher.metadata.backends import default_sqlite_uri

    monkeypatch.setenv("BATCHER_HOME", str(tmp_path))
    uri = default_sqlite_uri()
    assert uri == str(tmp_path / "metadata.db")

    backend = make_backend("sqlite")  # no uri → durable default
    backend.put("kyber.stats", ("sig",), b"v")
    assert backend.get("kyber.stats", ("sig",)) == b"v"
    assert (tmp_path / "metadata.db").exists()


def test_redis_backend_protocol_roundtrip():
    # CI-gated: runs where fakeredis is installed. Validates the same protocol.
    fakeredis = pytest.importorskip("fakeredis")
    import redis

    from batcher.metadata.backends.redis import RedisBackend

    be = RedisBackend("redis://localhost:6379/0")
    be._redis = fakeredis.FakeStrictRedis()  # swap in the in-memory server
    assert isinstance(be._redis, redis.Redis)
    assert be.get("t", ("a",)) is None
    be.put("t", ("a",), b"1")
    be.batch_put("t", [(("b", 2), b"2"), (("c",), b"3")])
    assert be.get("t", ("a",)) == b"1"
    assert dict(be.scan("t")) == {("a",): b"1", ("b", 2): b"2", ("c",): b"3"}
    assert dict(be.scan("t", ("b",))) == {("b", 2): b"2"}
