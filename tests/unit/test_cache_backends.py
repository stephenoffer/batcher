"""The Redis and RocksDB backends: their configuration edges, and a real round trip.

Neither driver is installed on CI, so the round-trip cases are gated and the ones that
matter most on a driver-free machine are the *configuration* ones — a bad URI, a missing
driver — because those are what a user hits first and what a bad error message makes
unfixable. `tests/unit/test_shared_cache_contract.py` covers everything above the driver
against a dict.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher._internal.errors import ConfigError, MissingDependencyError

pytestmark = pytest.mark.unit


# --- configuration edges, reachable with neither driver installed --------------------


@pytest.mark.parametrize("bad", ["localhost:6379", "http://localhost", "", None, 6379])
def test_the_redis_shared_cache_names_what_a_redis_url_looks_like(bad):
    from batcher.carbonite.cache_shared.redis import RedisSharedCache

    with pytest.raises(ConfigError) as excinfo:
        RedisSharedCache(bad)
    # A bare host:port is the mistake people actually make, and `redis.from_url` answers
    # it with an opaque failure, so the message has to carry a usable example.
    assert "redis://" in str(excinfo.value)


def test_an_absent_driver_names_the_extra_that_installs_it():
    try:
        import rocksdict  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("rocksdict is installed; the missing-driver path cannot be reached")
    from batcher.metadata.backends.rocksdb import RocksDBBackend

    with pytest.raises(MissingDependencyError) as excinfo:
        RocksDBBackend("/tmp/does-not-matter.rocksdb")
    assert "rocksdb" in str(excinfo.value)


def test_the_rocksdb_metadata_backend_requires_a_uri():
    from batcher.metadata.backends.rocksdb import RocksDBBackend

    with pytest.raises(ConfigError, match="needs a uri"):
        RocksDBBackend(None)


def test_rocksdb_is_a_known_metadata_backend_name():
    from batcher.metadata.backends import BACKEND_NAMES

    assert "rocksdb" in BACKEND_NAMES


def test_an_unknown_metadata_backend_lists_rocksdb_among_the_options():
    from batcher.metadata.backends import make_backend

    with pytest.raises(ConfigError) as excinfo:
        make_backend("rocks")
    assert "rocksdb" in str(excinfo.value)


# --- the stamp the RocksDB shared cache prepends, which needs no database ------------


@pytest.mark.parametrize(
    ("ttl", "now", "read_at", "alive"),
    [
        (None, 1000.0, 9_999_999.0, True),  # no expiry, however long it sits
        (0, 1000.0, 9_999_999.0, True),  # a non-positive ttl means no expiry
        (60, 1000.0, 1030.0, True),  # inside the window
        (60, 1000.0, 1061.0, False),  # past it
    ],
)
def test_the_expiry_stamp_round_trips(ttl, now, read_at, alive):
    from batcher.carbonite.cache_shared.rocksdb import _stamped, _unstamped

    raw = _stamped(b"payload", ttl, now)
    assert _unstamped(raw, read_at) == (b"payload" if alive else None)


@pytest.mark.parametrize("truncated", [b"", b"\x00", b"\x00" * 7])
def test_a_short_record_reads_as_a_miss_rather_than_raising(truncated):
    from batcher.carbonite.cache_shared.rocksdb import _unstamped

    # These bytes live in a file other processes and earlier versions have written to.
    assert _unstamped(truncated, 0.0) is None


# --- the physical key layout the RocksDB metadata backend scans by -------------------


def test_a_tables_keys_are_contiguous_and_prefix_seekable():
    from batcher.metadata.backends.rocksdb import _physical, _seek_prefix, _split

    key = ("ns", "col", 3)
    physical = _physical("column_stats", key)
    assert physical.startswith(_seek_prefix("column_stats", ()))
    assert physical.startswith(_seek_prefix("column_stats", ("ns",)))
    assert not physical.startswith(_seek_prefix("column_stats", ("other",)))
    assert not physical.startswith(_seek_prefix("op_stats", ()))
    assert _split(physical, "column_stats") == key
    assert _split(physical, "op_stats") is None


def test_a_prefix_seek_does_not_match_a_longer_sibling_name():
    from batcher.metadata.backends.rocksdb import _physical, _seek_prefix

    # `("ns",)` must not cover `("nsx", ...)`. The encoding is JSON, so the prefix ends
    # mid-string and a naive truncation would.
    assert not _physical("t", ("nsx", 1)).startswith(_seek_prefix("t", ("ns",)))


# --- a real round trip, when the driver is there ------------------------------------


def test_the_rocksdb_metadata_backend_round_trips(tmp_path):
    pytest.importorskip("rocksdict")
    from batcher.metadata.backends.rocksdb import RocksDBBackend

    backend = RocksDBBackend(str(tmp_path / "stats.rocksdb"))
    try:
        backend.put("t", ("a", 1), b"one")
        backend.batch_put("t", [(("a", 2), b"two"), (("b", 1), b"three")])
        assert backend.get("t", ("a", 1)) == b"one"
        assert backend.get("t", ("missing",)) is None
        assert dict(backend.scan("t", ("a",))) == {("a", 1): b"one", ("a", 2): b"two"}
        assert len(dict(backend.scan("t"))) == 3
        assert dict(backend.scan("other")) == {}
    finally:
        backend.close()


def test_the_rocksdb_shared_cache_round_trips(tmp_path):
    pytest.importorskip("rocksdict")
    from batcher.carbonite.cache_shared.rocksdb import RocksDBSharedCache

    cache = RocksDBSharedCache(str(tmp_path / "cache.rocksdb"))
    try:
        cache.put("k", b"payload")
        assert cache.get("k") == b"payload"
        cache.delete("k")
        assert cache.get("k") is None
        cache.delete("k")  # deleting an absent key is not an error
        cache.put("expired", b"gone", ttl_seconds=-1)
        assert cache.get("expired") is None
    finally:
        cache.close()


def test_the_rocksdb_shared_cache_sweeps_expired_entries(tmp_path):
    pytest.importorskip("rocksdict")
    from batcher.carbonite.cache_shared.rocksdb import RocksDBSharedCache

    cache = RocksDBSharedCache(str(tmp_path / "sweep.rocksdb"))
    try:
        cache.put("live", b"a")
        cache.put("dead", b"b", ttl_seconds=-1)
        assert cache.evict_expired() == 1
        assert cache.get("live") == b"a"
    finally:
        cache.close()


def test_an_arrow_result_survives_the_rocksdb_shared_cache(tmp_path):
    pytest.importorskip("rocksdict")
    from batcher.carbonite.cache_shared.rocksdb import RocksDBSharedCache
    from batcher.carbonite.cache_shared.store import SharedResultCache

    original = pa.table({"i": pa.array([1, 2], pa.int32()), "s": ["x", None]})
    cache = SharedResultCache(RocksDBSharedCache(str(tmp_path / "arrow.rocksdb")))
    try:
        cache.put("k", original)
        assert cache.get("k").equals(original)
    finally:
        cache.close()
