"""Every backend can forget, and no backend reads a whole table to answer a prefix.

Two properties, both invisible until a store has been running for a while. The hub bounds
its feedback table by calling `delete` on whatever backend it has — a backend without one
silently opts out of pruning, so the table grows for the life of the deployment and every
new process pays to read all of it. And `learned_params` holds every namespace at once, so
a `scan` that filters client-side moves the whole table to answer a question about one
namespace, on every query.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.backends.layered import LayeredBackend
from batcher.metadata.backends.object_storage import ObjectStorageBackend
from batcher.metadata.backends.redis import _match
from batcher.metadata.store import BACKEND_METHODS, Key, check_backend

pytestmark = pytest.mark.unit

fsspec = pytest.importorskip("fsspec")


def _object_backend(tmp_path) -> ObjectStorageBackend:
    return ObjectStorageBackend(f"file://{tmp_path}")


class TestEveryBackendCanForget:
    """`delete` is optional in the protocol but required for the store to stay bounded."""

    def test_in_process_deletes(self) -> None:
        backend = InProcessBackend()
        backend.put("t", ("a",), b"1")
        backend.put("t", ("b",), b"2")
        backend.delete("t", [("a",)])
        assert [k for k, _ in backend.scan("t")] == [("b",)]

    def test_object_storage_deletes(self, tmp_path) -> None:
        backend = _object_backend(tmp_path)
        backend.put("t", ("a",), b"1")
        backend.put("t", ("b",), b"2")
        backend.delete("t", [("a",), ("never-written",)])
        assert [k for k, _ in backend.scan("t")] == [("b",)]

    def test_layered_deletes_through_both_layers(self, tmp_path) -> None:
        shared = _object_backend(tmp_path)
        cache = InProcessBackend()
        layered = LayeredBackend(shared, cache)
        layered.put("t", ("a",), b"1")
        layered.put("t", ("b",), b"2")

        layered.delete("t", [("a",)])

        # A key left in the local cache would keep answering `get` after the shared store
        # had forgotten it, which is the one thing a cache must never do.
        assert cache.get("t", ("a",)) is None
        assert shared.get("t", ("a",)) is None
        assert layered.get("t", ("b",)) == b"2"

    def test_deleting_nothing_is_a_no_op_everywhere(self, tmp_path) -> None:
        for backend in (InProcessBackend(), _object_backend(tmp_path)):
            backend.put("t", ("a",), b"1")
            backend.delete("t", [])
            assert [k for k, _ in backend.scan("t")] == [("a",)]


class TestLayeredRefresh:
    """Refresh empties the cache; it must not swap a caller's cache for the default."""

    def test_refresh_keeps_the_supplied_cache_object(self, tmp_path) -> None:
        cache = InProcessBackend()
        layered = LayeredBackend(_object_backend(tmp_path), cache)
        layered.put("t", ("a",), b"1")

        layered.refresh()

        assert layered._cache is cache
        assert cache.get("t", ("a",)) is None  # emptied, not replaced

    def test_refresh_re_pulls_from_the_shared_store(self, tmp_path) -> None:
        shared = _object_backend(tmp_path)
        layered = LayeredBackend(shared, InProcessBackend())
        layered.put("t", ("a",), b"1")
        shared.put("t", ("a",), b"2")  # another driver's update

        assert layered.get("t", ("a",)) == b"1"  # served from the warm cache
        layered.refresh()
        assert layered.get("t", ("a",)) == b"2"

    def test_it_is_still_a_valid_backend(self, tmp_path) -> None:
        layered = LayeredBackend(_object_backend(tmp_path))
        assert check_backend(layered) is layered
        assert all(callable(getattr(layered, name)) for name in BACKEND_METHODS)


class TestPrefixPushdown:
    """A prefix scan must return exactly the tuple-prefix matches, however it is pushed down."""

    KEYS: ClassVar[list[Key]] = [
        ("ns",),
        ("ns", "k1"),
        ("ns", "k2"),
        ("nsX",),
        ("n",),
        ("ns2", "q"),
        (5,),
        (5, 1),
        (50,),
    ]

    @pytest.mark.parametrize(
        "prefix", [(), ("ns",), ("nsX",), ("n",), (5,), (50,), ("ns2",), ("absent",)]
    )
    def test_object_storage_prefix_scan(self, tmp_path, prefix: Key) -> None:
        backend = _object_backend(tmp_path)
        for i, key in enumerate(self.KEYS):
            backend.put("t", key, str(i).encode())

        got = sorted(repr(k) for k, _ in backend.scan("t", prefix))
        want = sorted(repr(k) for k in self.KEYS if k[: len(prefix)] == prefix)
        assert got == want

    def test_object_storage_scan_chunks_without_losing_rows(self, tmp_path) -> None:
        backend = _object_backend(tmp_path)
        backend.batch_put("t", [((i,), str(i).encode()) for i in range(1300)])
        rows = dict(backend.scan("t"))
        assert len(rows) == 1300
        assert rows[(7,)] == b"7"

    def test_redis_match_pattern_covers_the_prefix_and_its_extensions(self) -> None:
        # The pattern is a glob over the encoded key. It must cover both the prefix key
        # itself (which continues `]`) and every key extending it (which continues `,`).
        # The leading `[` of the JSON list is itself a glob metacharacter, so it is escaped.
        assert _match(("ns",)) == '\\["ns"*'
        assert _match(()) is None

    def test_redis_match_escapes_glob_metacharacters_in_a_namespace(self) -> None:
        # A namespace is a file path or a model name, so `[` and `*` appear literally.
        # Unescaped, the pattern becomes a character class and matches nothing — which
        # reads as "never learned" rather than as an error.
        pattern = _match(("io.source_stats:/data/[2024]/*.parquet",))
        assert pattern is not None
        assert "\\[" in pattern and "\\]" in pattern and "\\*" in pattern
        assert pattern.endswith("*")
        assert not pattern.endswith("\\*")
