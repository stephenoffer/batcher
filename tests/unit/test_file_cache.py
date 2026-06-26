"""Local-SSD read-through file cache (W5) — Carbonite's `FileBytesCache`.

The cache is transparent and result-invariant: a hit avoids re-fetching the remote
file, eviction keeps it within the byte budget, and an evicted key re-fetches. The
integration test confirms the wiring through `_ArrowFileSystem.open` serves a second
read from local disk.
"""

from __future__ import annotations

import dataclasses

import pyarrow.fs as pafs
import pytest

from batcher.config import active_config, config_context
from batcher.io.filesystem import FileBytesCache, _ArrowFileSystem

pytestmark = pytest.mark.unit


def _writer(data: bytes):
    def fetch(dst: str) -> None:
        with open(dst, "wb") as fh:
            fh.write(data)

    return fetch


def test_hit_avoids_refetch(tmp_path):
    cache = FileBytesCache(str(tmp_path / "c"), max_bytes=1 << 30)
    calls = []

    def fetch(dst: str) -> None:
        calls.append(dst)
        _writer(b"hello")(dst)

    p1 = cache.get_or_fetch("s3://bucket/a", fetch)
    p2 = cache.get_or_fetch("s3://bucket/a", fetch)
    assert p1 == p2
    assert len(calls) == 1  # the second call was a cache hit
    with open(p1, "rb") as fh:
        assert fh.read() == b"hello"


def test_lru_eviction_bounds_bytes_and_refetches(tmp_path):
    cache = FileBytesCache(str(tmp_path / "c"), max_bytes=10)  # holds two 5-byte files
    cache.get_or_fetch("a", _writer(b"xxxxx"))
    cache.get_or_fetch("b", _writer(b"yyyyy"))  # total 10 — at budget
    cache.get_or_fetch("c", _writer(b"zzzzz"))  # over budget → evict LRU ("a")
    assert cache.used_bytes <= 10

    refetch = []

    def fetch_a(dst: str) -> None:
        refetch.append(1)
        _writer(b"xxxxx")(dst)

    cache.get_or_fetch("a", fetch_a)
    assert len(refetch) == 1  # "a" was evicted, so it is fetched again


def test_second_open_served_from_cache(tmp_path, monkeypatch):
    # An object-store read (simulated with a local backend marked cacheable): the
    # second open must be served from the local cache, i.e. fetched exactly once.
    src = tmp_path / "remote.bin"
    src.write_bytes(b"payload-bytes")
    fs = _ArrowFileSystem(pafs.LocalFileSystem(), "", atomic_rename=True, cacheable=True)

    downloads: list[str] = []
    original = _ArrowFileSystem._download

    def counting(self, in_path: str, dst: str) -> None:
        downloads.append(in_path)
        original(self, in_path, dst)

    monkeypatch.setattr(_ArrowFileSystem, "_download", counting)

    cfg = active_config()
    cfg = dataclasses.replace(
        cfg, memory=dataclasses.replace(cfg.memory, file_cache_dir=str(tmp_path / "cache"))
    )
    with config_context(cfg):
        with fs.open(str(src)) as fh:
            assert fh.read() == b"payload-bytes"
        with fs.open(str(src)) as fh:
            assert fh.read() == b"payload-bytes"
    assert len(downloads) == 1  # fetched once; the second read hit the cache


def test_changed_remote_file_is_refetched_not_stale(tmp_path, monkeypatch):
    # Overwriting the same path with new content must be a cache miss (the key folds in
    # size + mtime), so the read returns the new bytes — never a stale cached copy.
    src = tmp_path / "remote.bin"
    src.write_bytes(b"v1-content")
    fs = _ArrowFileSystem(pafs.LocalFileSystem(), "", atomic_rename=True, cacheable=True)

    downloads: list[str] = []
    original = _ArrowFileSystem._download

    def counting(self, in_path: str, dst: str) -> None:
        downloads.append(in_path)
        original(self, in_path, dst)

    monkeypatch.setattr(_ArrowFileSystem, "_download", counting)

    cfg = active_config()
    cfg = dataclasses.replace(
        cfg, memory=dataclasses.replace(cfg.memory, file_cache_dir=str(tmp_path / "cache"))
    )
    with config_context(cfg):
        with fs.open(str(src)) as fh:
            assert fh.read() == b"v1-content"
        src.write_bytes(b"v2-different-length-content")  # different size → new key
        with fs.open(str(src)) as fh:
            assert fh.read() == b"v2-different-length-content"
    assert len(downloads) == 2  # the change forced a re-fetch, no stale hit


def test_local_reads_are_never_cached(tmp_path, monkeypatch):
    # A non-cacheable (local) filesystem must never route through the cache.
    src = tmp_path / "local.bin"
    src.write_bytes(b"data")
    fs = _ArrowFileSystem(pafs.LocalFileSystem(), "", atomic_rename=True, cacheable=False)

    downloads: list[str] = []
    monkeypatch.setattr(_ArrowFileSystem, "_download", lambda self, p, d: downloads.append(p))
    cfg = active_config()
    cfg = dataclasses.replace(
        cfg, memory=dataclasses.replace(cfg.memory, file_cache_dir=str(tmp_path / "cache"))
    )
    with config_context(cfg), fs.open(str(src)) as fh:
        assert fh.read() == b"data"
    assert downloads == []  # local path bypassed the cache entirely
