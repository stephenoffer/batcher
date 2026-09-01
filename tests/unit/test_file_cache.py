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


def test_stats_report_hit_rate(tmp_path):
    # Hit/miss counters make a warm-vs-cold read win visible: a low hit-rate over a repeated
    # read means the byte budget is too small, not that storage is slow.
    cache = FileBytesCache(str(tmp_path / "c"), max_bytes=10_000)
    fetch = _writer(b"x" * 100)
    cache.get_or_fetch("s3://b/a", fetch)  # miss
    cache.get_or_fetch("s3://b/a", fetch)  # hit
    cache.get_or_fetch("s3://b/a", fetch)  # hit
    s = cache.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["hit_rate"] == pytest.approx(2 / 3)
    assert s["used_bytes"] == 100
    assert FileBytesCache(str(tmp_path / "c2"), 10).stats()["hit_rate"] == 0.0  # cold cache


def test_a_file_bigger_than_the_budget_is_declined_not_handed_back_deleted(tmp_path):
    # Admitting an oversized file evicted the whole cache and then, with nothing else left
    # to drop, the entry itself -- deleting the file whose path was about to be returned.
    # The caller opened a path that no longer existed and the read raised FileNotFoundError.
    cache = FileBytesCache(str(tmp_path), max_bytes=1024)

    local = cache.get_or_fetch("s3://b/huge.parquet", _writer(b"x" * 4096))

    assert local is None, "an oversized file must be declined, so the caller reads remotely"
    assert cache.stats()["declined"] == 1
    assert cache.used_bytes == 0

    # A file that does fit is unaffected.
    small = cache.get_or_fetch("s3://b/small.parquet", _writer(b"y" * 16))
    assert small is not None
    with open(small, "rb") as fh:
        assert fh.read() == b"y" * 16


def test_a_size_hint_declines_before_the_download_is_paid(tmp_path):
    # Declining after the fetch is correct but still transfers the whole file for nothing.
    # The only caller stats the object before opening it, so the size is already known.
    cache = FileBytesCache(str(tmp_path), max_bytes=1024)
    fetched: list[int] = []
    payload = b"x" * 4096

    def fetch(dst: str) -> None:
        fetched.append(len(payload))
        with open(dst, "wb") as fh:
            fh.write(payload)

    assert cache.get_or_fetch("s3://b/huge.parquet", fetch, size_hint=4096) is None
    assert fetched == [], "the file was downloaded despite being known too large to keep"
    assert cache.stats()["declined"] == 1

    # A hint that fits does not suppress the fetch -- which is what makes the assertion
    # above a statement about the guard rather than about a fetch that never runs.
    payload = b"y" * 16
    assert cache.get_or_fetch("s3://b/ok.parquet", fetch, size_hint=16) is not None
    assert fetched == [16]


def test_concurrent_misses_on_one_file_cross_the_network_once(tmp_path):
    # Every reader that missed the same file downloaded the whole of it. A scan whose
    # worker threads all open one dimension table paid the transfer once per thread, and
    # the copies overwrote each other byte for byte.
    import threading

    cache = FileBytesCache(str(tmp_path), max_bytes=1 << 20)
    payload = b"z" * 4096
    fetches = 0
    counter_lock = threading.Lock()
    release = threading.Event()

    def slow_fetch(dst: str) -> None:
        nonlocal fetches
        with counter_lock:
            fetches += 1
        release.wait(5)  # hold the fetch open so every thread is inside the miss window
        with open(dst, "wb") as fh:
            fh.write(payload)

    results: list[str | None] = [None] * 8
    threads = [
        threading.Thread(
            target=lambda i=i: results.__setitem__(
                i, cache.get_or_fetch("s3://b/dim.parquet", slow_fetch)
            )
        )
        for i in range(8)
    ]
    for t in threads:
        t.start()
    # Let them all reach the miss before the leader is allowed to finish.
    threading.Event().wait(0.2)
    release.set()
    for t in threads:
        t.join(30)

    assert not any(t.is_alive() for t in threads), "a reader was left waiting"
    assert fetches == 1, f"the same file was downloaded {fetches} times concurrently"
    assert cache.stats()["coalesced"] == 7
    for path in results:
        assert path is not None
        with open(path, "rb") as fh:
            assert fh.read() == payload


def test_a_failed_leader_does_not_block_or_break_the_readers_waiting_on_it(tmp_path):
    # The coalescing must never make one thread's failure another thread's failure: a
    # waiter whose leader dies falls through and fetches for itself.
    import threading

    cache = FileBytesCache(str(tmp_path), max_bytes=1 << 20)
    started = threading.Event()

    def failing(dst: str) -> None:
        started.set()
        raise OSError("connection reset")

    leader_error: list[BaseException] = []

    def run_leader() -> None:
        try:
            cache.get_or_fetch("s3://b/x.parquet", failing)
        except BaseException as exc:  # recorded, then asserted below
            leader_error.append(exc)

    leader = threading.Thread(target=run_leader)
    leader.start()
    started.wait(5)
    leader.join(10)

    assert leader_error and isinstance(leader_error[0], OSError)

    # A later reader must still be able to fetch it, and must not first sit out the
    # coalescing wait -- the leader has to clear its claim even when it raises. Timed,
    # because the failure mode here is a 2-minute stall rather than a wrong answer.
    import time

    started_at = time.monotonic()
    local = cache.get_or_fetch("s3://b/x.parquet", _writer(b"ok"))
    elapsed = time.monotonic() - started_at

    assert local is not None
    assert elapsed < 5.0, f"the next reader waited {elapsed:.1f}s on a claim nobody holds"
    with open(local, "rb") as fh:
        assert fh.read() == b"ok"


def test_a_cached_file_deleted_underneath_the_ledger_is_re_fetched_not_handed_back(tmp_path):
    # The cache directory is a node volume, not a private one. `file_cache_dir="auto"` puts
    # it on shared scratch, several workers on one node keep separate ledgers over the same
    # files, and the node's own cleaner may sweep it. So one process evicting handed another
    # a hit for a file that was gone, and the caller opened a path that did not exist --
    # `FileNotFoundError` out of the read, past the containment in `_cached_local`.
    import os

    one = FileBytesCache(str(tmp_path), max_bytes=1 << 20)
    other = FileBytesCache(str(tmp_path), max_bytes=1 << 20)  # a second worker, same volume

    first = one.get_or_fetch("s3://b/dim.parquet", _writer(b"payload"))
    same = other.get_or_fetch("s3://b/dim.parquet", _writer(b"payload"))
    assert first is not None and first == same, "the two ledgers must address the same file"

    os.remove(same)  # the other worker evicts it, or the node's cleaner sweeps it

    recovered = one.get_or_fetch("s3://b/dim.parquet", _writer(b"payload"))

    assert recovered is not None
    with open(recovered, "rb") as fh:
        assert fh.read() == b"payload", "the ledger served a file it no longer had"
    stats = one.stats()
    assert stats["stale"] == 1
    assert one.used_bytes == len(b"payload"), "the deleted entry's bytes were never given back"


def test_the_file_cache_counters_are_reachable_from_cache_stats(tmp_path):
    # `FileBytesCache.stats` counted hits, coalesced fetches and declines, and no caller
    # anywhere read them -- so the one cache tier whose value is measured in bytes off the
    # network was the only one with no way to tell whether it was doing anything.
    import batcher as bt
    from batcher.config import Config, MemoryConfig, config_context

    assert not [k for k in bt.cache_stats() if k.startswith("file_")], (
        "file_* keys appear with no file cache configured"
    )

    configured = Config().replace(memory=MemoryConfig(file_cache_dir=str(tmp_path)))
    with config_context(configured):
        keys = {k for k in bt.cache_stats() if k.startswith("file_")}

    assert {"file_hits", "file_misses", "file_coalesced", "file_declined"} <= keys
