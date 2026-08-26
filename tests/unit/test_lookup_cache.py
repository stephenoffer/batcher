"""The lookup cache: deduplication, LRU, TTL, and the negative caching that pays for it.

The join around this cache is only fast because the cache answers most of the key stream
without a round trip, so what these cases pin is the *fetch count* — the one number a
correct-but-useless implementation gets wrong while returning perfectly right answers.

Negative caching gets the most attention here because it is the part most implementations
leave out and the part with the largest win: a key the store does not hold costs a full
round trip to learn nothing, and without a remembered absence it costs that on every batch
for the length of the stream.
"""

from __future__ import annotations

import pytest

from batcher.io.lookup.cache import LookupCache

pytestmark = pytest.mark.unit


class _Store:
    """A fetch function that records what it was asked for."""

    def __init__(self, rows: dict[str, object] | None = None) -> None:
        self.rows = rows or {}
        self.calls: list[list[str]] = []

    def __call__(self, keys: list[str]) -> dict[str, object]:
        self.calls.append(list(keys))
        return {key: self.rows[key] for key in keys if key in self.rows}

    @property
    def fetched(self) -> list[str]:
        return [key for call in self.calls for key in call]


def test_a_repeated_key_is_fetched_once_within_a_batch():
    store = _Store({"a": 1})
    cache = LookupCache()
    assert cache.resolve(["a", "a", "a"], store) == {"a": 1}
    assert store.fetched == ["a"], "duplicates must collapse before the fetch"


def test_a_repeated_key_is_fetched_once_across_batches():
    store = _Store({"a": 1, "b": 2})
    cache = LookupCache()
    cache.resolve(["a", "b"], store)
    cache.resolve(["a", "b"], store)
    assert store.calls == [["a", "b"]]
    assert cache.stats()["hits"] == 2


def test_an_absent_key_is_fetched_once_and_then_remembered():
    store = _Store({"a": 1})
    cache = LookupCache()
    assert cache.resolve(["a", "zz"], store) == {"a": 1, "zz": None}
    assert cache.resolve(["a", "zz"], store) == {"a": 1, "zz": None}
    assert cache.resolve(["zz"], store) == {"zz": None}
    # One fetch total for the missing key. Without negative caching this is three, and on
    # a real stream it is one round trip per batch, forever.
    assert store.fetched.count("zz") == 1
    assert cache.stats()["negative_hits"] == 2


def test_a_stored_null_is_not_confused_with_an_absence():
    # A dimension row whose column is genuinely null is a *hit*. Conflating it with a miss
    # would re-fetch it on every batch and, worse, make the two indistinguishable to the
    # caller building an inner join.
    store = _Store({"a": None})
    cache = LookupCache()
    assert cache.resolve(["a"], store) == {"a": None}
    assert cache.resolve(["a"], store) == {"a": None}
    assert store.fetched == ["a"]
    assert cache.stats()["negative_hits"] == 0


def test_only_the_unknown_keys_are_fetched():
    store = _Store({"a": 1, "b": 2, "c": 3})
    cache = LookupCache()
    cache.resolve(["a"], store)
    cache.resolve(["a", "b", "c"], store)
    assert store.calls == [["a"], ["b", "c"]]


def test_eviction_is_least_recently_used():
    store = _Store({"a": 1, "b": 2, "c": 3})
    cache = LookupCache(max_entries=2)
    cache.resolve(["a", "b"], store)
    cache.resolve(["a"], store)  # "a" is now the most recent, so "b" is the victim
    cache.resolve(["c"], store)
    store.calls.clear()
    cache.resolve(["a", "b", "c"], store)
    assert store.calls == [["b"]]


def test_a_zero_size_cache_fetches_every_time():
    # The setting that measures what the cache is buying, and the right one when the
    # store changes faster than any TTL could track.
    store = _Store({"a": 1})
    cache = LookupCache(max_entries=0)
    cache.resolve(["a"], store)
    cache.resolve(["a"], store)
    assert store.fetched == ["a", "a"]
    assert cache.stats()["entries"] == 0


def test_an_expired_entry_is_refetched(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("batcher.io.lookup.cache.time.monotonic", lambda: clock[0])
    store = _Store({"a": 1})
    cache = LookupCache(ttl_seconds=60)
    cache.resolve(["a"], store)
    clock[0] += 59
    cache.resolve(["a"], store)
    assert store.fetched == ["a"], "still inside the window"
    clock[0] += 2
    cache.resolve(["a"], store)
    assert store.fetched == ["a", "a"], "past the window, so the store is asked again"


def test_a_negative_entry_expires_too(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("batcher.io.lookup.cache.time.monotonic", lambda: clock[0])
    store = _Store({})
    cache = LookupCache(ttl_seconds=60)
    cache.resolve(["zz"], store)
    clock[0] += 61
    # A key that was absent may since have been written, and a negative entry that never
    # expired would hide it for the life of the worker.
    store.rows["zz"] = 7
    assert cache.resolve(["zz"], store) == {"zz": 7}


def test_an_expired_entry_does_not_count_as_a_hit(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("batcher.io.lookup.cache.time.monotonic", lambda: clock[0])
    store = _Store({"a": 1})
    cache = LookupCache(ttl_seconds=10)
    cache.resolve(["a"], store)
    clock[0] += 11
    cache.resolve(["a"], store)
    assert cache.stats()["hits"] == 0
    assert cache.stats()["misses"] == 2


def test_clear_drops_entries_but_keeps_the_counters():
    store = _Store({"a": 1})
    cache = LookupCache()
    cache.resolve(["a"], store)
    cache.resolve(["a"], store)
    cache.clear()
    stats = cache.stats()
    assert stats["entries"] == 0
    assert stats["hits"] == 1, "a lifetime counter must survive an emptying"


def test_a_fetch_returning_nothing_is_handled():
    cache = LookupCache()
    assert cache.resolve(["a", "b"], lambda keys: None) == {"a": None, "b": None}
