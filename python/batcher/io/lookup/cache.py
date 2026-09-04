"""The lookup cache: the part of an enrichment join that makes recall cheap.

A lookup join enriches rows by asking a key-value store for the keys in each batch. The
store answers in a millisecond or two, which is fine once and ruinous a million times, so
what decides whether the join is usable is not the store's latency but how often it is
asked at all. Two properties of real key distributions are what this exploits:

- **Keys repeat, heavily.** A fact stream joined to a dimension hits the same few thousand
  customers, products or regions over and over. An LRU over the *values* turns a Zipfian
  key stream into a handful of round trips.
- **Misses repeat too, and cost the same.** A key that is not in the store costs a full
  round trip to learn nothing, and the next batch asks again. Caching the absence — a
  **negative** entry — is what stops an unmatched key from being re-queried on every batch
  for the length of the stream. It is the single largest win on a dirty join key, and it is
  the one most lookup implementations leave out.

Both entry kinds expire, and the TTL is the whole freshness story: a dimension row changed
in the store becomes visible here when its entry expires, not before. That is the same
trade every lookup join makes (Flink's included), and the knob is the caller's.

Bounded by entry count rather than by bytes. A dimension row is small and uniform where a
cached query *result* is not, so the count is both the honest unit here and one that costs
nothing to maintain — where the result cache measures retained footprint because its
entries vary by orders of magnitude.

Not thread-safe on its own: one cache belongs to one worker, which is how `map_batches`
constructs it (one instance per worker, like a model). Sharing one across threads would
serialize the batches behind a lock for no gain, since the round trips it saves are what
cost the time.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

from batcher._internal.mathx import safe_div

__all__ = ["LookupCache"]

#: Stand-in for "this key is known to be absent from the store". A distinct sentinel rather
#: than `None`, because `None` is a perfectly good *value* for a dimension column and
#: conflating the two would turn a cached null into a cache miss on every batch.
_ABSENT = object()


class LookupCache:
    """A count-bounded, TTL'd LRU over key-value lookups, caching absences as well as hits.

    `resolve` is the whole surface: hand it the keys a batch needs and a function that
    fetches the ones it does not know, and it returns the full mapping while remembering
    what it learned. That shape is deliberate — a `get`/`put` pair invites a caller to
    fetch keys one at a time, and the batched fetch is the reason this is fast.
    """

    __slots__ = ("_entries", "_hits", "_max_entries", "_misses", "_negative_hits", "_ttl")

    def __init__(self, max_entries: int = 100_000, ttl_seconds: float | None = None) -> None:
        """Create the cache.

        Args:
            max_entries: How many keys to remember, hits and absences together. Zero
                disables caching entirely, which makes every batch a fetch — useful to
                measure what the cache is buying, and the right setting when the store's
                contents change faster than any TTL could track.
            ttl_seconds: How long an entry stays usable, or `None` for the life of the
                worker. This is the join's freshness bound: a dimension row updated in the
                store is invisible here until its entry expires.
        """
        self._max_entries = max(0, int(max_entries))
        self._ttl = ttl_seconds if ttl_seconds and ttl_seconds > 0 else None
        # key -> (value_or_absent, expiry). Insertion order is recency order; `resolve`
        # moves what it serves to the end and eviction takes from the front.
        self._entries: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._negative_hits = 0
        self._misses = 0

    def resolve(self, keys: list[str], fetch: Any) -> dict[str, Any]:
        """Return `{key: value}` for every key, fetching only what is not cached.

        Absent keys map to `None` in the result, and that absence is remembered — so a key
        the store does not have is fetched once, not once per batch.

        Args:
            keys: The keys this batch needs. Duplicates are collapsed before the fetch,
                which is the first and largest saving on a stream with repeated keys.
            fetch: A callable taking the list of unknown keys and returning
                `{key: value}` for those it found. Keys it omits are recorded as absent.

        Returns:
            A mapping from every requested key to its value, or `None` where the store has
            none.
        """
        now = time.monotonic()
        resolved: dict[str, Any] = {}
        unknown: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            entry = self._lookup(key, now)
            if entry is None:
                unknown.append(key)
            else:
                resolved[key] = None if entry[0] is _ABSENT else entry[0]

        if unknown:
            found = fetch(unknown) or {}
            for key in unknown:
                value = found.get(key)
                present = key in found
                resolved[key] = value if present else None
                self._remember(key, value if present else _ABSENT, now)
        return resolved

    def stats(self) -> dict[str, int | float]:
        """How much of the key stream this cache answered without a round trip.

        Returns:
            The hit and miss counts, the aggregate hit-rate, how many entries are held,
            and `negative_hits` — the requests answered from a remembered *absence*. That
            last figure is the one to read on a join that seems slow despite a decent hit
            rate: a high value means the key column is dirty, and the join is doing exactly
            the right thing about it.
        """
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "negative_hits": self._negative_hits,
            "misses": self._misses,
            "hit_rate": safe_div(self._hits, total),
            "entries": len(self._entries),
            "max_entries": self._max_entries,
        }

    def clear(self) -> None:
        """Drop every entry, keeping the counters."""
        self._entries.clear()

    def _lookup(self, key: str, now: float) -> tuple[Any, float] | None:
        """The live entry for `key`, counting the access, or `None` if it must be fetched."""
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry[1] and entry[1] <= now:
            del self._entries[key]  # expired; reclaim it now rather than at eviction
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        if entry[0] is _ABSENT:
            self._negative_hits += 1
        return entry

    def _remember(self, key: str, value: Any, now: float) -> None:
        """Store `key`, evicting the least-recently-used entries to stay within the count."""
        if self._max_entries == 0:
            return
        expiry = now + self._ttl if self._ttl else 0.0
        self._entries[key] = (value, expiry)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
