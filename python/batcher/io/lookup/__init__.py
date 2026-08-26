"""Lookup joins: enriching rows from a key-value store, one batched fetch per batch.

The alternative to broadcasting or shuffling a dimension that is far larger than what any
one batch touches. Cost scales with the *distinct keys the data contains* rather than with
the dimension's size, which is what makes a hundred-million-row dimension joinable from a
stream that sees ten thousand of them.

Re-exports only; `base` holds the contract and why the trade is what it is, `cache` holds
the LRU and the negative caching that make recall cheap, `backends` holds the stores, and
`join` holds the per-batch step `Dataset.lookup_join` runs.
"""

from __future__ import annotations

from batcher.io.lookup.backends import InMemoryLookup, RedisLookup, RocksDBLookup
from batcher.io.lookup.base import KeyValueLookup, lookup_arrays
from batcher.io.lookup.cache import LookupCache
from batcher.io.lookup.join import LookupEnricher
from batcher.io.lookup.spec import build_lookup, lookup_schema
from batcher.io.lookup.stage import LookupStage

__all__ = [
    "InMemoryLookup",
    "KeyValueLookup",
    "LookupCache",
    "LookupEnricher",
    "LookupStage",
    "RedisLookup",
    "RocksDBLookup",
    "build_lookup",
    "lookup_arrays",
    "lookup_schema",
]
