"""`metadata` — the MetadataHub: durable, learned, Ray-object-store-free.

The Hub is where the engine's adaptivity becomes *cross-execution* learning:
per-query traces, per-operator stats, column sketches, learned cardinality/cost
corrections, bandit posteriors, and (later) cached compiled artifacts. It is
deliberately decoupled from any execution runtime — it persists through a
pluggable `MetadataBackend` (in-process, SQLite, Redis, cloud object storage),
never the Ray object store.
"""

from __future__ import annotations

from batcher.metadata.hub import MetadataHub
from batcher.metadata.store import MetadataBackend

__all__ = ["MetadataBackend", "MetadataHub"]
