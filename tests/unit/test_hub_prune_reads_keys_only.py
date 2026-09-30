"""Pruning the stored operator feedback reads keys, never values.

`_prune_op_stats` needs only each key's sequence number to find the oldest rows. It used to
walk the store with `scan`, which on the in-process backend JSON-encodes every deferred row it
yields, and then sorted the whole key list: 219 ms per prune at the 65,536-row cap, paid every
4,096 records — a stall roughly every hundred small queries.
"""

from __future__ import annotations

import pytest

from batcher.metadata import hub as hub_mod
from batcher.metadata.backends.in_process import InProcessBackend

pytestmark = pytest.mark.unit


def _backend_with(n: int) -> InProcessBackend:
    backend = InProcessBackend()
    rows = backend._tables.setdefault(hub_mod._OP_STATS, {})
    for seq in range(n):
        rows[(seq % 7, seq)] = {"rows_in": seq}  # deferred: stored decoded, encoded on read
    return backend


def test_prune_keeps_the_newest_rows_without_encoding_any(monkeypatch):
    monkeypatch.setattr(hub_mod, "_OP_STATS_MAX", 100)
    backend = _backend_with(130)
    hub = hub_mod.MetadataHub(backend=backend)
    hub._prune_op_stats()

    rows = backend._tables[hub_mod._OP_STATS]
    assert sorted(k[1] for k in rows) == list(range(30, 130))
    # Nothing was read for its value: every surviving row is still the deferred dict.
    assert all(type(v) is dict for v in rows.values())


def test_keys_lists_a_prefix_without_touching_values():
    backend = _backend_with(10)
    keys = backend.keys(hub_mod._OP_STATS, (3,))
    assert sorted(keys) == [(3, 3)]
    assert all(type(v) is dict for v in backend._tables[hub_mod._OP_STATS].values())
