"""Keyed state costs what it expires, not what it retains.

`transform_with_state` is the one streaming operator whose state is an arbitrary user
mapping, so it is also the one whose key space has no bound but a TTL. That makes the
*per-trigger* cost of holding a large key space the operator's real ceiling: a trigger that
touches ten keys must not walk ten million.

It used to walk them three times — once looking for stale keys, then twice more sizing the
retained bytes for the budget check and the metrics. These tests pin the shape that fixed
it: last-touched ordering, so expiry stops at the first live key, and a running byte total
charged and credited per entry instead of a scan. They are written against `KeyedStateFold`
directly because the property is about how much of the state is touched, which no end-to-end
assertion can see.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.core.streaming.keyed_state import KeyedStateFold

pytestmark = pytest.mark.integration

_SCHEMA = pa.schema([("k", pa.string()), ("v", pa.int64())])


def _running_total(key, rows, state):
    total = (state or {"total": 0})["total"] + sum(rows.column("v").to_pylist())
    return {"k": [key[0]], "total": [total]}, {"total": total}


def _fold(ttl: str | None = "1 hour", fn=_running_total) -> KeyedStateFold:
    """A fold over `fn` (a running per-key total by default), built from a real plan node."""
    stream = bt.from_batches(lambda: iter(()), _SCHEMA, bounded=False)
    node = stream.transform_with_state(
        fn, group_by="k", output_columns=["k", "total"], state_ttl=ttl
    )._plan
    return KeyedStateFold(node)


def _batch(keys: list[str]) -> pa.RecordBatch:
    return pa.record_batch({"k": keys, "v": [1] * len(keys)}, schema=_SCHEMA)


def test_state_stays_in_last_touched_order(monkeypatch):
    """The order expiry depends on: least-recently-touched first, whatever the insert order."""
    fold = _fold()
    stamps = iter([1_000, 2_000, 3_000])
    monkeypatch.setattr(
        "batcher.core.streaming.keyed_state._now_micros", lambda: next(stamps, 3_000)
    )
    fold.push(_batch(["a", "b", "c"]))
    fold.push(_batch(["a"]))  # 'a' was first; touching it must move it last
    assert list(fold._state) == [("b",), ("c",), ("a",)]


def test_expiry_visits_only_the_stale_prefix(monkeypatch):
    """A trigger that expires nothing must not walk the retained key space."""
    fold = _fold()
    now = [1_000_000]
    monkeypatch.setattr("batcher.core.streaming.keyed_state._now_micros", lambda: now[0])
    fold.push(_batch([f"k{i}" for i in range(500)]))

    visited = 0
    real_items = type(fold._state).items

    class Counting(dict):
        def items(self):
            nonlocal visited
            for entry in real_items(self):
                visited += 1
                yield entry

    fold._state = Counting(fold._state)
    now[0] += 1  # well inside the one-hour TTL: nothing is stale
    fold.push(_batch(["k0"]))
    assert visited <= 2, f"expiry walked {visited} entries to expire none"


def test_expiry_still_forgets_everything_past_the_ttl(monkeypatch):
    fold = _fold(ttl="1 second")
    now = [1_000_000]
    monkeypatch.setattr("batcher.core.streaming.keyed_state._now_micros", lambda: now[0])
    fold.push(_batch(["a", "b", "c"]))
    assert len(fold._state) == 3
    now[0] += 5_000_000  # five seconds later, every key is past a one-second TTL
    fold.push(pa.record_batch({"k": [], "v": []}, schema=_SCHEMA))
    assert fold._state == {}
    assert fold.metrics().num_rows_removed == 3


def test_a_backwards_clock_does_not_stall_expiry(monkeypatch):
    """An NTP step must not file a key ahead of older ones and pin the prefix forever."""
    fold = _fold(ttl="1 second")
    now = [10_000_000]
    monkeypatch.setattr("batcher.core.streaming.keyed_state._now_micros", lambda: now[0])
    fold.push(_batch(["old"]))
    now[0] = 1_000  # the clock steps backwards
    fold.push(_batch(["new"]))
    now[0] = 10_000_000 + 5_000_000  # and recovers, five seconds past both
    fold.push(pa.record_batch({"k": [], "v": []}, schema=_SCHEMA))
    assert fold._state == {}, "a backwards step left keys unexpirable"


def test_footprint_is_stable_without_a_scan():
    fold = _fold()
    fold.push(_batch(["a"]))
    one_field = fold.nbytes()
    assert one_field > 0
    assert fold.nbytes() == one_field, "the estimate must not depend on when it is asked"


def _payload_fold(sizes: dict[int, int | None]):
    """A fold whose state for key 'a' on push `n` is a string of `sizes[n]` bytes (None forgets)."""
    calls = [0]

    def fn(key, rows, state):
        size = sizes[calls[0]]
        calls[0] += 1
        return {"k": [key[0]], "total": [0]}, (None if size is None else {"blob": "x" * size})

    return _fold(fn=fn)


def test_a_large_string_field_is_charged_its_payload():
    """BT-004: one key holding a 10 MiB string was charged 192 bytes by a per-field constant."""
    ten_mib = 10 * 1024 * 1024
    fold = _payload_fold({0: ten_mib})
    fold.push(_batch(["a"]))
    assert fold.nbytes() >= ten_mib, f"10 MiB of state charged as {fold.nbytes()} bytes"


def test_shrinking_and_forgetting_state_release_its_charge():
    """The charge is reversible: replacing a value credits the old one back, forgetting all."""
    fold = _payload_fold({0: 4_000_000, 1: 10, 2: None})
    fold.push(_batch(["a"]))
    big = fold.nbytes()
    fold.push(_batch(["a"]))
    small = fold.nbytes()
    assert small < 1_000, f"replacing 4 MB with 10 bytes left {small} bytes charged"
    assert big - small >= 4_000_000 - 10
    fold.push(_batch(["a"]))
    assert fold.nbytes() == 0, "a forgotten key must release everything it was charged"


def test_expiry_releases_the_charge(monkeypatch):
    fold = _fold(ttl="1 second")
    now = [1_000_000]
    monkeypatch.setattr("batcher.core.streaming.keyed_state._now_micros", lambda: now[0])
    fold.push(_batch(["a", "b"]))
    assert fold.nbytes() > 0
    now[0] += 5_000_000
    fold.push(pa.record_batch({"k": [], "v": []}, schema=_SCHEMA))
    assert fold.nbytes() == 0


def test_a_large_state_trips_the_budget(monkeypatch):
    """The budget the estimate feeds must now see a state that grows by payload, not by keys."""
    from batcher._internal.errors import ResourceError

    fold = _payload_fold({0: 2_000_000})
    fold._cap = 1_000_000
    with pytest.raises(ResourceError):
        fold.push(_batch(["a"]))


def test_restore_rebuilds_the_order_and_the_counters(monkeypatch):
    """A restored fold expires and reports exactly as the one that wrote the snapshot."""
    now = [1_000_000]
    monkeypatch.setattr("batcher.core.streaming.keyed_state._now_micros", lambda: now[0])
    original = _fold(ttl="1 second")
    original.push(_batch(["a", "b"]))
    now[0] += 10
    original.push(_batch(["c"]))
    snapshot = original.state()

    restored = _fold(ttl="1 second")
    restored.restore(snapshot)
    assert list(restored._state) == list(original._state)
    assert restored.nbytes() == original.nbytes()
    now[0] += 5_000_000
    restored.push(pa.record_batch({"k": [], "v": []}, schema=_SCHEMA))
    assert restored._state == {}
