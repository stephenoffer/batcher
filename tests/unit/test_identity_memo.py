"""`IdentityMemo`: hits by identity, never answers for a recycled `id()`, clears when full."""

from __future__ import annotations

import pytest

from batcher._internal.memo import MISSING, IdentityMemo

pytestmark = pytest.mark.unit


class _Key:
    """A plain object: hashable by identity, and freed as soon as nothing refers to it."""


def test_hit_returns_the_stored_value_and_equal_objects_miss():
    memo: IdentityMemo[list[int], str] = IdentityMemo()
    key = [1, 2]
    assert memo.get(key) is MISSING
    memo.put(key, "v")
    assert memo.get(key) == "v"
    # Equal by value, different by identity: a different key.
    assert memo.get([1, 2]) is MISSING


def test_a_memoized_none_is_a_hit_not_a_miss():
    memo: IdentityMemo[_Key, None] = IdentityMemo()
    key = _Key()
    memo.put(key, None)
    assert memo.get(key) is None


def test_the_entry_pins_its_key_so_a_recycled_id_cannot_hit():
    memo: IdentityMemo[_Key, str] = IdentityMemo()
    memo.put(_Key(), "stale")
    # The memo holds the only reference, so the stored key is still alive...
    assert len(memo) == 1
    # ...and no new object can land at its address while the entry exists.
    fresh = [_Key() for _ in range(1000)]
    assert all(memo.get(k) is MISSING for k in fresh)


def test_a_recycled_id_is_rejected_by_the_identity_check():
    # Positive control for the check itself: forge an entry whose id belongs to one live
    # object but whose pinned key is another, which is what an unpinned id memo would see
    # after the original was freed and its address reused.
    memo: IdentityMemo[_Key, str] = IdentityMemo()
    original, impostor = _Key(), _Key()
    memo.put(original, "original")
    memo._entries[id(impostor)] = (original, "original")
    assert memo.get(impostor) is MISSING
    assert memo.get(original) == "original"


def test_a_bounded_memo_clears_wholesale_when_full():
    memo: IdentityMemo[_Key, int] = IdentityMemo(max_entries=3)
    keys = [_Key() for _ in range(4)]
    for i, k in enumerate(keys[:3]):
        memo.put(k, i)
    assert len(memo) == 3
    assert [memo.get(k) for k in keys[:3]] == [0, 1, 2]
    memo.put(keys[3], 3)
    assert len(memo) == 1
    assert all(memo.get(k) is MISSING for k in keys[:3])
    assert memo.get(keys[3]) == 3


def test_an_unbounded_memo_never_clears():
    memo: IdentityMemo[_Key, int] = IdentityMemo()
    keys = [_Key() for _ in range(2000)]
    for i, k in enumerate(keys):
        memo.put(k, i)
    assert len(memo) == 2000
    assert memo.get(keys[0]) == 0
