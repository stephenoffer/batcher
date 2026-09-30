"""`IdentityMemo`: the one bounded identity-keyed memo the optimizer and sizers share."""

from __future__ import annotations

import pytest

from batcher._internal.registry import MISSING, IdentityMemo

pytestmark = pytest.mark.unit


def test_a_hit_requires_the_same_object_not_an_equal_one():
    memo: IdentityMemo[str] = IdentityMemo(8)
    a, b = [1], [1]
    memo.put(a, "a")
    assert memo.get(a) == "a"
    assert memo.get(b) is MISSING


def test_an_unhashable_key_is_memoized_by_identity():
    memo: IdentityMemo[int] = IdentityMemo(8)
    key: dict[str, int] = {}  # unhashable, like a plan node carrying `Expr`s
    assert memo.put(key, 3) == 3
    assert memo.get(key) == 3


def test_a_memoized_none_is_a_hit():
    memo: IdentityMemo[None] = IdentityMemo(8)
    key = object()
    assert memo.get(key) is MISSING
    memo.put(key, None)
    assert memo.get(key) is None


def test_extra_inputs_are_part_of_the_key():
    memo: IdentityMemo[float] = IdentityMemo(8)
    key = object()
    memo.put(key, 1.0, 16.0)
    assert memo.get(key, 16.0) == 1.0
    assert memo.get(key, 32.0) is MISSING
    assert memo.get(key) is MISSING


def test_the_entry_pins_its_key_so_an_id_cannot_be_recycled():
    memo: IdentityMemo[str] = IdentityMemo(8)
    memo.put(object(), "stale")
    # The first object is still referenced by the memo, so a fresh one cannot reuse its id.
    assert memo.get(object()) is MISSING


def test_a_full_memo_clears_rather_than_growing():
    memo: IdentityMemo[int] = IdentityMemo(2)
    keys = [object() for _ in range(3)]
    for i, k in enumerate(keys):
        memo.put(k, i)
    assert memo.get(keys[0]) is MISSING
    assert memo.get(keys[2]) == 2
