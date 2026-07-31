"""Locality-aware split-to-worker assignment (pure logic, no engine).

`assign_splits` decides which worker reads which split. For a split already resident on
a worker — the buckets of an intermediate a prior stage left on the shuffle fleet — that
decision is what makes the read a zero-copy local-store hit instead of a network fetch of
bytes already in the reading process. These pin the policy: affinity is honoured, order
preservation still outranks it, and a concentrated intermediate falls back to load
balancing rather than serializing the stage onto one worker.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.partition_io.assignment import (
    _balance,
    assign_splits,
    balance_with_affinity,
    has_affinity,
)

pytestmark = pytest.mark.unit


class _StorageSplit:
    """A split read from object storage: equidistant from every worker, so no affinity."""

    def __init__(self, rows: int) -> None:
        self.rows = rows

    def row_count(self) -> int:
        return self.rows


class _ResidentSplit(_StorageSplit):
    """A split whose bytes already sit on the worker serving `addr`."""

    def __init__(self, rows: int, addr: str) -> None:
        super().__init__(rows)
        self.addr = addr

    def affinity(self) -> str:
        return self.addr


ADDRS = ["10.0.0.1:100", "10.0.0.2:200", "10.0.0.3:300", "10.0.0.4:400"]


def test_storage_splits_have_no_affinity():
    assert not has_affinity([_StorageSplit(10), _StorageSplit(20)])
    assert has_affinity([_StorageSplit(10), _ResidentSplit(20, ADDRS[0])])


def test_resident_splits_go_to_the_worker_already_holding_them():
    # One evenly-sized bucket per worker, listed in an order that does NOT match the
    # worker order, so a positional coincidence cannot pass this.
    splits = [_ResidentSplit(100, a) for a in reversed(ADDRS)]
    groups = assign_splits(splits, len(ADDRS), worker_addrs=ADDRS)
    for i, addr in enumerate(ADDRS):
        assert [s.addr for s in groups[i]] == [addr]


def test_without_worker_addresses_assignment_is_unchanged():
    """The storage path must keep bin-packing: no addrs supplied ⇒ identical to `_balance`."""
    splits = [_ResidentSplit(rows, ADDRS[i % 4]) for i, rows in enumerate([9, 1, 5, 3, 7, 2])]
    assert assign_splits(splits, 4) == _balance(splits, 4)


def test_storage_splits_keep_bin_packing_even_with_addresses():
    splits = [_StorageSplit(r) for r in (9, 1, 5, 3, 7, 2)]
    assert assign_splits(splits, 4, worker_addrs=ADDRS) == _balance(splits, 4)


def test_order_preservation_outranks_locality():
    """`preserve_order` is a correctness requirement of the caller, not a preference.

    A locality reshuffle would put non-adjacent splits in one partition, which is exactly
    what breaks distributed `LIMIT` / `with_row_index`.
    """
    splits = [_ResidentSplit(10, a) for a in reversed(ADDRS)]
    groups = assign_splits(splits, 4, preserve_order=True, worker_addrs=ADDRS)
    # Contiguous source order: group i holds the i-th split, not the one it hosts.
    assert [s.addr for g in groups for s in g] == list(reversed(ADDRS))


def test_a_split_on_a_departed_worker_falls_to_the_least_loaded():
    """An address outside the fleet (a worker replaced since the intermediate was
    published) has no home here, so it is packed by load like any storage split."""
    splits = [
        _ResidentSplit(100, ADDRS[0]),
        _ResidentSplit(1, "10.9.9.9:999"),  # nobody in this fleet holds it
    ]
    groups = assign_splits(splits, 4, worker_addrs=ADDRS)
    assert [s.addr for s in groups[0]] == [ADDRS[0]]
    homeless = [s for g in groups[1:] for s in g]
    assert [s.addr for s in homeless] == ["10.9.9.9:999"]


def test_a_concentrated_intermediate_falls_back_to_load_balancing():
    """Locality's one failure mode: honouring it would idle three workers.

    Everything sits on worker 0, so a locality assignment gives it the whole stage. The
    balance check must catch that and hand back the bin-packed assignment instead.
    """
    splits = [_ResidentSplit(100, ADDRS[0]) for _ in range(8)]
    groups = assign_splits(splits, 4, worker_addrs=ADDRS)
    assert groups == _balance(splits, 4)
    assert all(len(g) == 2 for g in groups)  # spread, not piled on the holder


def test_mild_skew_still_keeps_locality():
    """Below the tolerance the local read is worth more than perfect balance."""
    splits = [_ResidentSplit(100, ADDRS[0]), _ResidentSplit(100, ADDRS[0])] + [
        _ResidentSplit(100, a) for a in ADDRS[1:]
    ]
    groups = assign_splits(splits, 4, worker_addrs=ADDRS)
    assert len(groups[0]) == 2  # worker 0 keeps both of its own buckets
    assert all(len(g) == 1 for g in groups[1:])


def test_every_split_is_assigned_exactly_once():
    """The invariant that makes this safe: placement moves work, it never drops or
    duplicates it. A lost split is silently missing rows."""
    splits = [_ResidentSplit(i + 1, ADDRS[i % 3]) for i in range(11)]
    groups = assign_splits(splits, 4, worker_addrs=ADDRS)
    assert sorted(id(s) for g in groups for s in g) == sorted(id(s) for s in splits)


def test_more_workers_than_addresses_falls_back():
    """A fleet whose address list is short of its worker count cannot be reasoned about
    positionally, so it keeps the load-only assignment rather than guessing."""
    splits = [_ResidentSplit(10, a) for a in ADDRS]
    assert balance_with_affinity(splits, 4, ADDRS[:2], _balance) == _balance(splits, 4)


def test_no_workers_yields_no_groups():
    assert assign_splits([_ResidentSplit(1, ADDRS[0])], 0, worker_addrs=ADDRS) == []
