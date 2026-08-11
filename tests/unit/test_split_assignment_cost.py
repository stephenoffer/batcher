"""What `assign_splits` costs the driver before a single worker starts.

A whole-file split does not carry its row count: asking for one rebuilds a single-file
reader and opens that file's footer. Assignment asked twice per split, so dividing N files
among workers cost 2N metadata round trips on the driver -- the very cost
`FileSource.splits` stops planning sub-file splits to avoid, paid again one layer up while
the cluster idles.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.partition_io.assignment import (
    _MAX_WEIGHED_SPLITS,
    _balance,
    assign_splits,
    balance_with_affinity,
)

pytestmark = pytest.mark.unit


class _CountingSplit:
    """A whole-file-shaped split: no `rows`, and `row_count()` is the expensive question."""

    def __init__(self, rows: int) -> None:
        self._rows = rows
        self.calls = 0

    def row_count(self) -> int:
        self.calls += 1
        return self._rows


class _CheapSplit:
    """A row-group-shaped split, which records its count when it is built."""

    def __init__(self, rows: int) -> None:
        self.rows = rows

    def row_count(self) -> int:  # pragma: no cover - must never be reached
        raise AssertionError("a split carrying `rows` must not be asked to count")


def test_a_splits_row_count_is_read_at_most_once():
    splits = [_CountingSplit(i + 1) for i in range(20)]
    _balance(splits, 4)
    assert max(s.calls for s in splits) == 1


def test_contiguous_assignment_also_asks_only_once():
    splits = [_CountingSplit(i + 1) for i in range(20)]
    assign_splits(splits, 4, preserve_order=True)
    assert max(s.calls for s in splits) == 1


def test_affinity_assignment_also_asks_only_once():
    splits = [_CountingSplit(i + 1) for i in range(20)]
    balance_with_affinity(splits, 4, ["a", "b", "c", "d"], _balance)
    assert max(s.calls for s in splits) == 1


def test_a_split_carrying_its_count_is_never_asked_to_recount():
    # Would raise from `_CheapSplit.row_count` if the cheap attribute were ignored.
    groups = _balance([_CheapSplit(i + 1) for i in range(20)], 4)
    assert sum(len(g) for g in groups) == 20


def test_a_very_large_split_set_is_not_weighed_off_storage():
    # Past the cap, an unknown weight is taken as 1 rather than read: N metadata round
    # trips on the driver is the cost this exists to refuse.
    splits = [_CountingSplit(1) for _ in range(_MAX_WEIGHED_SPLITS + 1)]
    groups = _balance(splits, 8)
    assert sum(s.calls for s in splits) == 0
    assert sum(len(g) for g in groups) == len(splits)
    # Equal weights still spread evenly rather than piling onto worker 0.
    assert max(len(g) for g in groups) - min(len(g) for g in groups) <= 1


def test_balancing_still_balances():
    splits = [_CheapSplit(rows) for rows in (100, 90, 80, 70, 60, 50, 40, 30)]
    groups = _balance(splits, 4)
    loads = sorted(sum(s.rows for s in g) for g in groups)
    assert sum(loads) == 520
    # Largest-first packing over four workers gives 130 each here; nothing may exceed it.
    assert loads[-1] == 130


def test_assignment_is_deterministic():
    def build():
        return [_CheapSplit(rows) for rows in (5, 5, 5, 5, 3, 3, 1)]

    first = [[s.rows for s in g] for g in _balance(build(), 3)]
    second = [[s.rows for s in g] for g in _balance(build(), 3)]
    assert first == second


def test_every_split_is_assigned_exactly_once():
    splits = [_CheapSplit(i) for i in range(37)]
    groups = _balance(splits, 5)
    assert sorted(id(s) for g in groups for s in g) == sorted(id(s) for s in splits)


def test_no_splits_still_yields_one_group_per_worker():
    assert _balance([], 3) == [[], [], []]
