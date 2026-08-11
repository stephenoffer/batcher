"""Ordering shortcuts — what the data is already sorted by, so a sort need not prove it.

A source can declare a physical ordering it maintains (a clustered table, a sorted Parquet
write, a lakehouse table with a sort key). `RelStats.sorted_by` carries it through every
order-preserving operator as a prefix of `SortOrder` keys, each naming a column, a
direction, and its null placement — so a descending ordering is recorded and usable, not
discarded.

Every function here is *one-sided*: a match proves sortedness, but the absence of a match
proves nothing (the data may well be sorted and unrecorded). So they return True or None —
never False. A caller that needs a definite "no" must execute, and a caller that only wants
to *skip work when it can* gets exactly what it needs.
"""

from __future__ import annotations

from collections.abc import Sequence

from batcher.kyber.shortcuts.facts import Facts
from batcher.plan.stats import SortOrder, as_sort_orders, orderings_satisfy

__all__ = ["is_sorted_by", "sort_prefix", "sorted_columns"]


def sorted_columns(facts: Facts) -> tuple[SortOrder, ...]:
    """The ordering the relation is known to be stored in, direction included.

    Empty means "no recorded ordering", which is not the same as "unordered".
    """
    return facts.sorted_by


def is_sorted_by(facts: Facts, columns: Sequence[SortOrder | str]) -> bool | None:
    """Whether the relation is known to be sorted by `columns`, or None if not recorded.

    True only when `columns` is a **prefix** of the recorded ordering: a relation sorted by
    ``(region, day)`` is sorted by ``(region,)``, but a relation sorted by ``(region,)`` says
    nothing about ``day``. A bare column name means ascending, nulls-last. Never returns
    False — see the module docstring.
    """
    keys = as_sort_orders(columns)
    if not keys:
        return True  # sorting by nothing is trivially satisfied
    non_nullable = frozenset(name for name, col in facts.columns.items() if col.null_count == 0)
    if orderings_satisfy(facts.sorted_by, keys, non_nullable=non_nullable):
        return True
    return None


def sort_prefix(facts: Facts, columns: Sequence[SortOrder | str]) -> int:
    """How many leading `columns` the relation is already sorted by — the work a sort can skip.

    Zero means the sort must do everything; `len(columns)` means it is a no-op. A partial
    match is the interesting case: a relation already sorted by ``region`` needs only to sort
    *within* each region to reach ``(region, day)``.
    """
    wanted = as_sort_orders(columns)
    matched = 0
    for want, have in zip(wanted, facts.sorted_by, strict=False):
        if want != have:
            break
        matched += 1
    return matched
