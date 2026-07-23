"""Ordering shortcuts — what the data is already sorted by, so a sort need not prove it.

A source can declare a physical ordering it maintains (a clustered table, a sorted Parquet
write, a lakehouse table with a sort key). `RelStats.sorted_by` carries it through every
order-preserving operator, and its contract is deliberately narrow: it lists only a
**canonical ascending, nulls-last prefix** — the one ordering a producer and a consumer can
compare without ambiguity. A descending or nulls-first ordering is simply not recorded.

That narrowness is why every function here is *one-sided*: a match proves sortedness, but the
absence of a match proves nothing (the data may well be sorted and unrecorded). So they
return True or None — never False. A caller that needs a definite "no" must execute, and a
caller that only wants to *skip work when it can* gets exactly what it needs.
"""

from __future__ import annotations

from collections.abc import Sequence

from batcher.kyber.shortcuts.facts import Facts

__all__ = ["is_sorted_by", "sort_prefix", "sorted_columns"]


def sorted_columns(facts: Facts) -> tuple[str, ...]:
    """The columns the relation is known to be ascending, nulls-last ordered by.

    Empty means "no recorded ordering", which is not the same as "unordered".
    """
    return facts.sorted_by


def is_sorted_by(facts: Facts, columns: Sequence[str]) -> bool | None:
    """Whether the relation is known to be sorted by `columns`, or None if not recorded.

    True only when `columns` is a **prefix** of the recorded ordering: a relation sorted by
    ``(region, day)`` is sorted by ``(region,)``, but a relation sorted by ``(region,)`` says
    nothing about ``day``. Never returns False — see the module docstring.
    """
    keys = tuple(columns)
    if not keys:
        return True  # sorting by nothing is trivially satisfied
    if tuple(facts.sorted_by[: len(keys)]) == keys:
        return True
    return None


def sort_prefix(facts: Facts, columns: Sequence[str]) -> int:
    """How many leading `columns` the relation is already sorted by — the work a sort can skip.

    Zero means the sort must do everything; `len(columns)` means it is a no-op. A partial
    match is the interesting case: a relation already sorted by ``region`` needs only to sort
    *within* each region to reach ``(region, day)``.
    """
    matched = 0
    for wanted, have in zip(columns, facts.sorted_by, strict=False):
        if wanted != have:
            break
        matched += 1
    return matched
