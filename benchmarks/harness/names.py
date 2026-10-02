"""The name a result column is compared under.

A derived column with no alias has no name in the query, so each engine invents one — and
they disagree in ways that are pure spelling. This is the one definition of what those
spellings have in common; `compare` keys its type reconciliation and its row comparison on
it, and `order` resolves an `ORDER BY` term to an output column with it.
"""

from __future__ import annotations

import re

import pyarrow as pa

__all__ = ["canonical_column_name", "canonical_names"]

# A derived column with no alias has no name in the query, so each engine invents one, and
# they disagree in ways that are pure spelling: DuckDB qualifies a built-in with its catalog
# and quotes it (``main."substring"(s_city, 1, 30)`` against ``substring(s_city, 1, 30)``) and
# parenthesizes sub-expressions it did not have to (``round((a / b), 2)`` against
# ``round(a / b, 2)``, ``((cast(a) / cast(b)) * 100)`` against ``cast(a) / cast(b) * 100``).
#
# `column_classes` already lowercased names for exactly this reason — the engines disagree on
# a generated name's *case* — but that covered only one of the three ways they disagree, so
# TPC-DS q2, q61, q79 and q85 were each reported as a correctness FAILURE over data that
# matched. Squeezing out the catalog prefix, the quotes, the whitespace and the parentheses
# leaves the one thing both engines do agree on, and a genuinely different column set still
# fails: two columns that squeeze to one name are two spellings of the same expression, and
# if they were not, the values would then disagree and the row would fail anyway.
_CATALOG_PREFIX = re.compile(r"\bmain\.")
_DROPPED_PUNCTUATION = str.maketrans("", "", ' "()')


def canonical_column_name(name: str) -> str:
    """The name a column is compared under, with each engine's spelling squeezed out."""
    return _CATALOG_PREFIX.sub("", name.lower()).translate(_DROPPED_PUNCTUATION)


def canonical_names(table: pa.Table) -> list[str]:
    """`table`'s column names canonicalized, falling back step by step until they are unique.

    Two columns of one result sharing a comparison name would silently drop one of them from
    the comparison, because the rowset is built as a name-keyed mapping. That is the one
    outcome worse than the false failure canonicalization fixes: with columns ``x`` and
    ``X``, the lowercased fallback that used to stand here collided too, and changing ``X``
    from 2 to 999 still passed. So the fallbacks are canonical, then lowercased, then the
    names exactly as given, and finally the exact names with a positional suffix on any
    repeat (Arrow permits duplicate names).

    The squeeze drops only spaces, quotes and parentheses, and only when doing so leaves
    every name in the result distinct, so two user aliases differing only in those
    characters are never merged *within* one result. Across engines a squeezed match only
    pairs the columns up; their values are still compared.
    """
    for candidate in (
        [canonical_column_name(n) for n in table.column_names],
        [n.lower() for n in table.column_names],
        list(table.column_names),
    ):
        if len(set(candidate)) == len(candidate):
            return candidate
    used: set[str] = set()
    unique = []
    for name in table.column_names:
        label, count = name, 0
        while label in used:
            count += 1
            label = f"{name}#{count}"
        used.add(label)
        unique.append(label)
    return unique
