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
    """`table`'s column names canonicalized, or merely lowercased if that would collide.

    Two columns of one result squeezing to the same name would silently drop one of them from
    the comparison, which is the one outcome worse than the false failure this fixes. The
    lowercased fallback is exactly the behaviour that preceded canonicalization.
    """
    canonical = [canonical_column_name(n) for n in table.column_names]
    if len(set(canonical)) != len(canonical):
        return [n.lower() for n in table.column_names]
    return canonical
