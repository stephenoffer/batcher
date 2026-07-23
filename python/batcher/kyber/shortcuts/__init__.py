"""EXACT-gated metadata shortcuts (façade) — the answers that need no scan.

`facts_for` turns a plan into the `Facts` that are provable without execution; the family
modules (`rows`, `nulls`, `bounds`, `distinct`, `moments`, `checks`, `ordering`, `approx`,
`storage`) are pure derivations over it. Every exact answer is `None` when it cannot be
proved, and `None` means the caller executes — so a shortcut is only ever an optimisation.

Import the families as modules (``from batcher.kyber import shortcuts`` then
``shortcuts.checks.all_positive(facts, "x")``) rather than re-exporting a hundred names into
one namespace: the family is part of the meaning, and a flat wall of names is exactly the
`__init__` dump the structure rules exist to prevent.
"""

from __future__ import annotations

from batcher.kyber.shortcuts import (
    approx,
    bounds,
    checks,
    distinct,
    joins,
    moments,
    nulls,
    ordering,
    rows,
    storage,
)
from batcher.kyber.shortcuts.facts import ColumnFacts, Facts, facts_for, facts_from_relstats

__all__ = [
    "ColumnFacts",
    "Facts",
    "approx",
    "bounds",
    "checks",
    "distinct",
    "facts_for",
    "facts_from_relstats",
    "joins",
    "moments",
    "nulls",
    "ordering",
    "rows",
    "storage",
]
