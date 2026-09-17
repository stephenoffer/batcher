"""The `sem.<name>` transforms registry templates call for what the template DSL cannot say.

Declining is the conservative answer to anything not provably equivalent (a pattern held in a
variable, a join condition that is a `Column`, a sort key the inference cannot type), and a
declined template leaves the call alone with a marker. The transforms are grouped by what they
restore: `columns` (column references, argument checks, positions, date patterns), `ordering`
(sort and window keys) and `relational` (joins, writes, sessions, constructors, aggregates).
"""

from __future__ import annotations

from batcher.migrate.semantics import columns, ordering, relational
from batcher.migrate.semantics.base import TRANSFORMS, Context, lookup
from batcher.migrate.semantics.columns import java_to_strftime

__all__ = [
    "TRANSFORMS",
    "Context",
    "columns",
    "java_to_strftime",
    "lookup",
    "ordering",
    "relational",
]
