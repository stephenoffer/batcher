"""Kyber rule modules.

Importing this package registers every rule it contains into the default
registry (via the `@rule` decorator). Each module groups related rules by kind;
`kyber.registry` imports this package so the default optimizer sees them. New
rules are added by dropping a decorated function into one of these modules (or a
new one imported here) — nothing else changes.
"""

from __future__ import annotations

from batcher.kyber.rules import algebraic as _algebraic  # noqa: F401  (registers rules on import)
from batcher.kyber.rules import fusion as _fusion  # noqa: F401  (rule bodies)
from batcher.kyber.rules import join_order as _join_order  # noqa: F401  (registers rules)
from batcher.kyber.rules import joins as _joins  # noqa: F401  (registers rules)
from batcher.kyber.rules import normalize as _normalize  # noqa: F401  (rule bodies)
from batcher.kyber.rules import ordering as _ordering  # noqa: F401  (registers rules)
from batcher.kyber.rules import projections as _projections  # noqa: F401  (registers rules)
from batcher.kyber.rules import pushdown as _pushdown  # noqa: F401  (registers rules)
from batcher.kyber.rules import selection as _selection  # noqa: F401  (rule bodies)
from batcher.kyber.rules import zonemap_pruning as _zonemap  # noqa: F401  (registers rules)

__all__: list[str] = []
