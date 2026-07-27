"""Kyber rule modules.

Importing this package registers every rule it contains into the default
registry (via the `@rule` decorator). Each module groups related rules by kind;
`kyber.registry` imports this package so the default optimizer sees them. New
rules are added by dropping a decorated function into one of these modules (or a
new one imported here) — nothing else changes.
"""

from __future__ import annotations

from batcher.kyber.gpu import sizing as _gpu_sizing  # noqa: F401  (registers the GPU sizing rule)
from batcher.kyber.rules import agg_algebra as _agg_algebra  # noqa: F401  (registers rules)
from batcher.kyber.rules import agg_pushdown as _agg_pushdown  # noqa: F401  (registers rules)
from batcher.kyber.rules import algebraic as _algebraic  # noqa: F401  (registers rules on import)
from batcher.kyber.rules import (
    exprs as _exprs,  # noqa: F401  (registers the expression-algebra families)
)
from batcher.kyber.rules import extra as _extra  # noqa: F401  (registers the extended families)
from batcher.kyber.rules import fusion as _fusion  # noqa: F401  (rule bodies)

# Registration order *is* within-phase run order, so the join family's submodules are
# imported exactly where the former flat modules were — sorting them would move
# `join_reorder` / `push_projection_through_join` behind the rules they used to precede.
# isort: off
from batcher.kyber.rules.joins import order as _join_order  # noqa: F401  (registers rules)
from batcher.kyber.rules.joins import projection as _join_projection  # noqa: F401  (registers rules)
from batcher.kyber.rules import joins as _joins  # noqa: F401  (registers the rewrites family)
from batcher.kyber.rules.joins import agg_semijoin as _agg_semijoin  # noqa: F401  (registers rules)

# isort: on
from batcher.kyber.rules import normalize as _normalize  # noqa: F401  (rule bodies)
from batcher.kyber.rules import ordering as _ordering  # noqa: F401  (registers rules)
from batcher.kyber.rules import projections as _projections  # noqa: F401  (registers rules)
from batcher.kyber.rules import pushdown as _pushdown  # noqa: F401  (registers rules)

# The range-join rewrite reads the *residue* of `derive_join_keys` (in `pushdown`), which
# absorbs equi-conjuncts into real join keys first — an equality is worth more than an
# inequality — so it registers after it rather than with the rest of the join family.
# isort: off
from batcher.kyber.rules.joins import range_join as _range_join  # noqa: F401  (registers rules)

# isort: on
from batcher.kyber.rules import (
    relational as _relational,  # noqa: F401  (registers the relational families)
)
from batcher.kyber.rules import selection as _selection  # noqa: F401  (rule bodies)
from batcher.kyber.rules import streaming as _streaming  # noqa: F401  (registers rules)
from batcher.kyber.rules import zonemap_pruning as _zonemap  # noqa: F401  (registers rules)

__all__: list[str] = []
