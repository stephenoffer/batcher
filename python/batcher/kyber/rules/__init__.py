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
from batcher.kyber.rules import (
    aggregate_algebra as _aggregate_algebra,  # noqa: F401  (registers the extreme families)
)
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
from batcher.kyber.rules import (
    collections_algebra as _collections_algebra,  # noqa: F401  (registers the list families)
)
from batcher.kyber.rules import (
    conditional_algebra as _conditional_algebra,  # noqa: F401  (registers the CASE families)
)
from batcher.kyber.rules import (
    math_algebra as _math_algebra,  # noqa: F401  (registers the numeric-range families)
)
from batcher.kyber.rules import normalize as _normalize  # noqa: F401  (rule bodies)
from batcher.kyber.rules import nulls as _nulls  # noqa: F401  (registers the null families)
from batcher.kyber.rules import ordering as _ordering  # noqa: F401  (registers rules)
from batcher.kyber.rules import (
    predicate_algebra as _predicate_algebra,  # noqa: F401  (registers the bound families)
)
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
from batcher.kyber.rules import (
    temporal_algebra as _temporal_algebra,  # noqa: F401  (registers the instant families)
)
from batcher.kyber.rules import (
    text_algebra as _text_algebra,  # noqa: F401  (registers the string families)
)
from batcher.kyber.rules import (
    window_algebra as _window_algebra,  # noqa: F401  (registers the window families)
)
from batcher.kyber.rules import zonemap_pruning as _zonemap  # noqa: F401  (registers rules)

__all__: list[str] = []
