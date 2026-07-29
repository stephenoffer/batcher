"""Extended Kyber rule families.

Grouped-by-family rule modules that expand the optimizer beyond the original set —
boolean/arithmetic expression algebra, sargable and temporal-sargable normalization,
predicate inference, set-op and join structural rewrites, limit/top-N, aggregate,
projection/scan and window simplification, cost-based filter splitting, and
metadata-JIT (provenance-aware) rules.
Kept in this subpackage so the parent ``rules/`` directory stays within the file-count
cap while the families grow (the dir itself is allowlisted in ``tools/lint_structure.py``
as the sanctioned "many small things" pattern for the optimizer's large rule set).

Importing this package runs each module's ``@rule`` decorators, registering every rule
into ``kyber.registry.DEFAULT_REGISTRY``. ``kyber.rules`` imports this package, so the
default optimizer sees them. Re-export/registration only — no logic here.
"""

from __future__ import annotations

from batcher.kyber.rules.extra import adaptive_meta as _adaptive_meta  # noqa: F401
from batcher.kyber.rules.extra import agg_extra as _agg_extra  # noqa: F401
from batcher.kyber.rules.extra import agg_rules as _agg_rules  # noqa: F401
from batcher.kyber.rules.extra import arith_algebra as _arith_algebra  # noqa: F401
from batcher.kyber.rules.extra import arith_extra as _arith_extra  # noqa: F401
from batcher.kyber.rules.extra import boolean_algebra as _boolean_algebra  # noqa: F401
from batcher.kyber.rules.extra import casts as _casts  # noqa: F401
from batcher.kyber.rules.extra import conditional as _conditional  # noqa: F401
from batcher.kyber.rules.extra import cse as _cse  # noqa: F401
from batcher.kyber.rules.extra import disjunction_infer as _disjunction_infer  # noqa: F401
from batcher.kyber.rules.extra import empty_relation as _empty_relation  # noqa: F401
from batcher.kyber.rules.extra import filter_split as _filter_split  # noqa: F401
from batcher.kyber.rules.extra import join_elim as _join_elim  # noqa: F401
from batcher.kyber.rules.extra import join_extra as _join_extra  # noqa: F401
from batcher.kyber.rules.extra import limit_extra as _limit_extra  # noqa: F401
from batcher.kyber.rules.extra import metadata_adaptive as _metadata_adaptive  # noqa: F401
from batcher.kyber.rules.extra import nullability as _nullability  # noqa: F401

# Registration order is run order: the shape-driven null rules registered directly after
# the nullability-driven ones when the two shared a module.
# isort: off
from batcher.kyber.rules.extra import null_shapes as _null_shapes  # noqa: F401

# isort: on
from batcher.kyber.rules.extra import predicate_infer as _predicate_infer  # noqa: F401
from batcher.kyber.rules.extra import projection_scan as _projection_scan  # noqa: F401
from batcher.kyber.rules.extra import pushdown_gaps as _pushdown_gaps  # noqa: F401
from batcher.kyber.rules.extra import runtime_filters as _runtime_filters  # noqa: F401
from batcher.kyber.rules.extra import sargable as _sargable  # noqa: F401
from batcher.kyber.rules.extra import setops as _setops  # noqa: F401
from batcher.kyber.rules.extra import setops_extra as _setops_extra  # noqa: F401
from batcher.kyber.rules.extra import strings as _strings  # noqa: F401

# Registration order is run order: the folding half of the string family registered
# immediately after the pattern half when the two shared one module, so it is imported
# here rather than in sorted position.
# isort: off
from batcher.kyber.rules.extra import string_folds as _string_folds  # noqa: F401

# isort: on
from batcher.kyber.rules.extra import temporal_extra as _temporal_extra  # noqa: F401

# Imported immediately after `temporal_extra`: the folding rules were split out of it and
# must keep registering in the same position, since registration order is run order.
from batcher.kyber.rules.extra import temporal_folds as _temporal_folds  # noqa: F401
from batcher.kyber.rules.extra import temporal_sargable as _temporal_sargable  # noqa: F401
from batcher.kyber.rules.extra import topn_limit as _topn_limit  # noqa: F401
from batcher.kyber.rules.extra import window_extra as _window_extra  # noqa: F401
from batcher.kyber.rules.extra import window_rules as _window_rules  # noqa: F401

__all__: list[str] = []
