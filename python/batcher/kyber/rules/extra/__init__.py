"""Extended Kyber rule families.

Grouped-by-family rule modules that expand the optimizer beyond the original set —
boolean/arithmetic expression algebra, sargable and temporal-sargable normalization,
predicate inference, set-op and join structural rewrites, limit/top-N, aggregate,
projection/scan and window simplification, and metadata-JIT (provenance-aware) rules.
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
from batcher.kyber.rules.extra import arith_algebra as _arith_algebra  # noqa: F401
from batcher.kyber.rules.extra import boolean_algebra as _boolean_algebra  # noqa: F401
from batcher.kyber.rules.extra import empty_relation as _empty_relation  # noqa: F401
from batcher.kyber.rules.extra import join_extra as _join_extra  # noqa: F401
from batcher.kyber.rules.extra import metadata_adaptive as _metadata_adaptive  # noqa: F401
from batcher.kyber.rules.extra import predicate_infer as _predicate_infer  # noqa: F401
from batcher.kyber.rules.extra import projection_scan as _projection_scan  # noqa: F401
from batcher.kyber.rules.extra import sargable as _sargable  # noqa: F401
from batcher.kyber.rules.extra import setops as _setops  # noqa: F401
from batcher.kyber.rules.extra import temporal_sargable as _temporal_sargable  # noqa: F401
from batcher.kyber.rules.extra import topn_limit as _topn_limit  # noqa: F401
from batcher.kyber.rules.extra import window_rules as _window_rules  # noqa: F401

__all__: list[str] = []
