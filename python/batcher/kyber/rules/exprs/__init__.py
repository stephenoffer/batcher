"""Expression-level Kyber rule families.

One module per algebraic surface the engine exposes -- numeric identities, complex
types, strings, temporal values, lists, hashing, and conditional forms. These sit
alongside `rules/extra`, which holds the earlier families; the split keeps each
directory inside the file-count cap while both grow, and keeps a rule discoverable
from the shape it rewrites.

Importing this package runs each module's ``@rule`` decorators, registering every
rule into ``kyber.registry.DEFAULT_REGISTRY``. Re-export and registration only -- no
logic here.
"""

from __future__ import annotations

from batcher.kyber.rules.exprs import boolean_normalize as _boolean_normalize  # noqa: F401
from batcher.kyber.rules.exprs import cast_unwrap as _cast_unwrap  # noqa: F401
from batcher.kyber.rules.exprs import comparisons as _comparisons  # noqa: F401
from batcher.kyber.rules.exprs import complex_types as _complex_types  # noqa: F401
from batcher.kyber.rules.exprs import conditionals as _conditionals  # noqa: F401
from batcher.kyber.rules.exprs import numeric as _numeric  # noqa: F401

# Registration order is run order: `round_with_zero_digits` registered directly after the
# rest of the numeric family when the two shared a module.
# isort: off
from batcher.kyber.rules.exprs import numeric_rounding as _numeric_rounding  # noqa: F401

# isort: on
from batcher.kyber.rules.exprs import temporal as _temporal  # noqa: F401
from batcher.kyber.rules.exprs import text as _text  # noqa: F401
from batcher.kyber.rules.exprs import text_algebra as _text_algebra  # noqa: F401
from batcher.kyber.rules.exprs import text_folds as _text_folds  # noqa: F401

__all__: list[str] = []
