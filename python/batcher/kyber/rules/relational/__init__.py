"""Relational Kyber rule families -- rewrites that move or reshape a plan node.

Distinct from `rules/exprs`, which rewrites the expressions a node carries: these
rules change the tree. One module per operator family, kept here so `rules/` stays
inside the file-count cap while both groups grow.

Importing this package runs each module's ``@rule`` decorators, registering every rule
into ``kyber.registry.DEFAULT_REGISTRY``. Re-export and registration only -- no logic
here.
"""

from __future__ import annotations

from batcher.kyber.rules.relational import windows as _windows  # noqa: F401

__all__: list[str] = []
