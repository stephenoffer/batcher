"""Benchmark suites, grouped one module per query family.

Importing this package imports every family module, which runs their registration
decorators and populates ``registry.REGISTRY``. To add a family, create a module
here and add it to ``load_all`` below; to add a query, add a ``@<family>.case``
function to the matching module. The file count stays flat as the suite grows.
"""

from __future__ import annotations


def load_all() -> None:
    """Import every family module so all cases register themselves."""
    from . import (  # noqa: F401  (imported for registration side effects)
        aggregation,
        conditional,
        dates,
        filtering,
        joins,
        mathfns,
        ordering,
        projection,
        ray_data,
        setops,
        strings,
        tpch,
        window,
    )
