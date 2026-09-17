"""The migration registry: every PySpark, Polars, Daft and Ray Data name, and its Batcher spelling.

Layer 0, so both the `Expr` guidance (layer 1) and the `Dataset` guidance (layer 5) can read
it, as can the `batcher.migrate` codemod and the tools that generate the migration docs.
Not part of the public API.
"""

from __future__ import annotations

from batcher._internal.migration.loader import (
    DATA_DIR,
    Registry,
    load_codemod_tables,
    load_registry,
    load_returns,
)
from batcher._internal.migration.renames import (
    OPERATORS,
    TRANSFORMS,
    KwargRename,
    Rename,
    load_kwarg_renames,
    load_renames,
)
from batcher._internal.migration.schema import (
    ENGINES,
    WAVES,
    Mapping,
    RegistryError,
    Status,
    Template,
    parse_template,
)

__all__ = [
    "DATA_DIR",
    "ENGINES",
    "OPERATORS",
    "TRANSFORMS",
    "WAVES",
    "KwargRename",
    "Mapping",
    "Registry",
    "RegistryError",
    "Rename",
    "Status",
    "Template",
    "load_codemod_tables",
    "load_kwarg_renames",
    "load_registry",
    "load_renames",
    "load_returns",
    "parse_template",
]
