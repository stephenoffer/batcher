"""Catalogs and tables: named, persistent (or in-memory) tables a session resolves names against.

`Catalog` owns namespaces and tables over one storage backend, `Table` is a handle on one of
them, and `SessionCatalog` is ``session.catalog``: the catalogs a `Session` has attached and
which one, and which namespace, an unqualified name resolves into.
"""

from __future__ import annotations

from batcher.api.catalog.catalog import Catalog
from batcher.api.catalog.session_catalog import SessionCatalog
from batcher.api.catalog.table import Table

__all__ = ["Catalog", "SessionCatalog", "Table"]
