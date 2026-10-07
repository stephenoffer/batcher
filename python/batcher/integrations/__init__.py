"""Adapters that let other tools drive Batcher: SQLAlchemy, dbt, Ibis, and Flight SQL clients.

Each subpackage speaks one external tool's contract over Batcher's public SQL surface, the
`Session` and the PEP 249 adapter in `batcher.dbapi`, and needs that tool installed through
its own extra. Nothing here is imported by ``import batcher``: a subpackage loads only when
its tool asks for it, through an entry point or an explicit import.

This is the `integrations` layer, the top of the import matrix. It may import `dbapi` and
`api`; nothing in Batcher imports it.
"""
