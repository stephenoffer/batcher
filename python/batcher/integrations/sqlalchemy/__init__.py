"""A SQLAlchemy 2.0 dialect over `batcher.dbapi`, registered as the ``batcher://`` URL scheme.

Not yet verified against a live SQLAlchemy application beyond this repository's tests; see
tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

from batcher.integrations.sqlalchemy.dialect import BatcherDialect

__all__ = ["BatcherDialect"]
