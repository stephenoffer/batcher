"""The SQL `Session`: a table catalog, a Python-function registry, and a read dialect."""

from __future__ import annotations

from batcher.api.sql_session.registry import RegisteredFunction
from batcher.api.sql_session.session import Session

__all__ = ["RegisteredFunction", "Session"]
