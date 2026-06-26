"""Engine adapters: one comparator per engine, behind a common contract.

Public surface re-exported here; the registry/resolution logic lives in ``lineup``
and the contract in ``base``.
"""

from __future__ import annotations

from .base import Engine
from .lineup import default_names, get, resolve

__all__ = ["Engine", "default_names", "get", "resolve"]
