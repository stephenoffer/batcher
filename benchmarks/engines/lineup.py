"""The engine lineup: which adapters exist, the per-tier defaults, and name resolution.

Kept out of ``__init__`` (a re-export shim) so the registry logic lives in a named
module. ``resolve`` turns a user-selected list of engine names into the concrete
adapters importable in this environment; ``default_names`` gives the per-tier default
lineup (Spark is opt-in everywhere because of its JVM startup cost).
"""

from __future__ import annotations

from .base import Engine
from .batcher import BatcherEngine
from .daft import DaftEngine
from .duckdb import DuckDBEngine
from .polars import PolarsEngine
from .pyarrow import PyArrowEngine
from .ray import RayEngine
from .spark import SparkEngine

# Registration order is also the report column order.
_ADAPTERS: dict[str, Engine] = {
    e.name: e
    for e in (
        BatcherEngine(),
        DuckDBEngine(),
        PolarsEngine(),
        PyArrowEngine(),
        SparkEngine(),
        DaftEngine(),
        RayEngine(),
    )
}

# Default lineups per tier. Batcher leads (it is the system under test); Spark is
# omitted from every default and added only when explicitly requested.
_DEFAULT_SINGLE = ("batcher", "duckdb", "polars", "pyarrow")
_DEFAULT_MULTI = ("batcher", "ray", "daft")


def get(name: str) -> Engine:
    """Return the adapter registered under ``name`` (raises ``KeyError`` if unknown)."""
    return _ADAPTERS[name]


def default_names(tier: str) -> list[str]:
    """The default engine lineup for ``tier`` (``"single"`` or ``"multi"``)."""
    return list(_DEFAULT_MULTI if tier == "multi" else _DEFAULT_SINGLE)


def resolve(names: list[str]) -> list[Engine]:
    """Map names to adapters, keeping only those importable here (others are dropped).

    Batcher is always kept (it is the package under test). An unknown name raises;
    a known-but-uninstalled engine is silently skipped so a partial environment
    still runs the engines it has.
    """
    out: list[Engine] = []
    for name in names:
        engine = _ADAPTERS[name]  # KeyError on a typo'd engine name
        if engine.available() or name == "batcher":
            out.append(engine)
    return out
