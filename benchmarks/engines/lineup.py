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
from .duckdb_arrow import DuckDBArrowEngine
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
        DuckDBArrowEngine(),
        PolarsEngine(),
        PyArrowEngine(),
        SparkEngine(),
        DaftEngine(),
        RayEngine(),
    )
}

# Default lineups per tier. Batcher leads (it is the system under test); Spark is
# omitted from every default and added only when explicitly requested.
#
# Daft is in *both* tiers. It was previously multi-node only, on the assumption that it is
# a distributed engine — but Daft's single-node runner is its default and the one most
# users meet, and `performance.md` names Daft as a system Batcher claims to beat. A claim
# nobody's default run measures is a claim nobody checks, which is exactly how a regression
# against it would survive. It costs one more column; `resolve` drops it where Daft is not
# installed, so a partial environment is unaffected.
#
# `ray` is registered above but is in no default lineup: `performance.md` no longer names
# it among the systems Batcher positions against, so a default run must not emit a column
# that reads as a published comparison. Pass `--engines ...,ray` to measure against it
# deliberately.
# `duckdb_arrow` is **not** here, and the reason is a measurement rather than an oversight.
# The README's methodology says both DuckDB bars are reported — `duckdb` on its native
# compressed store ("DuckDB at its best") and `duckdb_arrow` on the same zero-copy Arrow
# Batcher executes over ("the like-for-like execution bar") — so it was added to this lineup,
# and TPC-DS then could not finish: DuckDB over registered Arrow views has no storage
# statistics to order a many-way join with, and it was **SIGKILLed on q64** (`rc=137`, on a
# 184 GiB box) and still climbing through 22 GiB after 30 minutes on q72. Both answer in
# tens of milliseconds against DuckDB's *native* tables, so this is the Arrow-view binding
# and not the queries.
#
# One engine dying that way takes the whole suite's process down with it, so a default run
# that includes it reports nothing for 99 queries. It stays a deliberate `--engines
# batcher,duckdb,duckdb_arrow` opt-in, and `BENCHMARK_RESULTS.md` carries both bars for the
# suites where it survives. `b/duckdb` remains the number to quote either way.
_DEFAULT_SINGLE = ("batcher", "duckdb", "polars", "pyarrow", "daft")
_DEFAULT_MULTI = ("batcher", "daft")


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
