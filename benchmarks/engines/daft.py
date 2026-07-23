"""Daft adapter — distributed DataFrame comparator with a best-effort SQL surface.

Daft participates in the operator-mix (native ``DataFrame`` handle) and in the SQL
suites where its SQL planner can express the query. The SQL registration API has
shifted across Daft versions, so ``sql_runner`` tries the known shapes and degrades
to ``None`` (the suite then omits Daft) rather than crashing the run.
"""

from __future__ import annotations

import importlib.util
import os

import pyarrow as pa

from .base import Engine, Rename, SqlRunner

# Whether this run is the distributed (multi-node) tier. Daft defaults to its LOCAL
# native runner, so without this it would answer on the driver's cores while Batcher
# (``BENCH_BATCHER_DISTRIBUTED=1``) and Ray Data (``address="auto"``) both spread across
# the whole cluster — a 16-core engine timed against 128-core ones, which flatters
# Batcher and makes the multi-tier table meaningless. Reading Batcher's flag as well as
# the neutral one means an existing distributed invocation puts Daft on the cluster too,
# rather than silently keeping the old unfair comparison.
_DISTRIBUTED: bool = (
    os.environ.get("BENCH_DISTRIBUTED") == "1" or os.environ.get("BENCH_BATCHER_DISTRIBUTED") == "1"
)
_runner_selected = False


def _ensure_runner() -> None:
    """Put Daft on the same Ray cluster the other distributed engines use (once).

    Must run before Daft executes anything, so every entry point that produces a Daft
    handle or runner calls it first.

    This raises rather than degrading to the local runner, and that is deliberate: a
    silently-local Daft still *produces numbers*, and they would be a 16-core engine timed
    against 128-core ones and reported as if the comparison were fair. A loud failure is the
    only safe behavior. (An earlier version swallowed the error here and did exactly that —
    Daft moved `set_runner_ray` from `daft.context` to the top level in 0.7, so the call
    silently no-opped and the "distributed" Daft column was really local.)
    """
    global _runner_selected
    if _runner_selected or not _DISTRIBUTED:
        return
    import daft

    setter = getattr(daft, "set_runner_ray", None) or getattr(daft.context, "set_runner_ray", None)
    if setter is None:  # pragma: no cover - depends on the installed Daft
        raise RuntimeError(
            "this Daft exposes no set_runner_ray, so it cannot be put on the Ray cluster; "
            "a distributed-tier run would compare a local Daft against cluster engines"
        )
    setter(address=os.environ.get("BENCH_RAY_ADDRESS", "auto"))
    _runner_selected = True


class DaftEngine(Engine):
    name = "daft"
    tier = "multi"
    supports_sql = True

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("daft") is not None

    def handle(self, table: pa.Table):
        import daft

        _ensure_runner()
        return daft.from_arrow(table)

    def read_parquet(self, uri: str):
        import daft

        _ensure_runner()
        return daft.read_parquet(uri)

    def sql_runner(self, tables: dict[str, pa.Table]) -> SqlRunner | None:
        import daft

        _ensure_runner()
        frames = {name: daft.from_arrow(tbl) for name, tbl in tables.items()}
        # Current Daft: named DataFrames are passed to daft.sql as bindings.
        return lambda query: daft.sql(query, **frames).to_arrow()

    def sql_runner_scan(
        self, uris: dict[str, str], rename: Rename | None = None
    ) -> SqlRunner | None:
        import daft

        _ensure_runner()
        frames = {}
        for name, uri in uris.items():
            frame = daft.read_parquet(uri)
            cols = (rename or {}).get(name)
            if cols:
                frame = frame.select(
                    *(daft.col(stored).alias(canonical) for stored, canonical in cols.items())
                )
            frames[name] = frame
        return lambda query: daft.sql(query, **frames).to_arrow()

    def scan_sql_runner(self, glob: str) -> SqlRunner:
        import daft

        _ensure_runner()

        def run(query: str) -> pa.Table:
            return daft.sql(query, t=daft.read_parquet(glob)).to_arrow()

        return run
