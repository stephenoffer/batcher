"""Isolated single (engine, query) worker: fresh process, mmap tables, time best-of-N.

Prints one JSON line: {"ms": float|null, "rows": int, "sig": [...], "err": str|null}.
Run via ``run.py``, which spawns one of these per (engine, query) so no cross-query process
state can inflate any engine's timing. Engines come from the shared ``engines/`` adapters,
so they are configured exactly as they are everywhere else in the suite.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pyarrow as pa
import pyarrow.feather as feather

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envinfo import machine_fingerprint, require_quiet_box, require_release_build
from signature import result_signature

#: Where the memory-mappable copy of the TPC-H tables lives. Overridable so a run can point
#: at a faster disk; created on first use.
_DEFAULT_FEATHER = "~/bench-data/tpch-feather"
FEATHER_CACHE = os.path.expanduser(os.environ.get("BENCH_ISO_FEATHER", _DEFAULT_FEATHER))


def _ensure_feather(scale: int, names: list[str]) -> str:
    """Materialize the TPC-H tables as Feather under :data:`FEATHER_CACHE`, once.

    Feather rather than the shared Arrow the rest of the suite uses, because this harness
    spawns a process per (engine, query) — the whole point — and a process that re-read
    parquet from S3 would spend more time loading than measuring. `memory_map=True` then
    makes the per-process load a page-table operation instead of a copy.

    This used to read a hard-coded ``/home/ray/tpch_feather/sf{scale}`` that no code in the
    repository wrote, so every invocation died in :func:`_load` and the harness had never
    produced a number — while its docstring described it as "the fair, official-benchmark
    method". It is now built from the same :func:`sources.load_tables` every other benchmark
    reads, so what it measures is the same data the rest of the suite measures.
    """
    target = os.path.join(FEATHER_CACHE, f"sf{_sf(scale)}")
    if all(os.path.exists(os.path.join(target, f"{n}.feather")) for n in names):
        return target
    from sources import load_tables

    os.makedirs(target, exist_ok=True)
    tables = load_tables("tpch", float(scale), None)
    for name in names:
        feather.write_feather(tables[name], os.path.join(target, f"{name}.feather"))
    return target


def _sf(scale: float) -> str | int:
    return int(scale) if float(scale).is_integer() else scale


def _load(scale: int, names: list[str]) -> dict[str, pa.Table]:
    base = _ensure_feather(scale, names)
    return {
        n: feather.read_table(os.path.join(base, f"{n}.feather"), memory_map=True) for n in names
    }


_TPCH_TABLES = (
    "region",
    "nation",
    "supplier",
    "customer",
    "part",
    "partsupp",
    "orders",
    "lineitem",
)


def main() -> None:
    # Refuse to time a dev-profile engine: it is 8-60x slower, so a number taken from one
    # compares an unoptimized Batcher against release competitors. `BENCH_ALLOW_DEBUG_BUILD=1`
    # overrides deliberately.
    require_release_build()
    # Print the machine before any number: a timing is only reproducible beside the
    # box that produced it, and this file's own history has ratios quoted across four
    # different machines as if they were comparable.
    print(machine_fingerprint())
    # ...and refuse a contended one: a neighbour's load is not a fact about any
    # engine. `BENCH_ALLOW_BUSY_BOX=1` overrides.
    require_quiet_box()
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--scale", type=int, required=True)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--sql", required=True, help="the SQL text")
    args = p.parse_args()

    out = {"ms": None, "rows": 0, "sig": None, "err": None}
    try:
        tables = _load(args.scale, list(_TPCH_TABLES))
        runner = _make_runner(args.engine, tables)
        res = runner(args.sql)
        tbl = res if isinstance(res, pa.Table) else pa.table(res)
        out["rows"] = tbl.num_rows
        out["sig"] = result_signature(tbl)
        runner(args.sql)  # warm up
        best = float("inf")
        for _ in range(args.runs):
            t0 = time.perf_counter()
            runner(args.sql)
            best = min(best, (time.perf_counter() - t0) * 1000.0)
        out["ms"] = best
    except Exception as exc:
        out["err"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out))


def _make_runner(engine: str, tables: dict[str, pa.Table]):
    """The engine's pre-registered `query -> pa.Table` callable, from the shared adapters.

    This used to reimplement four runners inline, which is how it drifted from the rest of
    the suite in two ways that mattered. Its Polars runner called `pl.SQLContext` directly
    with none of `engines/polars.py`'s dialect handling, so TPC-H q6 — where Polars folds
    `0.06 + 0.01` to the double below `0.07` and drops every `l_discount = 0.07` row — ran
    here with no rewrite and no note. And its DuckDB comment still claimed a registered-Arrow
    scan was "~100x slower on joins", a figure `engines/duckdb_arrow.py` retired: on DuckDB
    1.5.x it is ~1.5-3x, which is why `duckdb_arrow` exists as the like-for-like bar at all.

    Reusing the adapters means this harness measures the same engines, configured the same
    way, as every other benchmark here — which is the only thing that makes its numbers
    comparable to theirs.
    """
    from engines import get

    runner = get(engine).sql_runner(tables)
    if runner is None:
        raise ValueError(f"engine {engine!r} has no SQL surface")
    return runner


if __name__ == "__main__":
    sys.exit(main())
