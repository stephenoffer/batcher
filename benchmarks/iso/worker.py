"""Isolated single (engine, query) worker: fresh process, mmap tables, time best-of-N.

Prints one JSON line: {"ms": float|null, "rows": int, "sig": [...], "err": str|null}.
Run via ``iso_run.py`` which spawns one of these per (engine, query) so no cross-query
process state can inflate any engine's timing (the fair, official-benchmark method).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pyarrow as pa
import pyarrow.feather as feather


def _load(scale: int, names: list[str]) -> dict[str, pa.Table]:
    base = f"/home/ray/tpch_feather/sf{scale}"
    return {n: feather.read_table(f"{base}/{n}.feather", memory_map=True) for n in names}


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


def _signature(tbl: pa.Table) -> list:
    """A small order-independent fingerprint of a result for cross-engine equality."""
    import math
    from decimal import Decimal

    cols = sorted(tbl.column_names)
    tbl = tbl.select(cols)
    n = tbl.num_rows
    pd = tbl.to_pydict()
    rows = []
    for i in range(n):
        r = []
        for c in cols:
            v = pd[c][i]
            if isinstance(v, Decimal):
                v = float(v)
            if hasattr(v, "isoformat"):
                v = v.isoformat()
            if isinstance(v, float):
                v = "nan" if math.isnan(v) else round(v, 4)
            r.append(v)
        rows.append(tuple(r))
    rows.sort(key=lambda r: tuple(repr(x) for x in r))
    return [n, rows[:5], rows[-5:] if n > 5 else []]


def main() -> None:
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
        out["sig"] = _signature(tbl)
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
    if engine == "batcher":
        import batcher as bt

        s = bt.Session()
        for n, tb in tables.items():
            s.register(n, tb)
        return lambda q: s.sql(q).collect(distributed=False)
    if engine == "duckdb":
        import duckdb

        con = duckdb.connect()
        # Load into DuckDB's NATIVE columnar storage (how DuckDB is actually used in
        # every official TPC-H/ClickBench result). Registering the Arrow table instead
        # forces DuckDB's slow zero-copy Arrow scan (~100x slower on joins) — an unfair
        # handicap. Ingestion is one-time and not timed, exactly as Batcher's Arrow is
        # already in memory and not timed.
        for n, tb in tables.items():
            con.register(f"_arrow_{n}", tb)
            con.execute(f"CREATE TABLE {n} AS SELECT * FROM _arrow_{n}")
            con.unregister(f"_arrow_{n}")
        return lambda q: con.sql(q).to_arrow_table()
    if engine == "duckdb_arrow":
        import duckdb

        con = duckdb.connect()
        for n, tb in tables.items():
            con.register(n, tb)
        return lambda q: con.sql(q).to_arrow_table()
    if engine == "polars":
        import polars as pl

        frames = {n: pl.from_arrow(tb) for n, tb in tables.items()}
        ctx = pl.SQLContext(frames=frames, eager=True)
        return lambda q: ctx.execute(q).to_arrow()
    raise ValueError(f"unknown engine {engine}")


if __name__ == "__main__":
    sys.exit(main())
