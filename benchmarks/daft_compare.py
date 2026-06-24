"""Head-to-head: Batcher vs Daft (and DuckDB/Polars) on core relational queries.

Runs a curated set of single-node queries on the same synthetic fact/dim data,
checks every engine's result against DuckDB, and reports best-of-N wall-clock time
plus the Batcher/Daft ratio. Build the **release** engine first (`just build-release`)
or the numbers are meaningless (the dev build is unoptimized).

    python benchmarks/daft_compare.py [n_rows]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import daft
import duckdb
import polars as pl

import batcher as bt
from batcher import col, count
from contexts import make_synthetic_data
from harness import bench, results_match

dc = daft.col


def _cases(fact, dim):
    bf, bd = bt.from_arrow(fact), bt.from_arrow(dim)
    df_f, df_d = daft.from_arrow(fact), daft.from_arrow(dim)
    con = duckdb.connect()
    con.register("fact", fact)
    con.register("dim", dim)
    pf, pd_ = pl.from_arrow(fact), pl.from_arrow(dim)

    return {
        "groupby-agg": {
            "batcher": lambda: (
                bf.group_by("k1")
                .agg(s=col("price").sum(), n=count(), a=col("price").mean())
                .collect()
            ),
            "daft": lambda: (
                df_f.groupby("k1")
                .agg(
                    dc("price").sum().alias("s"),
                    dc("price").count().alias("n"),
                    dc("price").mean().alias("a"),
                )
                .to_arrow()
            ),
            "duckdb": lambda: con.sql(
                "SELECT k1, SUM(price) s, COUNT(*) n, AVG(price) a FROM fact GROUP BY k1"
            ).fetch_arrow_table(),
            "polars": lambda: (
                pf.group_by("k1")
                .agg(
                    pl.col("price").sum().alias("s"),
                    pl.len().alias("n"),
                    pl.col("price").mean().alias("a"),
                )
                .to_arrow()
            ),
        },
        "filter-project": {
            "batcher": lambda: (
                bf.filter(col("price") > 500.0).select("k1", p2=col("price") * 2.0).collect()
            ),
            "daft": lambda: (
                df_f.where(dc("price") > 500.0)
                .select(dc("k1"), (dc("price") * 2.0).alias("p2"))
                .to_arrow()
            ),
            "duckdb": lambda: con.sql(
                "SELECT k1, price*2 p2 FROM fact WHERE price > 500"
            ).fetch_arrow_table(),
            "polars": lambda: (
                pf.filter(pl.col("price") > 500.0)
                .select("k1", (pl.col("price") * 2.0).alias("p2"))
                .to_arrow()
            ),
        },
        "join": {
            "batcher": lambda: bf.join(bd, on="dim_key").select("price", "weight").collect(),
            "daft": lambda: (
                df_f.join(df_d, on="dim_key").select(dc("price"), dc("weight")).to_arrow()
            ),
            "duckdb": lambda: con.sql(
                "SELECT f.price, d.weight FROM fact f JOIN dim d USING (dim_key)"
            ).fetch_arrow_table(),
            "polars": lambda: pf.join(pd_, on="dim_key").select("price", "weight").to_arrow(),
        },
        "top-n": {
            "batcher": lambda: bf.sort("price", descending=True).limit(100).collect(),
            "daft": lambda: df_f.sort("price", desc=True).limit(100).to_arrow(),
            "duckdb": lambda: con.sql(
                "SELECT * FROM fact ORDER BY price DESC LIMIT 100"
            ).fetch_arrow_table(),
            "polars": lambda: pf.sort("price", descending=True).head(100).to_arrow(),
        },
        "distinct": {
            "batcher": lambda: bf.select("k1", "k2").distinct().collect(),
            "daft": lambda: df_f.select(dc("k1"), dc("k2")).distinct().to_arrow(),
            "duckdb": lambda: con.sql("SELECT DISTINCT k1, k2 FROM fact").fetch_arrow_table(),
            "polars": lambda: pf.select("k1", "k2").unique().to_arrow(),
        },
    }


def main() -> None:
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
    print(f"Daft {daft.__version__} vs Batcher — {n_rows:,} rows, best-of-5\n")
    fact, dim = make_synthetic_data(n_rows)
    cases = _cases(fact, dim)

    header = (
        f"{'query':<16}{'batcher':>10}{'daft':>10}{'duckdb':>10}{'polars':>10}{'b/daft':>9}  status"
    )
    print(header)
    print("-" * len(header))
    for name, engines in cases.items():
        ref = engines["duckdb"]()
        ms, status = {}, "ok"
        for eng, fn in engines.items():
            try:
                out = fn()
                ok, msg = results_match(ref, out)
                if not ok:
                    status = f"{eng}:WRONG({msg[:30]})"
                ms[eng] = bench(fn)
            except Exception as exc:
                status = f"{eng}:ERR({type(exc).__name__})"
                ms[eng] = float("nan")
        ratio = f"{ms['batcher'] / ms['daft']:.2f}x" if ms.get("daft") else "-"
        print(
            f"{name:<16}{ms['batcher']:>9.1f}{ms['daft']:>10.1f}{ms['duckdb']:>10.1f}"
            f"{ms['polars']:>10.1f}{ratio:>9}  {status}"
        )


if __name__ == "__main__":
    main()
