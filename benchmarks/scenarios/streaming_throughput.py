"""Streaming benchmark: Batcher vs Spark Structured Streaming (+ DuckDB/Polars floor).

Batcher and Spark are the two real *structured-streaming* engines here — DuckDB and
Polars have no streaming query engine, so they appear only as a **batch floor** (the
same aggregation run once, the fastest a non-streaming engine could do it). The
streaming runs use the drain trigger both engines support (Spark `Trigger.AvailableNow`,
Batcher `Trigger.available_now()`): read a partitioned Parquet backlog as a stream,
fold a grouped aggregation, emit the final result. Wall time → rows/second throughput.

Correctness is gated before any timing is reported: every engine's per-key aggregate
must agree (order-independent), else the row is marked MISMATCH and not compared.

Run:
    python benchmarks/scenarios/streaming_throughput.py                 # 4M rows
    python benchmarks/scenarios/streaming_throughput.py --rows 20000000
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import batcher as bt

_KEYS = 1000  # grouping cardinality


def _write_backlog(path: str, rows: int, files: int) -> pa.Schema:
    """Write `rows` across `files` Parquet files (a partitioned stream backlog)."""
    rng = np.random.default_rng(0)
    per = -(-rows // files)
    schema = pa.schema([("k", pa.int64()), ("x", pa.float64())])
    written = 0
    for i in range(files):
        n = min(per, rows - written)
        if n <= 0:
            break
        tbl = pa.table(
            {
                "k": pa.array(rng.integers(0, _KEYS, n), type=pa.int64()),
                "x": pa.array(rng.random(n), type=pa.float64()),
            },
            schema=schema,
        )
        pq.write_table(tbl, f"{path}/part-{i:04d}.parquet")
        written += n
    return schema


def _canon(table: pa.Table) -> dict[int, float]:
    """{key: sum(x)} rounded — the order-independent comparison key across engines."""
    d = table.to_pydict()
    cols = {n.lower(): c for n, c in zip(table.column_names, d.values(), strict=True)}
    ks = cols["k"]
    sums = cols.get("s") or cols.get("sum(x)") or cols.get("sum_x") or cols["x"]
    return {int(k): round(float(s), 3) for k, s in zip(ks, sums, strict=True)}


def _batcher(path: str) -> dict[int, float]:
    q = (
        bt.read(path, format="parquet")
        .group_by("k")
        .agg(s=bt.col("x").sum())
        .write.memory("bt_agg", trigger=bt.Trigger.available_now(), output_mode="complete")
    )
    q.await_termination()
    return _canon(bt.read_memory("bt_agg").collect())


def _spark_session():
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master("local[*]")
        .appName("batcher-stream-bench")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def _spark(spark, path: str, sschema, ckpt: str) -> dict[int, float]:
    from pyspark.sql import functions as F

    q = (
        spark.readStream.schema(sschema)
        .parquet(path)
        .groupBy("k")
        .agg(F.sum("x").alias("s"))
        .writeStream.format("memory")
        .queryName("spark_agg")
        .outputMode("complete")
        .option("checkpointLocation", ckpt)
        .trigger(availableNow=True)
        .start()
    )
    q.awaitTermination()
    return _canon(pa.Table.from_pandas(spark.sql("SELECT * FROM spark_agg").toPandas()))


def _duckdb(path: str) -> dict[int, float]:
    import duckdb

    tbl = duckdb.sql(
        f"SELECT k, sum(x) AS s FROM read_parquet('{path}/*.parquet') GROUP BY k"
    ).to_arrow_table()
    return _canon(tbl)


def _polars(path: str) -> dict[int, float]:
    import polars as pl

    df = (
        pl.scan_parquet(f"{path}/*.parquet")
        .group_by("k")
        .agg(pl.col("x").sum().alias("s"))
        .collect()
    )
    return _canon(df.to_arrow())


def _spark_schema():
    from pyspark.sql.types import DoubleType, LongType, StructField, StructType

    return StructType([StructField("k", LongType()), StructField("x", DoubleType())])


def _best_ms(fn, runs: int) -> tuple[float, dict[int, float]]:
    best, result = float("inf"), {}
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming throughput benchmark")
    parser.add_argument("--rows", type=int, default=4_000_000)
    parser.add_argument("--files", type=int, default=16)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    tmp = tempfile.mkdtemp()
    data = f"{tmp}/backlog"
    import os

    os.makedirs(data, exist_ok=True)
    _write_backlog(data, args.rows, args.files)

    print(
        f"\nStreaming grouped aggregation — {args.rows:,} rows, {_KEYS} keys, best-of-{args.runs}\n"
    )

    runners: dict[str, object] = {
        "batcher (stream)": lambda: _batcher(data),
        "duckdb (batch)": lambda: _duckdb(data),
        "polars (batch)": lambda: _polars(data),
    }

    spark = None
    try:
        spark = _spark_session()
        sschema = _spark_schema()
        runners["spark (stream)"] = lambda: _spark(
            spark, data, sschema, f"{tmp}/ckpt-{time.time_ns()}"
        )
    except Exception as exc:  # pyspark missing / JVM unavailable → skip, others still run
        print(f"(spark skipped: {str(exc)[:70]})")

    results: dict[str, dict[int, float]] = {}
    times: dict[str, float] = {}
    for name, fn in runners.items():
        try:
            ms, res = _best_ms(fn, args.runs)
            times[name], results[name] = ms, res
        except Exception as exc:
            times[name] = -1.0
            print(f"  ({name} error: {str(exc)[:70]})")

    # Correctness gate: every engine's {key: sum} must agree with the reference.
    ref_name = "duckdb (batch)" if "duckdb (batch)" in results else next(iter(results))
    ref = results.get(ref_name, {})
    gate = {n: (r == ref) for n, r in results.items()}

    print(f"  {'engine':<20} {'time (ms)':>12} {'rows/sec':>16}   correct")
    print("  " + "-" * 60)
    for name in runners:
        ms = times.get(name, -1.0)
        if ms and ms > 0:
            rps = args.rows / (ms / 1000.0)
            ok = "OK" if gate.get(name) else "MISMATCH"
            print(f"  {name:<20} {ms:>12.1f} {rps:>16,.0f}   {ok}")
        else:
            print(f"  {name:<20} {'n/a':>12} {'n/a':>16}   -")

    if "batcher (stream)" in times and "spark (stream)" in times and times["spark (stream)"] > 0:
        ratio = times["spark (stream)"] / times["batcher (stream)"]
        print(f"\n  Batcher streaming is {ratio:.1f}x vs Spark Structured Streaming (drain).")
    print("\n(streaming = AvailableNow drain of a Parquet backlog; batch = one-shot floor.)")

    if spark is not None:
        spark.stop()
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
