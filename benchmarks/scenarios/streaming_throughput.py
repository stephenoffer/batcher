"""Streaming benchmark: Batcher vs Spark Structured Streaming (+ DuckDB/Polars floor).

Batcher and Spark are the two real *structured-streaming* engines here — DuckDB and
Polars have no streaming query engine, so they appear only as a **batch floor** (the
same aggregation run once, the fastest a non-streaming engine could do it). The
streaming runs use the drain trigger both engines support (Spark `Trigger.AvailableNow`,
Batcher `Trigger.available_now()`): read a partitioned Parquet backlog as a stream,
fold a grouped aggregation, emit the final result. Wall time → rows/second throughput.

Correctness is gated **before** any clock starts: every engine answers once, untimed, and
only the ones matching the oracle (DuckDB where present, never an engine under test) are
then timed. A mismatched engine is reported and not timed, and the Batcher-vs-Spark headline
is printed only when both sides passed. That untimed pass doubles as the warm-up.

Run:
    python benchmarks/scenarios/streaming_throughput.py                 # 4M rows
    python benchmarks/scenarios/streaming_throughput.py --rows 20000000
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import batcher as bt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envinfo import require_quiet_box, require_release_build

_KEYS = 1000  # grouping cardinality


def _write_backlog(path: str, rows: int, files: int) -> tuple[pa.Schema, float]:
    """Write `rows` across `files` Parquet files, and return the schema and the true sum of `x`.

    The total is the positive control the correctness gate needs. Comparing engines to each
    other proves they agree; it does not prove they *did the work*, and two engines that
    each drained nothing agree perfectly — the gate compares `{key: sum}` maps and two empty
    maps are equal. An empty or half-written backlog then reports every engine correct, at
    a throughput computed from the row count that was *requested*. Holding each result
    against the total actually written is what makes the agreement mean something.
    """
    rng = np.random.default_rng(0)
    per = -(-rows // files)
    schema = pa.schema([("k", pa.int64()), ("x", pa.float64())])
    written, total = 0, 0.0
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
        total += float(pc.sum(tbl.column("x")).as_py())
        written += n
    return schema, total


def _canon(table: pa.Table) -> dict[int, float]:
    """{key: sum(x)} rounded — the order-independent comparison key across engines."""
    d = table.to_pydict()
    cols = {n.lower(): c for n, c in zip(table.column_names, d.values(), strict=True)}
    ks = cols["k"]
    sums = cols.get("s") or cols.get("sum(x)") or cols.get("sum_x") or cols["x"]
    return {int(k): round(float(s), 3) for k, s in zip(ks, sums, strict=True)}


#: Distinguishes each Batcher memory sink, so no two runs share one.
_SINK_SEQ = itertools.count()


def _batcher(path: str) -> dict[int, float]:
    """One AvailableNow drain of the backlog into a **fresh** memory sink.

    The sink name used to be the constant `"bt_agg"` for every invocation, while Spark was
    handed a fresh `checkpointLocation` per call (`ckpt-{time.time_ns()}`). That asymmetry
    is the kind that decides a benchmark: the reported figure is the *minimum* of N runs,
    so any per-query-name state that let run 2 or 3 skip already-processed files would
    become the published number, and the comparison would be one engine re-reading the
    backlog against another engine not.

    Whether that state exists is beside the point — the two engines now start from the
    same place on every repeat, so the question cannot arise and nobody has to re-derive
    the answer from the streaming internals to trust the number.
    """
    sink = f"bt_agg_{next(_SINK_SEQ)}"
    q = (
        bt.read(path, format="parquet")
        .group_by("k")
        .agg(s=bt.col("x").sum())
        .write.memory(sink, trigger=bt.Trigger.available_now(), output_mode="complete")
    )
    q.await_termination()
    return _canon(bt.read_memory(sink).collect())


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


def _drained_everything(result: dict[int, float], expected_total: float) -> bool:
    """Whether `result` accounts for the whole backlog, not merely for the same rows twice.

    Engine-to-engine agreement cannot see a drain that covered nothing: two empty results
    are equal. This holds each result against the sum of `x` actually written, so an empty
    or partial drain fails rather than passing as a very fast one.
    """
    if not result:
        return False
    got = sum(result.values())
    return abs(got - expected_total) <= max(1e-6, abs(expected_total) * 1e-9)


def _gate_then_time(
    runners: dict[str, object], runs: int, expected_total: float
) -> tuple[dict[str, bool] | None, dict[str, float]]:
    """Check every engine's answer, then time only the ones that agreed.

    The order is the whole point, and it used to be the other way round: `_best_ms` ran
    over every engine and the gate was built afterwards, so a MISMATCH row still printed
    its ms and its rows/sec, and the Batcher-vs-Spark summary divided two of them without
    consulting the gate at all. The module docstring claimed the opposite. One untimed pass
    per engine settles correctness before any clock starts, which is what `harness.compare`
    does and what this script already promised.

    The untimed pass doubles as the warm-up `_best_ms` never had, so best-of-N now measures
    a warm engine the way the rest of the suite does.

    Args:
        runners: Engine label mapped to a zero-argument callable returning `{key: sum}`.
        runs: Repeats for the best-of-N timing.

    Returns:
        `(gate, times)`, where `gate` maps each engine to whether it matched the oracle.
        `gate` is `None` when no engine produced a result at all.
    """
    results: dict[str, dict[int, float]] = {}
    times: dict[str, float] = {}
    for name, fn in runners.items():
        try:
            results[name] = fn()
        except Exception as exc:
            times[name] = -1.0
            print(f"  ({name} error: {str(exc)[:70]})")

    # DuckDB is the oracle where present — never an engine under test. The fallback can
    # only pick Batcher or Spark, so it says so rather than quietly self-certifying.
    ref_name = "duckdb (batch)" if "duckdb (batch)" in results else next(iter(results), None)
    if ref_name is None:
        print("  (no engine produced a result)")
        return None, times
    if ref_name != "duckdb (batch)":
        print(f"  !! no independent oracle; checking against {ref_name!r}, an engine under test")
    ref = results[ref_name]
    gate = {n: (r == ref and _drained_everything(r, expected_total)) for n, r in results.items()}
    for n, r in results.items():
        if r == ref and not _drained_everything(r, expected_total):
            print(f"  ({n} agreed with {ref_name} but did not drain the backlog — not timed)")

    for name in results:
        if not gate[name]:
            print(f"  ({name} MISMATCH vs {ref_name} — not timed)")
            continue
        try:
            times[name], _ = _best_ms(runners[name], runs)
        except Exception as exc:
            times[name] = -1.0
            print(f"  ({name} error while timing: {str(exc)[:70]})")
    return gate, times


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming throughput benchmark")
    parser.add_argument("--rows", type=int, default=4_000_000)
    parser.add_argument("--files", type=int, default=16)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    # A dev build is 8-60x slower and a contended box measures the neighbour; neither
    # produces a throughput number. `BENCH_ALLOW_DEBUG_BUILD=1` / `BENCH_ALLOW_BUSY_BOX=1`
    # override. Single-node, so unlike the cluster benchmarks the local run queue *is* the
    # contention signal that matters here.
    require_release_build()
    require_quiet_box()

    tmp = tempfile.mkdtemp()
    data = f"{tmp}/backlog"
    import os

    os.makedirs(data, exist_ok=True)
    _, expected_total = _write_backlog(data, args.rows, args.files)

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

    gate, times = _gate_then_time(runners, args.runs, expected_total)
    if gate is None:
        return 1

    # `rows/sec` divides `args.rows` by the elapsed time, which is only meaningful because
    # the gate above proved each engine actually folded the whole backlog — see
    # `_drained_everything`. Without that check the numerator is a constant from the command
    # line and an engine that drained nothing still reports full throughput.
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

    # The headline is only printed when *both* sides passed the gate. Previously it was
    # computed from `times` alone and printed regardless, so a MISMATCH could be reported
    # as "Batcher streaming is Nx vs Spark".
    both_gated = gate.get("batcher (stream)") and gate.get("spark (stream)")
    if both_gated and times.get("spark (stream)", 0) > 0 and times.get("batcher (stream)", 0) > 0:
        ratio = times["spark (stream)"] / times["batcher (stream)"]
        print(f"\n  Batcher streaming is {ratio:.1f}x vs Spark Structured Streaming (drain).")
    elif "spark (stream)" in gate:
        print("\n  (no Batcher/Spark ratio: one of the two did not pass the correctness gate.)")
    print("\n(streaming = AvailableNow drain of a Parquet backlog; batch = one-shot floor.)")

    if spark is not None:
        spark.stop()
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
