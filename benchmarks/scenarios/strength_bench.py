"""Strength-representative multi-engine benchmark: batcher vs Ray Data vs Daft.

The SQL suites (``tpch``) are the optimizer/compute story; this script is the
*data-engine* story — the workloads Ray Data and Daft are actually built and marketed
for, run on the engines' own idiomatic APIs on the live cluster:

- ``udf-map``  — a streaming per-batch **numpy UDF** + grouped reduce. Ray Data's
  signature pattern (``map_batches``); batcher mirrors it with ``map_batches`` and Daft
  with a columnar ``@daft.udf``.
- ``expr-etl`` — a multi-stage **expression pipeline** (derived columns → filter →
  group-by with two aggregates). Daft's lazy-optimized DataFrame strength; batcher runs
  the same as SQL; Ray Data does it the only way it can — numpy in ``map_batches``.
- ``top-n``    — ``ORDER BY … DESC LIMIT k``. A fused top-N heap (batcher) vs a full
  global sort (what Ray Data must do, and Daft).

Every workload is **correctness-gated first** (all engines must agree, as a sorted row
multiset within float tolerance) before any timing is trusted — the same discipline as
``harness.py``, reused here. Batcher runs single-node (its in-process strength); Ray
Data attaches to the existing cluster (its distributed home turf); Daft runs its
default multithreaded local executor.

Run:
    python benchmarks/strength_bench.py                 # all three engines
    python benchmarks/strength_bench.py --rows 6000000  # cap lineitem rows
"""

from __future__ import annotations

import argparse
import os
import sys

# Daft must use its native multithreaded local runner — its fast single-node engine,
# the fair counterpart to batcher single-node. Without this, importing Daft *after*
# ray.init makes it auto-select its Ray "flotilla" distributed runner, which needs a
# version-matched worker env this cluster doesn't provide (it times out). Set before
# Daft is imported anywhere.
os.environ.setdefault("DAFT_RUNNER", "native")
from collections.abc import Callable

import numpy as np
import pyarrow as pa

sys.path.insert(0, "benchmarks")
from harness import bench, results_match
from sources import load_tables


# --------------------------------------------------------------------------- #
# The numpy UDF shared (in spirit) across engines — a per-row transform that is
# cheap to state and identical everywhere, so the engines' results must match.
# `charge = sqrt(extendedprice^2 + (quantity*1000)^2) * (1 - discount)`.
# --------------------------------------------------------------------------- #
def _charge(ep: np.ndarray, qty: np.ndarray, disc: np.ndarray) -> np.ndarray:
    return np.sqrt(ep * ep + (qty * 1000.0) * (qty * 1000.0)) * (1.0 - disc)


# --------------------------------------------------------------------------- #
# batcher (single-node, in-process)
# --------------------------------------------------------------------------- #
def _batcher_engine(tables: dict[str, pa.Table]):
    import batcher as bt

    li = bt.from_arrow(tables["lineitem"])
    sess = bt.Session()
    sess.register("lineitem", tables["lineitem"])

    def udf_map() -> pa.Table:
        def fn(batch: pa.RecordBatch) -> pa.RecordBatch:
            ep = batch.column("l_extendedprice").to_numpy(zero_copy_only=False)
            qty = batch.column("l_quantity").to_numpy(zero_copy_only=False)
            disc = batch.column("l_discount").to_numpy(zero_copy_only=False)
            return pa.record_batch(
                {
                    "l_returnflag": batch.column("l_returnflag"),
                    "charge": pa.array(_charge(ep, qty, disc)),
                }
            )

        return (
            li.map_batches(fn, output_columns=["l_returnflag", "charge"])
            .group_by("l_returnflag")
            .agg(charge=bt.col("charge").sum())
            .collect(distributed=False)
        )

    def expr_etl() -> pa.Table:
        return sess.sql(
            "SELECT l_returnflag, "
            "SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS charge, "
            "AVG(l_quantity) AS avg_qty "
            "FROM lineitem "
            "WHERE l_extendedprice * (1 - l_discount) > 30000 "
            "GROUP BY l_returnflag"
        ).collect(distributed=False)

    def top_n() -> pa.Table:
        return (
            li.select("l_orderkey", "l_extendedprice")
            .sort("l_extendedprice", "l_orderkey", descending=True)
            .limit(20)
            .collect(distributed=False)
        )

    return {"udf-map": udf_map, "expr-etl": expr_etl, "top-n": top_n}


# --------------------------------------------------------------------------- #
# Ray Data (attached to the existing cluster — its distributed home turf)
# --------------------------------------------------------------------------- #
def _ray_engine(tables: dict[str, pa.Table]):
    import ray.data

    from engines.ray import _ensure_ray

    _ensure_ray()
    rd = ray.data.from_arrow(tables["lineitem"])

    def udf_map() -> pa.Table:
        def fn(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            return {
                "l_returnflag": batch["l_returnflag"],
                "charge": _charge(
                    batch["l_extendedprice"], batch["l_quantity"], batch["l_discount"]
                ),
            }

        out = rd.map_batches(fn, batch_format="numpy").groupby("l_returnflag").sum("charge")
        df = out.to_pandas().rename(columns={"sum(charge)": "charge"})
        return pa.Table.from_pandas(df[["l_returnflag", "charge"]], preserve_index=False)

    def expr_etl() -> pa.Table:
        def fn(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            ep, disc, tax, qty = (
                batch["l_extendedprice"],
                batch["l_discount"],
                batch["l_tax"],
                batch["l_quantity"].astype("float64"),
            )
            net = ep * (1 - disc)
            keep = net > 30000
            return {
                "l_returnflag": batch["l_returnflag"][keep],
                "charge": (net * (1 + tax))[keep],
                "qty": qty[keep],
            }

        agg = (
            rd.map_batches(fn, batch_format="numpy")
            .groupby("l_returnflag")
            .aggregate(
                ray.data.aggregate.Sum("charge"),
                ray.data.aggregate.Mean("qty"),
            )
        )
        df = agg.to_pandas().rename(columns={"sum(charge)": "charge", "mean(qty)": "avg_qty"})
        return pa.Table.from_pandas(df[["l_returnflag", "charge", "avg_qty"]], preserve_index=False)

    def top_n() -> pa.Table:
        out = (
            rd.select_columns(["l_orderkey", "l_extendedprice"])
            .sort(["l_extendedprice", "l_orderkey"], descending=True)
            .limit(20)
        )
        return pa.Table.from_pandas(out.to_pandas(), preserve_index=False)

    return {"udf-map": udf_map, "expr-etl": expr_etl, "top-n": top_n}


# --------------------------------------------------------------------------- #
# Daft (default multithreaded local executor)
# --------------------------------------------------------------------------- #
def _daft_engine(tables: dict[str, pa.Table]):
    import daft
    from daft import col

    ddf = daft.from_arrow(tables["lineitem"])

    @daft.udf(return_dtype=daft.DataType.float64())
    def charge_udf(ep, qty, disc):  # columnar batched UDF over arrow chunks
        return _charge(
            np.asarray(ep.to_arrow()), np.asarray(qty.to_arrow()), np.asarray(disc.to_arrow())
        )

    def udf_map() -> pa.Table:
        out = (
            ddf.with_column(
                "charge",
                charge_udf(col("l_extendedprice"), col("l_quantity"), col("l_discount")),
            )
            .groupby("l_returnflag")
            .agg(col("charge").sum().alias("charge"))
        )
        return out.to_arrow()

    def expr_etl() -> pa.Table:
        net = col("l_extendedprice") * (1 - col("l_discount"))
        out = (
            ddf.with_column("charge", net * (1 + col("l_tax")))
            .with_column("net", net)
            .where(col("net") > 30000)
            .groupby("l_returnflag")
            .agg(col("charge").sum().alias("charge"), col("l_quantity").mean().alias("avg_qty"))
            .select("l_returnflag", "charge", "avg_qty")
        )
        return out.to_arrow()

    def top_n() -> pa.Table:
        out = (
            ddf.select("l_orderkey", "l_extendedprice")
            .sort(["l_extendedprice", "l_orderkey"], desc=True)
            .limit(20)
        )
        return out.to_arrow()

    return {"udf-map": udf_map, "expr-etl": expr_etl, "top-n": top_n}


ENGINES: dict[str, Callable] = {
    "batcher": _batcher_engine,
    "ray": _ray_engine,
    "daft": _daft_engine,
}
WORKLOADS = ("udf-map", "expr-etl", "top-n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="batcher,ray,daft")
    ap.add_argument("--rows", type=int, default=None, help="cap lineitem rows (default: full sf1)")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    names = [n for n in args.engines.split(",") if n]
    tables = load_tables("tpch", 1.0, None)
    if args.rows is not None:
        tables = {k: v.slice(0, args.rows) for k, v in tables.items()}
    print(f"lineitem rows: {tables['lineitem'].num_rows:,}")

    built = {}
    for n in names:
        try:
            built[n] = ENGINES[n](tables)
            print(f"engine ready: {n}")
        except Exception as exc:
            print(f"engine {n} unavailable: {type(exc).__name__}: {exc}")

    results: dict[str, dict[str, float | str]] = {}
    for w in WORKLOADS:
        outputs: dict[str, pa.Table] = {}
        for n, fns in built.items():
            try:
                outputs[n] = fns[w]()
            except Exception as exc:
                results.setdefault(w, {})[n] = f"ERR:{type(exc).__name__}"
                print(f"[{w}] {n} error: {type(exc).__name__}: {exc}")
        # correctness gate vs the first engine that produced a result
        ref_name = next(iter(outputs), None)
        if ref_name is not None:
            ref = outputs[ref_name]
            for n, out in outputs.items():
                ok, msg = results_match(ref, out)
                if not ok:
                    results.setdefault(w, {})[n] = f"WRONG:{msg[:40]}"
                    print(f"[{w}] {n} MISMATCH vs {ref_name}: {msg}")
        # time only the engines that produced a correct result
        for n in outputs:
            if isinstance(results.get(w, {}).get(n), str):
                continue
            results.setdefault(w, {})[n] = bench(built[n][w], runs=args.runs)

    # report
    print()
    hdr = ["workload"] + [f"{n}_ms" for n in names]
    comps = [n for n in names if n != "batcher"]
    if "batcher" in names:
        hdr += [f"{c}/batcher" for c in comps]
    print("  ".join(h.rjust(12) for h in hdr))
    print("-" * (14 * len(hdr)))
    for w in WORKLOADS:
        row = [w]
        cells = results.get(w, {})
        for n in names:
            v = cells.get(n)
            row.append(f"{v:.1f}" if isinstance(v, float) else str(v or "n/a"))
        if "batcher" in names:
            b = cells.get("batcher")
            for c in comps:
                cv = cells.get(c)
                if isinstance(b, float) and isinstance(cv, float) and b > 0:
                    row.append(f"{cv / b:.1f}x")
                else:
                    row.append("-")
        print("  ".join(str(c).rjust(12) for c in row))


if __name__ == "__main__":
    main()
