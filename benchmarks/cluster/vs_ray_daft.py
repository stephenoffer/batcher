"""Batcher vs Ray Data vs Daft on a live Ray cluster — distributed pipeline race.

Each engine reads the same public TPC-H parquet *directly from S3* (no driver-side
materialization — the distributed read is part of the work) and runs the same
pipeline, simple → complex. For every pipeline we report wall time per engine, the
Batcher/competitor speedup, a result signature (so a divergence is caught, not
hidden), and — for Batcher and Ray Data — how much of the cluster the run actually
used (mean/peak node-CPU busy%, active node count) via ``cluster_util``.

Run (from the conda env that carries ray/daft, with batcher installed):
    python benchmarks/vs_ray_daft.py            # sf10, pipelines 1..5
    python benchmarks/vs_ray_daft.py 100        # sf100
    BENCH_PIPES=groupby,join python benchmarks/vs_ray_daft.py 10
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import time

# Unbuffered, line-flushed prints so a long distributed run streams progress.
print = functools.partial(print, flush=True)

# TPC-H lineitem / orders are stored with positional column names; alias them here so
# the pipelines read like SQL without a per-engine rename op muddying the timing.
L_ORDERKEY, L_QTY, L_EXTPRICE, L_DISCOUNT = "column00", "column04", "column05", "column06"
L_RETURNFLAG, L_LINESTATUS = "column08", "column09"
# orders columns are NOT zero-padded (column0..column9), unlike lineitem (column00..16)
O_ORDERKEY, O_ORDERPRIORITY = "column0", "column5"


def tpch_uri(scale: int, table: str, glob: bool = True) -> str:
    base = os.environ.get("BENCH_TPCH_BASE", "s3://ray-benchmark-data/tpch/parquet")
    d = f"{base}/sf{scale}/{table}"
    # Ray Data wants a directory; Batcher/Daft accept (and prefer) an explicit glob.
    return f"{d}/*.parquet" if glob else f"{d}/"


# --------------------------------------------------------------------------- #
# Batcher pipelines (100% batcher API)
# --------------------------------------------------------------------------- #
def batcher_thunk(name: str, scale: int):
    import batcher as bt
    from batcher import col, count

    li = bt.read.parquet(tpch_uri(scale, "lineitem"))
    if name == "scan_count":
        ds = li.agg(n=count())
    elif name == "filter_count":
        ds = li.filter(col(L_QTY) > 30).agg(n=count())
    elif name == "groupby":
        ds = li.group_by(L_RETURNFLAG, L_LINESTATUS).agg(
            s=col(L_EXTPRICE).sum(), q=col(L_QTY).sum(), n=count()
        )
    elif name == "join":
        orders = bt.read.parquet(tpch_uri(scale, "orders")).rename({O_ORDERKEY: L_ORDERKEY})
        ds = (
            li.join(orders, on=L_ORDERKEY, how="inner")
            .group_by(O_ORDERPRIORITY)
            .agg(s=col(L_EXTPRICE).sum(), n=count())
        )
    elif name == "udf":
        # output_columns declares the UDF's new schema (column05 → score) so the agg
        # above can reference it — the control plane never runs the opaque UDF to infer it.
        ds = (
            li.select(L_EXTPRICE)
            .map_batches(_heavy_batch, output_columns=["score"])
            .agg(s=col("score").sum())
        )
    else:
        raise ValueError(name)

    def run():
        return _sig_arrow(ds.collect(distributed=True))

    return run


# --------------------------------------------------------------------------- #
# Ray Data pipelines
# --------------------------------------------------------------------------- #
def ray_thunk(name: str, scale: int):
    import ray.data as rd
    from ray.data.aggregate import Count, Sum

    li = rd.read_parquet(tpch_uri(scale, "lineitem", glob=False))
    if name == "scan_count":

        def run():
            return {"rows": 1, "checksum": float(li.count())}

    elif name == "filter_count":
        f = li.filter(expr=f"{L_QTY} > 30")

        def run():
            return {"rows": 1, "checksum": float(f.count())}

    elif name == "groupby":
        g = li.groupby([L_RETURNFLAG, L_LINESTATUS]).aggregate(Sum(L_EXTPRICE), Sum(L_QTY), Count())

        def run():
            return _sig_rows(g.take_all(), f"sum({L_EXTPRICE})")

    elif name == "join":
        orders = rd.read_parquet(tpch_uri(scale, "orders", glob=False)).rename_columns(
            {O_ORDERKEY: L_ORDERKEY}
        )
        j = li.join(orders, join_type="inner", on=(L_ORDERKEY,), num_partitions=128)
        g = j.groupby([O_ORDERPRIORITY]).aggregate(Sum(L_EXTPRICE), Count())

        def run():
            return _sig_rows(g.take_all(), f"sum({L_EXTPRICE})")

    elif name == "udf":
        m = li.select_columns([L_EXTPRICE]).map_batches(_heavy_batch, batch_format="pyarrow")

        def run():
            # ray.data aggregate over the whole dataset returns a dict, not a Dataset.
            d = m.aggregate(Sum("score"))
            return {"rows": 1, "checksum": round(float(d["sum(score)"]), 2)}

    else:
        raise ValueError(name)

    return run


# --------------------------------------------------------------------------- #
# Daft pipelines
# --------------------------------------------------------------------------- #
def daft_thunk(name: str, scale: int):
    import daft

    li = daft.read_parquet(tpch_uri(scale, "lineitem"))
    if name == "scan_count":

        def run():
            return {"rows": 1, "checksum": float(li.count_rows())}

    elif name == "filter_count":
        f = li.where(daft.col(L_QTY) > 30)

        def run():
            return {"rows": 1, "checksum": float(f.count_rows())}

    elif name == "groupby":
        g = li.groupby(L_RETURNFLAG, L_LINESTATUS).agg(
            daft.col(L_EXTPRICE).sum().alias("s"),
            daft.col(L_QTY).sum().alias("q"),
            daft.col(L_RETURNFLAG).count().alias("n"),
        )

        def run():
            return _sig_arrow(g.to_arrow(), sumcol="s")

    elif name == "join":
        orders = daft.read_parquet(tpch_uri(scale, "orders")).select(
            daft.col(O_ORDERKEY).alias(L_ORDERKEY), daft.col(O_ORDERPRIORITY)
        )
        j = li.join(orders, on=L_ORDERKEY, how="inner")
        g = j.groupby(O_ORDERPRIORITY).agg(
            daft.col(L_EXTPRICE).sum().alias("s"), daft.col(O_ORDERPRIORITY).count().alias("n")
        )

        def run():
            return _sig_arrow(g.to_arrow(), sumcol="s")

    elif name == "udf":
        return None  # daft UDF surface diverges; covered by batcher vs ray for udf
    else:
        raise ValueError(name)

    return run


# --------------------------------------------------------------------------- #
# Shared UDF + signature helpers
# --------------------------------------------------------------------------- #
def _heavy_batch(batch):
    """A CPU-heavy per-batch transform (ML/UDF proxy): several transcendental passes.

    Takes a pyarrow batch/table with the extendedprice column, returns one ``score``
    column. Identical math in every engine so the result signature matches.
    """
    import numpy as np
    import pyarrow as pa

    col = batch.column(0).to_numpy(zero_copy_only=False).astype("float64")
    x = col
    for _ in range(20):
        x = np.sqrt(np.abs(x) + 1.0) + np.log1p(np.abs(x))
    return pa.table({"score": pa.array(x)})


def _sig_arrow(tbl, sumcol: str | None = None) -> dict:
    d = tbl.to_pydict()
    rows = tbl.num_rows
    cols = [sumcol] if (sumcol and sumcol in d) else list(d)
    chk = 0.0
    for c in cols:  # checksum = sum of the first numeric column (skips string keys)
        vals = [v for v in d[c] if isinstance(v, (int, float))]
        if vals and len(vals) == len([v for v in d[c] if v is not None]):
            chk = float(sum(vals))
            break
    return {"rows": rows, "checksum": round(chk, 2)}


def _sig_rows(rows: list[dict], sumkey: str) -> dict:
    chk = float(sum(r.get(sumkey, 0) or 0 for r in rows))
    return {"rows": len(rows), "checksum": round(chk, 2)}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
ENGINES = {"batcher": batcher_thunk, "ray": ray_thunk, "daft": daft_thunk}
PIPELINES = ["scan_count", "filter_count", "groupby", "join", "udf"]


def _time_run(run, monitor=None) -> tuple[float, dict]:
    if monitor:
        monitor.start()
    t0 = time.perf_counter()
    sig = run()
    dt = time.perf_counter() - t0
    util = monitor.stop() if monitor else {}
    return dt, {"sig": sig, "util": util}


def _with_timeout(fn, timeout_s: float):
    """Wrap `fn` so each call raises `TimeoutError` if it runs past `timeout_s`.

    Runs the call on a **daemon** thread and waits up to `timeout_s`. A timed-out
    call's thread is abandoned but, being a daemon, never keeps the process alive — so
    a pathological engine (e.g. Ray Data's distributed join) cannot leave a zombie
    driver holding cluster actors after the sweep ends. (A plain ThreadPoolExecutor
    uses non-daemon threads, which did exactly that.)
    """
    import threading

    def wrapped():
        box: dict = {}

        def run():
            try:
                box["v"] = fn()
            except BaseException as e:
                box["e"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            raise TimeoutError
        if "e" in box:
            raise box["e"]
        return box.get("v")

    return wrapped


def bench_pipeline(name: str, scale: int, runs: int) -> dict:
    from cluster_util import ClusterMonitor

    out: dict = {}
    for eng, builder in ENGINES.items():
        thunk = builder(name, scale)
        if thunk is None:
            continue
        # Per-engine wall-clock cap so one pathological engine (e.g. Ray Data's join)
        # is recorded as a timeout instead of hanging the whole sweep.
        timeout_s = float(os.environ.get("BENCH_ENGINE_TIMEOUT", "180"))
        try:
            run = _with_timeout(thunk, timeout_s)
            print(f"  [{name}/{eng}] warmup ...")
            # warmup (untimed): pays cold scheduling / shipping / planning.
            warm_sig = run()
            # timed best-of-N; utilization sampled on the first timed run.
            best = float("inf")
            util = {}
            for i in range(runs):
                mon = ClusterMonitor() if (eng in ("batcher", "ray") and i == 0) else None
                dt, meta = _time_run(run, mon)
                if mon:
                    mon.shutdown()
                    util = meta["util"]
                best = min(best, dt)
            out[eng] = {"ms": best * 1000, "sig": warm_sig, "util": util}
        except TimeoutError:
            out[eng] = {"error": f"TIMEOUT (>{timeout_s:.0f}s)"}
        except Exception as e:
            out[eng] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            # Free batcher's warm session fleet so the next engine gets the whole
            # cluster (fair comparison); a no-op for ray/daft.
            if eng == "batcher":
                with contextlib.suppress(Exception):
                    from batcher.dist.fleet import release_session_fleet

                    release_session_fleet()
    return out


def _fmt_util(u: dict) -> str:
    if not u:
        return "-"
    mean, peak = u.get("mean_busy_pct", 0), u.get("peak_busy_pct", 0)
    nodes = f"{int(u.get('active_nodes', 0))}/{int(u.get('total_nodes', 0))}n"
    return f"{mean:.0f}%/{peak:.0f}%peak {nodes}"


def main() -> int:
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    runs = int(os.environ.get("BENCH_RUNS", "2"))
    pipes = os.environ.get("BENCH_PIPES")
    pipelines = pipes.split(",") if pipes else PIPELINES

    import ray

    if not ray.is_initialized():
        os.environ.setdefault("RAY_ADDRESS", "auto")
        ray.init(address="auto", logging_level="ERROR", log_to_driver=False)
    print(f"cluster: {ray.cluster_resources().get('CPU')} CPU, {len(ray.nodes())} nodes")
    print(f"TPC-H sf{scale}, best-of-{runs}\n")

    h = ("pipeline", "batcher_ms", "ray_ms", "daft_ms", "vs_ray", "vs_daft")
    w = (14, 12, 11, 11, 9, 9)
    cols = "".join(c.ljust(w[0]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(h))
    hdr = f"{cols}  util(batcher | ray)"
    print(hdr)
    print("-" * len(hdr))
    for name in pipelines:
        res = bench_pipeline(name, scale, runs)

        def ms(e, res=res):
            return res.get(e, {}).get("ms")

        bm, rm, dm = ms("batcher"), ms("ray"), ms("daft")

        def cell(v):
            return f"{v:.0f}" if isinstance(v, (int, float)) else "ERR"

        vs_ray = f"{rm / bm:.2f}x" if bm and rm else "-"
        vs_daft = f"{dm / bm:.2f}x" if bm and dm else "-"
        bu = _fmt_util(res.get("batcher", {}).get("util", {}))
        ru = _fmt_util(res.get("ray", {}).get("util", {}))
        cells = f"{name:<14}{cell(bm):>12}{cell(rm):>11}{cell(dm):>11}{vs_ray:>9}{vs_daft:>9}"
        print(f"{cells}  {bu} | {ru}")
        # surface signatures + errors below the row so divergence/failures are visible.
        sigs = {e: res[e].get("sig") for e in res if "sig" in res[e]}
        errs = {e: res[e]["error"] for e in res if "error" in res[e]}
        if len({(s["rows"], s["checksum"]) for s in sigs.values()}) > 1:
            print(f"    !! signature mismatch: {sigs}")
        for e, msg in errs.items():
            print(f"    !! {e}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
